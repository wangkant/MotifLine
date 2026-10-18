#!/usr/bin/env python
"""AlphaGenome DeepLIFT/SHAP hypothetical attributions.

Scalar: log1p of the ATAC track summed over the central 1000 bp. References: local
dinucleotide shuffles (n_shuffles, seeded by random_state), batch size 1.

A peak whose kept window (central `crop` bp) contains a non-ACGT base is dropped, as on
the ChromBPNet path. Non-ACGT bases outside the kept window are set to A so the input is
valid one-hot; those peaks are listed in n_filled.tsv.

tangermeme's convergence check runs at its default threshold and is recorded in
convergence.json rather than switched off.

Regions are processed in checkpointed chunks. Resuming onto chunks computed from other
inputs is refused (chunk_identity.json).

Runs in the compute env and must not import motifline (digests is a sibling file).
"""
import argparse
import hashlib
import json
import os
import sys

from convergence import THRESHOLD, Deltas, summary
from digests import file_sha256, git_commit, tree_sha256


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag_src", required=True,
                    help="src/ of the alphagenome-pytorch deeplift-support checkout")
    ap.add_argument("--weights", required=True, help="AlphaGenome checkpoint")
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--chrom_sizes", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--track", type=int, required=True, help="index into the ATAC head")
    ap.add_argument("--biosample", required=True,
                    help="biosample_name of that track in AlphaGenome's metadata, e.g. K562")
    ap.add_argument("--in_window", type=int, default=2048, help="model input, a multiple of 2048")
    ap.add_argument("--crop", type=int, default=2048, help="central width kept")
    ap.add_argument("--n_shuffles", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=100, help="regions per checkpoint")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--head", type=int, default=None, help="first N peaks only")
    ap.add_argument("--hypothetical", action="store_true", help="required")
    ap.add_argument("--n_shards", type=int, default=None)
    ap.add_argument("--shard_i", type=int, default=None)
    a = ap.parse_args()
    if not a.hypothetical:
        ap.error("--hypothetical is required; only hypothetical attributions are implemented")
    if a.crop > a.in_window:
        ap.error(f"--crop {a.crop} exceeds --in_window {a.in_window}")
    if a.in_window % 2048:
        ap.error(f"--in_window {a.in_window} is not a multiple of 2048")
    if (a.shard_i is None) != (a.n_shards is None):
        ap.error("--shard_i and --n_shards must be given together")
    if a.n_shards is not None:
        if a.head is not None:
            ap.error("--head and --shard_i/--n_shards are mutually exclusive")
        if not 0 <= a.shard_i < a.n_shards:
            ap.error(f"--shard_i must be in [0, {a.n_shards})")
    return a


def shard_bounds(n_rows, n_shards):
    import numpy as np
    return np.linspace(0, n_rows, n_shards + 1).astype(int)


def crop_center(arr, w):
    L = arr.shape[-1]; s = L // 2 - w // 2
    return arr[..., s:s + w]


def clean_in_crop(onehot_4xl, crop):
    """True when the central `crop` columns are all ACGT (no all-zero column)."""
    return not crop_center(onehot_4xl.sum(0) < 0.5, crop).any()


def fill_with_a(x):
    """Set the non-ACGT columns of a (1, 4, L) one-hot to A in place; return how many."""
    bad = x[0].sum(0) < 0.5
    n = int(bad.sum())
    if n:
        x[0, :, bad] = 0.0
        x[0, 0, bad] = 1.0
    return n


_NARROWPEAK_FILLER = ["."] * 6


def narrowpeak_text(rows):
    """(chr, start, end, summit) -> 11-column narrowPeak; finemo reads the summit from
    column 10. Same text as motifline.adopt.normalize_peaks (pinned by a test)."""
    return "".join("\t".join([str(c), str(s), str(e), *_NARROWPEAK_FILLER, str(m), "True"]) + "\n"
                   for c, s, e, m in rows)


def track_row(meta, track, biosample):
    rows = meta[(meta["output_type"] == "atac") & (meta["track_index"] == track)]
    if len(rows) != 1:
        raise SystemExit(f"ATAC track {track} is not in the track metadata")
    r = rows.iloc[0]
    if str(r["biosample_name"]).casefold() != biosample.casefold():
        raise SystemExit(f"ATAC track {track} is {r['biosample_name']!r} ({r['track_name']}), "
                         f"not {biosample!r}")
    return r


CHUNK_ARGS = ("track", "biosample", "head", "in_window", "crop", "n_shuffles",
              "random_state", "chunk", "shard_i", "n_shards")


def check_chunk_identity(cdir, want):
    path = os.path.join(cdir, "chunk_identity.json")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            json.dump(want, fh, indent=2, sort_keys=True)
        return
    have = json.load(open(path))
    if have != want:
        diff = {k: (have.get(k), want.get(k)) for k in sorted(set(have) | set(want))
                if have.get(k) != want.get(k)}
        raise SystemExit(f"{cdir} holds chunks computed from different inputs "
                         f"(was, now): {diff}. Delete it to recompute.")


def main():
    a = parse_args()
    sys.path.insert(0, a.ag_src)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    import pandas as pd
    import torch
    import pyfaidx
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "8")))
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.attribution import deeplift
    from tangermeme.deep_lift_shap import deep_lift_shap
    from ag_support import AlphaGenomeWrapper, _ag_nonlinear_ops, load_peaks, onehot, window_seq

    meta = pd.read_parquet(os.path.join(a.ag_src, "alphagenome_pytorch", "data",
                                        "track_metadata_human.parquet"))
    t = track_row(meta, a.track, a.biosample)
    print(f"[attr_ag] ATAC track {a.track}: {t['track_name']} ({t['biosample_name']})", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    df = load_peaks(a.peaks, a.chrom_sizes, a.in_window)
    center = df["chr"] + ":" + (df["start"] + df["summit"]).astype(str)
    if center.duplicated().any():
        raise SystemExit(f"{a.peaks}: {int(center.duplicated().sum())} peaks share a summit "
                         f"with an earlier peak")
    N = len(df)
    suffix = ""
    if a.n_shards is not None:
        bounds = shard_bounds(N, a.n_shards)
        lo, hi = int(bounds[a.shard_i]), int(bounds[a.shard_i + 1])
        df = df.iloc[lo:hi].reset_index(drop=True)
        suffix = f"_{a.shard_i}"
        print(f"[shard {a.shard_i}/{a.n_shards}] rows [{lo}:{hi}]", flush=True)
    if a.head:
        df = df.iloc[: a.head].reset_index(drop=True)

    fasta = pyfaidx.Fasta(a.fasta)
    seqs = [window_seq(fasta, r.chr, int(r.start + r.summit), a.in_window)
            for r in df.itertuples(index=False)]
    keep = [clean_in_crop(onehot(s).numpy(), a.crop) for s in seqs]
    n_drop = len(keep) - sum(keep)
    df = df[np.array(keep, dtype=bool)].reset_index(drop=True)
    seqs = [s for s, k in zip(seqs, keep, strict=True) if k]
    print(f"[attr_ag] {len(df)} peaks ({N} in bounds); {n_drop} dropped for non-ACGT "
          f"bases in the central {a.crop} bp", flush=True)
    if df.empty:
        raise SystemExit("no peaks left to attribute")

    h = hashlib.sha256()
    for r, s in zip(df.itertuples(index=False), seqs, strict=True):
        h.update(f"{r.chr}\t{r.start}\t{r.end}\t{r.summit}\t{s}\n".encode())
    here = os.path.dirname(os.path.abspath(__file__))
    identity = {**{k: getattr(a, k) for k in CHUNK_ARGS},
                "inputs_sha256": h.hexdigest(),
                "weights_sha256": file_sha256(a.weights),
                "ag_src_commit": git_commit(a.ag_src) or f"tree:{tree_sha256(a.ag_src)}",
                "step_sha256": {f: file_sha256(os.path.join(here, f))
                                for f in ("attr_ag.py", "ag_support.py")}}
    cdir = f"{a.out_dir}/_chunks{suffix}"
    os.makedirs(cdir, exist_ok=True)
    check_chunk_identity(cdir, identity)

    model = AlphaGenome.from_pretrained(a.weights, device="cuda"); model.eval()
    for p in model.parameters():
        p.requires_grad = False
    wrapper = AlphaGenomeWrapper(model, "atac", a.track, 1000, resolution=1)
    ops = _ag_nonlinear_ops(use_log_ratio_softmax=True, use_log_ratio_layernorm=False)

    def attr_one(i):
        x = torch.stack([onehot(seqs[i])])
        n_fill = fill_with_a(x)
        with torch.cuda.device(0), Deltas() as d:
            attr = deep_lift_shap(
                wrapper, x, target=0, batch_size=a.batch_size,
                references=deeplift.local_dinucleotide_shuffle, n_shuffles=a.n_shuffles,
                warning_threshold=THRESHOLD, additional_nonlinear_ops=ops,
                hypothetical=True, device="cuda", random_state=a.random_state)
        return (crop_center(attr.cpu().numpy().astype(np.float32), a.crop),
                crop_center(x.numpy().astype(np.int8), a.crop), n_fill, d)

    n = len(df)
    n_chunks = (n + a.chunk - 1) // a.chunk
    for c in range(n_chunks):
        af, of, ff = f"{cdir}/attr_{c}.npy", f"{cdir}/ohe_{c}.npy", f"{cdir}/n_filled_{c}.tsv"
        cf = f"{cdir}/convergence_{c}.json"
        if all(os.path.exists(p) for p in (af, of, ff, cf)):
            print(f"  chunk {c}/{n_chunks} exists, skipped", flush=True)
            continue
        aa, oo, filled, over, worst = [], [], [], 0, 0.0
        rows = range(c * a.chunk, min((c + 1) * a.chunk, n))
        for i in rows:
            at, oh, n_fill, d = attr_one(i)
            aa.append(at); oo.append(oh)
            over, worst = over + d.over, max(worst, d.max)
            if n_fill:
                r = df.iloc[i]
                filled.append((r["chr"], int(r["start"]), int(r["summit"]), n_fill))
        np.save(af, np.concatenate(aa, 0)); np.save(of, np.concatenate(oo, 0))
        with open(ff, "w") as fh:
            fh.writelines("\t".join(map(str, row)) + "\n" for row in filled)
        with open(cf, "w") as fh:
            json.dump(summary(len(rows) * a.n_shuffles, over, worst), fh)
        print(f"  chunk {c + 1}/{n_chunks} saved", flush=True)

    attr = np.concatenate([np.load(f"{cdir}/attr_{c}.npy") for c in range(n_chunks)], 0)
    ohe = np.concatenate([np.load(f"{cdir}/ohe_{c}.npy") for c in range(n_chunks)], 0)
    if not len(attr) == len(ohe) == n:
        raise SystemExit(f"chunks hold {len(attr)} rows, expected {n}")
    np.save(f"{a.out_dir}/attr{suffix}.npy", attr.astype(np.float32))
    np.save(f"{a.out_dir}/ohe{suffix}.npy", ohe.astype(np.int8))
    with open(f"{a.out_dir}/regions{suffix}.bed", "w") as fh:
        fh.write(narrowpeak_text(df[["chr", "start", "end", "summit"]]
                                 .itertuples(index=False, name=None)))
    n_filled = 0
    with open(f"{a.out_dir}/n_filled{suffix}.tsv", "w") as fh:
        fh.write("chr\tstart\tsummit\tn_filled\n")
        for c in range(n_chunks):
            for line in open(f"{cdir}/n_filled_{c}.tsv"):
                fh.write(line)
                n_filled += 1
    print(f"[attr_ag] {n_filled} peaks with non-ACGT bases outside the kept window set to A",
          flush=True)
    parts = [json.load(open(f"{cdir}/convergence_{c}.json")) for c in range(n_chunks)]
    conv = summary(sum(p["pairs"] for p in parts), sum(p["pairs_over"] for p in parts),
                   max(p["max_delta"] for p in parts))
    with open(f"{a.out_dir}/convergence{suffix}.json", "w") as fh:
        json.dump(conv, fh, indent=2)
    print(f"[attr_ag] DeepLIFT convergence: {conv['pairs_over']} of {conv['pairs']} "
          f"(sequence, reference) pairs over {THRESHOLD}; max delta {conv['max_delta']:.3g}",
          flush=True)
    print(f"DONE attr={attr.shape}", flush=True)


if __name__ == "__main__":
    main()
