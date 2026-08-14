#!/usr/bin/env python
"""AlphaGenome DeepLIFT HYPOTHETICAL attribution step (compute layer, chrombpnet_local py3.10).

Keeps the math of region_v3/k562/ag_cbp_deeplift_validation/ag_shap_shard.py:
same AlphaGenomeWrapper("atac", track, 1000, resolution=1), same nonlinear-op table
(softmax log-ratio, layernorm symmetric), same N->A fill, same deep_lift_shap call
(batch_size=1, local dinucleotide shuffle references, n_shuffles=20, hypothetical=True,
random_state pinned) and the same central crop, so its rows are directly comparable to
the archived interpret_ag_par_hyp/attr.npy prefix.

Dropped relative to the shard script: --two_sided (a control, not the canonical path),
--shuffle_input (a null control) and --subsample. Kept: the shard split (same linspace
bounds) for full-scale runs, and the resumable chunk checkpointing inside each shard --
AG runs for tens of hours, so a kill must cost at most one chunk. The two compose:
shard i writes ``_chunks_<i>/`` and then ``attr_<i>.npy``/``ohe_<i>.npy``/``regions_<i>.bed``.

stdlib + numpy/pandas/torch/pyfaidx + alphagenome-pytorch + chrombpnet only; this file must
never import motiflex or motifline. Site paths arrive as --ag_src/--chrombpnet_src, never
hardcoded. argparse runs before every heavy import so --help works anywhere.
"""
import argparse
import os
import sys


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag_src", required=True,
                    help="path to the alphagenome-pytorch deeplift-support checkout's src/")
    ap.add_argument("--chrombpnet_src", required=True,
                    help="path to chrombpnet-pytorch checkout (load_peaks needs read_chrom_sizes)")
    ap.add_argument("--weights", required=True, help="AlphaGenome .safetensors checkpoint")
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--chrom_sizes", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--track", type=int, default=60, help="AG output track index (60 = K562 ATAC)")
    ap.add_argument("--in_window", type=int, default=16384, help="validated AG DeepLIFT window")
    ap.add_argument("--crop", type=int, default=2114, help="central crop -> matches CBP attr length")
    ap.add_argument("--n_shuffles", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=100, help="checkpoint granularity (regions per chunk)")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--head", type=int, default=None, help="take only the first N peaks (slice runs)")
    ap.add_argument("--hypothetical", action="store_true",
                    help="REQUIRED; only hypothetical attributions are implemented on the AG path")
    ap.add_argument("--n_shards", type=int, default=None,
                    help="split the peak list into N contiguous shards (full-scale runs)")
    ap.add_argument("--shard_i", type=int, default=None, help="index of the shard to compute, 0-based")
    a = ap.parse_args()
    if not a.hypothetical:
        ap.error("--hypothetical is required: the AG path only implements hypothetical "
                 "attributions, and actual-score output would be silently mixable with them")
    if (a.shard_i is None) != (a.n_shards is None):
        ap.error("--shard_i and --n_shards must be given together")
    if a.n_shards is not None:
        if a.head is not None:
            ap.error("--head and --shard_i/--n_shards are mutually exclusive: a head-slice of a "
                     "shard is neither the slice nor the shard, and the merged output would be "
                     "silently short")
        if not 0 <= a.shard_i < a.n_shards:
            ap.error(f"--shard_i must be in [0, {a.n_shards})")
    return a


def shard_bounds(n_rows, n_shards):
    """Contiguous shard boundaries, verbatim from region_v3/k562/shap_shard.py:52-56."""
    import numpy as np
    return np.linspace(0, n_rows, n_shards + 1).astype(int)


def crop_center(arr, w):                       # (N,4,L) -> (N,4,w)
    L = arr.shape[-1]; s = L // 2 - w // 2
    return arr[..., s:s + w]


_NARROWPEAK_FILLER = ["."] * 6


def narrowpeak_text(rows):
    """``(chr, start, end, summit)`` rows -> the 11-column bed the next stage needs.

    AG's peak frame is 4-column, but ``finemo extract-regions-modisco-fmt -p``
    names the first ten columns of a bed positionally and reads the **summit
    from column 10** (`finemo/data_io.py:36-47`). Handed the 4-column frame it
    does not misread it, it refuses to open it -- polars ``ShapeError: The
    length of the new names list should be equal to or less than the original
    column length`` (verified against the installed finemo 0.30, 2026-08-14).

    The expansion is character-for-character ``motifline.adopt.normalize_peaks``,
    which is what the *adopted* AG runs write and which reproduces the archive's
    own ``peaks_np.bed`` byte-for-byte; a computed AG run has to write the same
    file as an adopted one or the two runs' hits are not comparable. It is
    duplicated rather than imported because a compute-layer step may not import
    motifline -- ``tests/test_step_attr_ag.py`` pins the two together.
    """
    return "".join("\t".join([str(c), str(s), str(e), *_NARROWPEAK_FILLER, str(m), "True"]) + "\n"
                   for c, s, e, m in rows)


def main():
    a = parse_args()
    sys.path.insert(0, a.chrombpnet_src)
    sys.path.insert(0, a.ag_src)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    import torch
    import pyfaidx
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "8")))
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.attribution import deeplift
    from tangermeme.deep_lift_shap import deep_lift_shap
    from ag_support import AlphaGenomeWrapper, _ag_nonlinear_ops, ag_seqs_centered, load_peaks

    os.makedirs(a.out_dir, exist_ok=True)

    model = AlphaGenome.from_pretrained(a.weights, device="cuda"); model.eval()
    for p in model.parameters():
        p.requires_grad = False
    wrapper = AlphaGenomeWrapper(model, "atac", a.track, 1000, resolution=1)
    ops = _ag_nonlinear_ops(use_log_ratio_softmax=True, use_log_ratio_layernorm=False)

    df = load_peaks(a.peaks, a.chrom_sizes, a.in_window)
    df = df.drop_duplicates(subset=["chr", "start", "summit"]).reset_index(drop=True)
    N = len(df)
    suffix = ""
    if a.n_shards is not None:
        bounds = shard_bounds(N, a.n_shards)
        lo, hi = int(bounds[a.shard_i]), int(bounds[a.shard_i + 1])
        df = df.iloc[lo:hi].reset_index(drop=True)
        suffix = f"_{a.shard_i}"
        print(f"[shard {a.shard_i}/{a.n_shards}] rows [{lo}:{hi}] = {len(df)} peaks", flush=True)
    if a.head:
        df = df.iloc[: a.head].reset_index(drop=True)
    print(f"[attr_ag] {len(df)} peaks (of {N} in-bounds, head={a.head})", flush=True)

    fasta = pyfaidx.Fasta(a.fasta)

    def attr_one(i):
        x = ag_seqs_centered(df.iloc[i:i + 1], fasta, a.in_window)          # (1,4,in_window)
        # N / non-one-hot columns make tangermeme reject the input. They occur ONLY in AG's large
        # flanks (the central 2114bp is N-free for all 33917 peaks, per CBP) and we crop the flanks
        # away anyway -> fill N -> A so the input is valid one-hot and the region set stays = CBP's.
        bad = x[0].sum(0) < 0.5                                             # (L,) all-zero columns = N
        if bool(bad.any()):
            x[0, :, bad] = 0.0
            x[0, 0, bad] = 1.0
        with torch.cuda.device(0):
            attr = deep_lift_shap(
                wrapper, x, target=0, batch_size=a.batch_size,
                references=deeplift.local_dinucleotide_shuffle, n_shuffles=a.n_shuffles,
                warning_threshold=1e8, additional_nonlinear_ops=ops,
                hypothetical=True, device="cuda", random_state=a.random_state)
        return (crop_center(attr.cpu().numpy().astype(np.float32), a.crop),
                crop_center(x.numpy().astype(np.int8), a.crop))

    # CHECKPOINTED: process in resumable chunks. A kill loses <= 1 chunk; restart skips finished chunks.
    n = len(df)
    n_chunks = (n + a.chunk - 1) // a.chunk
    # Chunk checkpoints live per shard: a shared _chunks/ would let shard 1's chunk 0 be
    # mistaken for shard 0's and resume-skipped with the wrong rows.
    cdir = f"{a.out_dir}/_chunks{suffix}"
    os.makedirs(cdir, exist_ok=True)
    for c in range(n_chunks):
        af, of = f"{cdir}/attr_{c}.npy", f"{cdir}/ohe_{c}.npy"
        if os.path.exists(af) and os.path.exists(of):
            print(f"  chunk {c}/{n_chunks} exists -> resume-skip", flush=True)
            continue
        aa, oo = [], []
        for i in range(c * a.chunk, min((c + 1) * a.chunk, n)):
            at, oh = attr_one(i)
            aa.append(at); oo.append(oh)
        np.save(af, np.concatenate(aa, 0)); np.save(of, np.concatenate(oo, 0))
        print(f"  chunk {c + 1}/{n_chunks} saved ({min((c+1)*a.chunk, n)}/{n})", flush=True)

    attr = np.concatenate([np.load(f"{cdir}/attr_{c}.npy") for c in range(n_chunks)], 0)
    ohe = np.concatenate([np.load(f"{cdir}/ohe_{c}.npy") for c in range(n_chunks)], 0)
    np.save(f"{a.out_dir}/attr{suffix}.npy", attr.astype(np.float32))
    np.save(f"{a.out_dir}/ohe{suffix}.npy", ohe.astype(np.int8))
    with open(f"{a.out_dir}/regions{suffix}.bed", "w") as fh:
        fh.write(narrowpeak_text(df[["chr", "start", "end", "summit"]]
                                 .itertuples(index=False, name=None)))
    print(f"DONE attr={attr.shape} ohe={ohe.shape}", flush=True)


if __name__ == "__main__":
    main()
