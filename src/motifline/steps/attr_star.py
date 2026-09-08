#!/usr/bin/env python
"""GPN-Star per-position contribution step (compute layer, runs in envs/gpn_star, py3.13).

What it physically does, per region: a window (default 1536bp, the model's context) centred
on the summit is read from the multiz100way alignment, all species. The human column is
masked in groups -- every position p with p % stride == r is masked in one forward pass, so
``stride`` passes cover the window -- and the masked-LM logits at the masked positions are
read out. Both strands are scored (the reverse strand is the reverse complement of the whole
alignment, as in gpn.star.inference.LogitsInference) and averaged in probability space.
The contribution is the recipe of songlab's own modisco_run.py
(analysis/gpn-msa_human/workflow/scripts/, branch analysis-archive-2026-08-18):
``softmax(logits) - 1/4`` per position, a 4-vector = the HYPOTHETICAL contribution;
MoDISco multiplies by the one-hot itself.

Why grouped masking: one position per forward (upstream's LogitsInference) costs about
2,900 GPU-h for 33,917 x 1536 positions. Masking every stride-th position at once was
measured against it on 2026-09-07 (new_experiment/star/results/stride_fidelity.tsv):
Pearson >= 0.9987 on the contributions for strides 4..64, x357 throughput at stride 4.
The model was trained with 15% of positions masked (gpn/star/train.py, mlm_probability),
so stride >= 7 stays inside the training density.

``--source none`` scores with the alignment (the published use); ``--source mask`` blanks
the 99 non-human columns to '?' so only the human sequence is seen. They are different
readouts (new_experiment/star/README.md, section 3) and a run declares which one it is.

Rows: a region is dropped when its window runs off the chromosome (as the other steps do),
when its chromosome is not in the alignment store, or when the human column holds anything
but A/C/G/T inside the window (the ChromBPNet policy, not AG's fill). Dropped rows are
counted per reason in ``n_dropped.tsv``.

Outputs the same triple as attr_cbp/attr_ag: ``attr.npy`` (N,4,W) float32 hypothetical,
``ohe.npy`` (N,4,W) int8, ``regions.bed`` 11 columns (summit in column 10), row-aligned.
Sharding (--shard_i/--n_shards, linspace bounds) and per-chunk checkpoints as attr_ag.
stdlib + numpy/pandas/torch/zarr + gpn only; never imports motifline. GPU selection is
the caller's (CUDA_VISIBLE_DEVICES); this uses cuda:0 of what it is given. argparse runs
before every heavy import so --help works anywhere.
"""
import argparse
import json
import os
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True, help="4-, 10- or 11-column bed; summit = start + col 10 (or col 4)")
    ap.add_argument("--chrom_sizes", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--msa_path", required=True, help="multiz100way directory holding the zarr stores")
    ap.add_argument("--repo", default="songlab/gpn-star-hg38-v100-200m")
    ap.add_argument("--in_window", type=int, default=1536,
                    help="window = stored width; 1536 is the model's max_position_embeddings")
    ap.add_argument("--stride", type=int, default=8,
                    help="positions p with p %% stride == r are masked together; stride passes per strand")
    ap.add_argument("--source", default="none", choices=["none", "mask"],
                    help="none = with the alignment; mask = non-human columns blanked to '?'")
    ap.add_argument("--batch", type=int, default=8, help="sequences per forward (the groups of one strand)")
    ap.add_argument("--chunk", type=int, default=100, help="checkpoint granularity (regions per chunk)")
    ap.add_argument("--head", type=int, default=None, help="take only the first N peaks (slice runs)")
    ap.add_argument("--n_shards", type=int, default=None,
                    help="split the peak list into N contiguous shards (full-scale runs)")
    ap.add_argument("--shard_i", type=int, default=None, help="index of the shard to compute, 0-based")
    a = ap.parse_args()
    if a.in_window > 1536:
        ap.error("--in_window above 1536 exceeds max_position_embeddings; the model raises a "
                 "device-side assert that poisons the CUDA context (new_experiment/star/README.md)")
    if a.stride < 1 or a.stride > a.in_window:
        ap.error("--stride must be in [1, in_window]")
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


_NARROWPEAK_FILLER = ["."] * 6


def narrowpeak_text(rows):
    """``(chr, start, end, summit)`` rows -> the 11-column bed the next stage needs.

    Character-for-character ``motifline.adopt.normalize_peaks`` / ``attr_ag.narrowpeak_text``
    (finemo reads the summit from column 10); duplicated because a compute-layer step may
    not import motifline. ``tests/test_step_attr_star.py`` pins the three together.
    """
    return "".join("\t".join([str(c), str(s), str(e), *_NARROWPEAK_FILLER, str(m), "True"]) + "\n"
                   for c, s, e, m in rows)


def read_peaks(peaks_bed):
    """``(chr, start, end, summit)`` rows from a 4-, 10- or 11-column bed. Row order kept."""
    rows = []
    with open(peaks_bed) as fh:
        for ln in fh:
            if not ln.strip():
                continue
            f = ln.rstrip("\n").split("\t")
            if len(f) == 4:
                rows.append((f[0], int(f[1]), int(f[2]), int(f[3])))
            elif len(f) in (10, 11):
                rows.append((f[0], int(f[1]), int(f[2]), int(f[9])))
            else:
                raise ValueError(f"{peaks_bed}: expected 4, 10 or 11 columns, saw {len(f)}")
    return rows


def read_chrom_sizes(path):
    out = {}
    with open(path) as fh:
        for ln in fh:
            f = ln.split()
            if len(f) >= 2:
                out[f[0]] = int(f[1])
    return out


def in_bounds(rows, sizes, in_window):
    """Keep rows whose (start + summit) +/- in_window//2 lies inside the chromosome."""
    flank = in_window // 2
    keep = []
    for c, s, e, m in rows:
        centre = s + m
        keep.append(centre - flank > 0 and centre + flank <= sizes.get(c, 0))
    return keep


def stride_groups(length, stride):
    """The index groups masked together: group r = every p with p % stride == r."""
    import numpy as np
    return [np.arange(r, length, stride) for r in range(stride)]


def contrib_from_probs(p_fwd, p_rev):
    """(W,4) ACGT probabilities on the forward window and on its reverse complement
    -> (4,W) float32 hypothetical contribution, ``softmax - 1/4`` averaged over strands.

    Reverse-strand position j is forward position W-1-j and its A,C,G,T are the forward
    T,G,C,A, so the reverse array is flipped on both axes before averaging.
    """
    import numpy as np
    p = (np.asarray(p_fwd, dtype=np.float64) + np.asarray(p_rev, dtype=np.float64)[::-1, ::-1]) / 2
    return (p - 0.25).T.astype(np.float32)


def human_onehot(tokens, nuc_ids):
    """Human-column token ids (W,) -> (4,W) int8 one-hot over A,C,G,T, plus ``ok``:
    False when any token in the window is not one of the four nucleotides."""
    import numpy as np
    tokens = np.asarray(tokens)
    ohe = np.zeros((4, len(tokens)), dtype=np.int8)
    ok = True
    for k, tid in enumerate(nuc_ids):
        ohe[k] = tokens == tid
    if ohe.sum(0).min() < 1:
        ok = False
    return ohe, ok


#: The arguments that decide what a chunk CONTAINS (attr_ag's rule): resuming onto
#: checkpoints computed under any other value of these is refused.
CHUNK_IDENTITY = ("peaks", "repo", "msa_path", "in_window", "stride", "source",
                  "head", "chunk", "shard_i", "n_shards")


def check_chunk_fingerprint(cdir, a):
    want = {k: getattr(a, k, None) for k in CHUNK_IDENTITY}
    path = os.path.join(cdir, "chunk_identity.json")
    if os.path.exists(path):
        have = json.load(open(path))
        if have != want:
            diff = {k: (have.get(k), want[k]) for k in want if have.get(k) != want[k]}
            raise SystemExit(
                f"{cdir} holds checkpoints computed with different parameters "
                f"(was -> now): {diff}. Delete the directory to recompute, or restore "
                f"the previous parameters.")
    else:
        with open(path, "w") as fh:
            json.dump(want, fh, indent=2, sort_keys=True, default=str)


def main():
    a = parse_args()
    import numpy as np
    import torch
    import zarr
    from gpn import register_auto_classes
    from gpn.data import Tokenizer
    from gpn.star.data import GenomeMSA
    from gpn.star.utils import find_directory_sum_paths
    import gpn.star.model  # noqa: F401  (registers the architecture)
    from transformers import AutoModelForMaskedLM

    os.makedirs(a.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True   # measured x2.06, log-softmax off by 0.0016 nats
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        sys.exit("FAIL: torch.cuda is not available in this env")
    dev = torch.device("cuda:0")
    print(f"[attr_star] cuda:0 = {torch.cuda.get_device_properties(0).uuid}", flush=True)

    register_auto_classes("star")
    model = AutoModelForMaskedLM.from_pretrained(a.repo).to(dev).eval()
    tok = Tokenizer()
    vocab = list(tok.vocab)
    nuc_ids = [vocab.index(c) for c in "ACGT"]
    mask_id = tok.mask_token_id()
    # complement permutation of the token ids, so the reverse strand is COMP[fwd[::-1]]
    comp = np.arange(len(vocab))
    for x, y in (("A", "T"), ("C", "G")):
        comp[vocab.index(x)], comp[vocab.index(y)] = vocab.index(y), vocab.index(x)

    W, half = a.in_window, a.in_window // 2
    rows = read_peaks(a.peaks)
    sizes = read_chrom_sizes(a.chrom_sizes)
    keep = in_bounds(rows, sizes, W)
    n_oob = int(len(rows) - sum(keep))
    rows = [r for r, k in zip(rows, keep) if k]
    N = len(rows)
    suffix = ""
    if a.n_shards is not None:
        bounds = shard_bounds(N, a.n_shards)
        lo, hi = int(bounds[a.shard_i]), int(bounds[a.shard_i + 1])
        rows = rows[lo:hi]
        suffix = f"_{a.shard_i}"
        print(f"[shard {a.shard_i}/{a.n_shards}] rows [{lo}:{hi}] = {len(rows)} peaks", flush=True)
    if a.head:
        rows = rows[: a.head]
    print(f"[attr_star] {len(rows)} peaks (of {N} in-bounds; {n_oob} off-chromosome dropped, "
          f"head={a.head}) window={W} stride={a.stride} source={a.source}", flush=True)

    msa_paths = find_directory_sum_paths(a.msa_path)
    zchroms = set(zarr.open(list(msa_paths.values())[0], mode="r").keys())
    # zarr chromosome names carry no 'chr' prefix; peaks do.
    cmap = {}
    for c in sorted({r[0] for r in rows}):
        for cand in (c, c.removeprefix("chr"), f"chr{c}"):
            if cand in zchroms:
                cmap[c] = cand
                break
    gm = [GenomeMSA(p, n_species=n, subset_chroms=sorted(set(cmap.values())), in_memory=False)
          for n, p in msa_paths.items()]
    groups = stride_groups(W, a.stride)

    def logprobs(base, source):
        """(W,S) tokens of one strand -> (W,4) ACGT probabilities, every position masked
        exactly once across the stride groups."""
        out = np.zeros((W, 4), dtype=np.float64)
        copies = []
        for g in groups:
            m = base.copy()
            if source == "mask":
                m[:, 1:] = mask_id
            m[g, 0] = mask_id
            copies.append(m)
        for b in range(0, len(copies), a.batch):
            msa = torch.tensor(np.stack(copies[b:b + a.batch]), device=dev)
            ts = torch.zeros((msa.shape[0], 1), dtype=torch.long, device=dev)
            with torch.no_grad():
                lg = model(input_ids=msa[:, :, :1], source_ids=msa, target_species=ts).logits
            pr = torch.softmax(lg[:, :, 0][:, :, nuc_ids].float(), dim=-1).cpu().numpy()
            for j, g in enumerate(groups[b:b + a.batch]):
                out[g] = pr[j, g]
        return out

    def score_one(chrom, centre):
        zc = cmap.get(chrom)
        if zc is None:
            return None, None, "no_msa"
        fwd = np.concatenate([g.get_msa(zc, centre - half, centre + half, strand="+", tokenize=True)
                              for g in gm], axis=-1).astype(np.int64)            # (W,S)
        ohe, ok = human_onehot(fwd[:, 0], nuc_ids)
        if not ok:
            return None, None, "non_acgt"
        rev = comp[fwd[::-1]]
        attr = contrib_from_probs(logprobs(fwd, a.source), logprobs(rev, a.source))
        return attr, ohe, None

    n = len(rows)
    n_chunks = (n + a.chunk - 1) // a.chunk
    cdir = f"{a.out_dir}/_chunks{suffix}"
    os.makedirs(cdir, exist_ok=True)
    check_chunk_fingerprint(cdir, a)
    t0 = time.time()
    for c in range(n_chunks):
        af, of, rf, df = (f"{cdir}/attr_{c}.npy", f"{cdir}/ohe_{c}.npy",
                          f"{cdir}/regions_{c}.bed", f"{cdir}/dropped_{c}.tsv")
        if all(os.path.exists(p) for p in (af, of, rf, df)):
            print(f"  chunk {c}/{n_chunks} exists -> resume-skip", flush=True)
            continue
        aa, oo, kept, dropped = [], [], [], []
        for i in range(c * a.chunk, min((c + 1) * a.chunk, n)):
            chrom, start, end, summit = rows[i]
            attr, ohe, why = score_one(chrom, start + summit)
            if why:
                dropped.append((chrom, start, summit, why))
                continue
            aa.append(attr); oo.append(ohe); kept.append(rows[i])
        np.save(af, np.stack(aa) if aa else np.zeros((0, 4, W), np.float32))
        np.save(of, np.stack(oo) if oo else np.zeros((0, 4, W), np.int8))
        with open(rf, "w") as fh:
            fh.write(narrowpeak_text(kept))
        with open(df, "w") as fh:
            for row in dropped:
                fh.write("\t".join(str(v) for v in row) + "\n")
        done = min((c + 1) * a.chunk, n)
        el = time.time() - t0
        print(f"  chunk {c + 1}/{n_chunks} saved ({done}/{n})  {done / max(el, 1e-9):.2f} regions/s",
              flush=True)

    attr = np.concatenate([np.load(f"{cdir}/attr_{c}.npy") for c in range(n_chunks)], 0)
    ohe = np.concatenate([np.load(f"{cdir}/ohe_{c}.npy") for c in range(n_chunks)], 0)
    np.save(f"{a.out_dir}/attr{suffix}.npy", attr.astype(np.float32))
    np.save(f"{a.out_dir}/ohe{suffix}.npy", ohe.astype(np.int8))
    with open(f"{a.out_dir}/regions{suffix}.bed", "w") as fh:
        for c in range(n_chunks):
            fh.write(open(f"{cdir}/regions_{c}.bed").read())
    n_dropped = 0
    with open(f"{a.out_dir}/n_dropped{suffix}.tsv", "w") as fh:
        fh.write("chr\tstart\tsummit\treason\n")
        for c in range(n_chunks):
            for line in open(f"{cdir}/dropped_{c}.tsv"):
                if line.strip():
                    fh.write(line); n_dropped += 1
    print(f"[attr_star] dropped inside the run: {n_dropped} region(s) (see n_dropped{suffix}.tsv)",
          flush=True)
    print(f"DONE attr={attr.shape} ohe={ohe.shape}", flush=True)


if __name__ == "__main__":
    main()
