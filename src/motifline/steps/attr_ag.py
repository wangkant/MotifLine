#!/usr/bin/env python
"""AlphaGenome DeepLIFT HYPOTHETICAL attribution step (compute layer, chrombpnet_local py3.10).

Keeps the math of region_v3/k562/ag_cbp_deeplift_validation/ag_shap_shard.py:
same AlphaGenomeWrapper("atac", track, 1000, resolution=1), same nonlinear-op table
(softmax log-ratio, layernorm symmetric), same N->A fill, same deep_lift_shap call
(batch_size=1, local dinucleotide shuffle references, n_shuffles=20, hypothetical=True,
random_state pinned) and the same central crop, so its rows are directly comparable to
the archived interpret_ag_par_hyp/attr.npy prefix.

Dropped relative to the shard script: --two_sided (a control, not the canonical path),
--shuffle_input (a null control), --subsample, and the shard split -- this plan attributes
a head-slice in one process. Kept: the resumable chunk checkpointing, because a kill in
an hour-long AG run must not cost the whole run.

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
    a = ap.parse_args()
    if not a.hypothetical:
        ap.error("--hypothetical is required: the AG path only implements hypothetical "
                 "attributions, and actual-score output would be silently mixable with them")
    return a


def crop_center(arr, w):                       # (N,4,L) -> (N,4,w)
    L = arr.shape[-1]; s = L // 2 - w // 2
    return arr[..., s:s + w]


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
    cdir = f"{a.out_dir}/_chunks"
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
    np.save(f"{a.out_dir}/attr.npy", attr.astype(np.float32))
    np.save(f"{a.out_dir}/ohe.npy", ohe.astype(np.int8))
    df.to_csv(f"{a.out_dir}/regions.bed", sep="\t", header=False, index=False)
    print(f"DONE attr={attr.shape} ohe={ohe.shape}", flush=True)


if __name__ == "__main__":
    main()
