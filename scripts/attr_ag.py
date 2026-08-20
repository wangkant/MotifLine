#!/usr/bin/env python
"""AlphaGenome DeepLIFT/SHAP hypothetical attributions for one contiguous shard of the peaks.

Target: log1p of the ATAC track summed over the central 1000 bp. References: local
dinucleotide shuffles, batch size 1, seeded by random_state + shuffle index.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pyfaidx
import torch

ap = argparse.ArgumentParser()
for k in ("ag_src", "weights", "peaks", "fasta", "chrom_sizes", "out_dir"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--track", type=int, required=True, help="index into the ATAC head")
ap.add_argument("--shard_i", type=int, default=0)
ap.add_argument("--n_shards", type=int, default=1)
ap.add_argument("--in_window", type=int, default=2048)
ap.add_argument("--n_shuffles", type=int, default=20)
ap.add_argument("--random_state", type=int, default=42)
ap.add_argument("--chunk", type=int, default=100, help="peaks per checkpoint")
a = ap.parse_args()

sys.path.insert(0, a.ag_src)
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.attribution import deeplift
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor
from tangermeme.deep_lift_shap import deep_lift_shap
from ag_deeplift import AlphaGenomeWrapper, nonlinear_ops

os.makedirs(a.out_dir, exist_ok=True)
p = pd.read_csv(a.peaks, sep="\t", header=None)
df = pd.DataFrame({"chr": p[0], "start": p[1], "end": p[2], "summit": p[9]})
sizes = dict(pd.read_csv(a.chrom_sizes, sep="\t", header=None).values)
half = a.in_window // 2
center = df.start + df.summit
df = df[(center - half > 0) & (center + half <= df.chr.map(sizes))].reset_index(drop=True)
lo, hi = np.linspace(0, len(df), a.n_shards + 1).astype(int)[[a.shard_i, a.shard_i + 1]]
df = df.iloc[lo:hi].reset_index(drop=True)

fasta = pyfaidx.Fasta(a.fasta)
X = torch.stack([sequence_to_onehot_tensor(fasta[c][m - half:m + half].seq).T
                 for c, m in zip(df.chr, df.start + df.summit)])
# non-ACGT bases are all-zero columns
keep = ~(X.sum(1) == 0).any(1)
X, df = X[keep], df[keep.numpy()].reset_index(drop=True)

model = AlphaGenome.from_pretrained(a.weights, device="cuda").eval().requires_grad_(False)
wrapper = AlphaGenomeWrapper(model, "atac", a.track, 1000)
# a killed shard resumes from its saved chunks; other arguments or input files get another directory
inputs = [(os.path.getsize(p), os.path.getmtime(p)) for p in (a.peaks, a.fasta, a.weights)]
tag = hashlib.sha1(repr((sorted(vars(a).items()), inputs)).encode()).hexdigest()[:12]
cdir = f"{a.out_dir}/chunks_{tag}"
os.makedirs(cdir, exist_ok=True)
for c in range(0, len(X), a.chunk):
    f = f"{cdir}/attr_{c}.npy"
    if not os.path.exists(f):
        attr = deep_lift_shap(wrapper, X[c:c + a.chunk], target=0, batch_size=1,
                              references=deeplift.local_dinucleotide_shuffle,
                              n_shuffles=a.n_shuffles, additional_nonlinear_ops=nonlinear_ops(),
                              hypothetical=True, device="cuda", random_state=a.random_state)
        np.save(f, attr.cpu().numpy().astype(np.float32))
        print(f"{c + len(attr)}/{len(X)}", flush=True)
attr = np.concatenate([np.load(f"{cdir}/attr_{c}.npy") for c in range(0, len(X), a.chunk)])
np.save(f"{a.out_dir}/attr.npy", attr)
np.save(f"{a.out_dir}/ohe.npy", X.numpy().astype(np.int8))
# narrowPeak, as the ChromBPNet path writes it: finemo reads the summit from column 10
with open(f"{a.out_dir}/regions.bed", "w") as fh:
    for r in df.itertuples(index=False):
        fh.write("\t".join(map(str, [r.chr, r.start, r.end, *["."] * 6, r.summit, True])) + "\n")
