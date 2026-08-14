#!/usr/bin/env python
"""AlphaGenome DeepLIFT/SHAP hypothetical attributions.

Target: log1p of the ATAC track summed over the central 1000 bp. References: local
dinucleotide shuffles, batch size 1.
"""
import argparse
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
ap.add_argument("--in_window", type=int, default=2048)
ap.add_argument("--n_shuffles", type=int, default=20)
ap.add_argument("--random_state", type=int, default=42)
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

fasta = pyfaidx.Fasta(a.fasta)
X = torch.stack([sequence_to_onehot_tensor(fasta[c][m - half:m + half].seq).T
                 for c, m in zip(df.chr, df.start + df.summit)])
# non-ACGT bases are all-zero columns
keep = ~(X.sum(1) == 0).any(1)
X, df = X[keep], df[keep.numpy()].reset_index(drop=True)

model = AlphaGenome.from_pretrained(a.weights, device="cuda").eval().requires_grad_(False)
wrapper = AlphaGenomeWrapper(model, "atac", a.track, 1000)
attr = deep_lift_shap(wrapper, X, target=0, batch_size=1,
                      references=deeplift.local_dinucleotide_shuffle, n_shuffles=a.n_shuffles,
                      additional_nonlinear_ops=nonlinear_ops(), hypothetical=True,
                      device="cuda", random_state=a.random_state, verbose=True)
np.save(f"{a.out_dir}/attr.npy", attr.cpu().numpy().astype(np.float32))
np.save(f"{a.out_dir}/ohe.npy", X.numpy().astype(np.int8))
df.to_csv(f"{a.out_dir}/regions.bed", sep="\t", header=False, index=False)
