#!/usr/bin/env python
"""ChromBPNet DeepLIFT/SHAP hypothetical attributions, counts head of the model without its
bias.

References: dinucleotide shuffles, seeded by random_state + shuffle index.
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import pyfaidx
import torch
from tangermeme.deep_lift_shap import deep_lift_shap

ap = argparse.ArgumentParser()
for k in ("wo_bias", "peaks", "fasta", "chrom_sizes", "out_dir"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--in_window", type=int, default=2114)
ap.add_argument("--n_shuffles", type=int, default=20)
ap.add_argument("--random_state", type=int, default=42)
a = ap.parse_args()


class BPNetCounts(torch.nn.Module):
    """BPNet trunk and counts head: a wide convolution, dilated residual convolutions with
    centre cropping, global average pooling and a linear layer. Each ReLU is its own module
    so that DeepLIFT sees every non-linearity."""

    def __init__(self, n_filters=512, n_layers=8):
        super().__init__()
        self.iconv = torch.nn.Conv1d(4, n_filters, 21)
        self.irelu = torch.nn.ReLU()
        self.rconvs = torch.nn.ModuleList(
            torch.nn.Conv1d(n_filters, n_filters, 3, dilation=2 ** i) for i in range(1, n_layers + 1))
        self.rrelus = torch.nn.ModuleList(torch.nn.ReLU() for _ in range(n_layers))
        self.linear = torch.nn.Linear(n_filters, 1)

    def forward(self, x):
        x = self.irelu(self.iconv(x))
        for conv, relu in zip(self.rconvs, self.rrelus):
            y = relu(conv(x))
            c = (x.shape[2] - y.shape[2]) // 2
            x = x[:, :, c:-c] + y
        return self.linear(x.mean(2))


os.makedirs(a.out_dir, exist_ok=True)
model = BPNetCounts()
# weights straight from the Keras h5; its kernels are (width, in, out), .T gives the
# (out, in, width) used here. The profile head is not needed for the counts.
h5 = h5py.File(a.wo_bias, "r")["model_weights"]
sd = {}
for name, layer in [("iconv", "1st_conv"), *((f"rconvs.{i - 1}", f"{i}conv") for i in range(1, 9)),
                    ("linear", "logcount_predictions")]:
    g = h5[f"wo_bias_bpnet_{layer}/wo_bias_bpnet_{layer}"]
    sd[f"{name}.weight"] = torch.from_numpy(g["kernel:0"][:].T.copy())
    sd[f"{name}.bias"] = torch.from_numpy(g["bias:0"][:])
model.load_state_dict(sd)
model = model.eval().requires_grad_(False).cuda()

p = pd.read_csv(a.peaks, sep="\t", header=None, usecols=range(10))
p[10] = True
sizes = dict(pd.read_csv(a.chrom_sizes, sep="\t", header=None).values)
half = a.in_window // 2
center = p[1] + p[9]
p = p[(center - half > 0) & (center + half <= p[0].map(sizes))].reset_index(drop=True)

fasta = pyfaidx.Fasta(a.fasta)
acgt = np.frombuffer(b"ACGT", np.uint8)
X = torch.tensor(np.stack([
    np.frombuffer(fasta[c][m - half:m + half].seq.upper().encode(), np.uint8) == acgt[:, None]
    for c, m in zip(p[0], p[1] + p[9])]), dtype=torch.float32)
# non-ACGT bases are all-zero columns
keep = ~(X.sum(1) == 0).any(1)
X, p = X[keep], p[keep.numpy()].reset_index(drop=True)

attr = deep_lift_shap(model, X, target=0, batch_size=32, n_shuffles=a.n_shuffles,
                      hypothetical=True, device="cuda", random_state=a.random_state, verbose=True)
np.save(f"{a.out_dir}/attr.npy", attr.cpu().numpy().astype(np.float32))
np.save(f"{a.out_dir}/ohe.npy", X.numpy().astype(np.int8))
p.to_csv(f"{a.out_dir}/regions.bed", sep="\t", header=False, index=False)
