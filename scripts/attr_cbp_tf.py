#!/usr/bin/env python
"""ChromBPNet DeepSHAP hypothetical attributions, counts head, with the official TensorFlow
code (what `chrombpnet contribs_bw` runs), for one contiguous shard of the peaks.

One change to that code: shuffle k of every sequence is drawn from
RandomState(random_state + k), as tangermeme seeds them, instead of an unseeded one.
"""
import argparse
import os
import shutil
from argparse import Namespace

import numpy as np

ap = argparse.ArgumentParser()
for k in ("model_h5", "peaks", "fasta", "out_dir"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--chrom_sizes")
ap.add_argument("--shard_i", type=int, default=0)
ap.add_argument("--n_shards", type=int, default=1)
ap.add_argument("--n_shuffles", type=int, default=20)
ap.add_argument("--random_state", type=int, default=42)
a = ap.parse_args()

os.makedirs(a.out_dir, exist_ok=True)
lines = ["\t".join(ln.rstrip("\n").split("\t")[:10]) + "\n" for ln in open(a.peaks) if ln.strip()]
lo, hi = np.linspace(0, len(lines), a.n_shards + 1).astype(int)[[a.shard_i, a.shard_i + 1]]
bed = f"{a.out_dir}/peaks.bed"
with open(bed, "w") as fh:
    fh.writelines(lines[lo:hi])

# importing interpret turns off eager execution, which shap's DeepExplainer needs
import chrombpnet.evaluation.interpret.interpret as interpret
import chrombpnet.evaluation.interpret.shap_utils as shap_utils
from deeplift.dinuc_shuffle import dinuc_shuffle


def shuffle_several_times(s):
    refs = [np.array([dinuc_shuffle(s[0], rng=np.random.RandomState(a.random_state + k))
                      for k in range(a.n_shuffles)])]
    if len(s) == 2:
        refs.append(np.array([s[1]] * a.n_shuffles))
    return refs


shap_utils.shuffle_several_times = shuffle_several_times
prefix = f"{a.out_dir}/chrombpnet"
interpret.main(Namespace(genome=a.fasta, regions=bed, model_h5=a.model_h5, output_prefix=prefix,
                         debug_chr=None, profile_or_counts=["counts"]))

import tables
with tables.open_file(f"{prefix}.counts_scores.h5") as f:
    np.save(f"{a.out_dir}/attr.npy", f.get_node("/shap/seq").read().astype(np.float32))
    np.save(f"{a.out_dir}/ohe.npy", f.get_node("/raw/seq").read().astype(np.int8))
shutil.copy(f"{prefix}.interpreted_regions.bed", f"{a.out_dir}/regions.bed")
