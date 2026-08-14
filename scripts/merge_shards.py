#!/usr/bin/env python
"""Concatenate attribution shards, in the order given, into one attr/ohe/regions triple."""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", required=True)
ap.add_argument("shards", nargs="+")
a = ap.parse_args()

os.makedirs(a.out_dir, exist_ok=True)
for name in ("attr.npy", "ohe.npy"):
    np.save(f"{a.out_dir}/{name}", np.concatenate([np.load(f"{d}/{name}") for d in a.shards]))
with open(f"{a.out_dir}/regions.bed", "w") as fh:
    for d in a.shards:
        fh.write(open(f"{d}/regions.bed").read())
