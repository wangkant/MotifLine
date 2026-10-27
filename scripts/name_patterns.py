#!/usr/bin/env python
"""Name MoDISco patterns by TomTom against the reference union.

Each pattern is trimmed as in modiscolite's report (|CWM| >= 30% of max) and named after
the gene symbol of its best match (lowest q, then p); `family` is that TF's TFClass family
from the H14 annotation; `confident` is q <= qval_threshold.
"""
import argparse
import json
import os
import re
import subprocess

import h5py
import numpy as np
import pandas as pd

HEADER = ("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
          "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")

ap = argparse.ArgumentParser()
for k in ("h5", "target", "h14_annotation", "tomtom", "out_dir"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--qval_threshold", type=float, default=0.05)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

pats = []
with h5py.File(a.h5, "r") as f:
    for group in ("pos_patterns", "neg_patterns"):
        for name in sorted(f.get(group, {}), key=lambda n: int(n.split("_")[-1])):
            g = f[group][name]
            score = np.abs(g["contrib_scores"][:]).sum(1)
            keep = np.where(score >= 0.3 * score.max())[0]
            pats.append(dict(key=f"{group}.{name}", n_seqlets=int(g["seqlets"]["n_seqlets"][0]),
                             ppm=g["sequence"][keep.min():keep.max() + 1]))

query = f"{a.out_dir}/query.meme"
with open(query, "w") as fh:
    fh.write(HEADER)
    for p in pats:
        fh.write(f"MOTIF {p['key']}\nletter-probability matrix: alength= 4 w= {len(p['ppm'])}\n")
        fh.writelines(" ".join(f"{v:.6f}" for v in row) + "\n" for row in p["ppm"])
        fh.write("\n")

# modiscolite's TomTom line without -mi 1, which would keep only the first query motif
subprocess.run([a.tomtom, "-no-ssc", "-oc", f"{a.out_dir}/tomtom", "-verbosity", "1",
                "-min-overlap", "5", "-dist", "pearson", "-evalue", "-thresh", "10.0",
                query, a.target], check=True)

# HOCOMOCO ids go through the H14 annotation; JASPAR ids keep the name after '_'
h14, family = {}, {}
for line in filter(str.strip, open(a.h14_annotation)):
    r = json.loads(line)
    h14[r["name"]] = r["masterlist_info"]["species"]["HUMAN"]["gene_symbol"]
    family[h14[r["name"]]] = r["masterlist_info"]["tfclass_family"]


def symbol(motif_id):
    return h14.get(motif_id) or re.sub(r"^[A-Z]{2}\d+\.\d+_", "", motif_id).upper()


t = pd.read_csv(f"{a.out_dir}/tomtom/tomtom.tsv", sep="\t", comment="#")
t.columns = [c.replace("-", "_") for c in t.columns]
t = t.sort_values(["Query_ID", "q_value", "p_value"])
top = {q: g.head(3) for q, g in t.groupby("Query_ID")}
rows = []
for p in pats:
    rec = dict(key=p["key"], n_seqlets=p["n_seqlets"])
    hits = top.get(p["key"], t.iloc[:0])
    for i, h in enumerate(hits.itertuples(index=False)):
        rec[f"match{i}"] = h.Target_ID
        rec[f"tf{i}"] = symbol(h.Target_ID)
        rec[f"qval{i}"] = h.q_value
    rec["name"] = rec.get("tf0", "")
    # TFClass family of the best match; JASPAR dimers such as FOS::JUN keep their name
    rec["family"] = family.get(rec["name"], rec["name"])
    rec["confident"] = rec.get("qval0", np.nan) <= a.qval_threshold
    rows.append(rec)
pd.DataFrame(rows).to_csv(f"{a.out_dir}/pattern_names.tsv", sep="\t", index=False)
