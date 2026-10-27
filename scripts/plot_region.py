#!/usr/bin/env python
"""Per-base contributions over one region, one track per run, with the run's FiNeMo hits of
positive patterns boxed and named by TomTom (naming/pattern_names.tsv): the best match's TF,
with "-like" when the match is not confident.

Contributions are the projected scores (hypothetical x one-hot) of the attributed peak whose
window covers the region and whose centre is closest to it; the hits are those called in
that peak.
"""
import argparse

import logomaker
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

ap = argparse.ArgumentParser()
ap.add_argument("--region", required=True, help="chr:start-end")
ap.add_argument("--run", action="append", required=True, help="label=results/<run>, repeatable")
ap.add_argument("--out", required=True)
a = ap.parse_args()

chrom, span = a.region.split(":")
rs, re_ = map(int, span.split("-"))

tracks = []
for spec in a.run:
    label, d = spec.split("=", 1)
    attr = np.load(f"{d}/attr/attr.npy", mmap_mode="r")
    ohe = np.load(f"{d}/attr/ohe.npy", mmap_mode="r")
    W = attr.shape[-1]
    bed = pd.read_csv(f"{d}/attr/regions.bed", sep="\t", header=None)
    lo = bed[1] + bed[9] - W // 2
    row = (lo + W // 2 - (rs + re_) // 2)[(bed[0] == chrom) & (lo <= rs) & (lo + W >= re_)].abs().idxmin()
    proj = (attr[row].astype(np.float32) * ohe[row])[:, rs - lo[row]:re_ - lo[row]]
    hits = pd.read_csv(f"{d}/hits/hits.tsv", sep="\t")
    hits = hits[(hits.peak_id == row) & (hits.start >= rs) & (hits.end <= re_)
                & hits.motif_name.str.startswith("pos_patterns")]
    n = pd.read_csv(f"{d}/naming/pattern_names.tsv", sep="\t").set_index("key")
    hits = hits.assign(tf=hits.motif_name.map(n["name"]),
                       name=hits.motif_name.map(n["name"].where(n.confident, n["name"] + "-like")))
    tracks.append((label, proj, hits))

# one colour per TF across tracks
color = {t: PALETTE[i % len(PALETTE)]
         for i, t in enumerate(dict.fromkeys(t for _, _, h in tracks for t in h.tf))}

fig, axes = plt.subplots(len(tracks), 1, figsize=(14, 2.4 * len(tracks) + 0.5), sharex=True, squeeze=False)
for ax, (label, proj, hits) in zip(axes[:, 0], tracks):
    logomaker.Logo(pd.DataFrame(proj.T, columns=list("ACGT"), index=np.arange(rs, re_)),
                   ax=ax, color_scheme="classic")
    col = proj.sum(0)
    for h in hits.itertuples():
        top = col[h.start - rs:h.end - rs].max()
        ax.add_patch(plt.Rectangle((h.start - 0.5, 0), h.end - h.start, top,
                                   color=color[h.tf], alpha=0.25, lw=0))
        ax.text((h.start + h.end) / 2 - 0.5, top, h.name, ha="center", va="bottom", fontsize=9)
    ax.set_xlim(rs - 0.5, re_ - 0.5)
    ax.set_ylim(min(col.min(), 0) * 1.1, col.max() * 1.25)
    ax.axhline(0, color="#8a8984", lw=0.6)
    ax.set_ylabel("DeepLIFT/SHAP\ncontribution")
    ax.set_title(label, loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[-1, 0].set_xlabel(f"{chrom} position (bp)")
axes[-1, 0].ticklabel_format(axis="x", style="plain", useOffset=False)
fig.suptitle(f"Per-base contributions and FiNeMo motif hits, {chrom}:{rs:,}-{re_:,}", fontsize=12)
fig.legend(handles=[Patch(color="#8a8984", alpha=0.25, lw=0, label=(
    "FiNeMo hit of a TF-MoDISco pattern; label: the pattern's best TomTom match "
    "in HOCOMOCO H14 + JASPAR 2026, marked -like when q > 0.05"))],
    loc="lower center", frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.5 / (2.4 * len(tracks) + 0.5), 1, 1))
fig.savefig(a.out, dpi=200)
