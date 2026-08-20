#!/usr/bin/env python
"""Cross-model motif merge step (compute layer, envs/motifcompendium py3.11).

Takes the MoDISco h5 of two or more leaves -- in this pipeline, the same region set
under ChromBPNet (native 2114bp) and AlphaGenome (native 2048bp) -- and clusters their
patterns together with kundajelab/MotifCompendium 1.0.18, producing ONE merged lexicon
plus the bookkeeping that says which entries the two models share.

What the stage does NOT do:

* It does not call hits against the merged lexicon. The line stops at the lexicon.
* It does not decide that a cluster is one motif. MotifCompendium clusters on CWM
  similarity at a threshold; the threshold is in config, the per-cluster quality
  statistics are written out next to the clusters, and reading them is the caller's job.
* It does not let a positive pattern merge with a negative one. Similarity is computed
  on IC-scaled PPMs, which are sign-blind, so an activating and a repressive pattern of
  the same shape would otherwise land in one cluster; ``--cluster_within posneg``
  forbids it. Pass ``--no_cluster_within_posneg`` to allow it deliberately.

Outputs in --out_dir:

  motifs.tsv       one row per INPUT pattern: leaf label, MoDISco pattern name, seqlets,
                   posneg, cluster id
  clusters.tsv     one row per cluster: size, seqlet total, per-leaf pattern counts,
                   membership (shared / <label>_only), and MotifCompendium's own
                   cluster-quality columns when they were computed
  merged.meme      the merged lexicon: one cluster-average PPM per cluster, MEME format
  compendium.mc    the MotifCompendium object holding every input pattern + its cluster
  merged.mc        the cluster-average MotifCompendium (what merged.meme was written from)

stdlib + MotifCompendium's own dependency set only; this file must never import
motifline. The MotifCompendium checkout arrives as --mc_src, never hardcoded, and
argparse runs before every heavy import so --help works anywhere.
"""
import argparse
import os
import sys


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc_src", required=True,
                    help="path to the kundajelab/MotifCompendium checkout")
    ap.add_argument("--modisco", action="append", required=True, metavar="LABEL=PATH",
                    help="a leaf: its label and its modisco.h5. Repeat, at least twice.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--region_width", type=int, required=True,
                    help="MoDISco's -w for these leaves; sets avg_dist_from_summit's origin")
    ap.add_argument("--algorithm", default="cpm_leiden", help="MotifCompendium's own default")
    ap.add_argument("--similarity_threshold", type=float, default=0.9,
                    help="MotifCompendium's own default")
    ap.add_argument("--seeds", default="100,200",
                    help="comma-separated Leiden seeds; MotifCompendium's own default is 100,200")
    ap.add_argument("--max_cpus", type=int, default=1)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--no_cluster_within_posneg", action="store_true",
                    help="allow positive and negative patterns into the same cluster")
    ap.add_argument("--no_quality_stats", action="store_true",
                    help="skip MotifCompendium's per-cluster quality statistics")
    a = ap.parse_args()
    leaves = []
    for spec in a.modisco:
        if "=" not in spec:
            ap.error(f"--modisco takes LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        if set("-.") & set(label):
            ap.error(f"leaf label {label!r} contains '-' or '.', which MotifCompendium "
                     f"uses as separators inside motif names")
        if not os.path.exists(path):
            ap.error(f"leaf {label!r}: no such file {path}")
        leaves.append((label, path))
    if len(leaves) < 2:
        ap.error("a merge needs at least two --modisco leaves")
    if len({l for l, _ in leaves}) != len(leaves):
        ap.error("duplicate leaf labels")
    a.leaves = leaves
    a.seed_list = [int(s) for s in a.seeds.split(",") if s.strip()]
    return a


def membership(labels_present, all_labels):
    """``shared`` when every leaf contributed, ``<label>_only`` when one did.

    A three-leaf merge in which two of three contributed is neither, and is named for
    what it is rather than collapsed into "shared": the distinction between "both
    models found it" and "some models found it" is the whole output of this stage.
    """
    present = sorted(labels_present)
    if len(present) == len(all_labels):
        return "shared"
    if len(present) == 1:
        return f"{present[0]}_only"
    return "+".join(present)


def main():
    a = parse_args()
    sys.path.insert(0, a.mc_src)
    import pandas as pd
    import MotifCompendium as mcp
    from MotifCompendium.utils.analysis import export_compendium_meme

    os.makedirs(a.out_dir, exist_ok=True)
    labels = [l for l, _ in a.leaves]
    mcp.set_compute_options(max_cpus=a.max_cpus, use_gpu=a.use_gpu, progress_bar=False)

    mc = mcp.build_from_modisco({l: p for l, p in a.leaves}, modisco_region_width=a.region_width)
    counts = mc.metadata["model"].value_counts().to_dict()
    print(f"[merge_compendium] loaded {len(mc)} patterns: "
          + ", ".join(f"{l}={counts.get(l, 0)}" for l in labels), flush=True)
    empty = [l for l in labels if not counts.get(l)]
    if empty:
        # Not fatal: a leaf with no patterns is a finding about that model, and the merge
        # of the rest is still well defined. It must not pass silently, though.
        print(f"[merge_compendium] WARNING: leaves with 0 patterns: {empty}", flush=True)

    kwargs = dict(algorithm=a.algorithm, similarity_threshold=a.similarity_threshold,
                  save_name="cluster", seeds=a.seed_list)
    if not a.no_cluster_within_posneg:
        kwargs["cluster_within"] = "posneg"
    mc.cluster(**kwargs)

    merged = mc.cluster_averages(
        "cluster",
        aggregations=[("name", "count", "num_motifs"),
                      ("num_seqlets", "sum", "num_seqlets"),
                      ("model", "concat_counted", "leaves"),
                      ("posneg", "concat", "posneg"),
                      ("avg_dist_from_summit", "mean", "avg_dist_from_summit")],
        weight_col="num_seqlets",
        compute_quality_stats=not a.no_quality_stats,
    )

    # Per-input-pattern table. `name` is "<label>-<pos|neg>.pattern_<i>" as MotifCompendium
    # builds it; the MoDISco-side name is kept separately so the two can be joined back.
    motifs = mc.metadata.copy()
    motifs["pattern"] = [n.split("-", 1)[1] for n in motifs["name"]]
    motifs.to_csv(f"{a.out_dir}/motifs.tsv", sep="\t", index=False)

    # Per-cluster table: MotifCompendium's own aggregate columns, plus the membership
    # bookkeeping it does not do -- it clusters on shape and never looks at `model`.
    by_cluster = motifs.groupby("cluster")["model"]
    # `source_cluster` is cluster_averages' own key back into the per-pattern table
    # (its `name` is the display string "cluster#<id>").
    clusters = merged.metadata.copy()
    per_leaf = by_cluster.value_counts().unstack(fill_value=0)
    for l in labels:
        clusters[f"n_{l}"] = clusters["source_cluster"].map(
            per_leaf[l] if l in per_leaf else {}).fillna(0).astype(int)
    clusters["membership"] = clusters["source_cluster"].map(
        {c: membership(set(g), labels) for c, g in by_cluster})
    clusters.to_csv(f"{a.out_dir}/clusters.tsv", sep="\t", index=False)

    export_compendium_meme(merged, f"{a.out_dir}/merged.meme", name_col="name")
    mc.save(f"{a.out_dir}/compendium.mc")
    merged.save(f"{a.out_dir}/merged.mc")

    tally = clusters["membership"].value_counts().to_dict()
    print(f"[merge_compendium] {len(mc)} patterns -> {len(merged)} clusters; "
          f"membership {tally}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
