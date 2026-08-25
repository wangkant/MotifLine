#!/usr/bin/env python
"""Cross-model merge: the motif registry -> merge groups, by RSAT's clustering criterion.

This replaces the `mlex merge` subcommand, which motiflex removed on 2026-08-23
(commit 204cb3f, "the merge tree removed from the package") when it repositioned as an
auditor of *someone else's* consolidation -- `mlex selectivity` now says "Any tool's
merge can be scored" and `mlex compile --decisions` reads a decisions JSON "from any
tool". This step is that tool. Everything around it is unchanged: `mlex ingest` still
normalises the patterns, `align` still builds the alignment evidence, `annotate` still
attaches the labels, and `compile` still writes the finemo-callable lexicon.

WHAT CHANGED SCIENTIFICALLY. The removed stage cut an average-linkage tree over
`1 - ppm_similarity` at one scalar height (`cut_height: 0.05`, i.e. similarity >= 0.95).
A single height on a single score is not how the established motif-clustering software
decides a merge. RSAT `matrix-clustering` -- the tool that built this project's own
reference lexicon -- evaluates every internal node against SEVERAL thresholds at once:

  * the scores are read at one registration per pair (Ncor, cor, and w, the number of
    aligned columns), not at each score's own optimum;
  * under average linkage the node's score is the MEAN over all cross-pairs between the
    two child clusters (`TFBMclust/R/check_alignment.R:80-81`);
  * the node merges only if EVERY declared threshold is satisfied --
    `if(sum(unlist(th)) == length(thresholds))` (`check_alignment.R:106-109`);
  * a node that fails breaks the chain: its ancestors are not re-tested, they inherit
    the failure (`TFBMclust/R/align.motifs.R:101`, and `find_clusters.R`'s bottom-up
    walk stops at the first `alignment_flag == 0`).

So a cluster here is a maximal subtree in which every internal node passed every arm.
The thresholds are declared per merge in config (`thresholds: {Ncor:, cor:, w:}`); they
are not defaulted, for the same reason RSAT standalone refuses to default them.

The representative is the medoid -- an observed pattern, never a cross-model average.
Ties are broken explicitly (seqlet count, then node id) and flagged in the output,
because on small groups the medoid is genuinely undefined: for a two-member group both
mean distances are equal by construction.

Outputs
  merge_decisions.json   {"merges": [{"members": [...], "representative": ...}]} -- what
                         `mlex compile` reads. Singletons are absent: they survive as
                         themselves.
  merge_nodes.tsv        every internal node of the tree with its mean Ncor/cor/w, the
                         verdict, and WHICH arm failed -- this is what a threshold is
                         chosen by reading.
  merge_groups.tsv       per group: members, representative, membership (shared /
                         <label>_only), seqlet counts, within-group Ncor range.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motif_matrix import ncor_matrix  # noqa: E402

#: The arms RSAT's own default threshold set uses, in report order.
ARMS = ("Ncor", "cor", "w")


def read_registry(registry_dir):
    """``(node_ids, ppms, seqlet_counts, leaf_of)`` from an ``mlex ingest`` registry.

    Read as plain JSON + HDF5 rather than through motiflex: this step runs in the
    compute env, which has no motiflex. The layout is ingest's own
    (``ingest/__init__.py:288-304``): ``registry.json`` holds the node records without
    arrays, ``arrays.h5`` holds one group per node id with ``ppm``/``cwm``.

    The leaf a node came from is recovered from ``analysis_id`` minus its ``context``
    (the merge name), which is how ``mlex.project`` composed the id in the first place
    -- not by pattern-matching the string against a list of labels.
    """
    d = Path(registry_dir)
    blob = json.loads((d / "registry.json").read_text(encoding="utf-8"))
    nodes = blob["nodes"] if isinstance(blob, dict) else blob
    ids, seqlets, leaf_of = [], {}, {}
    for n in nodes:
        nid = n["node_id"]
        ids.append(nid)
        seqlets[nid] = int(n.get("seqlet_count") or 0)
        analysis_id = nid.split("::", 1)[0]
        context = str(n.get("context") or "")
        leaf_of[nid] = (analysis_id[: -(len(context) + 1)]
                        if context and analysis_id.endswith(f"_{context}") else analysis_id)
    with h5py.File(d / "arrays.h5") as f:
        ppms = {i: np.asarray(f[i]["ppm"][:], dtype=float) for i in ids}
    return ids, ppms, seqlets, leaf_of


def node_verdicts(link, scores, thresholds, n_leaves):
    """RSAT's bottom-up walk: mean cross-pair score per arm, ANDed, failure inherited.

    ``scores`` maps arm -> (n, n) leaf matrix. Returns one record per internal node and
    the leaf sets of the maximal all-pass subtrees.
    """
    members = {i: [i] for i in range(n_leaves)}
    broken = {i: False for i in range(n_leaves)}
    records = []
    for row, (a, b, height, _) in enumerate(link):
        a, b = int(a), int(b)
        node = n_leaves + row
        left, right = members[a], members[b]
        idx = np.ix_(left, right)
        means = {arm: float(scores[arm][idx].mean()) for arm in ARMS}
        failed = [arm for arm in ARMS if means[arm] < thresholds[arm]]
        inherited = broken[a] or broken[b]
        # A failed child breaks the chain; RSAT does not re-test above it.
        broken[node] = inherited or bool(failed)
        members[node] = left + right
        records.append(dict(node=node, height=float(height),
                            n_left=len(left), n_right=len(right),
                            **{f"mean_{arm}": round(means[arm], 4) for arm in ARMS},
                            failed_arms=",".join(failed),
                            inherited_break=inherited,
                            merged=not broken[node]))
    groups, taken = [], set()
    # A group is a node that passed and whose parent did not; walk top-down so the
    # largest surviving subtree wins.
    for node in range(n_leaves + len(link) - 1, -1, -1):
        if broken.get(node, True) or node in taken:
            continue
        leaves = members[node]
        if any(l in taken for l in leaves):
            continue
        groups.append(sorted(leaves))
        taken.update(leaves)
    for leaf in range(n_leaves):
        if leaf not in taken:
            groups.append([leaf])
    return pd.DataFrame(records), sorted(groups)


def choose_representative(idx, ncor, ids, seqlets):
    """Medoid by mean within-group distance; ties broken on seqlets, then node id.

    Returns ``(representative, tied)``. ``tied`` is the honest part: for a two-member
    group every member has the same mean distance, so the medoid carries no information
    and the tie-break decides. Downstream should not read such a representative as
    "the most typical member".
    """
    if len(idx) == 1:
        return ids[idx[0]], False
    d = (1.0 - ncor[np.ix_(idx, idx)]).mean(1)
    best = d.min()
    tied_at = [i for i, v in zip(idx, d) if v <= best + 1e-12]
    pick = max(tied_at, key=lambda i: (seqlets.get(ids[i], 0), ids[i]))
    return ids[pick], len(tied_at) > 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", required=True, help="registry/ from mlex ingest")
    p.add_argument("--out_dir", required=True)
    for arm in ARMS:
        p.add_argument(f"--{arm}", type=float, required=True,
                       help=f"lower threshold on the node's mean {arm}")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = {arm: getattr(a, arm) for arm in ARMS}

    ids, ppms, seqlets, leaf_of = read_registry(a.registry)
    n = len(ids)
    print(f"registry: {n} patterns; thresholds " +
          ", ".join(f"{k}>={v}" for k, v in thresholds.items()), flush=True)

    ncor, cor, w_overlap, _ = ncor_matrix([ppms[i] for i in ids])
    scores = {"Ncor": ncor, "cor": cor, "w": w_overlap}
    d = 1.0 - ncor
    np.fill_diagonal(d, 0.0)
    d = np.clip((d + d.T) / 2.0, 0.0, None)     # squareform demands exact symmetry
    link = linkage(squareform(d, checks=False), method="average")

    nodes, groups = node_verdicts(link, scores, thresholds, n)
    nodes.to_csv(out / "merge_nodes.tsv", sep="\t", index=False)

    rows, merges = [], []
    for gi, idx in enumerate(groups):
        names = [ids[i] for i in idx]
        rep, tied = choose_representative(idx, ncor, ids, seqlets)
        labs = sorted({leaf_of[x] for x in names})
        membership = "shared" if len(labs) > 1 else f"{labs[0]}_only"
        sub = ncor[np.ix_(idx, idx)]
        rows.append(dict(group=f"group_{gi}", size=len(idx), representative=rep,
                         representative_tied=tied, membership=membership,
                         min_ncor=round(float(sub.min()), 4) if len(idx) > 1 else np.nan,
                         n_seqlets=sum(seqlets.get(x, 0) for x in names),
                         members=";".join(names)))
        if len(idx) > 1:
            merges.append({"members": sorted(names), "representative": rep})

    pd.DataFrame(rows).to_csv(out / "merge_groups.tsv", sep="\t", index=False)
    (out / "merge_decisions.json").write_text(json.dumps({
        "method": "average",
        "criterion": "RSAT matrix-clustering: mean cross-pair score per arm, all arms ANDed",
        "thresholds": thresholds,
        "merges": merges,
    }, indent=2) + "\n")

    g = pd.DataFrame(rows)
    kept = int((g["size"] > 1).sum())
    print(f"{n} patterns -> {len(g)} groups ({kept} merged, {len(g) - kept} singletons)")
    print(g.membership.value_counts().to_string())
    tied = int(g.representative_tied.sum())
    if tied:
        print(f"representative undetermined by medoid in {tied} group(s); "
              f"broken on seqlet count then node id")
    print(f"\n-> {out}/merge_decisions.json, merge_groups.tsv, merge_nodes.tsv")


if __name__ == "__main__":
    main()
