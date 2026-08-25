#!/usr/bin/env python
"""RSAT's partition -> one representative motif per cluster (the reference library).

Input is RSAT ``matrix-clustering``'s own ``clusters.tab`` -- the partition is never
recomputed here, only read. What this step decides is the *representative*, and it is
always the **medoid**: the member with the smallest mean distance to the rest of its
cluster. Never an average. Averaging two real secondary modes produces a matrix that
matches neither, and the whole point of a cluster-level vocabulary is that every entry
is an observed matrix somebody can look up.

Distances come from the Ncor computed here (``motif_matrix.ncor_matrix``) and are
written out as ``ncor.npy`` -- a declared output of the rule, kept because it is the
similarity every cluster in this library was judged on and nothing else on disk records
it. It is NOT a cache: no stage reads it back.

Outputs
  lexicon_rsat.meme          one medoid PPM per cluster
  lexicon_rsat.metadata.tsv  per cluster: size, medoid, per-source counts, member TFs
  clusters_rsat.tsv          the flat motif -> cluster table, medoid flagged
  ncor.npy                   all-pairs Ncor over the union, in source order (evidence,
                             not a cache -- nothing reads it back)

This library is a reference and a plotting/publication convenience. It is NOT the
TomTom target -- see lexicon_cltagged.py for why.
"""
import argparse
import re
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motif_matrix import ncor_matrix, read_meme, write_meme, meme_block  # noqa: E402


def symbol(name):
    """The TF symbol a library id carries: ``AHR.H14CORE.0.P.B`` -> ``AHR``."""
    if ".H14CORE." in name:
        return name.split(".H14CORE.")[0].upper()
    return name.split("_", 1)[1].upper() if "_" in name else name.upper()


def cluster_label(cluster_no, medoid, symbols, size):
    """``cl<N>_<medoid symbol>[-<up to 2 other roots>][-etc]_n<size>``.

    The medoid's own symbol always leads: a label whose representative matrix belongs
    to a TF the label does not name is unreadable. Trailing digits are stripped to a
    root so that SP1/SP3/SP4 count as one root rather than three.
    """
    med_sym = re.sub(r"[^A-Za-z0-9]", "", symbol(medoid)) or "NA"
    roots = [r for r, _ in Counter(re.sub(r"\d+$", "", s) or s for s in symbols).most_common()]
    med_root = re.sub(r"\d+$", "", symbol(medoid)) or symbol(medoid)
    extra = [re.sub(r"[^A-Za-z0-9]", "", r) for r in roots if r != med_root][:2]
    label = "-".join([med_sym] + [e for e in extra if e])
    if len(roots) > len(extra) + 1:
        label += "-etc"
    return f"cl{cluster_no}_{label}_n{size}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters_tab", required=True, help="RSAT rsat_tables/clusters.tab")
    p.add_argument("--source", action="append", required=True,
                   help="tag=path, same order as the union build")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--w_th", type=int, required=True,
                   help="minimum comparable overlap in columns (config w_th)")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- partition: cluster_N \t union_m17_NAME,union_m18_NAME,... ----
    rows = []
    for line in open(a.clusters_tab):
        cl, members = line.rstrip("\n").split("\t")
        for m in members.split(","):
            rows.append((cl, re.sub(r"^union_m\d+_", "", m)))
    cl = pd.DataFrame(rows, columns=["cluster", "motif"])
    assert cl.motif.is_unique, "a motif appears in more than one cluster"

    # ---- library, in the source order the union was written in ----
    ppms, src_of, names_order = {}, {}, []
    tags = []
    for spec in a.source:
        tag, path = spec.split("=", 1)
        tags.append(tag)
        d = read_meme(path)
        ppms.update(d)
        for n in d:
            src_of[n] = tag
            names_order.append(n)
    missing = set(cl.motif) - set(ppms)
    assert not missing, f"clusters.tab names {len(missing)} motifs absent from the sources: {sorted(missing)[:5]}"
    cl["source"] = cl.motif.map(src_of)
    print(f"RSAT partition: {cl.cluster.nunique()} clusters / {len(cl)} motifs  ("
          + ", ".join(f"{t} {(cl.source == t).sum()}" for t in tags) + ")", flush=True)

    pos = {n: i for i, n in enumerate(names_order)}
    S, _, _, _ = ncor_matrix([ppms[n][0] for n in names_order], w_th=a.w_th)
    np.save(out / "ncor.npy", S)
    D = 1.0 - S

    meta, blocks, used = [], [], set()
    for c, g in cl.groupby("cluster", sort=False):
        idx = np.array([pos[m] for m in g.motif])
        med = names_order[idx[int(D[np.ix_(idx, idx)].mean(1).argmin())]]
        symbols = sorted({symbol(n) for n in g.motif})
        cluster_no = int(c.split("_")[1])
        name = cluster_label(cluster_no, med, symbols, len(g))
        assert name not in used, f"duplicate cluster label {name}"
        used.add(name)
        ppm, nsites = ppms[med]
        blocks.append(meme_block(name, ppm, nsites))
        sub = S[np.ix_(idx, idx)]
        meta.append(dict(lexicon_id=name, cluster=cluster_no, size=len(g), medoid=med,
                         width=ppm.shape[0],
                         **{f"n_{t}": int((g.source == t).sum()) for t in tags},
                         min_ncor=round(float(sub.min()), 3) if len(g) > 1 else np.nan,
                         tf_symbols=";".join(symbols), members=";".join(sorted(g.motif))))

    write_meme(out / "lexicon_rsat.meme", blocks)
    md = pd.DataFrame(meta)
    md.to_csv(out / "lexicon_rsat.metadata.tsv", sep="\t", index=False)
    cl["is_medoid"] = cl.motif.isin(set(md.medoid))
    cl.to_csv(out / "clusters_rsat.tsv", sep="\t", index=False)

    single = int((md["size"] == 1).sum())
    print(f"lexicon_rsat.meme: {len(md)} entries  singletons {single} ({single / len(md):.1%})  "
          f"largest {md['size'].max()}")
    print(md.nlargest(8, "size")[["lexicon_id", "size", "min_ncor"]].to_string(index=False))


if __name__ == "__main__":
    main()
