#!/usr/bin/env python
"""Name MoDISco patterns by TomTom against the reference union.

Each pattern is trimmed as in modiscolite's report (|CWM| >= 30% of max), queried
against the union, and named after the gene symbol of its best match. Every pattern
keeps its name; `confident` marks those whose best match has q <= qval_threshold.
"""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from digests import file_sha256

MEME_HEADER = ("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
               "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
GROUPS = ("pos_patterns", "neg_patterns")


def modisco_trim(ppm, cwm, threshold=0.3):
    score = np.abs(cwm).sum(1)
    keep = np.where(score >= score.max() * threshold)[0]
    if len(keep) == 0:
        return ppm, 0, len(ppm)
    return ppm[keep.min():keep.max() + 1], int(keep.min()), int(keep.max() + 1)


def patterns_of(h5_path):
    out = []
    with h5py.File(h5_path, "r") as f:
        for group in GROUPS:
            if group not in f:
                continue
            for name in sorted(f[group], key=lambda n: int(n.split("_")[-1])):
                g = f[group][name]
                out.append(dict(key=f"{group}.{name}", group=group.split("_")[0],
                                pattern=name,
                                ppm=g["sequence"][:], cwm=g["contrib_scores"][:],
                                n_seqlets=int(g["seqlets"]["n_seqlets"][0])))
    return out


def tomtom_argv(tomtom, query, target, out_dir):
    # modiscolite's TomTom line without -mi 1: it queries one motif per file, we
    # query all patterns in one file, and -mi 1 would keep only the first.
    return [tomtom, "-no-ssc", "-oc", str(out_dir), "-verbosity", "1",
            "-min-overlap", "5", "-dist", "pearson", "-evalue", "-thresh", "10.0",
            str(query), str(target)]


def h14_symbols(annotation_jsonl):
    out = {}
    for line in open(annotation_jsonl):
        if line.strip():
            r = json.loads(line)
            out[r["name"]] = r["masterlist_info"]["species"]["HUMAN"]["gene_symbol"]
    return out


def tf_name(motif_id, h14):
    """HOCOMOCO ids go through the H14 annotation (TYY1.H14CORE... -> YY1). JASPAR ids
    keep the name after '_', upper-cased; that is not always an HGNC symbol."""
    s = str(motif_id)
    if not s:
        return ""
    if s in h14:
        return h14[s]
    m = re.match(r"^(?:MA|UN|PB)\d+\.\d+_(.+)$", s)
    if m:
        return m.group(1).upper()
    raise KeyError(f"{s}: not in the HOCOMOCO annotation and not a JASPAR id")


def confident(qvals, threshold):
    """q <= threshold; a pattern without a match (NaN) is never confident."""
    q = np.asarray(qvals, dtype=float)
    return ~np.isnan(q) & (q <= threshold)


def _top(tsv, n=3):
    t = pd.read_csv(tsv, sep="\t", comment="#").dropna(subset=["Query_ID"])
    t = t.sort_values(["Query_ID", "q-value", "p-value"])
    cols = ["Target_ID", "p-value", "q-value", "Overlap"]
    return {q: g.head(n)[cols].values.tolist() for q, g in t.groupby("Query_ID")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", action="append", required=True, help="label=modisco.h5, repeatable")
    p.add_argument("--target", required=True)
    p.add_argument("--h14_annotation", required=True)
    p.add_argument("--tomtom", required=True)
    p.add_argument("--qval_threshold", type=float, required=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    leaves = [s.split("=", 1) for s in a.h5]
    h14 = h14_symbols(a.h14_annotation)

    pats = {}
    for label, h5 in leaves:
        pats[label] = patterns_of(h5)
        blocks = []
        for r in pats[label]:
            trimmed, start, end = modisco_trim(r["ppm"], r["cwm"])
            r["trim"], r["trim_start"], r["trim_end"] = trimmed, start, end
            blocks.append(f"MOTIF {r['key']}\nletter-probability matrix: alength= 4 "
                          f"w= {len(trimmed)}\n"
                          + "\n".join(" ".join(f"{v:.6f}" for v in row) for row in trimmed) + "\n")
        (out / f"{label}_query.meme").write_text(MEME_HEADER + "\n".join(blocks))
        print(f"{label}: {len(pats[label])} patterns", flush=True)

    # TomTom always reruns: a cached result would be joined by pattern key to a
    # modisco.h5 whose pattern_N may no longer be the same motif.
    for label, _ in leaves:
        d = out / f"tomtom_{label}"
        if d.exists():
            shutil.rmtree(d)
        subprocess.run(tomtom_argv(a.tomtom, out / f"{label}_query.meme", a.target, d),
                       check=True)

    rows = []
    for label, _ in leaves:
        hits = _top(out / f"tomtom_{label}" / "tomtom.tsv")
        for r in pats[label]:
            h = hits.get(r["key"], [])
            rec = dict(label=label, key=r["key"], group=r["group"], pattern=r["pattern"],
                       n_seqlets=r["n_seqlets"], trim_len=len(r["trim"]),
                       trim_start=r["trim_start"], trim_end=r["trim_end"])
            for i in range(3):
                tid = h[i][0] if len(h) > i else ""
                rec[f"match{i}"] = tid
                rec[f"tf{i}"] = tf_name(tid, h14)
                rec[f"pval{i}"] = float(h[i][1]) if len(h) > i else np.nan
                rec[f"qval{i}"] = float(h[i][2]) if len(h) > i else np.nan
                rec[f"overlap{i}"] = int(h[i][3]) if len(h) > i else -1
            rec["name"] = rec["tf0"]
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["confident"] = confident(df["qval0"], a.qval_threshold)
    df["qval_threshold"] = a.qval_threshold
    df["target"] = Path(a.target).name
    df["target_sha256"] = file_sha256(a.target)[:12]
    df["h14_annotation_sha256"] = file_sha256(a.h14_annotation)[:12]
    df.to_csv(out / "pattern_names.tsv", sep="\t", index=False)
    print(f"pattern_names.tsv: {len(df)} patterns, {int(df.confident.sum())} with q <= "
          f"{a.qval_threshold}, {int((df.match0 == '').sum())} without a match")


if __name__ == "__main__":
    main()
