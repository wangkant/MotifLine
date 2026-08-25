#!/usr/bin/env python
"""MoDISco patterns -> cluster-level names, through TomTom against the tagged union.

The naming chain, in one stage:

    modisco.h5  --modiscolite trim-->  query.meme
                --TomTom vs union_cltagged.meme-->  best library match
                --the cluster id already in that match's name-->  assigned_cluster
                --gate on (q, best_ncor)-->  annotated yes/no

Three details that are load-bearing:

1. **The query trim is modiscolite's**, not ours: ``score = sum(|CWM|, axis=1)``, keep
   the span between the first and last column at or above 30% of the max, and cut the
   PPM there (modiscolite ``report.py``). The CWM used is ``contrib_scores`` -- what
   modiscolite's own report trims on -- even when the attributions fed to MoDISco were
   hypothetical. Changing it would change the query and so the match.
2. **``-mi 1`` is deliberately absent** from the TomTom command line. modiscolite passes
   it because it writes one motif per temporary file; against a batch query file it
   silently queries only the first motif and returns nothing for the rest.
3. **``best_ncor`` is a second, independent measurement**, on the IC>0.3 core rather
   than the |CWM| trim, from the same Ncor that built the cluster tree. The gate is
   ``q < gate_q AND best_ncor >= gate_ncor``: q says "closer than chance", Ncor says
   "close enough that the clustering would have merged them". q alone passes matches
   that are statistically real and biologically wrong -- a repeat element whose core
   happens to contain a TF half-site is the standard case.

An ungated pattern is reported with its match and marked ``annotated=False``; it is not
renamed and it is not proposed to ``mlex annotate``. Nothing here assigns a *family*:
grouping cluster ids into site types is a human judgement made once per project, not a
derivable function of the matrices.
"""
import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motif_matrix import MEME_HEADER, ic_core, ncor_matrix, read_meme  # noqa: E402

GROUPS = ("pos_patterns", "neg_patterns")


def _digest(path):
    """sha256 of the target library, so a rebuilt lexicon cannot be silently reused."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _complete(tsv):
    """TomTom closes its TSV with a '# Tomtom ... Version' comment block.

    A run killed mid-write leaves a file that `pd.read_csv(comment='#')` parses happily
    and that yields fewer matches than it should, so completeness is checked, not
    assumed."""
    try:
        tail = open(tsv, "rb").read()[-4096:].decode("utf-8", "replace")
    except OSError:
        return False
    return any(l.startswith("# Tomtom") for l in tail.splitlines())


def modisco_trim(ppm, cwm, threshold=0.3):
    """modiscolite report.py's trim: |CWM| row sums at >= 30% of max bound the PPM."""
    score = np.abs(cwm).sum(1)
    keep = np.where(score >= score.max() * threshold)[0]
    if len(keep) == 0:
        return ppm, 0, len(ppm)
    return ppm[keep.min():keep.max() + 1], int(keep.min()), int(keep.max() + 1)


def patterns_of(h5_path):
    """Every pattern in the file, pos then neg, in numeric index order."""
    out = []
    with h5py.File(h5_path) as f:
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
    """modiscolite report.py's TomTom line, minus ``-mi 1``.

    ``-mi 1`` means "query only motif number 1". modiscolite can pass it because it
    writes one motif per temporary file; here the query file holds every pattern, so
    it would return matches for the first and silence for the rest.
    """
    return [tomtom, "-no-ssc", "-oc", str(out_dir), "-verbosity", "1",
            "-min-overlap", "5", "-dist", "pearson", "-evalue", "-thresh", "10.0",
            str(query), str(target)]


def gated(qval, best_ncor, gate_q, gate_ncor):
    """Both arms, or not annotated. A missing Ncor (no comparable IC core) fails."""
    return bool(qval == qval and qval < gate_q
                and best_ncor == best_ncor and best_ncor >= gate_ncor)


def db_matches(df, node_ids):
    """The gated rows as ``mlex annotate`` database entries.

    The proposed label is the RSAT cluster, never the individual library entry: the
    matched entry is recorded alongside as the evidence for it.
    """
    out = []
    for r in df[df.annotated].itertuples():
        m = dict(node_id=f"{node_ids.get(r.label, r.label)}::{r.key}",
                 proposed_family_id=r.assigned_cluster,
                 matched_motif_id=r.match0,
                 q_value=float(r.qval0))
        if r.overlap0 >= 0:
            m["aligned_span"] = int(r.overlap0)
        out.append(m)
    return out


def _top(tsv, n=3):
    """``{query: [[target, q, overlap], ...]}``, best q first."""
    t = pd.read_csv(tsv, sep="\t", comment="#").dropna(subset=["Query_ID"])
    t = t.sort_values(["Query_ID", "q-value", "p-value"])
    cols = ["Target_ID", "q-value"] + (["Overlap"] if "Overlap" in t.columns else [])
    return {q: g.head(n)[cols].values.tolist() for q, g in t.groupby("Query_ID")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", action="append", required=True, help="label=modisco.h5, repeatable")
    p.add_argument("--node_id", action="append", default=[],
                   help="label=<mlex analysis id>; db.yaml node ids are <id>::<pattern key>")
    p.add_argument("--target", required=True, help="union_cltagged.meme")
    p.add_argument("--source", action="append", required=True,
                   help="tag=path of the untagged union sources, for the Ncor arm")
    p.add_argument("--tomtom", required=True)
    p.add_argument("--w_th", type=int, required=True,
                   help="minimum comparable overlap in columns for the Ncor arm")
    p.add_argument("--gate_q", type=float, required=True)
    p.add_argument("--gate_ncor", type=float, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--db_yaml", help="also emit an mlex annotate database for the gated matches")
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    leaves = [s.split("=", 1) for s in a.h5]
    node_ids = dict(s.split("=", 1) for s in a.node_id)

    # ---- 1. query MEME per leaf, on modiscolite's trim ----
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
        print(f"{label}: {len(pats[label])} patterns, trimmed width median "
              f"{np.median([len(r['trim']) for r in pats[label]]):.0f}", flush=True)

    # ---- 2. TomTom against the tagged union (batch query: NO -mi 1) ----
    target_key = _digest(a.target)
    for label, _ in leaves:
        d = out / f"tomtom_{label}"
        stamp = d / "target.sha256"
        # Reuse only when the SAME target library produced a COMPLETE run. Existence
        # alone was not enough: a rebuilt reference lexicon left the old tomtom.tsv in
        # place (it is not a rule output, so snakemake never removed it) and the names
        # came from the previous library while db.yaml recorded the new one; and an
        # interrupted TomTom left a truncated tsv that parsed without complaint.
        usable = (d / "tomtom.tsv").exists() and _complete(d / "tomtom.tsv") \
            and stamp.exists() and stamp.read_text().strip() == target_key
        if not usable:
            if d.exists():
                shutil.rmtree(d)
            subprocess.run(tomtom_argv(a.tomtom, out / f"{label}_query.meme", a.target, d),
                           check=True)
            stamp.write_text(target_key + "\n")
        print(f"  tomtom {label} done", flush=True)

    # ---- 3. Ncor of each pattern's IC core against every library matrix ----
    lib_names, lib_mats = [], []
    for spec in a.source:
        _, path = spec.split("=", 1)
        for n, (ppm, _) in read_meme(path).items():
            lib_names.append(n)
            lib_mats.append(ppm)
    nl = len(lib_mats)
    core_of = []                       # (label, key) for each appended pattern row
    mats = list(lib_mats)
    for label, _ in leaves:
        for r in pats[label]:
            core = ic_core(r["ppm"], w_th=a.w_th)
            r["ic_core_len"] = 0 if core is None else len(core)
            if core is None:           # IC core shorter than the comparable minimum
                continue
            core_of.append((label, r["key"]))
            mats.append(core)
    S, _, _, _ = ncor_matrix(mats, w_th=a.w_th)
    best_ncor = {}
    for i, k in enumerate(core_of):
        row = S[nl + i, :nl]
        j = int(row.argmax())
        best_ncor[k] = (round(float(row[j]), 3), lib_names[j])

    # ---- 4. assemble: the cluster is already in the matched name ----
    rows = []
    for label, _ in leaves:
        hits = _top(out / f"tomtom_{label}" / "tomtom.tsv")
        for r in pats[label]:
            h = hits.get(r["key"], [])
            rec = dict(label=label, key=r["key"], group=r["group"], pattern=r["pattern"],
                       n_seqlets=r["n_seqlets"], trim_len=len(r["trim"]),
                       trim_start=r["trim_start"], trim_end=r["trim_end"],
                       ic_core_len=r["ic_core_len"])
            for i in range(3):
                tid = h[i][0] if len(h) > i else ""
                rec[f"match{i}"] = tid.split("__", 1)[1] if "__" in tid else tid
                rec[f"cluster{i}"] = tid.split("__", 1)[0] if "__" in tid else ""
                rec[f"qval{i}"] = float(h[i][1]) if len(h) > i else np.nan
                rec[f"overlap{i}"] = int(h[i][2]) if len(h) > i and len(h[i]) > 2 else -1
            rec["assigned_cluster"] = rec["cluster0"]
            seen = {rec[f"cluster{i}"] for i in range(3) if rec[f"cluster{i}"]}
            rec["cluster_stable"] = bool(rec["cluster0"]) and len(seen) == 1
            nc, nc_match = best_ncor.get((label, r["key"]), (np.nan, ""))
            rec["best_ncor"] = nc
            rec["best_ncor_match"] = nc_match
            rec["annotated"] = gated(rec["qval0"], nc, a.gate_q, a.gate_ncor)
            rows.append(rec)
    df = pd.DataFrame(rows)
    # Provenance on the table itself: someone holding pattern_names.tsv must be able to
    # say which library and which gate produced these names, and config/default.yaml is
    # deliberately untracked, so it cannot be the answer.
    df["target"] = Path(a.target).name
    df["target_sha256"] = target_key[:12]
    df["gate_q"] = a.gate_q
    df["gate_ncor"] = a.gate_ncor
    df["w_th"] = a.w_th
    df.to_csv(out / "pattern_names.tsv", sep="\t", index=False)
    print(f"\npattern_names.tsv: {len(df)} patterns; annotated {int(df.annotated.sum())}, "
          f"below the gate {int((~df.annotated).sum())} "
          f"(q arm {int((df.qval0 >= a.gate_q).sum())}, "
          f"Ncor arm {int((~(df.best_ncor >= a.gate_ncor)).sum())})")
    for label, _ in leaves:
        s = df[df.label == label]
        print(f"\n===== {label} =====")
        print(s[["key", "n_seqlets", "assigned_cluster", "qval0", "best_ncor",
                 "cluster_stable", "annotated"]].to_string(index=False))

    # ---- 5. optional: the mlex annotate database, gated matches only ----
    if a.db_yaml:
        import yaml
        # Fail loudly: motiflex copies this string into every annotation row's
        # source_version, and "MEME unknown" is a provenance hole, not a value.
        v = subprocess.run([a.tomtom, "-version"], capture_output=True, text=True,
                           check=True).stdout.strip()
        matches = db_matches(df, node_ids)
        Path(a.db_yaml).parent.mkdir(parents=True, exist_ok=True)
        Path(a.db_yaml).write_text(yaml.safe_dump(
            {"tomtom": {"version": f"MEME {v} / {Path(a.target).name} sha256:{target_key[:12]}",
                        "matches": matches},
             "family_level_output": True}, sort_keys=False, width=200))
        print(f"\n{a.db_yaml}: {len(matches)} gated matches proposed to mlex annotate")


if __name__ == "__main__":
    main()
