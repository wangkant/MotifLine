#!/usr/bin/env python
"""MoDISco patterns -> names, through TomTom against the source union.

The naming chain, in one stage (user, 2026-09-27: TomTom against the current union
library; every hand-set gate dropped):

    modisco.h5  --modiscolite trim-->  query.meme
                --TomTom vs lexicon_sources_union.meme-->  top 3 library matches
                --TF symbol of the first match-->  name

Nothing is gated: every pattern is named after its first match whatever its q-value,
and the q-value is reported beside it. No RSAT cluster, no Ncor arm, no family, no
repeat call, no name shared across models.

Two details that are load-bearing:

1. **The query trim is modiscolite's**, not ours: ``score = sum(|CWM|, axis=1)``, keep
   the span between the first and last column at or above 30% of the max, and cut the
   PPM there (modiscolite ``report.py``). The CWM used is ``contrib_scores`` -- what
   modiscolite's own report trims on -- even when the attributions fed to MoDISco were
   hypothetical. Changing it would change the query and so the match.
2. **``-mi 1`` is deliberately absent** from the TomTom command line. modiscolite passes
   it because it writes one motif per temporary file; against a batch query file it
   silently queries only the first motif and returns nothing for the rest.
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
from motif_matrix import MEME_HEADER  # noqa: E402

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


def tf_name(motif_id):
    """TF symbol out of a union id. JASPAR: MA0140.3_GATA1::TAL1 -> GATA1::TAL1.
    HOCOMOCO v14: CTCF.H14CORE.0.P.B -> CTCF."""
    s = str(motif_id)
    m = re.match(r"^(?:MA|UN|PB)\d+\.\d+_(.+)$", s)
    if m:
        return m.group(1)
    return s.split(".")[0] if s else ""


def _top(tsv, n=3):
    """``{query: [[target, p, q, overlap], ...]}``, best q first, ties broken by p."""
    t = pd.read_csv(tsv, sep="\t", comment="#").dropna(subset=["Query_ID"])
    t = t.sort_values(["Query_ID", "q-value", "p-value"])
    cols = ["Target_ID", "p-value", "q-value", "Overlap"]
    return {q: g.head(n)[cols].values.tolist() for q, g in t.groupby("Query_ID")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", action="append", required=True, help="label=modisco.h5, repeatable")
    p.add_argument("--target", required=True, help="lexicon_sources_union.meme")
    p.add_argument("--tomtom", required=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    leaves = [s.split("=", 1) for s in a.h5]

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

    # ---- 2. TomTom against the union (batch query: NO -mi 1) ----
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

    # ---- 3. assemble: name = TF of the first match, no gate ----
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
                rec[f"tf{i}"] = tf_name(tid)
                rec[f"pval{i}"] = float(h[i][1]) if len(h) > i else np.nan
                rec[f"qval{i}"] = float(h[i][2]) if len(h) > i else np.nan
                rec[f"overlap{i}"] = int(h[i][3]) if len(h) > i else -1
            rec["name"] = rec["tf0"]
            rows.append(rec)
    df = pd.DataFrame(rows)
    # Provenance on the table itself: someone holding pattern_names.tsv must be able to
    # say which library produced these names, and config/default.yaml is deliberately
    # untracked, so it cannot be the answer.
    df["target"] = Path(a.target).name
    df["target_sha256"] = target_key[:12]
    df.to_csv(out / "pattern_names.tsv", sep="\t", index=False)
    print(f"\npattern_names.tsv: {len(df)} patterns; without any TomTom match "
          f"{int((df.match0 == '').sum())}")
    for label, _ in leaves:
        s = df[df.label == label]
        print(f"\n===== {label} =====")
        print(s[["key", "n_seqlets", "name", "match0", "qval0"]].to_string(index=False))


if __name__ == "__main__":
    main()
