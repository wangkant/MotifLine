"""Equivalence reports: new-pipeline products vs the archived ones.

Level 1 (``compare_attr``) keeps the hard, pre-registered gate it was written
with: aligned regions AND per-row pearson >= 0.999. Levels 2 and 3
(``compare_modisco``, ``compare_hits``) are consistency checks, not gates --
the canonical runs use the tools' official parameters, which are not always the
archive's, so a difference here is evidence about a parameter, not a verdict.
They therefore report the real numbers plus a *soft* tolerance band and a list
of flags; ``within_tolerance`` is a summary of that band and never suppresses a
number.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROW_PEARSON_GATE = 0.999

# Soft tolerances (buildout plan, Global Constraints): out-of-band is a flagged
# finding to attribute to a parameter change, not a failure.
CWM_COSINE_TOL = 0.85
PATTERN_COUNT_TOL = 2
HIT_RATIO_BAND = (0.8, 1.25)


def compare_attr(new_dir, old_attr_npy, old_regions_bed, n, out_json) -> dict:
    new_dir = Path(new_dir)
    A = np.load(new_dir / "attr.npy").astype(np.float64)
    B = np.asarray(np.load(old_attr_npy, mmap_mode="r")[:n], dtype=np.float64)
    ra = pd.read_csv(new_dir / "regions.bed", sep="\t", header=None).iloc[:, :3]
    rb = pd.read_csv(old_regions_bed, sep="\t", header=None).iloc[:n, :3]
    aligned = bool(len(ra) == n == len(A) == len(B)
                   and ra.reset_index(drop=True).astype(str).equals(
                       rb.reset_index(drop=True).astype(str)))
    fa, fb = A.reshape(len(A), -1), B.reshape(len(B), -1)
    m = min(len(fa), len(fb))
    rows_r = [float(np.corrcoef(fa[i], fb[i])[0, 1]) for i in range(m)]
    diff = np.abs(fa[:m] - fb[:m])
    report = {
        "n": int(m),
        "regions_aligned": aligned,
        "bitwise_equal_rows": int((diff.max(axis=1) == 0).sum()),
        "max_abs_diff": float(diff.max()) if m else float("nan"),
        "min_row_pearson": float(min(rows_r)) if rows_r else float("nan"),
        "mean_row_pearson": float(np.mean(rows_r)) if rows_r else float("nan"),
        "row_pearson_gate": ROW_PEARSON_GATE,
    }
    report["passed"] = bool(aligned and rows_r and min(rows_r) >= ROW_PEARSON_GATE)
    Path(out_json).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _cwms(h5path) -> dict:
    """Top-level MoDISco patterns as ``{"<metacluster>/<name>": (L,4) contrib_scores}``."""
    import h5py
    out = {}
    with h5py.File(h5path) as f:
        for mc in ("pos_patterns", "neg_patterns"):
            if mc not in f:
                continue
            for name, g in f[mc].items():
                if "contrib_scores" in g:
                    out[f"{mc}/{name}"] = np.asarray(g["contrib_scores"])
    return out


def _best_cosine(a, b, max_offset=20) -> float:
    """Best cosine over both orientations and +-max_offset alignments of two CWMs."""
    best = -1.0
    for bb in (b, b[::-1, ::-1]):
        for off in range(-max_offset, max_offset + 1):
            lo_a, lo_b = max(0, off), max(0, -off)
            n = min(len(a) - lo_a, len(bb) - lo_b)
            if n < 8:
                continue
            x = a[lo_a:lo_a + n].ravel()
            y = bb[lo_b:lo_b + n].ravel()
            d = float(np.linalg.norm(x) * np.linalg.norm(y))
            if d > 0:
                best = max(best, float(x @ y) / d)
    return best


def compare_modisco(new_h5, old_h5, out_json) -> dict:
    """Level-2 consistency: does the new lexicon contain the archived motifs?"""
    A, B = _cwms(new_h5), _cwms(old_h5)
    rows = []
    for ko, cwm_o in B.items():
        scores = {kn: _best_cosine(A[kn], cwm_o) for kn in A}
        best = max(scores, key=scores.get) if scores else None
        rows.append({"old": ko, "best_new": best,
                     "cosine": float(scores[best]) if best else float("nan")})
    counts = {f"{s}_{w}": sum(k.startswith(s) for k in (A if w == "new" else B))
              for s in ("pos", "neg") for w in ("new", "old")}
    rep = {"new_h5": str(new_h5), "old_h5": str(old_h5),
           "n_patterns_new": len(A), "n_patterns_old": len(B), **counts,
           "per_old_pattern": rows,
           "min_best_cosine": min((r["cosine"] for r in rows), default=float("nan")),
           "mean_best_cosine": float(np.mean([r["cosine"] for r in rows])) if rows else float("nan"),
           "n_old_below_cosine_tol": sum(r["cosine"] < CWM_COSINE_TOL for r in rows),
           "cwm_cosine_tol": CWM_COSINE_TOL, "pattern_count_tol": PATTERN_COUNT_TOL}
    flags = []
    for s in ("pos", "neg"):
        d = counts[f"{s}_new"] - counts[f"{s}_old"]
        if abs(d) > PATTERN_COUNT_TOL:
            flags.append(f"{s} pattern count off by {d:+d} "
                         f"(new={counts[f'{s}_new']} old={counts[f'{s}_old']}, tol +-{PATTERN_COUNT_TOL})")
    low = [r for r in rows if r["cosine"] < CWM_COSINE_TOL]
    if low:
        flags.append(f"{len(low)}/{len(rows)} archived patterns below best-match cosine "
                     f"{CWM_COSINE_TOL}: " + ", ".join(f"{r['old']}={r['cosine']:.3f}" for r in low[:10]))
    rep["flags"] = flags
    rep["within_tolerance"] = not flags
    Path(out_json).write_text(json.dumps(rep, indent=2) + "\n")
    return rep


def compare_hits(new_tsv, old_tsv, out_json, tol_bp=2) -> dict:
    """Level-3 consistency: per-motif hit counts and hit-set overlap, new vs archived."""
    def load(p):
        df = pd.read_csv(p, sep="\t")
        need = [c for c in ("chr", "start", "end", "motif_name") if c not in df.columns]
        if need:
            raise ValueError(f"{p}: missing columns {need}; adapt keys to the actual header")
        return df

    a, b = load(new_tsv), load(old_tsv)
    counts = pd.DataFrame({"new": a.groupby("motif_name").size(),
                           "old": b.groupby("motif_name").size()}).fillna(0).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        counts["ratio_new_over_old"] = counts["new"] / counts["old"].replace(0, np.nan)
    ka = set(zip(a["chr"], a["start"] // (tol_bp + 1), a["motif_name"]))
    kb = set(zip(b["chr"], b["start"] // (tol_bp + 1), b["motif_name"]))
    exact_a = set(zip(a["chr"], a["start"], a["end"], a["motif_name"]))
    exact_b = set(zip(b["chr"], b["start"], b["end"], b["motif_name"]))
    lo, hi = HIT_RATIO_BAND
    per_motif = counts.reset_index().rename(columns={"index": "motif_name"})
    per_motif = per_motif.where(pd.notnull(per_motif), None).to_dict("records")
    rep = {"new_tsv": str(new_tsv), "old_tsv": str(old_tsv),
           "n_new": int(len(a)), "n_old": int(len(b)),
           "ratio_total_new_over_old": float(len(a) / len(b)) if len(b) else float("nan"),
           "exact_jaccard": len(exact_a & exact_b) / max(1, len(exact_a | exact_b)),
           "tolerant_jaccard": len(ka & kb) / max(1, len(ka | kb)), "tol_bp": tol_bp,
           "hit_ratio_band": list(HIT_RATIO_BAND),
           "n_motifs_new": int((counts["new"] > 0).sum()), "n_motifs_old": int((counts["old"] > 0).sum()),
           "per_motif_counts": per_motif}
    flags = []
    for r in per_motif:
        ratio = r["ratio_new_over_old"]
        if ratio is None or (isinstance(ratio, float) and np.isnan(ratio)):
            flags.append(f"{r['motif_name']}: absent from the archived hits (new={r['new']}, old=0)")
        elif not (lo <= ratio <= hi):
            flags.append(f"{r['motif_name']}: count ratio {ratio:.3f} outside [{lo}, {hi}] "
                         f"(new={r['new']} old={r['old']})")
    rep["flags"] = flags
    rep["within_tolerance"] = not flags
    Path(out_json).write_text(json.dumps(rep, indent=2) + "\n")
    return rep
