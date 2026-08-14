"""Level-1 equivalence: new-pipeline attributions vs the archived prefix.

Acceptance is fixed before any run: aligned regions AND per-row pearson >=
0.999. The report records the real values either way; a failing comparison is
recorded, never re-thresholded.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROW_PEARSON_GATE = 0.999


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
