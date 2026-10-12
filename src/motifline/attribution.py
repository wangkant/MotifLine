"""Tell hypothetical from projected attributions by the array itself.

Both have the same shape and dtype. frac(|attr| > 1e-9) is ~1.0 for hypothetical
scores and ~0.25 for projected ones. The check is recorded and warned about, never fatal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HYP_FLOOR = 0.99
SAMPLE_ROWS = 64


def hyp_fraction(path, rows: int = SAMPLE_ROWS) -> float:
    arr = np.load(Path(path), mmap_mode="r")
    return float((np.abs(np.asarray(arr[:rows])) > 1e-9).mean())


def hyp_check(name: str = "attr.npy", rows: int = SAMPLE_ROWS):
    def check(out_dir) -> dict:
        path = Path(out_dir) / name
        try:
            frac = hyp_fraction(path, rows=rows)
        except (OSError, ValueError) as e:
            return {"looks_hypothetical": None,
                    "hypothetical_check": f"{path.name}: {type(e).__name__}: {e}"}
        out = {"hypothetical_fraction": round(frac, 4),
               "looks_hypothetical": frac >= HYP_FLOOR,
               "hypothetical_check": f"frac(|attr| > 1e-9) over {rows} rows; "
                                     f"~1.0 hypothetical, ~0.25 projected"}
        if frac < HYP_FLOOR:
            out["warning"] = (f"{path.name}: {frac:.4f} of entries are nonzero; this does "
                              f"not look like a hypothetical attribution")
        return out
    return check
