"""Was this attribution hypothetical? Ask the array, not the log.

``--hypothetical`` decides whether a score is stored for every base at every position
or only for the base that is actually there. The two come out as arrays of the same
shape, the same dtype and the same filename, and every stage downstream will consume
either without complaint: MoDISco is documented to take hypothetical attributions,
and giving it projected ones changes what it finds without failing.

Nothing in a run log distinguishes them. The array does:

    frac = (|attr| > 1e-9).mean()      ~1.00 hypothetical,  ~0.25 projected

``attr_ag.py`` refuses to run without the flag and ``attr_cbp.py`` now does too, so
this measurement should never disagree with what was asked for. It is recorded anyway,
in every attribution stage's ``run_info.json``, because the whole point is that the
flag's effect was previously unobservable after the fact.

It reports; it does not raise. An attribution that cost GPU-days is not deleted by its
own label, and a measurement that could fail a finished run would eventually be
switched off.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: Above this, an array carries a score on essentially every base: hypothetical.
#: A projected array sits at ~0.25 (one base in four), far below anything reachable
#: from the other side, so the exact floor is not a tuned number.
HYP_FLOOR = 0.99

#: Enough rows to tell 1.0 from 0.25; the arrays run to tens of thousands of regions.
SAMPLE_ROWS = 64


def hyp_fraction(path, rows: int = SAMPLE_ROWS) -> float:
    """Fraction of entries carrying a score, over the first ``rows`` regions."""
    arr = np.load(Path(path), mmap_mode="r")
    return float((np.abs(np.asarray(arr[:rows])) > 1e-9).mean())


def hyp_check(name: str = "attr.npy", rows: int = SAMPLE_ROWS):
    """A stage check: measure the written attribution and say what it looks like."""
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
            out["warning"] = (
                f"{path.name} scores {frac:.4f} of entries, not ~1.0: this does NOT "
                f"look like a hypothetical attribution (~0.25 means projected). "
                f"Everything downstream will consume it without complaining.")
        return out
    return check
