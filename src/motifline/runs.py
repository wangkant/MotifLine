"""Which runs the default target list still asks for.

A cancelled run's config entry is **not** deleted. The entry is what keeps a
half-finished stage tree resumable (`snakemake -- results/<run>/…` still
resolves) and what lets its parameters be read back later; deleting it would
turn the directory on disk into an orphan whose provenance no longer names a
run definition. What cancellation changes is narrower: `rule all` stops asking
for the run, so typing `snakemake` with no target cannot restart GPU-weeks of
work that a person decided to stop.

The cancelled run must say where the decision is written -- `cancelled_note` in
config points at the `CANCELLED.md` next to its artifacts -- because a flag in a
yaml file is not a reason.
"""
from __future__ import annotations


def _runs(cfg: dict) -> dict:
    return cfg.get("runs") or {}


def is_cancelled(run_cfg: dict) -> bool:
    return bool(run_cfg.get("cancelled"))


def cancelled_runs(cfg: dict) -> list[str]:
    """Names of the runs config marks cancelled, sorted."""
    return sorted(k for k, v in _runs(cfg).items() if is_cancelled(v))


def active_runs(cfg: dict, names) -> list[str]:
    """``names`` minus the cancelled runs, order preserved.

    An unknown name raises rather than being dropped: a typo in the default
    target list would otherwise shrink `rule all` in exactly the same way a
    cancellation does, and the two must not be indistinguishable.
    """
    runs = _runs(cfg)
    unknown = [n for n in names if n not in runs]
    if unknown:
        raise KeyError(f"not defined under config.runs: {unknown}")
    return [n for n in names if not is_cancelled(runs[n])]
