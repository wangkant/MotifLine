"""Runs that rule all asks for. A cancelled run stays reachable by explicit target."""
from __future__ import annotations


def _runs(cfg: dict) -> dict:
    return cfg.get("runs") or {}


def is_cancelled(run_cfg: dict) -> bool:
    return bool(run_cfg.get("cancelled"))


def cancelled_runs(cfg: dict) -> list[str]:
    return sorted(k for k, v in _runs(cfg).items() if is_cancelled(v))


def active_runs(cfg: dict, names) -> list[str]:
    runs = _runs(cfg)
    unknown = [n for n in names if n not in runs]
    if unknown:
        raise KeyError(f"not defined under config.runs: {unknown}")
    return [n for n in names if not is_cancelled(runs[n])]
