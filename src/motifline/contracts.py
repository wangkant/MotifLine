"""Stage contract: provenance+params land BEFORE the body, run_status lands last.

A record that arrives only on success is a record the runs you most want to
explain never get (motiflex.run_status docstring); this module keeps that
ordering for every motifline stage. Refusal = a failing GuardResult: its
outcome is on disk (guard_outcomes.json) before the raise, and the stage body
never runs, so a REFUSED directory holds no result written by this run.
"""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
from motiflex.guards import GuardError
from motiflex.guard_log import GuardLog
from motiflex.provenance import ProvenanceRecord
from motiflex.run_status import read_run_status, write_run_status

PARAMS_FILENAME = "params.json"


def append_params(out_dir: str | os.PathLike, params: dict) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / PARAMS_FILENAME
    entries = json.loads(path.read_text()) if path.exists() else []
    if not isinstance(entries, list):
        raise ValueError(f"{path} is not a JSON list; refusing to overwrite it")
    entries.append(params)
    staged = path.with_name(f".{PARAMS_FILENAME}.{os.getpid()}.partial")
    staged.write_text(json.dumps(entries, indent=2) + "\n")
    os.replace(staged, path)
    return len(entries)


def params_of_success(out_dir: str | os.PathLike) -> dict | None:
    """Params of the run the directory's artifacts belong to (join on record count)."""
    rs = read_run_status(out_dir)
    if rs is None:
        return None
    afo = rs.get("artifacts_are_from")
    if not isinstance(afo, dict):
        return None
    k = afo.get("provenance_records")
    path = Path(out_dir) / PARAMS_FILENAME
    if not isinstance(k, int) or not path.exists():
        return None
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not (1 <= k <= len(entries)):
        return None
    return entries[k - 1]


def run_stage(out_dir, stage: str, argv: list, params: dict, pre_guards=(),
              post_guards=(), inputs=(), software: dict | None = None,
              seed: int | None = None, env: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    append_params(out, params)                                   # before body
    rec = ProvenanceRecord(command=" ".join(str(a) for a in argv),
                           subcommand=stage, random_seed=seed,
                           software=dict(software) if software else {})
    for p in inputs:
        rec.add_input(p)
    rec.write(out)                                               # before body
    log = GuardLog(stage, out_dir=out)
    cmd = rec.command

    def _run_guards(guards):
        for thunk in guards:
            result, subject = thunk()
            log.record(result, subject=subject).raise_if_failed()

    try:
        _run_guards(pre_guards)
    except GuardError as e:
        write_run_status(out, status="REFUSED", command=cmd, subcommand=stage, detail=str(e))
        raise
    try:
        subprocess.run([str(a) for a in argv], check=True,
                       env={**os.environ, **env} if env else None)
    except subprocess.CalledProcessError as e:
        write_run_status(out, status="CRASHED", command=cmd, subcommand=stage,
                         detail=f"body exited {e.returncode}")
        raise
    try:
        _run_guards(post_guards)
    except GuardError as e:
        write_run_status(out, status="REFUSED", command=cmd, subcommand=stage,
                         detail=f"post-run guard: {e}")
        raise
    write_run_status(out, status="SUCCESS", command=cmd, subcommand=stage)
