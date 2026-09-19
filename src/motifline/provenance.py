"""One ``run_info.json`` per stage directory: what produced the files next to it.

A results tree is only worth handing to somebody else if each directory can answer
"which version of which tool, at which parameters, when". Nothing else in this
pipeline records that: the attribution/discovery/hit-calling versions live in a conda
prefix that is not part of the artifacts, and the parameters live in a config file that
gets edited. A stage that finished six months ago cannot be read back from either.

This is deliberately not the audit layer that was removed on 2026-08-14 -- it writes
one file, it is never read back by the pipeline, and no stage refuses to run because of
it. It is a label on the output, not a gate on the input.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from motifline import __version__

#: What the compute env contributes to a result, in the order worth reading.
PACKAGES = ("finemo", "modisco-lite", "tangermeme", "chrombpnet-pytorch",
            "gpn", "torch", "numpy")

RUN_INFO = "run_info.json"

_PROBE = """
import importlib.metadata as m, json, sys
out = {}
for p in sys.argv[1:]:
    try:
        out[p] = m.version(p)
    except Exception:
        out[p] = None
print(json.dumps(out))
"""


def now() -> str:
    """UTC, to the second, so two stage directories sort by when they ran."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@lru_cache(maxsize=None)
def tool_versions(env_prefix: str, packages: tuple[str, ...] = PACKAGES) -> dict:
    """Installed versions of ``packages`` inside ``env_prefix``, as that env reports them.

    Asking the env itself is the only honest answer -- the stage bodies run under a
    different interpreter than snakemake does. A missing package maps to ``None``; an
    env that cannot be queried at all reports why under ``unavailable`` rather than
    raising, because provenance must never be the thing that fails a run.
    """
    python = Path(env_prefix) / "bin" / "python"
    if not os.access(python, os.X_OK):
        return {"unavailable": f"{python} is not an executable interpreter"}
    try:
        done = subprocess.run([str(python), "-c", _PROBE, *packages],
                              capture_output=True, text=True, timeout=120, check=True)
        return json.loads(done.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        return {"unavailable": f"{type(e).__name__}: {e}"}


def record(out_dir, *, stage: str, argv, param_sig: str | None = None,
           started: str | None = None, env_prefix: str | None = None,
           packages: tuple[str, ...] = PACKAGES) -> Path:
    """Write ``run_info.json`` into a finished stage directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    finished = now()
    info = {
        "stage": stage,
        "motifline": __version__,
        "started": started or finished,
        "finished": finished,
        "argv": [str(a) for a in argv],
        "param_sig": param_sig,
        "env": env_prefix,
        "tools": tool_versions(env_prefix, tuple(packages)) if env_prefix else None,
        "host": os.uname().nodename,
        "orchestrator_python": sys.version.split()[0],
    }
    path = out_dir / RUN_INFO
    path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return path
