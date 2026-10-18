"""run_info.json: argv, parameters, tool versions and model code behind a stage."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from motifline import __version__
from motifline.steps.digests import git_commit, tree_sha256

PACKAGES = ("finemo", "modisco-lite", "tangermeme", "torch", "numpy")
# the env the official ChromBPNet attribution runs in
CBP_PACKAGES = ("chrombpnet", "tensorflow", "kundajelab-shap", "deeplift", "numpy")

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
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _ProbeFailed(Exception):
    pass


@lru_cache(maxsize=None)
def _probe(env_prefix: str, packages: tuple[str, ...]) -> dict:
    """Cached only when it returns; a failure raises, so the next call probes again."""
    python = Path(env_prefix) / "bin" / "python"
    if not os.access(python, os.X_OK):
        raise _ProbeFailed(f"{python} is not an executable interpreter")
    try:
        done = subprocess.run([str(python), "-c", _PROBE, *packages],
                              capture_output=True, text=True, timeout=120, check=True)
        return json.loads(done.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        raise _ProbeFailed(f"{type(e).__name__}: {e}") from e


def tool_versions(env_prefix: str, packages: tuple[str, ...] = PACKAGES) -> dict:
    """Package versions as the stage's own env reports them; never raises."""
    try:
        return _probe(env_prefix, tuple(packages))
    except _ProbeFailed as e:
        return {"unavailable": str(e)}


def code_entry(path) -> dict:
    """The commit of a checkout; outside git (an installed package), a digest of its code."""
    commit = git_commit(path)
    entry = {"path": str(path), "commit": commit}
    if commit is None and Path(path).is_dir():
        entry["sha256"] = tree_sha256(path)
    return entry


def code_entries(code: dict) -> dict:
    """Entries already taken (a dict) are kept as they are."""
    return {k: v if isinstance(v, dict) else code_entry(v) for k, v in code.items()}


def record(out_dir, *, stage: str, argv, param_sig: str | None = None,
           started: str | None = None, env_prefix: str | None = None,
           packages: tuple[str, ...] = PACKAGES, checks: dict | None = None,
           code: dict | None = None, device: str | None = None) -> Path:
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
        "code": code_entries(code) if code else None,
        "checks": checks,
        "device": device,
        "host": platform.node(),
        "orchestrator_python": sys.version.split()[0],
    }
    path = out_dir / RUN_INFO
    path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return path
