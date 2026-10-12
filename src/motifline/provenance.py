"""run_info.json: argv, parameters, tool versions and model code behind a stage."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from motifline import __version__

PACKAGES = ("finemo", "modisco-lite", "tangermeme", "torch", "numpy")

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


@lru_cache(maxsize=None)
def tool_versions(env_prefix: str, packages: tuple[str, ...] = PACKAGES) -> dict:
    """Package versions as the stage's own env reports them; never raises."""
    python = Path(env_prefix) / "bin" / "python"
    if not os.access(python, os.X_OK):
        return {"unavailable": f"{python} is not an executable interpreter"}
    try:
        done = subprocess.run([str(python), "-c", _PROBE, *packages],
                              capture_output=True, text=True, timeout=120, check=True)
        return json.loads(done.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        return {"unavailable": f"{type(e).__name__}: {e}"}


def git_commit(path) -> str | None:
    """HEAD of the checkout holding `path`, plus a digest of uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                              check=True).stdout
    try:
        head, diff = git("rev-parse", "HEAD").decode().strip(), git("diff", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return None
    return head if not diff else f"{head}+{hashlib.sha256(diff).hexdigest()[:12]}"


def record(out_dir, *, stage: str, argv, param_sig: str | None = None,
           started: str | None = None, env_prefix: str | None = None,
           packages: tuple[str, ...] = PACKAGES, checks: dict | None = None,
           code: dict | None = None) -> Path:
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
        "code": {k: {"path": str(v), "commit": git_commit(v)} for k, v in code.items()}
                if code else None,
        "checks": checks,
        "host": platform.node(),
        "orchestrator_python": sys.version.split()[0],
    }
    path = out_dir / RUN_INFO
    path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return path
