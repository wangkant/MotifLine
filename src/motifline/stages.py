"""Run one stage body, then label the directory it finished in (``run_info.json``).

The label is written only after the body succeeded. ``checks`` are measurements on the
finished output recorded in the label; they never fail the stage, and one that returns
a ``warning`` gets it printed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from motifline.provenance import PACKAGES, now, record


def run_body(out_dir, argv, *, stage: str, param_sig: str | None = None,
             env: dict | None = None, env_prefix: str | None = None,
             packages: tuple[str, ...] = PACKAGES, checks=()) -> Path:
    """Run one stage body and label its directory. Raises if the body fails."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now()
    subprocess.run([str(a) for a in argv], check=True,
                   env={**os.environ, **env} if env else None)

    results: dict = {}
    for fn in checks:
        try:
            results.update(fn(out_dir))
        except Exception as e:
            results[f"{getattr(fn, '__name__', 'check')}_error"] = f"{type(e).__name__}: {e}"
    if results.get("warning"):
        print(f"WARNING [{stage} {out_dir}]: {results['warning']}", file=sys.stderr)

    return record(out_dir, stage=stage, argv=argv, param_sig=param_sig,
                  started=started, env_prefix=env_prefix, packages=packages,
                  checks=results or None)
