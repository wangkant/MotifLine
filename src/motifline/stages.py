"""Launching a stage body, and labelling the directory it finished in.

Every rule in the Snakefile does the same three things: make the stage directory, run
one subprocess in it, and -- once that subprocess has succeeded -- write the
``run_info.json`` that says what produced the files. Keeping it here rather than in the
Snakefile is what makes those three things testable.

The label is written **after** the body succeeds and never before. A ``run_info.json``
next to a half-written array would be worse than no label at all: it would say a stage
finished that did not.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from motifline.provenance import PACKAGES, now, record


def run_body(out_dir, argv, *, stage: str, param_sig: str | None = None,
             env: dict | None = None, env_prefix: str | None = None,
             packages: tuple[str, ...] = PACKAGES) -> Path:
    """Run one stage body and label its directory. Raises if the body fails."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now()
    subprocess.run([str(a) for a in argv], check=True,
                   env={**os.environ, **env} if env else None)
    return record(out_dir, stage=stage, argv=argv, param_sig=param_sig,
                  started=started, env_prefix=env_prefix, packages=packages)
