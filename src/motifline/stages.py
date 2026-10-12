"""Run one stage body, then write run_info.json into its directory.

run_info.json is removed before the body starts and written only after it succeeds.
`checks` measure the finished output; they are recorded, never fatal.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from motifline.provenance import PACKAGES, RUN_INFO, now, record


def run_body(out_dir, argv, *, stage: str, param_sig: str | None = None,
             env: dict | None = None, env_prefix: str | None = None,
             packages: tuple[str, ...] = PACKAGES, checks=(),
             code: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / RUN_INFO).unlink(missing_ok=True)
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
                  checks=results or None, code=code)
