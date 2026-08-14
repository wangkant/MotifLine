"""Package version as installed on disk, read from dist-info directory names.

`finemo.__version__` does not exist in the installed 0.30, so import-based
detection cannot be the mechanism; the dist-info directory name can.
"""
from __future__ import annotations
import re
from pathlib import Path


def dist_info_version(site_packages: str | Path, package: str) -> str:
    sp = Path(site_packages)
    pat = re.compile(rf"^{re.escape(package)}-(?P<v>.+)\.dist-info$")
    hits = sorted(m.group("v") for d in sp.iterdir() if (m := pat.match(d.name)))
    if len(hits) != 1:
        raise LookupError(
            f"expected exactly one {package}-*.dist-info under {sp}, found {hits or 'none'}")
    return hits[0]
