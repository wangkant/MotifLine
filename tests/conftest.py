"""Where the machine-specific tests get their paths.

No site path is hardcoded here: the repository is public and a prior commit
(`chore: sanitize site paths for public release`) took them out on purpose. A test that
needs a real compute environment reads it from MOTIFLINE_COMPUTE_ENV, or from the
config the pipeline is actually driven with, and skips when there is neither.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def _from_config(*keys):
    cfg_path = REPO / "config" / "default.yaml"
    if not cfg_path.is_file():
        return None
    try:
        node = yaml.safe_load(cfg_path.read_text())
        for k in keys:
            node = node[k]
        return str(node)
    except (yaml.YAMLError, KeyError, TypeError, OSError):
        return None


#: The env holding modisco/finemo, or None when this machine has none configured.
COMPUTE = os.environ.get("MOTIFLINE_COMPUTE_ENV") or _from_config("envs", "compute")

#: A ChromBPNet checkout, for the import probe.
CHROMBPNET_SRC = os.environ.get("MOTIFLINE_CHROMBPNET_SRC") or _from_config("chrombpnet_src")

has_compute = pytest.mark.skipif(
    not (COMPUTE and shutil.which("modisco", path=f"{COMPUTE}/bin")),
    reason="set MOTIFLINE_COMPUTE_ENV, or configure envs.compute, to run this")

has_chrombpnet_src = pytest.mark.skipif(
    not (CHROMBPNET_SRC and Path(CHROMBPNET_SRC).is_dir()),
    reason="set MOTIFLINE_CHROMBPNET_SRC, or configure chrombpnet_src, to run this")
