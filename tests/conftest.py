"""Paths for tests that need a real compute env: MOTIFLINE_COMPUTE_ENV /
MOTIFLINE_CHROMBPNET_ENV, or config/default.yaml. Those tests
skip without one."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

import motifline

REPO = Path(__file__).resolve().parents[1]
# the installed package, so CI exercises what a wheel ships
PACKAGE = Path(motifline.__file__).resolve().parent
STEPS = PACKAGE / "steps"

# steps import their sibling modules, as they do when run as scripts
sys.path.insert(0, str(STEPS))


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

has_compute = pytest.mark.skipif(
    not (COMPUTE and shutil.which("modisco", path=f"{COMPUTE}/bin")),
    reason="set MOTIFLINE_COMPUTE_ENV, or configure envs.compute, to run this")

#: The official ChromBPNet env (TensorFlow), or None.
CBP_ENV = os.environ.get("MOTIFLINE_CHROMBPNET_ENV") or _from_config("envs", "chrombpnet")

has_cbp_env = pytest.mark.skipif(
    not (CBP_ENV and Path(CBP_ENV, "bin", "python").is_file()),
    reason="set MOTIFLINE_CHROMBPNET_ENV, or configure envs.chrombpnet, to run this")

