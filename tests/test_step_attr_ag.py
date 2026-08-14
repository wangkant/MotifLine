import subprocess
import sys
from pathlib import Path

STEP = Path(__file__).resolve().parents[1] / "src/motifline/steps/attr_ag.py"


def test_help_runs_without_heavy_imports():
    # argparse must come before alphagenome/torch imports so --help works anywhere
    r = subprocess.run([sys.executable, str(STEP), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "--hypothetical" in r.stdout and "--track" in r.stdout


def test_requires_ag_src():
    r = subprocess.run([sys.executable, str(STEP), "--weights", "w", "--peaks", "p",
                        "--fasta", "f", "--chrom_sizes", "c", "--out_dir", "o",
                        "--chrombpnet_src", "s", "--hypothetical"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "ag_src" in (r.stderr + r.stdout)
