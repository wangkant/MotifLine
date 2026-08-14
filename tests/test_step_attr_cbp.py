import subprocess, sys
from pathlib import Path

STEP = Path(__file__).resolve().parents[1] / "src/motifline/steps/attr_cbp.py"


def test_help_runs_without_heavy_imports():
    # argparse must come before torch/chrombpnet imports so --help works anywhere
    r = subprocess.run([sys.executable, str(STEP), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--hypothetical" in r.stdout


def test_requires_chrombpnet_src():
    r = subprocess.run([sys.executable, str(STEP), "--peaks", "x", "--out_dir", "y",
                        "--wo_bias", "m", "--fasta", "f", "--chrom_sizes", "c"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "chrombpnet_src" in (r.stderr + r.stdout)
