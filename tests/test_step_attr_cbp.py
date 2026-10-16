import importlib.util
import subprocess, sys
import pytest
from pathlib import Path

from conftest import STEPS

STEP = STEPS / "attr_cbp.py"


def _load_step():
    spec = importlib.util.spec_from_file_location("attr_cbp_step", STEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROW10 = "chr1\t100\t200\tPeak_1\t197\t.\t1.6\t19.7\t17.9\t50"


def test_help_runs_without_heavy_imports():
    # argparse must come before torch/chrombpnet imports so --help works anywhere
    r = subprocess.run([sys.executable, str(STEP), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--hypothetical" in r.stdout


def test_requires_chrombpnet_src():
    r = subprocess.run([sys.executable, str(STEP), "--peaks", "x", "--out_dir", "y",
                        "--wo_bias", "m", "--fasta", "f", "--chrom_sizes", "c"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "chrombpnet_src" in (r.stderr + r.stdout)

def test_normalize_passes_through_10_column_bed(tmp_path):
    src = tmp_path / "p.bed"; src.write_text(ROW10 + "\n" + ROW10 + "\n")
    assert _load_step().normalize_peaks_bed(str(src), str(tmp_path / "out")) == str(src)


def test_normalize_drops_is_peak_column_keeping_rows(tmp_path):
    src = tmp_path / "p.bed"; src.write_text((ROW10 + "\tTrue\n") * 3)
    out = _load_step().normalize_peaks_bed(str(src), str(tmp_path / "out"))
    lines = Path(out).read_text().splitlines()
    assert out != str(src)
    assert lines == [ROW10] * 3


def test_normalize_refuses_non_true_is_peak(tmp_path):
    src = tmp_path / "p.bed"; src.write_text(ROW10 + "\tTrue\n" + ROW10 + "\tFalse\n")
    with pytest.raises(ValueError, match="is_peak=True"):
        _load_step().normalize_peaks_bed(str(src), str(tmp_path / "out"))


def test_normalize_refuses_unexpected_width(tmp_path):
    src = tmp_path / "p.bed"; src.write_text(ROW10 + "\tTrue\textra\n")
    with pytest.raises(ValueError, match="expected 10 or 11 columns"):
        _load_step().normalize_peaks_bed(str(src), str(tmp_path / "out"))


def test_refuses_without_hypothetical_like_the_ag_path_does():
    """Without the flag the step still runs and writes projected scores under the same
    filenames -- indistinguishable downstream. attr_ag.py already refuses; this path
    must too, or the two models are not symmetric in the one way that matters."""
    r = subprocess.run([sys.executable, str(STEP), "--chrombpnet_src", "s",
                        "--peaks", "p", "--out_dir", "o", "--wo_bias", "m",
                        "--fasta", "f", "--chrom_sizes", "c"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "hypothetical" in (r.stderr + r.stdout)
