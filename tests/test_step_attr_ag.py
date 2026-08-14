import importlib.util
import subprocess
import sys
from pathlib import Path

from motifline.adopt import normalize_peaks

STEP = Path(__file__).resolve().parents[1] / "src/motifline/steps/attr_ag.py"

_spec = importlib.util.spec_from_file_location("attr_ag_step", STEP)
attr_ag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(attr_ag)          # top of the file is stdlib-only by contract


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


ROWS = [("chr1", 120939636, 120940065, 303), ("chr2", 500, 900, 12)]


def test_regions_bed_is_the_narrowpeak_form_finemo_can_parse():
    # finemo's -p parser reads the summit from column 10 and refuses a bed with
    # fewer than its ten schema columns, so a 4-column regions.bed cannot reach
    # extract_regions at all.
    text = attr_ag.narrowpeak_text(ROWS)
    lines = text.splitlines()
    assert len(lines) == 2
    for line, (chrom, start, end, summit) in zip(lines, ROWS):
        f = line.split("\t")
        assert len(f) == 11
        assert f[:3] == [chrom, str(start), str(end)]
        assert f[3:9] == ["."] * 6
        assert f[9] == str(summit)          # column 10 = summit, what finemo reads
        assert f[10] == "True"


def test_computed_ag_regions_match_what_an_adopted_ag_run_writes(tmp_path):
    # A computed AG run and an adopted one must produce the same regions.bed, or
    # their hits are not comparable. adopt.normalize_peaks is the reference; it
    # reproduces the archive's own peaks_np.bed.
    src = tmp_path / "peaks.bed"
    src.write_text("".join(f"{c}\t{s}\t{e}\t{m}\n" for c, s, e, m in ROWS))
    dst = tmp_path / "regions.bed"
    normalize_peaks(src, dst)
    assert dst.read_text() == attr_ag.narrowpeak_text(ROWS)
