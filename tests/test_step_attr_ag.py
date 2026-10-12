import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motifline.adopt import normalize_peaks

STEP = Path(__file__).resolve().parents[1] / "src/motifline/steps/attr_ag.py"

_spec = importlib.util.spec_from_file_location("attr_ag_step", STEP)
attr_ag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(attr_ag)


def test_help_runs_without_heavy_imports():
    r = subprocess.run([sys.executable, str(STEP), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "--hypothetical" in r.stdout and "--biosample" in r.stdout


def test_requires_ag_src():
    r = subprocess.run([sys.executable, str(STEP), "--weights", "w", "--peaks", "p",
                        "--fasta", "f", "--chrom_sizes", "c", "--out_dir", "o",
                        "--chrombpnet_src", "s", "--hypothetical"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "ag_src" in (r.stderr + r.stdout)


ROWS = [("chr1", 120939636, 120940065, 303), ("chr2", 500, 900, 12)]


def test_regions_bed_is_the_narrowpeak_form_finemo_can_parse():
    text = attr_ag.narrowpeak_text(ROWS)
    lines = text.splitlines()
    assert len(lines) == 2
    for line, (chrom, start, end, summit) in zip(lines, ROWS, strict=True):
        f = line.split("\t")
        assert len(f) == 11
        assert f[:3] == [chrom, str(start), str(end)]
        assert f[3:9] == ["."] * 6
        assert f[9] == str(summit)
        assert f[10] == "True"


def test_computed_ag_regions_match_what_an_adopted_ag_run_writes(tmp_path):
    src = tmp_path / "peaks.bed"
    src.write_text("".join(f"{c}\t{s}\t{e}\t{m}\n" for c, s, e, m in ROWS))
    dst = tmp_path / "regions.bed"
    normalize_peaks(src, dst)
    assert dst.read_text() == attr_ag.narrowpeak_text(ROWS)


META = pd.DataFrame({"track_index": [0, 60, 60], "output_type": ["atac", "atac", "dnase"],
                     "biosample_name": ["T-cell", "K562", "HepG2"],
                     "track_name": ["CL:0000084 ATAC-seq", "EFO:0002067 ATAC-seq", "x"]})


def test_the_track_must_be_the_declared_biosample():
    assert attr_ag.track_row(META, 60, "k562")["biosample_name"] == "K562"
    with pytest.raises(SystemExit, match="T-cell"):
        attr_ag.track_row(META, 0, "K562")
    with pytest.raises(SystemExit, match="not in the track metadata"):
        attr_ag.track_row(META, 7, "K562")


def test_chunks_from_other_inputs_are_refused(tmp_path):
    want = {"track": 60, "inputs_sha256": "a"}
    attr_ag.check_chunk_identity(str(tmp_path), want)
    attr_ag.check_chunk_identity(str(tmp_path), dict(want))
    with pytest.raises(SystemExit, match="inputs_sha256"):
        attr_ag.check_chunk_identity(str(tmp_path), {**want, "inputs_sha256": "b"})


def test_crop_center_keeps_the_middle():
    assert list(attr_ag.crop_center(np.arange(10), 4)) == [3, 4, 5, 6]
