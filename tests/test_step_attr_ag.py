import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import STEPS

from motifline.adopt import normalize_peaks

STEP = STEPS / "attr_ag.py"

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
                        "--hypothetical"],
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


def _onehot(seq):
    x = np.zeros((4, len(seq)), np.float32)
    for j, b in enumerate(seq):
        if b in "ACGT":
            x["ACGT".index(b), j] = 1
    return x


def test_a_peak_with_n_inside_the_kept_window_is_dropped():
    assert not attr_ag.clean_in_crop(_onehot("ACGTANGTAC"), 4)      # N at 5, crop is 3..6
    assert attr_ag.clean_in_crop(_onehot("NCGTACGTAN"), 4)          # N only in the flanks


def test_flank_n_is_filled_with_a_and_counted():
    x = _onehot("NCGTACGTAN")[None]
    assert attr_ag.fill_with_a(x) == 2
    assert (x.sum(1) == 1).all()
    assert x[0, 0, 0] == 1 and x[0, 0, 9] == 1
    assert attr_ag.fill_with_a(_onehot("ACGT")[None]) == 0


def _ag_support(monkeypatch):
    import types
    for name in ("torch", "torch.nn", "alphagenome_pytorch", "alphagenome_pytorch.layers",
                 "alphagenome_pytorch.attribution", "alphagenome_pytorch.attribution.deeplift",
                 "alphagenome_pytorch.utils", "alphagenome_pytorch.utils.sequence",
                 "tangermeme", "tangermeme.deep_lift_shap"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["torch.nn"].Module = object
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["alphagenome_pytorch"].layers = sys.modules["alphagenome_pytorch.layers"]
    sys.modules["alphagenome_pytorch.attribution"].deeplift = \
        sys.modules["alphagenome_pytorch.attribution.deeplift"]
    sys.modules["alphagenome_pytorch.utils.sequence"].sequence_to_onehot_tensor = None
    sys.modules["tangermeme.deep_lift_shap"]._nonlinear = None
    spec = importlib.util.spec_from_file_location("ag_support_t", STEPS / "ag_support.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Rec:
    def __init__(self, seq):
        self.seq = seq

    def __getitem__(self, sl):
        return _Rec(self.seq[sl])


def test_window_seq_uses_the_chromosome_name_as_written(monkeypatch):
    ags = _ag_support(monkeypatch)
    fasta = {"chr1": _Rec("AACCGGTT"), "1": _Rec("TTGGCCAA")}
    assert ags.window_seq(fasta, "chr1", 4, 4) == "CCGG"
    assert ags.window_seq(fasta, "1", 4, 4) == "GGCC"
