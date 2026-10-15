"""Importing official ChromBPNet contribution scores."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

STEP = Path(__file__).resolve().parents[1] / "src/motifline/steps/import_chrombpnet.py"
_spec = importlib.util.spec_from_file_location("import_chrombpnet_step", STEP)
imp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(imp)

GENOME = "ACGTACGTNNacgtACGTTTGGCCAAGTACGTACGATCGATCGA"


def test_encode_matches_chrombpnet_one_hot():
    x = imp.encode("AcgNT")
    assert x.dtype == np.int8 and x.shape == (4, 5)
    assert x[:, 3].sum() == 0
    assert [int(x[:, j].argmax()) for j in (0, 1, 2, 4)] == [0, 1, 2, 3]


def _rows(*summits, start=0):
    return [("chr1", start, s, f"chr1\t{start}\t40\t.\t0\t.\t0\t0\t0\t{s}") for s in summits]


def _fetch(c, s, e):
    return GENOME[s:e]


def _raw(rows, width):
    return np.stack([imp.encode(_fetch(c, st + sm - width // 2, st + sm + width // 2))
                     for c, st, sm, _ in rows])


def test_rows_that_match_the_genome_pass_and_a_shifted_one_is_caught():
    rows = _rows(10, 20, 30)
    raw = _raw(rows, 8)
    assert imp.mismatches(raw, rows, _fetch, 8) == []
    assert imp.mismatches(raw, _rows(10, 21, 30), _fetch, 8) == [1]


def _arrays(rows, width, seed=0):
    rng = np.random.default_rng(seed)
    raw = _raw(rows, width)
    shap = rng.normal(size=raw.shape).astype(np.float16)
    return raw, shap, (raw * shap).astype(np.float16)


def test_check_accepts_consistent_arrays():
    rows = _rows(10, 20)
    raw, shap, proj = _arrays(rows, 8)
    imp.check(raw, shap, proj, rows, expected_rows=2, expected_width=8)


@pytest.mark.parametrize("break_it, message", [
    (lambda raw, shap, proj, rows: (raw, proj, proj, rows), "projected scores"),
    (lambda raw, shap, proj, rows: (raw, shap, -proj, rows), "inconsistent"),
    (lambda raw, shap, proj, rows: (raw, shap, proj, rows[:1]), "regions"),
    (lambda raw, shap, proj, rows: (raw * 2, shap, None, rows), "not one-hot"),
])
def test_check_refuses_inconsistent_inputs(break_it, message):
    rows = _rows(10, 20)
    args = break_it(*_arrays(rows, 8), rows)
    with pytest.raises(SystemExit, match=message):
        imp.check(*args, expected_rows=len(args[3]), expected_width=8)


def test_check_refuses_wrong_row_count_and_width():
    rows = _rows(10, 20)
    raw, shap, proj = _arrays(rows, 8)
    with pytest.raises(SystemExit, match="expected 3 rows"):
        imp.check(raw, shap, proj, rows, expected_rows=3, expected_width=8)
    with pytest.raises(SystemExit, match="expected width 2114"):
        imp.check(raw, shap, proj, rows, expected_rows=2, expected_width=2114)


def test_regions_must_be_ten_columns(tmp_path):
    p = tmp_path / "r.bed"
    p.write_text("chr1\t0\t40\t.\t0\t.\t0\t0\t0\t10\tTrue\n")
    with pytest.raises(SystemExit, match="10-column"):
        imp.read_regions(p)


def test_end_to_end_on_a_deepdish_style_file(tmp_path):
    tables = pytest.importorskip("tables")
    pytest.importorskip("pyfaidx")
    (tmp_path / "g.fa").write_text(">chr1\n" + GENOME + "\n")
    rows = _rows(10, 20, 30)
    raw, shap, proj = _arrays(rows, 8)
    h5 = tmp_path / "k.counts_scores.h5"
    filters = tables.Filters(complevel=9, complib="blosc", shuffle=True)
    with tables.open_file(str(h5), "w") as f:
        for name, arr in (("raw", raw), ("shap", shap), ("projected_shap", proj)):
            g = f.create_group("/", name)
            f.create_carray(g, "seq", obj=arr, filters=filters)
    bed = tmp_path / "k.interpreted_regions.bed"
    bed.write_text("".join(r[3] + "\n" for r in rows))
    out = tmp_path / "attr"
    argv = [sys.executable, str(STEP), "--scores", str(h5), "--regions", str(bed),
            "--fasta", str(tmp_path / "g.fa"), "--out_dir", str(out), "--expected_rows", "3"]
    r = subprocess.run(argv, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    np.testing.assert_array_equal(np.load(out / "attr.npy"), shap.astype(np.float32))
    np.testing.assert_array_equal(np.load(out / "ohe.npy"), raw)
    assert (out / "regions.bed").read_text() == bed.read_text()
    assert '"rows_with_non_acgt": 1' in r.stdout

    bed.write_text("".join(r[3] + "\n" for r in _rows(10, 21, 30)))
    r = subprocess.run(argv, capture_output=True, text=True)
    assert r.returncode != 0 and "do not match" in r.stderr
