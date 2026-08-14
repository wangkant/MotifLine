import json
import numpy as np
from motifline.equivalence import compare_attr


def _mk(tmp_path, arr, bed_rows):
    d = tmp_path / "new"; d.mkdir(exist_ok=True)
    np.save(d / "attr.npy", arr)
    (d / "regions.bed").write_text("".join(f"chr1\t{i}\t{i+10}\n" for i in bed_rows))
    return d


def test_identical_arrays_pass(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 4, 10)).astype(np.float32)
    old = tmp_path / "old.npy"; np.save(old, np.concatenate([a, a], 0))  # old is longer
    ob = tmp_path / "old.bed"
    ob.write_text("".join(f"chr1\t{i}\t{i+10}\n" for i in range(10)))
    d = _mk(tmp_path, a, range(5))
    rep = compare_attr(d, old, ob, 5, tmp_path / "rep.json")
    assert rep["passed"] and rep["bitwise_equal_rows"] == 5 and rep["max_abs_diff"] == 0.0
    assert json.loads((tmp_path / "rep.json").read_text())["n"] == 5


def test_region_misalignment_fails(tmp_path):
    a = np.zeros((3, 4, 10), np.float32)
    old = tmp_path / "old.npy"; np.save(old, a)
    ob = tmp_path / "old.bed"; ob.write_text("".join(f"chr2\t{i}\t{i+10}\n" for i in range(3)))
    d = _mk(tmp_path, a, range(3))
    rep = compare_attr(d, old, ob, 3, tmp_path / "rep.json")
    assert rep["passed"] is False and rep["regions_aligned"] is False


def test_noise_below_pearson_gate_fails(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 4, 10)).astype(np.float32)
    old = tmp_path / "old.npy"; np.save(old, rng.normal(size=(4, 4, 10)).astype(np.float32))
    ob = tmp_path / "old.bed"; ob.write_text("".join(f"chr1\t{i}\t{i+10}\n" for i in range(4)))
    d = _mk(tmp_path, a, range(4))
    assert compare_attr(d, old, ob, 4, tmp_path / "rep.json")["passed"] is False
