"""Measuring, on the array itself, whether an attribution is hypothetical."""
from __future__ import annotations

import numpy as np

from motifline.attribution import HYP_FLOOR, hyp_check, hyp_fraction


def hyp_array(n=8, w=50, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 4, w)).astype(np.float32)


def actual_array(n=8, w=50, seed=0):
    """What `hypothetical=False` produces: a score on the observed base only."""
    rng = np.random.default_rng(seed)
    attr = rng.normal(size=(n, 4, w)).astype(np.float32)
    ohe = np.zeros((n, 4, w), dtype=np.int8)
    np.put_along_axis(ohe, rng.integers(0, 4, size=(n, w))[:, None, :], 1, axis=1)
    return attr * ohe


def test_a_hypothetical_array_scores_about_one(tmp_path):
    p = tmp_path / "attr.npy"
    np.save(p, hyp_array())
    assert hyp_fraction(p) > 0.99


def test_an_actual_array_scores_about_a_quarter(tmp_path):
    p = tmp_path / "attr.npy"
    np.save(p, actual_array())
    assert 0.2 < hyp_fraction(p) < 0.3


def test_the_floor_separates_the_two_cases(tmp_path):
    for name, arr in (("hyp", hyp_array()), ("act", actual_array())):
        np.save(tmp_path / f"{name}.npy", arr)
    assert hyp_fraction(tmp_path / "hyp.npy") >= HYP_FLOOR
    assert hyp_fraction(tmp_path / "act.npy") < HYP_FLOOR


def test_only_a_sample_of_a_large_array_is_read(tmp_path):
    p = tmp_path / "attr.npy"
    np.save(p, hyp_array(n=500, w=40))
    assert hyp_fraction(p, rows=10) > 0.99


def test_the_check_reports_the_fraction_and_the_verdict(tmp_path):
    np.save(tmp_path / "attr.npy", hyp_array())
    got = hyp_check()(tmp_path)
    assert got["looks_hypothetical"] is True
    assert got["hypothetical_fraction"] >= HYP_FLOOR


def test_the_check_calls_an_actual_array_what_it_is(tmp_path):
    np.save(tmp_path / "attr.npy", actual_array())
    assert hyp_check()(tmp_path)["looks_hypothetical"] is False


def test_a_shard_names_its_own_array(tmp_path):
    np.save(tmp_path / "attr_3.npy", hyp_array())
    assert hyp_check("attr_3.npy")(tmp_path)["looks_hypothetical"] is True


def test_a_missing_array_gives_a_reason_and_never_raises(tmp_path):
    got = hyp_check()(tmp_path)
    assert got["looks_hypothetical"] is None
    assert "attr.npy" in got["hypothetical_check"]


def test_an_unreadable_array_gives_a_reason_and_never_raises(tmp_path):
    (tmp_path / "attr.npy").write_text("not an npy file")
    got = hyp_check()(tmp_path)
    assert got["looks_hypothetical"] is None


def test_an_actual_array_asks_for_attention(tmp_path):
    np.save(tmp_path / "attr.npy", actual_array())
    got = hyp_check()(tmp_path)
    assert "hypothetical" in got["warning"] and "0.2" in got["warning"]


def test_a_hypothetical_array_asks_for_nothing(tmp_path):
    np.save(tmp_path / "attr.npy", hyp_array())
    assert "warning" not in hyp_check()(tmp_path)
