import json
import numpy as np
from motiflex.run_status import write_run_status
from motifline.contracts import append_params
from motifline import guards

def _success_dir(tmp_path, name, params):
    d = tmp_path / name
    d.mkdir()
    append_params(d, params)
    (d / "provenance.json").write_text(json.dumps([{"subcommand": name}]))
    write_run_status(d, status="SUCCESS", command="x", subcommand=name)
    return d

def test_upstream_success_fails_on_refused(tmp_path):
    d = tmp_path / "attr"; d.mkdir()
    write_run_status(d, status="REFUSED", command="x", subcommand="attribute", detail="no")
    assert guards.upstream_success(d).passed is False

def test_refuses_attr_without_hypothetical_flag(tmp_path):
    d = _success_dir(tmp_path, "attr", {"note": "no flag recorded"})
    assert guards.attr_is_hypothetical(d).passed is False

def test_refuses_actual_score_attr(tmp_path):
    d = _success_dir(tmp_path, "attr", {"hypothetical": False})
    assert guards.attr_is_hypothetical(d).passed is False

def test_accepts_hypothetical_attr(tmp_path):
    d = _success_dir(tmp_path, "attr", {"hypothetical": True})
    assert guards.attr_is_hypothetical(d).passed is True

def test_refuses_hyp_act_mismatch(tmp_path):
    a = _success_dir(tmp_path, "attr_hits", {"hypothetical": True})
    b = _success_dir(tmp_path, "attr_lex", {"hypothetical": False})
    assert guards.hyp_flags_match(a, b).passed is False
    c = _success_dir(tmp_path, "attr_lex2", {"hypothetical": True})
    assert guards.hyp_flags_match(a, c).passed is True

def test_refuses_row_and_width_mismatch(tmp_path):
    attr, ohe, bed = tmp_path / "attr.npy", tmp_path / "ohe.npy", tmp_path / "r.bed"
    np.save(attr, np.zeros((3, 4, 10), np.float32))
    np.save(ohe, np.zeros((3, 4, 10), np.int8))
    bed.write_text("chr1\t0\t10\n" * 3)
    assert guards.rows_and_width_match(attr, ohe, bed, 10).passed is True
    bed.write_text("chr1\t0\t10\n" * 2)                      # row mismatch
    assert guards.rows_and_width_match(attr, ohe, bed, 10).passed is False
    bed.write_text("chr1\t0\t10\n" * 3)
    assert guards.rows_and_width_match(attr, ohe, bed, 12).passed is False  # width

def test_refuses_finemo_version_drift(tmp_path):
    d = _success_dir(tmp_path, "regions", {"software": {"finemo": "0.30"}})
    sp = tmp_path / "sp"; sp.mkdir(); (sp / "finemo-0.41.dist-info").mkdir()
    assert guards.finemo_version_matches(d, sp).passed is False
    sp30 = tmp_path / "sp30"; sp30.mkdir(); (sp30 / "finemo-0.30.dist-info").mkdir()
    assert guards.finemo_version_matches(d, sp30).passed is True


def test_declared_finemo_version_must_be_the_one_on_disk(tmp_path):
    """The cross-version stage cannot use finemo_version_matches (regions are
    deliberately from the other version), so it still has to prove that the
    binary it says it ran is the binary it ran."""
    sp41 = tmp_path / "sp41"; sp41.mkdir(); (sp41 / "finemo-0.41.dist-info").mkdir()
    assert guards.finemo_version_declared("0.41", sp41).passed is True
    assert guards.finemo_version_declared("0.30", sp41).passed is False


def test_cross_version_stage_checks_what_produced_its_regions(tmp_path):
    d = _success_dir(tmp_path, "regions", {"software": {"finemo": "0.30"}})
    assert guards.recorded_finemo_version_is(d, "0.30").passed is True
    assert guards.recorded_finemo_version_is(d, "0.41").passed is False
    bare = _success_dir(tmp_path, "regions_nosoft", {})
    assert guards.recorded_finemo_version_is(bare, "0.30").passed is False


def test_external_lexicon_must_exist_before_the_stage_runs(tmp_path):
    h5 = tmp_path / "core.h5"
    assert guards.lexicon_exists(h5).passed is False
    h5.write_bytes(b"not really h5 but present")
    assert guards.lexicon_exists(h5).passed is True
    assert guards.lexicon_exists(tmp_path).passed is False        # a directory is not a lexicon
