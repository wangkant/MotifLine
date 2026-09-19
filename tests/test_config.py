"""The config validator: every way a config can be wrong, named before the DAG runs."""
from __future__ import annotations

import textwrap

import pytest
import yaml

from motifline.config import ConfigError, check, load, problems, skeleton


def cfg_cbp(**over):
    """A structurally valid config with one computed ChromBPNet run."""
    base = {
        "envs": {"compute": "/e/compute"},
        "chrombpnet_src": "/m/chrombpnet",
        "genome": {"fasta": "/g/hg38.fa", "chrom_sizes": "/g/hg38.chrom.sizes"},
        "attribution": {"n_shuffles": 20, "random_state": 42},
        "modisco": {"max_seqlets": 1000, "window": 400},
        "finemo": {"mode": "pp", "threshold": 0.3, "alpha": 0.7, "batch": 2000},
        "reference_lexicon": {
            "sources": {"H14": "/l/h14.meme"},
            "rsat": {"bin": "/e/rsat/bin/rsat", "hclust_method": "average",
                     "metric": "Ncor", "lth": {"cor": 0.75, "Ncor": 0.55}, "quick": True},
            "w_th": 5,
            "tomtom": "/e/meme/bin/tomtom",
            "gate": {"qval": 0.05, "ncor": 0.55},
        },
        "runs": {"r1": {"model": "cbp", "wo_bias": "/m/wo_bias.pt", "peaks": "p.bed"}},
    }
    base.update(over)
    return base


def only(prob_list, needle):
    return [p for p in prob_list if needle in p]


# --- structure -------------------------------------------------------------

def test_valid_config_has_no_problems():
    assert problems(cfg_cbp(), paths=False) == []


def test_missing_top_level_section_is_named():
    c = cfg_cbp()
    del c["genome"]
    assert only(problems(c, paths=False), "genome")


def test_missing_run_key_names_both_run_and_key():
    c = cfg_cbp()
    del c["runs"]["r1"]["wo_bias"]
    p = only(problems(c, paths=False), "wo_bias")
    assert p and "r1" in p[0]


def test_misspelled_run_key_is_reported_not_ignored():
    """`n_shard` would silently make a sharded run single-process."""
    c = cfg_cbp()
    c["runs"]["r1"]["n_shard"] = 4
    assert only(problems(c, paths=False), "n_shard")


def test_unknown_model_is_reported():
    c = cfg_cbp()
    c["runs"]["r1"]["model"] = "enformer"
    assert only(problems(c, paths=False), "enformer")


def test_chrombpnet_src_not_required_when_every_run_is_adopted():
    c = cfg_cbp(runs={"a1": {"model": "cbp", "adopt_from": "/arch/x", "rows": 10}})
    del c["chrombpnet_src"]
    assert not only(problems(c, paths=False), "chrombpnet_src")


def test_gpn_star_is_no_longer_a_model():
    """The star line was removed; a config still declaring one must say so, not run."""
    c = cfg_cbp(runs={"s1": {"model": "star", "peaks": "p.bed",
                             "stride": 8, "source": "none"}})
    assert only(problems(c, paths=False), "'star' is not one of")


def test_a_computed_ag_run_declares_ag_src_in_its_own_entry():
    """attr_argv reads r["ag_src"]; a top-level key of that name is never read."""
    c = cfg_cbp(runs={"g1": {"model": "ag", "weights": "/m/w.pt", "peaks": "p.bed",
                             "track": 0, "chunk": 100}})
    c["ag_src"] = "/m/alphagenome"
    p = problems(c, paths=False)
    assert only(p, "runs.g1.ag_src")
    assert only(p, "unknown top-level key")


def test_adopted_run_needs_adopt_from_and_rows_not_the_model_keys():
    c = cfg_cbp(runs={"a1": {"model": "cbp", "adopt_from": "/arch/x"}})
    p = problems(c, paths=False)
    assert only(p, "rows")
    assert not only(p, "wo_bias")


def test_cancelled_run_must_say_where_the_decision_is_written():
    c = cfg_cbp()
    c["runs"]["r1"]["cancelled"] = True
    assert only(problems(c, paths=False), "cancelled_note")


def test_empty_runs_is_reported():
    assert only(problems(cfg_cbp(runs={}), paths=False), "runs")


def test_shipped_example_config_matches_the_schema(tmp_path):
    """config/example.yaml is what a new user copies; it has to be valid."""
    from pathlib import Path
    example = Path(__file__).resolve().parents[1] / "config" / "example.yaml"
    assert problems(load(example), paths=False) == []


# --- filesystem ------------------------------------------------------------

def test_missing_path_is_reported_and_an_existing_one_is_not(tmp_path):
    fasta = tmp_path / "hg38.fa"
    fasta.write_text(">chr1\nACGT\n")
    c = cfg_cbp()
    c["genome"]["fasta"] = str(fasta)
    p = problems(c, paths=True)
    assert not only(p, str(fasta))
    assert only(p, "hg38.chrom.sizes")


def test_peaks_file_without_a_summit_column_is_reported(tmp_path):
    peaks = tmp_path / "p.bed"
    peaks.write_text("chr1\t100\t200\t150\n")
    c = cfg_cbp()
    c["runs"]["r1"]["peaks"] = str(peaks)
    assert only(problems(c, paths=True), "narrowPeak")


def test_ten_column_narrowpeak_passes(tmp_path):
    peaks = tmp_path / "p.bed"
    peaks.write_text("chr1\t100\t200\tpk\t0\t.\t1.0\t-1\t-1\t50\n")
    c = cfg_cbp()
    c["runs"]["r1"]["peaks"] = str(peaks)
    assert not only(problems(c, paths=True), "narrowPeak")


def test_compute_env_missing_its_binaries_is_reported(tmp_path):
    c = cfg_cbp()
    c["envs"]["compute"] = str(tmp_path)
    assert only(problems(c, paths=True), "finemo")


# --- entry points ----------------------------------------------------------

def test_check_raises_listing_every_problem_at_once():
    c = cfg_cbp()
    del c["genome"]
    del c["runs"]["r1"]["wo_bias"]
    with pytest.raises(ConfigError) as e:
        check(c, paths=False)
    assert "genome" in str(e.value) and "wo_bias" in str(e.value)


def test_check_returns_none_on_a_valid_config():
    assert check(cfg_cbp(), paths=False) is None


def test_load_reports_the_file_that_failed_to_parse(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("runs:\n  r1:\n   - oops\n  bad indent\n")
    with pytest.raises(ConfigError) as e:
        load(bad)
    assert "bad.yaml" in str(e.value)


def test_skeleton_is_valid_yaml_that_names_every_required_section():
    text = skeleton()
    parsed = yaml.safe_load(text)
    for section in ("envs", "genome", "modisco", "finemo", "reference_lexicon", "runs"):
        assert section in parsed, section


def test_a_cancelled_runs_vanished_archive_is_not_an_error(tmp_path):
    """Cancelling a run is how a frozen stage tree is recorded; its inputs may be gone."""
    gone = str(tmp_path / "archive-that-was-deleted")
    active = cfg_cbp(runs={
        "dead": {"model": "cbp", "adopt_from": gone, "rows": 10,
                 "cancelled": True, "cancelled_note": "results/dead/CANCELLED.md"},
        "live": {"model": "cbp", "adopt_from": gone, "rows": 10},
    })
    p = problems(active, paths=True)
    assert not only(p, "runs.dead.adopt_from")
    assert only(p, "runs.live.adopt_from")


def test_a_cancelled_run_is_still_checked_structurally(tmp_path):
    c = cfg_cbp(runs={"dead": {"model": "cbp", "cancelled": True,
                               "cancelled_note": "n.md", "adopt_from": "/x"}})
    assert only(problems(c, paths=True), "rows")
