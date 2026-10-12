"""The config validator: every way a config can be wrong, named before the DAG runs."""
from __future__ import annotations

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
            "tomtom": "/e/meme/bin/tomtom",
            "h14_annotation": "/l/H14CORE_annotation.jsonl",
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
    c = cfg_cbp(runs={"s1": {"model": "star", "peaks": "p.bed",
                             "stride": 8, "source": "none"}})
    assert only(problems(c, paths=False), "'star' is not one of")


def test_a_computed_ag_run_declares_ag_src_in_its_own_entry():
    """attr_argv reads r["ag_src"]; a top-level key of that name is never read."""
    c = cfg_cbp(runs={"g1": {"model": "ag", "weights": "/m/w.pt", "peaks": "p.bed",
                             "track": 60, "biosample": "K562", "chunk": 100}})
    c["ag_src"] = "/m/alphagenome"
    p = problems(c, paths=False)
    assert only(p, "runs.g1.ag_src")
    assert only(p, "unknown top-level key")


def test_adopted_run_needs_adopt_from_and_rows_not_the_model_keys():
    c = cfg_cbp(runs={"a1": {"model": "cbp", "adopt_from": "/arch/x"}})
    p = problems(c, paths=False)
    assert only(p, "rows")
    assert not only(p, "wo_bias")


def test_empty_runs_is_reported():
    assert only(problems(cfg_cbp(runs={}), paths=False), "runs")


def test_shipped_example_config_matches_the_schema(tmp_path):
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


def test_an_ag_run_must_name_the_biosample_of_its_track():
    c = cfg_cbp(runs={"g1": {"model": "ag", "ag_src": "/a", "weights": "/m/w.pt",
                             "peaks": "p.bed", "track": 60, "chunk": 100}})
    assert only(problems(c, paths=False), "runs.g1.biosample")


def test_unknown_keys_inside_a_section_are_reported():
    c = cfg_cbp()
    c["modisco"]["n_leidn"] = 2
    assert only(problems(c, paths=False), "n_leidn")


def test_head_and_n_shards_together_are_reported():
    c = cfg_cbp()
    c["runs"]["r1"].update(head=10, n_shards=2)
    assert only(problems(c, paths=False), "head and n_shards")


def test_n_shards_must_be_a_positive_integer():
    c = cfg_cbp()
    c["runs"]["r1"]["n_shards"] = 0
    assert only(problems(c, paths=False), "n_shards: must be a positive integer")


def _peaks_cfg(tmp_path, text, sizes="chr1\t1000000\n"):
    peaks = tmp_path / "p.bed"
    peaks.write_text(text)
    cs = tmp_path / "hg38.chrom.sizes"
    cs.write_text(sizes)
    c = cfg_cbp()
    c["genome"]["chrom_sizes"] = str(cs)
    c["runs"]["r1"]["peaks"] = str(peaks)
    return problems(c, paths=True)


ROW = "chr1\t100\t200\tpk\t0\t.\t1.0\t-1\t-1\t{summit}\n"


def test_every_peak_line_is_checked_not_only_the_first(tmp_path):
    assert only(_peaks_cfg(tmp_path, ROW.format(summit=50) + "chr1\t100\t200\t150\n"), "line 2")


def test_a_summit_outside_its_peak_is_reported(tmp_path):
    assert only(_peaks_cfg(tmp_path, ROW.format(summit=150)), "outside the peak")


def test_a_chromosome_missing_from_chrom_sizes_is_reported(tmp_path):
    assert only(_peaks_cfg(tmp_path, ROW.format(summit=50).replace("chr1", "chrUn", 1)),
                "'chrUn' is not in genome.chrom_sizes")


def test_peaks_sharing_a_summit_are_reported(tmp_path):
    text = ROW.format(summit=50) + "chr1\t120\t300\tpk2\t0\t.\t1.0\t-1\t-1\t30\n"
    assert only(_peaks_cfg(tmp_path, text), "repeats an earlier peak")


def test_a_header_line_is_reported(tmp_path):
    assert only(_peaks_cfg(tmp_path, "track name=x\n" + ROW.format(summit=50)), "line 1")


def test_rows_on_a_single_process_run_is_reported():
    c = cfg_cbp()
    c["runs"]["r1"]["rows"] = 100
    assert only(problems(c, paths=False), "rows: only checked")


def test_an_eleventh_column_other_than_true_is_reported(tmp_path):
    assert only(_peaks_cfg(tmp_path, ROW.format(summit=50).replace("\n", "\tFalse\n")),
                "column 11 must be True")
