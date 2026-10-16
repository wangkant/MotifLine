"""The Snakefile builds a DAG for every kind of run (snakemake -n on dummy inputs)."""
from __future__ import annotations

import subprocess
import sys

import yaml

from motifline.cli import SNAKEFILE


def _config(tmp_path):
    def f(name):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
        return str(p)

    archive = tmp_path / "archive"
    archive.mkdir()
    return {
        "envs": {"compute": str(tmp_path / "env")},
        "chrombpnet_src": str(tmp_path / "cbp_src"),
        "genome": {"fasta": f("hg38.fa"), "chrom_sizes": f("hg38.chrom.sizes")},
        "attribution": {"n_shuffles": 20, "random_state": 42},
        "modisco": {"max_seqlets": 1000000, "window": 400, "n_leiden": 50},
        "finemo": {"mode": "pp", "threshold": 0.3, "alpha": 0.7, "batch": 2000},
        "reference_lexicon": {"sources": {"H14": f("h14.meme"), "JAS": f("jas.meme")},
                              "h14_annotation": f("h14.jsonl"), "qval_threshold": 0.05,
                              "tomtom": str(tmp_path / "tomtom")},
        "runs": {
            "single": {"model": "cbp", "wo_bias": f("m.pt"), "peaks": f("p.bed")},
            "sharded": {"model": "cbp", "wo_bias": f("m.pt"), "peaks": f("p.bed"),
                        "n_shards": 3},
            "ag": {"model": "ag", "ag_src": str(tmp_path / "ag_src"), "weights": f("w.st"),
                   "peaks": f("p.bed"), "track": 60, "biosample": "K562", "chunk": 100},
            "adopted": {"model": "cbp", "adopt_from": str(archive), "rows": 10},
            "off": {"model": "cbp", "adopt_from": str(archive), "rows": 10,
                    "cancelled": True},
        },
    }


def _dry_run(tmp_path, cfg):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return subprocess.run(
        [sys.executable, "-m", "snakemake", "-n", "-q", "rules", "--snakefile",
         str(SNAKEFILE), "--directory", str(tmp_path), "--configfile", str(path)],
        capture_output=True, text=True)


def test_a_dot_in_a_run_name_does_not_match_another_run(tmp_path):
    cfg = _config(tmp_path)
    cfg["runs"]["k.v2"] = cfg["runs"].pop("adopted")
    cfg["runs"]["kxv2"] = cfg["runs"].pop("single")
    r = _dry_run(tmp_path, cfg)
    assert r.returncode == 0, r.stderr


def test_every_run_kind_builds(tmp_path):
    r = _dry_run(tmp_path, _config(tmp_path))
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    for rule in ("attribute", "attribute_sharded", "merge_shards", "import_attr",
                 "discover", "call_hits", "name_patterns", "lexicon_union"):
        assert rule in out, rule


def test_no_config_is_refused(tmp_path):
    r = subprocess.run([sys.executable, "-m", "snakemake", "-n", "--snakefile",
                        str(SNAKEFILE), "--directory", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "no config" in (r.stdout + r.stderr)


def test_a_missing_attribution_input_stops_the_dag(tmp_path):
    cfg = _config(tmp_path)
    cfg["runs"]["single"]["wo_bias"] = str(tmp_path / "moved.pt")
    r = _dry_run(tmp_path, cfg)
    assert r.returncode != 0 and "moved.pt" in (r.stdout + r.stderr)


def _frozen(tmp_path, with_attr):
    cfg = _config(tmp_path)
    cfg["runs"]["frozen"] = {"model": "cbp", "wo_bias": str(tmp_path / "gone.pt"),
                             "peaks": cfg["runs"]["single"]["peaks"], "cancelled": True}
    if with_attr:
        d = tmp_path / "results/frozen/attr"
        d.mkdir(parents=True)
        for f in ("attr.npy", "ohe.npy", "regions.bed"):
            (d / f).write_text("x")
    return cfg


def test_a_cancelled_run_with_missing_inputs_does_not_block_the_rest(tmp_path):
    assert _dry_run(tmp_path, _frozen(tmp_path, with_attr=False)).returncode == 0


def test_a_cancelled_run_is_rebuilt_from_its_frozen_attr_without_recomputing(tmp_path):
    cfg = _frozen(tmp_path, with_attr=True)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    r = subprocess.run(
        [sys.executable, "-m", "snakemake", "-n", "-q", "rules", "--snakefile",
         str(SNAKEFILE), "--directory", str(tmp_path), "--configfile", str(path),
         "--", "results/frozen/modisco/modisco.h5"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    assert "discover" in out and "attribute" not in out


def test_a_cancelled_run_without_attr_is_an_error_when_targeted(tmp_path):
    cfg = _frozen(tmp_path, with_attr=False)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    r = subprocess.run(
        [sys.executable, "-m", "snakemake", "-n", "--snakefile", str(SNAKEFILE),
         "--directory", str(tmp_path), "--configfile", str(path),
         "--", "results/frozen/modisco/modisco.h5"], capture_output=True, text=True)
    assert r.returncode != 0 and "attr" in (r.stdout + r.stderr)


def test_a_chrombpnet_h5_run_is_imported_by_the_compute_env_step(tmp_path):
    cfg = _config(tmp_path)
    cfg["runs"]["tf"] = {"model": "cbp", "adopt_from": str(tmp_path / "archive"),
                         "adopt_format": "chrombpnet_h5", "adopt_scores": "k.counts_scores.h5",
                         "adopt_peaks": "k.interpreted_regions.bed", "rows": 10}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    r = subprocess.run(
        [sys.executable, "-m", "snakemake", "-n", "-p", "--snakefile", str(SNAKEFILE),
         "--directory", str(tmp_path), "--configfile", str(path),
         "--", "results/tf/attr/attr.npy"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "import_attr" in r.stdout + r.stderr
