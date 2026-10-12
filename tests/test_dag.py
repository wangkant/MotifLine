"""The Snakefile builds a DAG for every kind of run (snakemake -n on dummy inputs)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


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
        "modisco": {"max_seqlets": 1000000, "window": 400},
        "finemo": {"mode": "pp", "threshold": 0.3, "alpha": 0.7, "batch": 2000},
        "reference_lexicon": {"sources": {"H14": f("h14.meme"), "JAS": f("jas.meme")},
                              "h14_annotation": f("h14.jsonl"),
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
         str(REPO / "Snakefile"), "--directory", str(tmp_path), "--configfile", str(path)],
        capture_output=True, text=True)


def test_every_run_kind_builds(tmp_path):
    r = _dry_run(tmp_path, _config(tmp_path))
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    for rule in ("attribute", "attribute_sharded", "merge_shards", "import_attr",
                 "discover", "call_hits", "name_patterns", "lexicon_union"):
        assert rule in out, rule


def test_no_config_is_refused(tmp_path):
    r = subprocess.run([sys.executable, "-m", "snakemake", "-n", "--snakefile",
                        str(REPO / "Snakefile"), "--directory", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "no config" in (r.stdout + r.stderr)


def test_a_missing_attribution_input_stops_the_dag(tmp_path):
    cfg = _config(tmp_path)
    cfg["runs"]["single"]["wo_bias"] = str(tmp_path / "moved.pt")
    r = _dry_run(tmp_path, cfg)
    assert r.returncode != 0 and "moved.pt" in (r.stdout + r.stderr)
