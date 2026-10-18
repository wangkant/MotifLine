"""Naming edge cases: a source without a trailing newline, a modisco.h5 without patterns."""
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from conftest import STEPS


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_step", STEPS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


union_step = _load("lexicon_union")
name_step = _load("name_patterns")


def test_union_source_without_trailing_newline_is_not_fused(tmp_path, monkeypatch):
    a = tmp_path / "a.meme"
    a.write_text("MOTIF A1\nletter-probability matrix: alength= 4 w= 1 E= 0\n"
                 " 0.25 0.25 0.25 0.25")                        # no trailing newline
    b = tmp_path / "b.meme"
    b.write_text("MOTIF B1\nletter-probability matrix: alength= 4 w= 1 E= 0\n"
                 " 0.25 0.25 0.25 0.25\n")
    assert all(l.endswith("\n") for l in union_step.blocks(a))
    out = tmp_path / "union.meme"
    monkeypatch.setattr(sys, "argv", ["x", "--source", f"A={a}", "--source", f"B={b}",
                                      "--out", str(out)])
    union_step.main()
    assert "MOTIF B1\n" in out.read_text().splitlines(keepends=True)


def _h5(path, n_patterns):
    with h5py.File(path, "w") as f:
        if n_patterns:
            g = f.create_group("pos_patterns")
            for i in range(n_patterns):
                p = g.create_group(f"pattern_{i}")
                p["sequence"] = np.tile([0.7, 0.1, 0.1, 0.1], (6, 1))
                p["contrib_scores"] = np.ones((6, 4))
                p.create_group("seqlets")["n_seqlets"] = np.array([12])
    return path


def _argv(tmp_path, leaves):
    ann = tmp_path / "h14.jsonl"
    ann.write_text(json.dumps({"name": "TYY1.H14CORE.0.P.B", "masterlist_info": {
        "species": {"HUMAN": {"gene_symbol": "YY1"}}}}) + "\n")
    target = tmp_path / "union.meme"
    target.write_text("MEME version 4\n")
    argv = ["x", "--target", str(target), "--h14_annotation", str(ann),
            "--tomtom", "/bin/tomtom", "--qval_threshold", "0.05",
            "--out_dir", str(tmp_path / "out")]
    for label, h5 in leaves:
        argv += ["--h5", f"{label}={h5}"]
    return argv


def test_no_patterns_at_all_exits_naming_the_h5(tmp_path, monkeypatch):
    h5 = _h5(tmp_path / "empty.h5", 0)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, [("run", h5)]))
    monkeypatch.setattr(name_step.subprocess, "run",
                        lambda *a, **k: pytest.fail("TomTom ran on an empty query"))
    with pytest.raises(SystemExit, match=r"modisco.h5 has no patterns: .*empty\.h5"):
        name_step.main()


def test_a_label_without_patterns_skips_tomtom(tmp_path, monkeypatch):
    full, empty = _h5(tmp_path / "full.h5", 1), _h5(tmp_path / "empty.h5", 0)
    calls = []

    def fake_run(argv, check):
        calls.append(argv)
        d = Path(argv[argv.index("-oc") + 1])
        d.mkdir(parents=True)
        (d / "tomtom.tsv").write_text(
            "Query_ID\tTarget_ID\tOptimal_offset\tp-value\tE-value\tq-value\tOverlap\n"
            "pos_patterns.pattern_0\tTYY1.H14CORE.0.P.B\t0\t1e-5\t1e-3\t0.01\t6\n")

    monkeypatch.setattr(sys, "argv", _argv(tmp_path, [("a", full), ("b", empty)]))
    monkeypatch.setattr(name_step.subprocess, "run", fake_run)
    name_step.main()
    assert len(calls) == 1 and calls[0][-2].endswith("a_query.meme")
    df = pd.read_csv(tmp_path / "out" / "pattern_names.tsv", sep="\t")
    assert list(df.label) == ["a"] and list(df.name) == ["YY1"]
