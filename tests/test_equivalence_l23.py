import json

import h5py
import numpy as np
import pandas as pd

from motifline.equivalence import compare_hits, compare_modisco

HITS_HEADER = ["chr", "start", "end", "start_untrimmed", "end_untrimmed", "motif_name",
               "hit_coefficient", "hit_coefficient_global", "hit_similarity",
               "hit_correlation", "hit_importance", "hit_importance_sq", "strand",
               "peak_name", "peak_id"]


def _h5(path, patterns):
    with h5py.File(path, "w") as f:
        for mc, mats in patterns.items():
            g = f.create_group(mc)
            for i, m in enumerate(mats):
                g.create_group(f"pattern_{i}").create_dataset("contrib_scores", data=m)
    return path


def _cwm(rng, length=20):
    return rng.normal(size=(length, 4))


def _hits(path, rows):
    df = pd.DataFrame(rows, columns=["chr", "start", "end", "motif_name"])
    for c in HITS_HEADER:
        if c not in df.columns:
            df[c] = "."
    df[HITS_HEADER].to_csv(path, sep="\t", index=False)
    return path


def test_compare_modisco_matches_identical_files(tmp_path):
    rng = np.random.default_rng(0)
    pats = {"pos_patterns": [_cwm(rng), _cwm(rng)], "neg_patterns": [_cwm(rng)]}
    a = _h5(tmp_path / "new.h5", pats)
    b = _h5(tmp_path / "old.h5", pats)
    rep = compare_modisco(a, b, tmp_path / "rep.json")
    assert rep["n_patterns_new"] == rep["n_patterns_old"] == 3
    assert rep["pos_new"] == rep["pos_old"] == 2 and rep["neg_new"] == rep["neg_old"] == 1
    assert rep["min_best_cosine"] > 0.999
    assert rep["within_tolerance"] is True and rep["flags"] == []
    assert json.loads((tmp_path / "rep.json").read_text())["n_patterns_old"] == 3


def test_compare_modisco_finds_reverse_complement_match(tmp_path):
    rng = np.random.default_rng(1)
    m = _cwm(rng)
    a = _h5(tmp_path / "new.h5", {"pos_patterns": [m[::-1, ::-1]]})
    b = _h5(tmp_path / "old.h5", {"pos_patterns": [m]})
    rep = compare_modisco(a, b, tmp_path / "rep.json")
    assert rep["min_best_cosine"] > 0.999


def test_compare_modisco_flags_pattern_count_gap_and_low_cosine(tmp_path):
    rng = np.random.default_rng(2)
    a = _h5(tmp_path / "new.h5", {"pos_patterns": [_cwm(rng)]})
    b = _h5(tmp_path / "old.h5", {"pos_patterns": [_cwm(rng) for _ in range(5)]})
    rep = compare_modisco(a, b, tmp_path / "rep.json")
    assert rep["within_tolerance"] is False
    assert any("pattern count" in f for f in rep["flags"])
    assert any("cosine" in f for f in rep["flags"])


def test_compare_hits_identical_files(tmp_path):
    rows = [("chr1", 10, 20, "pos_patterns.pattern_0"),
            ("chr1", 50, 60, "pos_patterns.pattern_1"),
            ("chr2", 10, 20, "pos_patterns.pattern_0")]
    a = _hits(tmp_path / "new.tsv", rows)
    b = _hits(tmp_path / "old.tsv", rows)
    rep = compare_hits(a, b, tmp_path / "rep.json")
    assert rep["n_new"] == rep["n_old"] == 3
    assert rep["exact_jaccard"] == 1.0 and rep["tolerant_jaccard"] == 1.0
    assert rep["within_tolerance"] is True and rep["flags"] == []
    counts = {r["motif_name"]: r for r in rep["per_motif_counts"]}
    assert counts["pos_patterns.pattern_0"]["new"] == 2
    assert counts["pos_patterns.pattern_0"]["ratio_new_over_old"] == 1.0


def test_compare_hits_flags_count_ratio_outside_band(tmp_path):
    old = [("chr1", i * 100, i * 100 + 10, "pos_patterns.pattern_0") for i in range(10)]
    new = [("chr1", i * 100, i * 100 + 10, "pos_patterns.pattern_0") for i in range(3)]
    rep = compare_hits(_hits(tmp_path / "new.tsv", new), _hits(tmp_path / "old.tsv", old),
                       tmp_path / "rep.json")
    assert rep["within_tolerance"] is False
    assert any("pos_patterns.pattern_0" in f for f in rep["flags"])
    assert rep["n_new"] == 3 and rep["n_old"] == 10


def test_compare_hits_tolerates_small_coordinate_shift(tmp_path):
    old = [("chr1", 100, 110, "m")]
    new = [("chr1", 101, 111, "m")]
    rep = compare_hits(_hits(tmp_path / "new.tsv", new), _hits(tmp_path / "old.tsv", old),
                       tmp_path / "rep.json", tol_bp=2)
    assert rep["exact_jaccard"] == 0.0 and rep["tolerant_jaccard"] == 1.0


def test_compare_hits_rejects_unexpected_header(tmp_path):
    (tmp_path / "bad.tsv").write_text("a\tb\n1\t2\n")
    good = _hits(tmp_path / "old.tsv", [("chr1", 1, 2, "m")])
    try:
        compare_hits(tmp_path / "bad.tsv", good, tmp_path / "rep.json")
    except ValueError as e:
        assert "missing columns" in str(e)
    else:
        raise AssertionError("expected ValueError on an unexpected hits header")


def test_identity_map_reads_matches_above_the_cosine_floor(tmp_path):
    from motifline.equivalence import identity_map_from_modisco_report
    rep = tmp_path / "modisco_report.json"
    rep.write_text(json.dumps({"per_old_pattern": [
        {"old": "pos_patterns/pattern_0", "best_new": "pos_patterns/pattern_3", "cosine": 0.99},
        {"old": "pos_patterns/pattern_1", "best_new": "pos_patterns/pattern_0", "cosine": 0.40},
        {"old": "neg_patterns/pattern_0", "best_new": None, "cosine": float("nan")}]}))
    assert identity_map_from_modisco_report(rep) == {
        "pos_patterns.pattern_0": "pos_patterns.pattern_3"}


def test_compare_hits_renames_archived_motifs_before_joining(tmp_path):
    # Same motif, renumbered between lexicons: a name join would call it two
    # different motifs; the identity map must recover the 1:1 count.
    new = [("chr1", 10, 20, "pos_patterns.pattern_3")] * 1
    old = [("chr1", 10, 20, "pos_patterns.pattern_0")] * 1
    imap = {"pos_patterns.pattern_0": "pos_patterns.pattern_3"}
    naive = compare_hits(_hits(tmp_path / "n1.tsv", new), _hits(tmp_path / "o1.tsv", old),
                         tmp_path / "r1.json")
    assert naive["within_tolerance"] is False and naive["join"] == "motif_name"
    mapped = compare_hits(_hits(tmp_path / "n2.tsv", new), _hits(tmp_path / "o2.tsv", old),
                          tmp_path / "r2.json", identity_map=imap)
    assert mapped["join"] == "cwm_identity" and mapped["within_tolerance"] is True
    assert mapped["exact_jaccard"] == 1.0
    assert mapped["old_motifs_without_cwm_counterpart"] == []


def test_compare_hits_reports_archived_motifs_with_no_counterpart(tmp_path):
    new = [("chr1", 10, 20, "pos_patterns.pattern_0")]
    old = [("chr1", 10, 20, "pos_patterns.pattern_0"), ("chr2", 5, 15, "pos_patterns.pattern_9")]
    rep = compare_hits(_hits(tmp_path / "n.tsv", new), _hits(tmp_path / "o.tsv", old),
                       tmp_path / "r.json", identity_map={"pos_patterns.pattern_0": "pos_patterns.pattern_0"})
    assert rep["old_motifs_without_cwm_counterpart"] == ["pos_patterns.pattern_9"]
    assert any("no new pattern above cosine" in f for f in rep["flags"])
