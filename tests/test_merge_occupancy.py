"""The occupancy/agreement arithmetic, end to end through the step's own main().

What these falsify: the peak x motif counting (a hit landing on the wrong row or
column), the binarisation caliber (occupancy counts peaks, not hits), the
agreement quadrants and jaccard, the zero-row guarantee for motifs a model never
calls, and the refusal to align leaves whose beds disagree.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

STEP = Path(__file__).resolve().parents[1] / "src/motifline/steps/merge_occupancy.py"
_spec = importlib.util.spec_from_file_location("merge_occupancy_step", STEP)
step = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step)

BED = "chr1\t100\t200\nchr1\t300\t400\nchr2\t100\t200\n"  # 3 peaks
MANIFEST = ("tier\tindex\tpattern_tag\tnode_id\tvariant_id\tmetacluster\n"
            "core\t0\tpos_patterns.pattern_0\tn0\tV0\tpos\n"
            "core\t1\tpos_patterns.pattern_1\tn1\tV1\tpos\n")


def _hits(rows):
    df = pd.DataFrame(rows, columns=["motif_name", "peak_id"])
    df.insert(0, "chr", "chr1")  # the step reads only motif_name and peak_id
    return df


def _run(tmp_path, ag_rows, cbp_rows, cbp_bed=BED):
    (tmp_path / "manifest.tsv").write_text(MANIFEST)
    (tmp_path / "ag.bed").write_text(BED)
    (tmp_path / "cbp.bed").write_text(cbp_bed)
    _hits(ag_rows).to_csv(tmp_path / "ag_hits.tsv", sep="\t", index=False)
    _hits(cbp_rows).to_csv(tmp_path / "cbp_hits.tsv", sep="\t", index=False)
    out = tmp_path / "occupancy"
    step.main(["--manifest", str(tmp_path / "manifest.tsv"), "--out_dir", str(out),
               "--hits", f"ag={tmp_path}/ag_hits.tsv", "--bed", f"ag={tmp_path}/ag.bed",
               "--hits", f"cbp={tmp_path}/cbp_hits.tsv", "--bed", f"cbp={tmp_path}/cbp.bed"])
    return out


def test_counts_binarisation_agreement_and_zero_rows(tmp_path):
    # ag: motif0 twice on peak0 + once on peak1; motif1 never.
    # cbp: motif0 on peak1; motif1 on peak2.
    out = _run(tmp_path,
               ag_rows=[("pos_patterns.pattern_0", 0), ("pos_patterns.pattern_0", 0),
                        ("pos_patterns.pattern_0", 1)],
               cbp_rows=[("pos_patterns.pattern_0", 1), ("pos_patterns.pattern_1", 2)])

    occ = pd.read_csv(out / "occupancy.tsv", sep="\t").set_index(["motif", "leaf"])
    r = occ.loc[("pos_patterns.pattern_0", "ag")]
    assert (r["n_hits"], r["n_peaks"]) == (3, 2) and r["frac_peaks"] == pytest.approx(2 / 3)
    r = occ.loc[("pos_patterns.pattern_1", "ag")]     # the zero row exists
    assert (r["n_hits"], r["n_peaks"], r["frac_peaks"]) == (0, 0, 0)
    assert occ.loc[("pos_patterns.pattern_1", "cbp")]["n_peaks"] == 1

    agr = pd.read_csv(out / "agreement.tsv", sep="\t").set_index("motif")
    m0 = agr.loc["pos_patterns.pattern_0"]
    assert (m0["leaf_a"], m0["leaf_b"]) == ("ag", "cbp")
    assert (m0["n_both"], m0["n_only_a"], m0["n_only_b"], m0["n_neither"]) == (1, 1, 0, 1)
    assert m0["jaccard"] == pytest.approx(0.5)
    m1 = agr.loc["pos_patterns.pattern_1"]
    assert (m1["n_both"], m1["n_only_a"], m1["n_only_b"], m1["jaccard"]) == (0, 0, 1, 0.0)

    mat = pd.read_csv(out / "peak_by_motif_ag.tsv.gz", sep="\t")
    assert list(mat["V0"]) == [2, 1, 0] and list(mat["V1"]) == [0, 0, 0]
    assert list(mat.columns[:4]) == ["peak_id", "chrom", "start", "end"]


def test_bed_mismatch_and_unknown_motif_refuse(tmp_path):
    with pytest.raises(SystemExit, match="refusing to align"):
        _run(tmp_path, ag_rows=[], cbp_rows=[],
             cbp_bed="chr1\t100\t200\nchr1\t999\t400\nchr2\t100\t200\n")
    with pytest.raises(SystemExit, match="absent from the manifest"):
        _run(tmp_path, ag_rows=[("pos_patterns.pattern_9", 0)], cbp_rows=[])
    with pytest.raises(SystemExit, match="outside"):
        _run(tmp_path, ag_rows=[("pos_patterns.pattern_0", 3)], cbp_rows=[])
