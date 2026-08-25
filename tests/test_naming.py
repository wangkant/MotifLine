"""The naming chain's load-bearing claims, each testable in isolation.

Not a unit-test suite for its own sake: every case here is a way the chain has been
seen to fail silently, or a parameter whose value is a decision rather than a default.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motifline import mlex as mlx
from motifline import naming as nm

STEPS = Path(__file__).resolve().parents[1] / "src/motifline/steps"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_step", STEPS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


union_step = _load("lexicon_union")
medoid_step = _load("lexicon_medoid")
name_step = _load("name_patterns")

CFG = {"reference_lexicon": {
    "sources": {"H14": "/lib/h14.meme", "JAS": "/lib/jaspar.meme"},
    "rsat": {"bin": "/env/rsat/bin/rsat", "hclust_method": "average",
             "metric": "Ncor", "lth": {"cor": 0.75, "Ncor": 0.55}, "quick": True},
    "tomtom": "/env/meme/bin/tomtom",
    "w_th": 5,
    "gate": {"qval": 0.05, "ncor": 0.55}}}

HOCOMOCO_STYLE = """MEME version 4

ALPHABET= ACGT

MOTIF AHR.H14CORE.0.P.B
letter-probability matrix: alength= 4 w= 2 nsites= 1003
0.1754735792622134\t0.3060817547357926\t0.390827517447657\t0.127617148554337
0.0019940179461615\t0.0\t0.0\t0.9980059820538385
URL https://hocomoco14.autosome.org/motif/AHR.H14CORE.0.P.B
"""

JASPAR_STYLE = """MEME version 4

ALPHABET= ACGT

MOTIF MA0004.1 Arnt
letter-probability matrix: alength= 4 w= 1 nsites= 20 E= 0
 0.200000  0.800000  0.000000  0.000000
"""


def test_union_adds_the_E_field_hocomoco_omits(tmp_path):
    """Without ``E= 0`` RSAT drops the whole HOCOMOCO half and says nothing."""
    p = tmp_path / "h14.meme"
    p.write_text(HOCOMOCO_STYLE)
    lines = union_step.blocks(p)
    matrix_line = next(l for l in lines if l.startswith("letter-probability"))
    assert matrix_line.rstrip("\n").endswith("E= 0")
    # and the probabilities are copied through, not reformatted
    assert "0.1754735792622134\t0.3060817547357926\t0.390827517447657\t0.127617148554337\n" in lines


def test_union_does_not_double_the_E_field_or_split_jaspar_names(tmp_path):
    p = tmp_path / "jaspar.meme"
    p.write_text(JASPAR_STYLE)
    lines = union_step.blocks(p)
    assert lines[0] == "MOTIF MA0004.1_Arnt\n"          # one id, not id + symbol
    matrix_line = next(l for l in lines if l.startswith("letter-probability"))
    assert matrix_line.count("E=") == 1


def test_cluster_label_names_the_medoid_it_represents():
    """A label whose representative belongs to a TF the label omits is unreadable."""
    label = medoid_step.cluster_label(
        181, "MA0139.2_CTCF", ["CTCF", "CTCFL", "CTCFL2"], 3)
    assert label.startswith("cl181_CTCF")
    assert label.endswith("_n3")


def test_gate_is_an_AND_so_a_significant_repeat_match_fails():
    """The failure this arm exists for: an Alu B-box matching HSF1 at q ~ 1e-7 because
    TTCGAA is an HSF1 half-site palindrome. Significant, and the wrong label."""
    assert name_step.gated(1e-7, 0.90, 0.05, 0.55) is True
    assert name_step.gated(1e-7, 0.31, 0.05, 0.55) is False     # Ncor arm
    assert name_step.gated(0.63, 0.90, 0.05, 0.55) is False     # q arm
    assert name_step.gated(1e-7, float("nan"), 0.05, 0.55) is False   # no comparable core


def test_tomtom_line_omits_mi_1():
    """-mi 1 against a batch query file returns matches for pattern 0 and nothing else."""
    argv = name_step.tomtom_argv("/bin/tomtom", "q.meme", "t.meme", "out")
    assert "-mi" not in argv
    assert argv[argv.index("-dist") + 1] == "pearson"


def test_modisco_trim_is_the_thirty_percent_of_max_span():
    ppm = np.tile([0.25, 0.25, 0.25, 0.25], (5, 1))
    cwm = np.zeros((5, 4))
    cwm[1] = [1.0, 0, 0, 0]          # max
    cwm[2] = [0.1, 0, 0, 0]          # inside the span, below threshold: kept anyway
    cwm[3] = [0.5, 0, 0, 0]
    trimmed, start, end = name_step.modisco_trim(ppm, cwm, threshold=0.3)
    assert (start, end) == (1, 4) and len(trimmed) == 3


def test_db_yaml_proposes_the_cluster_and_only_gated_rows():
    df = pd.DataFrame([
        dict(label="cbp", key="pos_patterns.pattern_0", assigned_cluster="cl181_CTCF_n3",
             match0="MA0139.2_CTCF", qval0=1e-9, overlap0=15, annotated=True),
        dict(label="ag", key="pos_patterns.pattern_9", assigned_cluster="cl77_HSF1_n2",
             match0="HSF1.H14CORE.0.P.B", qval0=1e-7, overlap0=11, annotated=False),
    ])
    out = name_step.db_matches(df, {"cbp": "cbp_islandA"})
    assert len(out) == 1
    assert out[0]["node_id"] == "cbp_islandA::pos_patterns.pattern_0"
    assert out[0]["proposed_family_id"] == "cl181_CTCF_n3"     # cluster, not the entry
    assert out[0]["matched_motif_id"] == "MA0139.2_CTCF"       # entry kept as evidence


def test_rsat_thresholds_are_passed_explicitly():
    """RSAT standalone defaults these to empty; the web server's values are not inherited."""
    argv = nm.rsat_argv(CFG, "union.meme", "out/rsat")
    for field, value in (("cor", "0.75"), ("Ncor", "0.55")):
        i = [k for k, a in enumerate(argv) if a == "-lth" and argv[k + 1] == field]
        assert i, f"-lth {field} missing"
        assert argv[i[0] + 2] == value
    assert argv[argv.index("-matrix") + 1] == "union"          # the union_m<N>_ id prefix


def test_a_threshold_quick_would_discard_is_refused():
    """RSAT accepts `-lth w 5 ... -quick` and then drops the w arm: the run reaches R as
    `lthsp = 'Ncor_0.55_cor_0.75_'`. Config must state the thresholds in force."""
    cfg = {"reference_lexicon": {**CFG["reference_lexicon"],
                                 "rsat": {**CFG["reference_lexicon"]["rsat"],
                                          "lth": {"w": 5, "cor": 0.75, "Ncor": 0.55}}}}
    with pytest.raises(nm.NamingError, match="quick"):
        nm.rsat_argv(cfg, "union.meme", "out/rsat")
    # ...unless quick is off, where RSAT does honour it
    cfg["reference_lexicon"]["rsat"]["quick"] = False
    assert "-quick" not in nm.rsat_argv(cfg, "union.meme", "out/rsat")


def test_merge_annotate_has_a_database_by_default():
    """Before 2026-08-25 db_yaml returned None for every merge and annotate was skipped,
    so merges carried no proposed labels at all."""
    cfg = {"compendium": {"merges": {"islandA": {"leaves": {"cbp": {"h5": "a"},
                                                            "ag": {"h5": "b"}}}}}}
    assert mlx.db_yaml(cfg, "islandA") == "results/merges/islandA/naming/db.yaml"
    cfg["compendium"]["merges"]["islandA"]["db_yaml"] = "/frozen/db.yaml"
    assert mlx.db_yaml(cfg, "islandA") == "/frozen/db.yaml"


def test_db_node_ids_match_the_ids_mlex_ingest_really_writes(tmp_path):
    """The half of the join that comes from motiflex, exercised rather than assumed.

    `mlex annotate` matches a database entry to a candidate on
    `<analysis id>::<pattern key>` and proposes nothing -- silently, backend logged
    UNVERIFIED -- for an id that does not exist. The sibling test below compares two
    strings this repo computes, so it cannot see a motiflex-side format change; this one
    runs the real ingest and reads the ids back out of registry.json. motiflex removed a
    whole subcommand on 2026-08-23, so a format change is not hypothetical.
    """
    import subprocess

    import h5py
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config/default.yaml").read_text())
    mlex_bin = ((cfg.get("motiflex") or {}).get("mlex") or "")
    if not Path(mlex_bin).exists():
        pytest.skip("mlex binary not available at config.motiflex.mlex")

    rng = np.random.default_rng(0)
    n_regions, width = 4, 200
    analyses = []
    for label in ("cbp", "ag"):
        d = tmp_path / label
        d.mkdir()
        ohe = np.zeros((n_regions, 4, width), dtype=np.int8)
        ohe[:, 0, :] = 1
        attr = rng.normal(size=(n_regions, 4, width)).astype(np.float32)
        np.save(d / "ohe.npy", ohe)
        np.save(d / "attr.npy", attr)
        (d / "regions.bed").write_text("".join(
            f"chr1\t{i * 1000}\t{i * 1000 + width}\tp{i}\t0\t.\t0\t0\t0\t{width // 2}\tTrue\n"
            for i in range(n_regions)))
        with h5py.File(d / "modisco.h5", "w") as f:
            for group in ("pos_patterns", "neg_patterns"):
                g = f.create_group(group)
                for k in range(2):
                    pat = g.create_group(f"pattern_{k}")
                    ppm = rng.random((10, 4))
                    pat.create_dataset("sequence", data=ppm / ppm.sum(1, keepdims=True))
                    pat.create_dataset("contrib_scores", data=rng.normal(size=(10, 4)))
                    pat.create_dataset("hypothetical_contribs", data=rng.normal(size=(10, 4)))
                    sq = pat.create_group("seqlets")
                    sq.create_dataset("n_seqlets", data=np.array([7]))
        analyses.append({
            "id": nm.node_prefix("smoke", label),
            "model": "chrombpnet" if label == "cbp" else "alphagenome",
            "readout": "counts" if label == "cbp" else "atac_track60",
            "union_id": label.upper(), "context": "smoke",
            "readout_celltype": "K562", "training_celltype": "K562",
            "input_width": width, "context_window": width,
            "modisco_h5": str(d / "modisco.h5"), "contributions": str(d / "attr.npy"),
            "one_hot": str(d / "ohe.npy"), "regions_bed": str(d / "regions.bed"),
            "search_metaclusters": {"pos_patterns": True, "neg_patterns": True},
        })
    project = tmp_path / "project.yaml"
    project.write_text(yaml.safe_dump(
        {"project": "smoke", "peak_universe_id": "SMOKE", "analyses": analyses,
         "shared_attribution_groups": None, "provenance": {"random_seed": 42}},
        sort_keys=False))
    r = subprocess.run([mlex_bin, "ingest", str(project), "--seed", "42",
                        "--out", str(tmp_path / "registry")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    registered = {n["node_id"] for n in
                  json.loads((tmp_path / "registry/registry.json").read_text())["nodes"]}
    # what merge_naming would put in db.yaml for the same patterns
    ours = {f"{nm.node_prefix('smoke', label)}::{group}.pattern_{k}"
            for label in ("cbp", "ag") for group in ("pos_patterns", "neg_patterns")
            for k in range(2)}
    assert ours == registered, (
        f"db.yaml node ids and ingest's registered ids have diverged; annotate would "
        f"propose nothing. ours-only={sorted(ours - registered)[:3]} "
        f"ingest-only={sorted(registered - ours)[:3]}")


def test_db_node_ids_match_the_ids_mlex_project_declares():
    """The cheap half: db.yaml and the ingest project must agree inside this repo."""
    cfg = {"motiflex": {"mlex": "mlex", "celltype": "K562", "peak_universe_id": "U"},
           "compendium": {"merges": {"islandA": {"leaves": {
               "cbp": {"h5": "a", "attr": "x", "ohe": "y", "bed": "z",
                       "meta": {"model": "chrombpnet", "readout": "counts", "union_id": "CBP",
                                "training_celltype": "K562", "input_width": 2114,
                                "context_window": 2114}},
               "ag": {"h5": "b", "attr": "x", "ohe": "y", "bed": "z",
                      "meta": {"model": "alphagenome", "readout": "atac_track60",
                               "union_id": "AG", "training_celltype": "MULTI",
                               "input_width": 2048, "context_window": 2048}}}}}}}
    ingest_ids = {a["id"] for a in mlx.project(cfg, "islandA")["analyses"]}
    assert {nm.node_prefix("islandA", l) for l in mlx.leaf_labels(cfg, "islandA")} == ingest_ids


# --- the merge criterion (2026-08-25): RSAT's rule, replacing the removed `mlex merge` ---

merge_step = _load("merge_tree")


def _link3():
    """3 leaves: (0,1) merge first at node 3, then (3,2) at node 4."""
    return np.array([[0.0, 1.0, 0.1, 2.0], [3.0, 2.0, 0.2, 3.0]])


def _scores(ncor, cor, w):
    """Constant off-diagonal score matrices for 3 leaves."""
    def m(v):
        a = np.full((3, 3), float(v))
        np.fill_diagonal(a, 1.0)
        return a
    return {"Ncor": m(ncor), "cor": m(cor), "w": m(w)}


def test_merge_gate_is_a_conjunction_not_a_single_height():
    """RSAT merges only when EVERY arm clears: check_alignment.R:106-109."""
    th = {"Ncor": 0.55, "cor": 0.75, "w": 5}
    passing = merge_step.node_verdicts(_link3(), _scores(0.9, 0.9, 8), th, 3)[0]
    assert passing.merged.all()
    # same Ncor and cor, but the overlap arm alone fails -> nothing merges
    nodes, groups = merge_step.node_verdicts(_link3(), _scores(0.9, 0.9, 4), th, 3)
    assert not nodes.merged.any()
    assert nodes.failed_arms.iloc[0] == "w"
    assert groups == [[0], [1], [2]]


def test_a_failed_node_breaks_the_chain_for_its_ancestors():
    """align.motifs.R:101 -- an ancestor of a failed node is not re-tested."""
    th = {"Ncor": 0.55, "cor": 0.75, "w": 5}
    scores = _scores(0.9, 0.9, 8)
    for arm in ("Ncor", "cor", "w"):        # make ONLY the 0-1 pair fail
        scores[arm][0, 1] = scores[arm][1, 0] = 0.0
    nodes, groups = merge_step.node_verdicts(_link3(), scores, th, 3)
    assert nodes.merged.iloc[0] is np.False_ or not nodes.merged.iloc[0]
    assert not nodes.merged.iloc[1], "the parent merged despite a broken child"
    assert bool(nodes.inherited_break.iloc[1])
    assert groups == [[0], [1], [2]], "a broken subtree was resurrected by its parent"


def test_two_member_representative_is_flagged_as_undetermined():
    """Both members have the same mean distance by construction, so 'medoid' is empty."""
    ncor = np.array([[1.0, 0.8], [0.8, 1.0]])
    rep, tied = merge_step.choose_representative(
        [0, 1], ncor, ["a::p0", "b::p0"], {"a::p0": 10, "b::p0": 99})
    assert tied is True
    assert rep == "b::p0"        # tie broken on seqlet count, deterministically


def test_cut_height_is_refused_rather_than_reinterpreted():
    """0.05 on 1 - ppm_similarity has no image under the RSAT criterion."""
    cfg = {"compendium": {"merges": {"m": {"cut_height": 0.05,
                                           "leaves": {"a": {"h5": "x"}, "b": {"h5": "y"}}}}}}
    with pytest.raises(mlx.MlexError, match="cut_height"):
        mlx.thresholds(cfg, "m")


def test_every_threshold_arm_is_required():
    """An absent arm is an undeclared threshold, not a disabled one."""
    cfg = {"compendium": {"merges": {"m": {"thresholds": {"Ncor": 0.55, "cor": 0.75},
                                           "leaves": {"a": {"h5": "x"}, "b": {"h5": "y"}}}}}}
    with pytest.raises(mlx.MlexError, match="w"):
        mlx.thresholds(cfg, "m")


# --- regressions the audit of 2026-08-25 asked for ---

def test_ncor_matches_a_brute_force_oracle():
    """The 55 lines of prefix-sum/offset/revcomp arithmetic under every medoid."""
    mm = _load("motif_matrix")
    rng = np.random.default_rng(0)
    mats = []
    for _ in range(6):
        w = int(rng.integers(6, 12))
        a = rng.random((w, 4)) + 0.05
        mats.append(a / a.sum(1, keepdims=True))

    def brute(x, y, w_th=5):
        best = -np.inf
        for b in (y, y[::-1, ::-1]):
            for d in range(-(len(b) - 1), len(x)):
                lo, hi = max(0, d), min(len(x), d + len(b))
                if hi - lo < w_th:
                    continue
                u = x[lo:hi].ravel()
                v = b[lo - d:hi - d].ravel()
                c = np.corrcoef(u, v)[0, 1]
                best = max(best, c * (hi - lo) / max(len(x), len(b)))
        return best

    got, _, _, _ = mm.ncor_matrix(mats)
    for i in range(len(mats)):
        for j in range(len(mats)):
            if i == j:
                continue
            assert abs(got[i, j] - brute(mats[i], mats[j])) < 1e-9, (i, j)


def test_example_config_is_runnable_for_every_run_it_declares():
    """The tracked template is the whole bootstrap; a key missing from it is a repo
    that cannot be cloned and run. geometry() reads config.geometry BEFORE a run's own
    in_window, so dropping that block breaks every geometry-dependent rule."""
    import yaml
    from motifline.geometry import attr_width, geometry
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config/example.yaml").read_text())
    assert cfg["runs"], "example.yaml declares no runs"
    for name, run in cfg["runs"].items():
        g = geometry(cfg, run)
        assert g["crop"] <= g["in_window"]
        assert attr_width(cfg, run) == g["crop"], name
