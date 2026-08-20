"""Cross-call leaf resolution: the arrays a leaf is scanned over must be the ones
its lexicon entries were discovered on -- a wrong path or width here corrupts the
occupancy comparison silently, so resolution refuses rather than guesses."""
import pytest

from motifline.mlex import MlexError, cross_call_inputs, cross_call_leaf, leaf_labels

CFG = {
    "runs": {"k562_ag_2048": {"model": "ag", "track": 60}},
    "geometry": {"ag": {"in_window": 2048, "crop": 2048},
                 "cbp": {"in_window": 2114}},
    "motiflex": {"mlex": "/x/mlex", "celltype": "K562", "peak_universe_id": "U"},
    "compendium": {"merges": {
        "m": {"cut_height": 0.05, "leaves": {
            "ag2048": {"run": "k562_ag_2048"},
            "cbp": {"h5": "/arch/modisco.h5", "attr": "/arch/attr.npy",
                    "ohe": "/arch/ohe.npy", "bed": "/arch/peaks.bed",
                    "meta": {"model": "chrombpnet", "readout": "counts",
                             "union_id": "CBP2114", "training_celltype": "K562",
                             "input_width": 2114, "context_window": 2114}},
        }},
    }},
}


def test_run_leaf_resolves_to_its_own_attr_stage_at_the_run_width():
    assert leaf_labels(CFG, "m") == ["ag2048", "cbp"]
    leaf = cross_call_leaf(CFG, "m", "ag2048")
    assert leaf == {"attr": "results/k562_ag_2048/attr/attr.npy",
                    "ohe": "results/k562_ag_2048/attr/ohe.npy",
                    "bed": "results/k562_ag_2048/attr/regions.bed",
                    "width": 2048}
    assert cross_call_inputs(CFG, "m", "ag2048") == [
        "results/k562_ag_2048/attr/attr.npy",
        "results/k562_ag_2048/attr/ohe.npy",
        "results/k562_ag_2048/attr/regions.bed"]


def test_h5_leaf_reads_exactly_what_it_declared_and_is_not_a_dag_input():
    leaf = cross_call_leaf(CFG, "m", "cbp")
    assert leaf == {"attr": "/arch/attr.npy", "ohe": "/arch/ohe.npy",
                    "bed": "/arch/peaks.bed", "width": 2114}
    assert cross_call_inputs(CFG, "m", "cbp") == []


def test_unknown_leaf_and_underdeclared_leaf_refuse():
    with pytest.raises(MlexError, match="no leaf"):
        cross_call_leaf(CFG, "m", "nope")
    broken = {**CFG, "compendium": {"merges": {"m": {"leaves": {
        "a": {"run": "k562_ag_2048"},
        "b": {"h5": "/arch/modisco.h5", "meta": {}},   # arrays undeclared
    }}}}}
    with pytest.raises(MlexError, match="missing"):
        cross_call_leaf(broken, "m", "b")
