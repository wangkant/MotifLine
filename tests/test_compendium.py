"""Cross-model merge definitions, and the membership call the merge exists to make.

The failure these guard against is a merge that runs and produces a lexicon that means
something other than it says: a leaf silently missing (one model's patterns absent, so
every cluster looks model-specific), a label MotifCompendium will chop up inside motif
names, or an archived leaf discovered at a different MoDISco width being read as if it
had been discovered at this pipeline's.
"""
import importlib.util
from pathlib import Path

import pytest
from motifline.compendium import MergeError, dag_inputs, leaves, merge_names, region_width

STEP = Path(__file__).resolve().parents[1] / "src/motifline/steps/merge_compendium.py"
_spec = importlib.util.spec_from_file_location("merge_compendium_step", STEP)
step = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step)             # top of the file is stdlib-only by contract

CFG = {
    "runs": {"k562_cbp_adopted": {"model": "cbp"}, "k562_ag_adopted": {"model": "ag"}},
    "modisco": {"window": 400},
    "compendium": {"merges": {
        "islandA": {"leaves": {"cbp": {"run": "k562_cbp_adopted"},
                               "ag2048": {"run": "k562_ag_adopted"}}},
        "archived": {"leaves": {"cbp": {"h5": "/arch/cbp/modisco.h5"},
                                "ag2048": {"run": "k562_ag_adopted"}},
                     "region_width": 500},
    }},
}


def test_a_run_leaf_is_a_dag_input_and_an_h5_leaf_is_not():
    assert merge_names(CFG) == ["archived", "islandA"]
    assert leaves(CFG, "islandA") == [("cbp", "results/k562_cbp_adopted/modisco/modisco.h5"),
                                      ("ag2048", "results/k562_ag_adopted/modisco/modisco.h5")]
    assert dag_inputs(CFG, "islandA") == ["results/k562_cbp_adopted/modisco/modisco.h5",
                                          "results/k562_ag_adopted/modisco/modisco.h5"]
    # the archived leaf is taken as given: declaring it an input would make a missing
    # archive look like a rule that failed to produce its output
    assert leaves(CFG, "archived")[0] == ("cbp", "/arch/cbp/modisco.h5")
    assert dag_inputs(CFG, "archived") == ["results/k562_ag_adopted/modisco/modisco.h5"]


def test_region_width_defaults_to_this_pipelines_window_and_can_be_overridden():
    assert region_width(CFG, "islandA") == 400        # config.modisco.window
    assert region_width(CFG, "archived") == 500       # the width THOSE leaves were found at


def test_malformed_merges_raise():
    def bad(spec, **extra):
        cfg = dict(CFG, compendium={"merges": {"m": dict(leaves=spec, **extra)}})
        with pytest.raises(MergeError):
            leaves(cfg, "m")
    bad({"cbp": {"run": "k562_cbp_adopted"}})                       # one leaf is not a merge
    bad({"cbp": {"run": "nosuchrun"}, "ag": {"run": "k562_ag_adopted"}})
    bad({"cbp": {}, "ag": {"run": "k562_ag_adopted"}})              # neither run nor h5
    bad({"cbp": {"run": "k562_cbp_adopted", "h5": "/x.h5"},         # both
         "ag": {"run": "k562_ag_adopted"}})
    bad({"cbp-2114": {"run": "k562_cbp_adopted"},                   # '-' is a name separator
         "ag": {"run": "k562_ag_adopted"}})
    with pytest.raises(MergeError):
        leaves(CFG, "nosuchmerge")


def test_membership_separates_shared_from_model_specific():
    labels = ["cbp", "ag2048"]
    assert step.membership({"cbp", "ag2048"}, labels) == "shared"
    assert step.membership({"ag2048"}, labels) == "ag2048_only"
    assert step.membership({"cbp"}, labels) == "cbp_only"
    # 2-of-3 is neither shared nor exclusive and must not be reported as either
    assert step.membership({"cbp", "ag2048"}, ["cbp", "ag2048", "ag16k"]) == "ag2048+cbp"
