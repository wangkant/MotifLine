import pytest

from motifline.runs import active_runs, cancelled_runs

CFG = {"runs": {"a": {"model": "cbp"},
                "b": {"model": "ag", "cancelled": True,
                      "cancelled_note": "results/b/CANCELLED.md"},
                "c": {"model": "cbp"}}}


def test_cancelled_runs_lists_only_the_cancelled_ones():
    assert cancelled_runs(CFG) == ["b"]


def test_active_runs_drops_cancelled_and_keeps_order():
    assert active_runs(CFG, ["c", "b", "a"]) == ["c", "a"]


def test_active_runs_leaves_an_uncancelled_list_alone():
    assert active_runs(CFG, ["a", "c"]) == ["a", "c"]


def test_active_runs_refuses_a_name_that_is_not_a_run():
    # A typo in the default target list must not look like a cancelled run:
    # silently dropping it would silently shrink `rule all`.
    with pytest.raises(KeyError, match="k562_typo"):
        active_runs(CFG, ["a", "k562_typo"])


def test_cancelled_flag_must_be_explicit_not_merely_present():
    cfg = {"runs": {"a": {"cancelled": False}}}
    assert cancelled_runs(cfg) == []
    assert active_runs(cfg, ["a"]) == ["a"]
