"""Recording tangermeme's convergence warnings."""
import warnings

import pytest

from convergence import THRESHOLD, Deltas, summary


def test_deltas_are_counted_from_tangermemes_warning():
    with Deltas() as d:
        warnings.warn("Convergence deltas too high: tensor([2.5000e-03, 1.0000e-06, "
                      "4.0e-02], device='cuda:0', grad_fn=<AbsBackward0>)", RuntimeWarning, stacklevel=2)
        warnings.warn("Convergence deltas too high: tensor([0.0011])", RuntimeWarning, stacklevel=2)
    assert d.over == 3
    assert d.max == pytest.approx(0.04)


def test_other_warnings_still_reach_the_caller():
    with pytest.warns(UserWarning, match="unrelated"):
        with Deltas() as d:
            warnings.warn("unrelated", UserWarning, stacklevel=2)
    assert d.over == 0 and d.max == 0.0


def test_summary_states_the_threshold():
    assert summary(40, 2, 0.5) == {"threshold": THRESHOLD, "pairs": 40, "pairs_over": 2,
                                   "max_delta": 0.5}
