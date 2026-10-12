"""The synthetic input `motifline smoke` recovers a known motif from.

The generator is tested here; the end-to-end recovery it feeds is `motifline smoke`,
which needs modisco and finemo and so is not part of this suite.
"""
from __future__ import annotations

import numpy as np

from motifline.smoke import MOTIF, decode, synthetic


def test_shapes_and_dtypes_match_what_a_stage_writes():
    attr, ohe, planted, bed = synthetic(n=20, width=200)
    assert attr.shape == (20, 4, 200) and attr.dtype == np.float32
    assert ohe.shape == (20, 4, 200) and ohe.dtype == np.int8


def test_every_position_is_exactly_one_base():
    _, ohe, _, _ = synthetic(n=20, width=200)
    assert (ohe.sum(axis=1) == 1).all()


def test_the_motif_is_planted_in_the_requested_fraction_of_regions():
    _, _, planted, _ = synthetic(n=100, width=200, frac=0.6)
    assert sum(p is not None for p in planted) == 60


def test_a_planted_region_really_reads_as_the_motif():
    _, ohe, planted, _ = synthetic(n=20, width=200)
    i = next(k for k, p in enumerate(planted) if p is not None)
    start = planted[i]
    assert decode(ohe[i][:, start:start + len(MOTIF)]) == MOTIF


def test_attribution_is_hypothetical_shaped_nonzero_on_all_four_bases():
    attr, _, _, _ = synthetic(n=20, width=200)
    assert (np.abs(attr) > 1e-9).mean() > 0.99


def test_the_signal_sits_on_the_planted_motif_and_not_elsewhere():
    attr, ohe, planted, _ = synthetic(n=40, width=200)
    actual = (attr * ohe).sum(axis=1)
    inside, outside = [], []
    for i, p in enumerate(planted):
        if p is None:
            outside.append(actual[i])
            continue
        inside.append(actual[i][p:p + len(MOTIF)])
        outside.append(np.delete(actual[i], np.arange(p, p + len(MOTIF))))
    assert np.mean(np.abs(np.concatenate(inside))) > \
        10 * np.mean(np.abs(np.concatenate(outside)))


def test_regions_bed_carries_the_summit_in_column_ten():
    _, _, _, bed = synthetic(n=5, width=200)
    lines = bed.strip().split("\n")
    assert len(lines) == 5
    for line in lines:
        cols = line.split("\t")
        assert len(cols) == 10
        assert int(cols[9]) == 100


def test_the_same_seed_gives_the_same_arrays():
    a1, o1, p1, _ = synthetic(n=10, width=100, seed=7)
    a2, o2, p2, _ = synthetic(n=10, width=100, seed=7)
    assert np.array_equal(a1, a2) and np.array_equal(o1, o2) and p1 == p2


def test_a_different_seed_gives_different_arrays():
    a1, _, _, _ = synthetic(n=10, width=100, seed=7)
    a2, _, _, _ = synthetic(n=10, width=100, seed=8)
    assert not np.array_equal(a1, a2)


def test_decode_round_trips_an_arbitrary_one_hot_block():
    _, ohe, _, _ = synthetic(n=3, width=40, seed=1)
    assert len(decode(ohe[0])) == 40
    assert set(decode(ohe[0])) <= set("ACGT")
