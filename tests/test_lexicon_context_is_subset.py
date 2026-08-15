"""Why `k562_cbp_cl1_adopted` is allowed to be compared to the archived cl1 hits.

The external-lexicon routes are validated against
`region_v4/05_allcluster_lexicon/hits_{gw,mmv}/cbp/cl1/finemo_hits/hits.tsv`.
That is only a legitimate comparison object if the attributions the archive
scored are the attributions this pipeline scores. They are: `prep_allclusters.py`
built each cluster leaf by row-subsetting `interpret_cbp_par_hyp/{attr,ohe}.npy`
on an explicit `(chrom, summit_abs)` join (that script's lines 44-80), which is
the same archive `k562_cbp_adopted` links.

This test is the executable form of that claim -- byte equality, not a citation.
It is skipped where the archive is not mounted.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_cfg = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
_kant = Path(yaml.safe_load(_cfg.read_text())["kant"]) if _cfg.exists() else Path("/nonexistent")
V3 = _kant / "new_experiment/region_v3/k562/interpret_cbp_par_hyp"
CL1 = _kant / "new_experiment/region_v4/04_cbp_only_islands/allclusters/cl1"

pytestmark = pytest.mark.skipif(
    not (V3 / "attr.npy").exists() or not (CL1 / "attr.npy").exists(),
    reason="region_v3/v4 archive not available")


def _summit_index(bed):
    df = pd.read_csv(bed, sep="\t", header=None)
    return [(c, int(s) + int(o)) for c, s, o in zip(df[0], df[1], df[9])]


def test_cl1_is_a_byte_identical_row_subset_of_the_adopted_attribution():
    key2row = {k: i for i, k in enumerate(_summit_index(V3 / "peaks.bed"))}
    assert len(key2row) == 33917, "(chrom, summit) is not unique in the canonical bed"
    idx = np.array([key2row[k] for k in _summit_index(CL1 / "peaks_np.bed")])
    assert len(idx) == 2399 and (np.diff(idx) > 0).all()

    for name in ("attr.npy", "ohe.npy"):
        whole = np.load(V3 / name, mmap_mode="r")
        leaf = np.load(CL1 / name, mmap_mode="r")
        assert leaf.shape == (2399, 4, 2114)
        # Compare in blocks: the whole matrix is 1.1 GiB and fancy-indexing it
        # materialises the subset, which is fine, but the memmap read is not.
        for lo in range(0, len(idx), 500):
            hi = min(lo + 500, len(idx))
            assert np.array_equal(np.asarray(whole[idx[lo:hi]]),
                                  np.asarray(leaf[lo:hi])), f"{name} differs at rows {lo}:{hi}"
