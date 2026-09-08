"""attr_star.py's pure parts: the contribution recipe, the masking groups, the row policy.

Nothing here touches the model or the alignment; those are the GPU run's business. What
is pinned is everything a wrong version of would silently change MoDISco's input.
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from motifline.adopt import normalize_peaks

STEP = Path(__file__).resolve().parents[1] / "src/motifline/steps/attr_star.py"

_spec = importlib.util.spec_from_file_location("attr_star_step", STEP)
attr_star = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(attr_star)          # top of the file is stdlib-only by contract


def test_help_runs_without_heavy_imports():
    r = subprocess.run([sys.executable, str(STEP), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "--stride" in r.stdout and "--source" in r.stdout


def test_window_above_the_context_is_refused():
    r = subprocess.run([sys.executable, str(STEP), "--peaks", "p", "--chrom_sizes", "c",
                        "--out_dir", "o", "--msa_path", "m", "--in_window", "2048"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "1536" in (r.stderr + r.stdout)


@pytest.mark.parametrize("stride", [1, 4, 7, 8, 64])
def test_stride_groups_mask_every_position_exactly_once(stride):
    groups = attr_star.stride_groups(1536, stride)
    assert len(groups) == stride
    for r, g in enumerate(groups):
        assert (g % stride == r).all()
    assert np.array_equal(np.sort(np.concatenate(groups)), np.arange(1536))


def _probs(rng, w):
    p = rng.random((w, 4))
    return p / p.sum(1, keepdims=True)


def test_contrib_is_softmax_minus_a_quarter():
    # the reverse strand is the reverse complement of the forward one -> both strands
    # say the same thing and the result is exactly p - 1/4, transposed to (4, W)
    rng = np.random.default_rng(0)
    p = _probs(rng, 12)
    out = attr_star.contrib_from_probs(p, p[::-1, ::-1])
    assert out.shape == (4, 12) and out.dtype == np.float32
    assert np.allclose(out, (p - 0.25).T)


def test_contrib_averages_the_two_strands_after_flipping():
    rng = np.random.default_rng(1)
    pf, pr = _probs(rng, 10), _probs(rng, 10)
    out = attr_star.contrib_from_probs(pf, pr)
    expect = ((pf + pr[::-1, ::-1]) / 2 - 0.25).T
    assert np.allclose(out, expect)
    # reverse position j is forward position W-1-j, and A<->T, C<->G
    assert np.isclose(out[0, 0], (pf[0, 0] + pr[9, 3]) / 2 - 0.25)


def test_uniform_probabilities_contribute_nothing():
    p = np.full((8, 4), 0.25)
    assert np.allclose(attr_star.contrib_from_probs(p, p), 0)


VOCAB = ["-", "A", "C", "G", "T", "?"]
NUC = [VOCAB.index(c) for c in "ACGT"]


def test_human_onehot_maps_tokens_to_acgt_channels():
    ohe, ok = attr_star.human_onehot([1, 2, 3, 4, 4], NUC)
    assert ok and ohe.dtype == np.int8
    assert np.array_equal(ohe, np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0],
                                         [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]]))


@pytest.mark.parametrize("bad", [0, 5])
def test_human_onehot_flags_gap_or_unknown(bad):
    _, ok = attr_star.human_onehot([1, 2, bad, 4], NUC)
    assert not ok


def test_in_bounds_drops_windows_off_the_chromosome():
    sizes = {"chr1": 10000}
    rows = [("chr1", 4000, 6000, 1000),    # centre 5000: in
            ("chr1", 0, 1000, 500),        # centre 500: 500 - 768 < 0
            ("chr1", 9000, 10000, 500),    # centre 9500: 9500 + 768 > 10000
            ("chrUn", 4000, 6000, 1000)]   # not in chrom sizes
    assert attr_star.in_bounds(rows, sizes, 1536) == [True, False, False, False]


def test_read_peaks_accepts_four_and_eleven_columns(tmp_path):
    four = tmp_path / "four.bed"
    four.write_text("chr1\t100\t400\t50\n")
    eleven = tmp_path / "eleven.bed"
    eleven.write_text("chr1\t100\t400\t.\t.\t.\t.\t.\t.\t50\tTrue\n")
    assert attr_star.read_peaks(four) == [("chr1", 100, 400, 50)]
    assert attr_star.read_peaks(eleven) == [("chr1", 100, 400, 50)]
    bad = tmp_path / "bad.bed"
    bad.write_text("chr1\t100\t400\n")
    with pytest.raises(ValueError):
        attr_star.read_peaks(bad)


ROWS = [("chr1", 120939636, 120940065, 303), ("chr2", 500, 900, 12)]


def test_regions_bed_matches_what_adopt_writes(tmp_path):
    src = tmp_path / "peaks.bed"
    src.write_text("".join(f"{c}\t{s}\t{e}\t{m}\n" for c, s, e, m in ROWS))
    dst = tmp_path / "regions.bed"
    normalize_peaks(src, dst)
    assert dst.read_text() == attr_star.narrowpeak_text(ROWS)


def test_chunk_fingerprint_refuses_other_parameters(tmp_path):
    a = argparse.Namespace(peaks="p", repo="r", msa_path="m", in_window=1536, stride=8,
                           source="none", head=None, chunk=100, shard_i=None, n_shards=None)
    attr_star.check_chunk_fingerprint(tmp_path, a)
    attr_star.check_chunk_fingerprint(tmp_path, a)      # same parameters: fine
    a.stride = 4
    with pytest.raises(SystemExit, match="stride"):
        attr_star.check_chunk_fingerprint(tmp_path, a)
