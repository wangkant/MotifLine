"""Sharding is only safe if concat(shards) is bit-identical to the unsharded run."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from motifline.merge_shards import merge_shards

STEPS = Path(__file__).resolve().parents[1] / "src/motifline/steps"
AG_ARGS = ["--ag_src", "a", "--weights", "w", "--track", "60", "--biosample", "K562"]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_step", STEPS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("step", ["attr_cbp", "attr_ag"])
def test_bounds_partition_the_rows_contiguously(step):
    b = _load(step).shard_bounds(33917, 4)
    assert list(b) == [0, 8479, 16958, 25437, 33917]
    assert b[0] == 0 and b[-1] == 33917
    assert all(b[i] <= b[i + 1] for i in range(len(b) - 1))


@pytest.mark.parametrize("step", ["attr_cbp", "attr_ag"])
def test_bounds_are_the_archive_formula(step):
    for n_rows, k in [(7, 2), (33917, 6), (200, 3), (5, 5)]:
        assert list(_load(step).shard_bounds(n_rows, k)) == \
            list(np.linspace(0, n_rows, k + 1).astype(int))


def _write_shard(d, i, attr, ohe, rows):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"attr_{i}.npy", attr)
    np.save(d / f"ohe_{i}.npy", ohe)
    (d / f"regions_{i}.bed").write_text("".join(r + "\n" for r in rows))


def test_two_shards_merge_to_the_unsharded_arrays(tmp_path):
    rng = np.random.default_rng(0)
    attr = rng.normal(size=(3, 4, 6)).astype(np.float32)
    ohe = rng.integers(0, 2, size=(3, 4, 6)).astype(np.int8)
    rows = [f"chr1\t{i}\t{i + 6}\tPeak_{i}" for i in range(3)]
    lo, hi = 0, int(np.linspace(0, 3, 3).astype(int)[1])   # bounds for n_shards=2 -> [0,1,3]
    _write_shard(tmp_path / "s0", 0, attr[lo:hi], ohe[lo:hi], rows[lo:hi])
    _write_shard(tmp_path / "s1", 1, attr[hi:], ohe[hi:], rows[hi:])

    rep = merge_shards([tmp_path / "s0", tmp_path / "s1"], tmp_path / "merged")
    assert rep["rows"] == 3 and rep["n_shards"] == 2
    np.testing.assert_array_equal(np.load(tmp_path / "merged/attr.npy"), attr)
    np.testing.assert_array_equal(np.load(tmp_path / "merged/ohe.npy"), ohe)
    assert (tmp_path / "merged/regions.bed").read_text() == "".join(r + "\n" for r in rows)


def test_merge_refuses_shards_given_out_of_order(tmp_path):
    a = np.zeros((2, 4, 6), np.float32)
    _write_shard(tmp_path / "s0", 0, a, a.astype(np.int8), ["r1", "r2"])
    _write_shard(tmp_path / "s1", 1, a, a.astype(np.int8), ["r3", "r4"])
    with pytest.raises(FileNotFoundError, match="shard order is row order"):
        merge_shards([tmp_path / "s1", tmp_path / "s0"], tmp_path / "merged")


def test_merge_refuses_row_count_disagreement(tmp_path):
    a = np.zeros((2, 4, 6), np.float32)
    _write_shard(tmp_path / "s0", 0, a, a.astype(np.int8), ["only-one-row"])
    with pytest.raises(ValueError, match="rows disagree"):
        merge_shards([tmp_path / "s0"], tmp_path / "merged")


@pytest.mark.parametrize("step", ["attr_cbp", "attr_ag"])
def test_head_and_shard_are_mutually_exclusive(step):
    argv = [sys.executable, str(STEPS / f"{step}.py"), "--peaks", "p", "--out_dir", "o",
            "--fasta", "f", "--chrom_sizes", "c", "--chrombpnet_src", "s",
            "--hypothetical", "--head", "10", "--shard_i", "0", "--n_shards", "4"]
    argv += ["--wo_bias", "m"] if step == "attr_cbp" else AG_ARGS
    r = subprocess.run(argv, capture_output=True, text=True)
    assert r.returncode != 0 and "mutually exclusive" in r.stderr


@pytest.mark.parametrize("step", ["attr_cbp", "attr_ag"])
def test_shard_args_must_come_in_pairs(step):
    argv = [sys.executable, str(STEPS / f"{step}.py"), "--peaks", "p", "--out_dir", "o",
            "--fasta", "f", "--chrom_sizes", "c", "--chrombpnet_src", "s",
            "--hypothetical", "--shard_i", "0"]
    argv += ["--wo_bias", "m"] if step == "attr_cbp" else AG_ARGS
    r = subprocess.run(argv, capture_output=True, text=True)
    assert r.returncode != 0 and "must be given together" in r.stderr


def test_a_short_merge_is_refused_before_anything_is_written(tmp_path):
    a = np.zeros((2, 4, 6), np.float32)
    _write_shard(tmp_path / "s0", 0, a, a.astype(np.int8), ["r1", "r2"])
    with pytest.raises(ValueError, match="expected 3"):
        merge_shards([tmp_path / "s0"], tmp_path / "merged", expected_rows=3)
    assert not (tmp_path / "merged").exists()
