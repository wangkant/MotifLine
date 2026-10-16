"""Sharding is only safe if concat(shards) is bit-identical to the unsharded run."""
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
from conftest import STEPS

from motifline.merge_shards import merge_shards

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


def test_side_files_are_merged_across_shards(tmp_path):
    import json
    a = np.zeros((1, 4, 6), np.float32)
    for i in range(2):
        d = tmp_path / f"s{i}"
        _write_shard(d, i, a, a.astype(np.int8), [f"r{i}"])
        (d / f"n_filled_{i}.tsv").write_text("chr\tstart\tsummit\tn_filled\n"
                                             + (f"chr1\t{i}\t5\t2\n" if i else ""))
        (d / f"convergence_{i}.json").write_text(json.dumps(
            {"threshold": 1e-3, "pairs": 20, "pairs_over": i, "max_delta": 0.1 * i}))
    merge_shards([tmp_path / "s0", tmp_path / "s1"], tmp_path / "m")
    assert (tmp_path / "m/n_filled.tsv").read_text() == \
        "chr\tstart\tsummit\tn_filled\nchr1\t1\t5\t2\n"
    conv = json.loads((tmp_path / "m/convergence.json").read_text())
    assert conv == {"threshold": 1e-3, "pairs": 40, "pairs_over": 1, "max_delta": 0.1}


def test_a_side_file_missing_from_one_shard_is_refused(tmp_path):
    a = np.zeros((1, 4, 6), np.float32)
    for i in range(2):
        _write_shard(tmp_path / f"s{i}", i, a, a.astype(np.int8), [f"r{i}"])
    (tmp_path / "s0/n_filled_0.tsv").write_text("chr\tstart\tsummit\tn_filled\n")
    with pytest.raises(FileNotFoundError, match="n_filled"):
        merge_shards([tmp_path / "s0", tmp_path / "s1"], tmp_path / "m")
    assert not (tmp_path / "m").exists()
