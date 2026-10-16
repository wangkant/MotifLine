"""The official ChromBPNet attribution step: what can be checked without a model."""
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import CBP_ENV, STEPS, has_cbp_env

STEP = STEPS / "attr_cbp_tf.py"
_spec = importlib.util.spec_from_file_location("attr_cbp_tf_step", STEP)
step = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step)

ROW10 = "chr1\t100\t200\tPeak_1\t197\t.\t1.6\t19.7\t17.9\t50"
REQUIRED = ["--model_h5", "m.h5", "--peaks", "p", "--fasta", "f", "--out_dir", "o"]


def _run(*args, python=sys.executable):
    return subprocess.run([python, str(STEP), *args], capture_output=True, text=True)


def test_help_runs_without_tensorflow():
    r = _run("--help")
    assert r.returncode == 0 and "--hypothetical" in r.stdout


def test_refuses_without_hypothetical_like_the_other_steps():
    r = _run(*REQUIRED)
    assert r.returncode != 0 and "hypothetical" in r.stderr


def test_head_and_shards_cannot_be_combined():
    r = _run(*REQUIRED, "--hypothetical", "--head", "5", "--n_shards", "2", "--shard_i", "0")
    assert r.returncode != 0 and "mutually exclusive" in r.stderr


def test_peaks_are_read_as_chrombpnet_s_ten_columns(tmp_path):
    p = tmp_path / "p.bed"
    p.write_text((ROW10 + "\tTrue\n") * 2)
    assert step.peak_lines(str(p)) == [ROW10] * 2
    p.write_text(ROW10 + "\n")
    assert step.peak_lines(str(p)) == [ROW10]
    p.write_text(ROW10 + "\tFalse\n")
    with pytest.raises(ValueError, match="is_peak=True"):
        step.peak_lines(str(p))
    p.write_text(ROW10 + "\n" + ROW10 + "\tTrue\n")
    with pytest.raises(ValueError, match="10 or 11 columns"):
        step.peak_lines(str(p))


def test_peaks_sharing_a_summit_are_counted():
    shifted = ROW10.replace("\t100\t", "\t101\t")
    assert step.repeated_summits([ROW10, shifted]) == 0
    assert step.repeated_summits([ROW10, shifted, ROW10]) == 1


def _permute(seq, rng):
    return seq[rng.permutation(len(seq))]


SEQ = np.eye(4, dtype=np.int8)[np.arange(40) % 4]


def test_shuffle_k_is_seeded_with_random_state_plus_k_whatever_came_before():
    """Seeded per shuffle, not per call: a sequence gets the same references wherever
    it falls in a shard, so sharding cannot change a run."""
    f = step.seeded_shuffles(5, 42, shuffle=_permute)
    first = f([SEQ])[0]
    f([SEQ[::-1].copy()])
    assert first.shape == (5, 40, 4)
    assert np.array_equal(f([SEQ])[0], first)
    assert np.array_equal(first[2], _permute(SEQ, np.random.RandomState(44)))
    assert not np.array_equal(step.seeded_shuffles(5, 43, shuffle=_permute)([SEQ])[0], first)
    assert f.calls == 3


def test_a_second_model_input_is_repeated_as_in_the_official_function():
    refs = step.seeded_shuffles(4, 0, shuffle=_permute)([SEQ, np.ones(3)])
    assert len(refs) == 2 and refs[1].shape == (4, 3)


def test_the_input_length_is_read_from_the_saved_keras_config(tmp_path):
    h5py = pytest.importorskip("h5py")
    cfg = {"class_name": "Functional", "config": {"layers": [
        {"class_name": "InputLayer",
         "config": {"batch_input_shape": [None, 2114, 4], "name": "sequence"}},
        {"class_name": "Conv1D", "config": {}}]}}
    p = tmp_path / "m.h5"
    with h5py.File(p, "w") as f:
        f.attrs["model_config"] = json.dumps(cfg)
    assert step.model_input_length(str(p)) == 2114
    with h5py.File(p, "w"):
        pass
    assert step.model_input_length(str(p)) is None


@has_cbp_env
def test_the_step_starts_in_the_chrombpnet_env():
    r = _run("--help", python=f"{CBP_ENV}/bin/python")
    assert r.returncode == 0, r.stderr


@has_cbp_env
def test_the_official_code_takes_the_seeded_references():
    """interpret() must look shuffle_several_times up on shap_utils when it runs, and
    deeplift's dinuc_shuffle must take the rng: otherwise the seeding would not apply."""
    probe = f"""
import inspect, sys
import numpy as np
sys.path.insert(0, {str(STEPS)!r})
import attr_cbp_tf as step
import chrombpnet.evaluation.interpret.interpret as interpret
from deeplift.dinuc_shuffle import dinuc_shuffle
assert "shap_utils.shuffle_several_times" in inspect.getsource(interpret.interpret)
seq = np.eye(4, dtype=np.int8)[np.random.RandomState(0).randint(0, 4, 300)]
a = step.seeded_shuffles(3, 7)([seq])[0]
b = step.seeded_shuffles(3, 7)([seq])[0]
assert a.shape == (3, 300, 4) and np.array_equal(a, b) and not np.array_equal(a[0], a[1])
assert np.array_equal(a[1], dinuc_shuffle(seq, rng=np.random.RandomState(8)))
print("OK")
"""
    r = subprocess.run([f"{CBP_ENV}/bin/python", "-c", probe], capture_output=True, text=True,
                       env={"TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": "",
                            "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]
