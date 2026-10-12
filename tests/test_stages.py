"""run_body: the one place a stage body is launched and labelled."""
from __future__ import annotations

import json
import subprocess

import pytest

from motifline.stages import run_body


def test_the_stage_directory_is_created_before_the_body_runs(tmp_path):
    out = tmp_path / "hits"
    run_body(out, ["true"], stage="call_hits")
    assert out.is_dir()


def test_a_finished_stage_is_labelled_with_run_info(tmp_path):
    run_body(tmp_path, ["true"], stage="discover")
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["stage"] == "discover" and info["argv"] == ["true"]


def test_a_failed_stage_leaves_no_run_info_behind(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run_body(tmp_path, ["false"], stage="discover")
    assert not (tmp_path / "run_info.json").exists()


def test_a_rerun_replaces_the_previous_label(tmp_path):
    run_body(tmp_path, ["true"], stage="discover")
    run_body(tmp_path, ["echo", "again"], stage="discover")
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["argv"] == ["echo", "again"]


def test_the_parameter_signature_is_carried_into_the_label(tmp_path):
    run_body(tmp_path, ["true"], stage="call_hits", param_sig='{"threshold": 0.3}')
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["param_sig"] == '{"threshold": 0.3}'


def test_extra_environment_reaches_the_child(tmp_path):
    target = tmp_path / "seen.txt"
    run_body(tmp_path, ["sh", "-c", f"echo $NUMBA_NUM_THREADS > {target}"],
             stage="discover", env={"NUMBA_NUM_THREADS": "8"})
    assert target.read_text().strip() == "8"


def test_the_label_records_when_the_body_started_not_only_when_it_ended(tmp_path):
    run_body(tmp_path, ["sh", "-c", "sleep 1.1"], stage="discover")
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["started"] < info["finished"]


def test_path_arguments_survive_into_the_label(tmp_path):
    run_body(tmp_path, ["true", tmp_path / "x.npy"], stage="discover")
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["argv"][1] == str(tmp_path / "x.npy")


# --- post-run checks ---------------------------------------------------------

def test_a_check_runs_after_the_body_and_lands_in_the_label(tmp_path):
    def saw_the_output(out_dir):
        return {"body_wrote": (out_dir / "made.txt").read_text().strip()}
    run_body(tmp_path, ["sh", "-c", f"echo hello > {tmp_path}/made.txt"],
             stage="attribute", checks=(saw_the_output,))
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["checks"]["body_wrote"] == "hello"


def test_a_check_that_raises_is_recorded_and_does_not_lose_the_stage(tmp_path):
    def broken(out_dir):
        raise RuntimeError("probe exploded")
    run_body(tmp_path, ["true"], stage="attribute", checks=(broken,))
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert "probe exploded" in str(info["checks"])


def test_a_stage_with_no_checks_records_none(tmp_path):
    run_body(tmp_path, ["true"], stage="discover")
    assert json.loads((tmp_path / "run_info.json").read_text())["checks"] is None


def test_a_checks_warning_is_printed_to_stderr(tmp_path, capfd):
    def warns(out_dir):
        return {"warning": "this array does not look hypothetical"}
    run_body(tmp_path, ["true"], stage="attribute", checks=(warns,))
    assert "does not look hypothetical" in capfd.readouterr().err


def test_a_failed_rerun_removes_the_old_label(tmp_path):
    run_body(tmp_path, ["true"], stage="discover")
    with pytest.raises(subprocess.CalledProcessError):
        run_body(tmp_path, ["false"], stage="discover")
    assert not (tmp_path / "run_info.json").exists()
