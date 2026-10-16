"""run_info.json: what a stage directory says about how it was produced."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from motifline.provenance import code_entry, record, tool_versions
from motifline.steps.digests import git_commit, tree_sha256

posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell, paths and env layout")


def test_record_writes_run_info_naming_the_stage_and_the_command(tmp_path):
    record(tmp_path, stage="discover", argv=["/e/bin/modisco", "motifs", "-w", "400"])
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["stage"] == "discover"
    assert info["argv"] == ["/e/bin/modisco", "motifs", "-w", "400"]


def test_record_stores_the_parameter_signature_the_rule_reruns_on(tmp_path):
    record(tmp_path, stage="call_hits", argv=["finemo"], param_sig='{"threshold": 0.3}')
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["param_sig"] == '{"threshold": 0.3}'


def test_record_stamps_the_motifline_version(tmp_path):
    from motifline import __version__
    record(tmp_path, stage="discover", argv=["x"])
    assert json.loads((tmp_path / "run_info.json").read_text())["motifline"] == __version__


def test_record_timestamps_are_utc_iso_and_ordered(tmp_path):
    record(tmp_path, stage="discover", argv=["x"])
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["started"].endswith("Z") and info["finished"].endswith("Z")
    assert info["started"] <= info["finished"]


def test_record_replaces_a_previous_run_info(tmp_path):
    record(tmp_path, stage="discover", argv=["old"])
    record(tmp_path, stage="discover", argv=["new"])
    assert json.loads((tmp_path / "run_info.json").read_text())["argv"] == ["new"]


def test_record_creates_the_stage_directory_if_it_is_missing(tmp_path):
    out = tmp_path / "hits"
    record(out, stage="call_hits", argv=["finemo"])
    assert (out / "run_info.json").exists()


@posix
def test_argv_is_stringified_so_paths_survive_json(tmp_path):
    from pathlib import Path
    record(tmp_path, stage="discover", argv=[Path("/e/bin/modisco"), 400])
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["argv"] == ["/e/bin/modisco", "400"]


@posix
def test_tool_versions_reports_a_present_package_and_a_missing_one(tmp_path):
    got = tool_versions(sys.prefix, packages=("pytest", "no-such-package-xyz"))
    assert isinstance(got["pytest"], str) and got["pytest"]
    assert got["no-such-package-xyz"] is None


def test_tool_versions_on_a_broken_env_reports_why_instead_of_raising(tmp_path):
    got = tool_versions(str(tmp_path / "not-an-env"), packages=("pytest",))
    assert "unavailable" in got


@posix
def test_record_embeds_the_versions_of_the_env_that_ran_the_stage(tmp_path):
    record(tmp_path, stage="discover", argv=["x"], env_prefix=sys.prefix,
           packages=("pytest",))
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["env"] == sys.prefix
    assert isinstance(info["tools"]["pytest"], str)


has_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t",
                    *args], check=True, capture_output=True)


@has_git
def test_record_names_the_commit_of_the_model_code(tmp_path):
    repo, out, plain = tmp_path / "repo", tmp_path / "out", tmp_path / "plain"
    repo.mkdir()
    plain.mkdir()
    _git(repo, "init", "-q")
    (repo / "m.py").write_text("x = 1\n")
    _git(repo, "add", "m.py")
    _git(repo, "commit", "-qm", "m")
    record(out, stage="attribute", argv=["x"], code={"src": repo, "none": plain})
    code = json.loads((out / "run_info.json").read_text())["code"]
    assert code["src"]["commit"] and len(code["src"]["commit"]) >= 40
    assert code["none"]["commit"] is None
    (repo / "m.py").write_text("x = 2\n")
    record(out, stage="attribute", argv=["x"], code={"src": repo})
    dirty = json.loads((out / "run_info.json").read_text())["code"]["src"]["commit"]
    assert dirty.startswith(code["src"]["commit"] + "+")


@has_git
def test_an_untracked_directory_inside_another_repository_has_no_commit(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-qm", "a")
    pkg = tmp_path / "venv" / "motifline"
    pkg.mkdir(parents=True)
    (pkg / "b.py").write_text("y = 2\n")
    assert git_commit(tmp_path) is not None
    assert git_commit(pkg) is None
    entry = code_entry(pkg)
    assert entry["commit"] is None and entry["sha256"] == tree_sha256(pkg)


def test_the_code_digest_follows_the_code_and_ignores_bytecode(tmp_path):
    (tmp_path / "steps").mkdir()
    (tmp_path / "steps" / "a.py").write_text("x = 1\n")
    (tmp_path / "Snakefile").write_text("rule all:\n")
    before = tree_sha256(tmp_path)
    (tmp_path / "steps" / "__pycache__").mkdir()
    (tmp_path / "steps" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    (tmp_path / "notes.txt").write_text("n")
    assert tree_sha256(tmp_path) == before
    (tmp_path / "steps" / "a.py").write_text("x = 2\n")
    assert tree_sha256(tmp_path) != before


@has_git
def test_an_untracked_py_file_in_a_checkout_marks_it_changed(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-qm", "a")
    clean = git_commit(tmp_path)
    (tmp_path / "notes.txt").write_text("not code")
    assert git_commit(tmp_path) == clean
    (tmp_path / "patch.py").write_text("y = 2\n")
    first = git_commit(tmp_path)
    assert first.startswith(clean + "+")
    (tmp_path / "patch.py").write_text("y = 3\n")
    assert git_commit(tmp_path) not in (clean, first)
