"""run_info.json: what a stage directory says about how it was produced."""
from __future__ import annotations

import json
import sys

from motifline.provenance import record, tool_versions


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


def test_argv_is_stringified_so_paths_survive_json(tmp_path):
    from pathlib import Path
    record(tmp_path, stage="discover", argv=[Path("/e/bin/modisco"), 400])
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["argv"] == ["/e/bin/modisco", "400"]


def test_tool_versions_reports_a_present_package_and_a_missing_one(tmp_path):
    got = tool_versions(sys.prefix, packages=("pytest", "no-such-package-xyz"))
    assert isinstance(got["pytest"], str) and got["pytest"]
    assert got["no-such-package-xyz"] is None


def test_tool_versions_on_a_broken_env_reports_why_instead_of_raising(tmp_path):
    got = tool_versions(str(tmp_path / "not-an-env"), packages=("pytest",))
    assert "unavailable" in got


def test_record_embeds_the_versions_of_the_env_that_ran_the_stage(tmp_path):
    record(tmp_path, stage="discover", argv=["x"], env_prefix=sys.prefix,
           packages=("pytest",))
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["env"] == sys.prefix
    assert isinstance(info["tools"]["pytest"], str)


def test_record_names_the_commit_of_the_model_code(tmp_path):
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    record(tmp_path, stage="attribute", argv=["x"], code={"src": repo, "none": tmp_path})
    code = json.loads((tmp_path / "run_info.json").read_text())["code"]
    assert code["src"]["commit"] and len(code["src"]["commit"]) >= 40
    assert code["none"]["commit"] is None
