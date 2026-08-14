import json, sys
from pathlib import Path
import pytest
from motiflex.guards import GuardResult, GuardError
from motifline.contracts import append_params, params_of_success, run_stage

OK = lambda: (GuardResult("g_ok", True, "ok"), "subject")
BAD = lambda: (GuardResult("g_bad", False, "hypothetical flag is absent"), "subject")
PY = sys.executable

def _touch_cmd(path):  # body that writes a result file
    return [PY, "-c", f"open({str(path)!r}, 'w').write('x')"]

def test_provenance_and_params_written_before_crashing_body(tmp_path):
    out = tmp_path / "attr"
    with pytest.raises(Exception):
        run_stage(out, "attribute", [PY, "-c", "import sys; sys.exit(3)"],
                  params={"hypothetical": True})
    prov = json.loads((out / "provenance.json").read_text())
    assert len(prov) == 1
    assert json.loads((out / "params.json").read_text()) == [{"hypothetical": True}]
    rs = json.loads((out / "run_status.json").read_text())
    assert rs["status"] == "CRASHED"
    assert rs["artifacts_are_from"] == "NO_SUCCESSFUL_RUN_RECORDED"

def test_refusal_logs_guard_and_writes_no_result(tmp_path):
    out = tmp_path / "hits"
    res = out / "hits.tsv"
    with pytest.raises(GuardError):
        run_stage(out, "call_hits", _touch_cmd(res), params={}, pre_guards=[BAD])
    assert not res.exists()
    rs = json.loads((out / "run_status.json").read_text())
    assert rs["status"] == "REFUSED" and "hypothetical" in rs["detail"]
    outcomes = json.loads((out / "guard_outcomes.json").read_text())
    assert outcomes[-1]["guard_id"] == "g_bad" and outcomes[-1]["passed"] is False

def test_success_then_refusal_carries_artifacts_are_from(tmp_path):
    out = tmp_path / "hits"
    res = out / "hits.tsv"
    run_stage(out, "call_hits", _touch_cmd(res), params={"v": 1}, pre_guards=[OK])
    first = json.loads((out / "run_status.json").read_text())["artifacts_are_from"]
    with pytest.raises(GuardError):
        run_stage(out, "call_hits", _touch_cmd(res), params={"v": 2}, pre_guards=[BAD])
    rs = json.loads((out / "run_status.json").read_text())
    assert rs["status"] == "REFUSED" and rs["artifacts_are_from"] == first

def test_params_of_success_joins_on_record_count(tmp_path):
    out = tmp_path / "hits"
    res = out / "hits.tsv"
    with pytest.raises(GuardError):
        run_stage(out, "call_hits", _touch_cmd(res), params={"v": "refused"}, pre_guards=[BAD])
    run_stage(out, "call_hits", _touch_cmd(res), params={"v": "good"})
    with pytest.raises(GuardError):
        run_stage(out, "call_hits", _touch_cmd(res), params={"v": "refused2"}, pre_guards=[BAD])
    assert params_of_success(out)["v"] == "good"

def test_post_guard_failure_is_refused(tmp_path):
    out = tmp_path / "attr"
    res = out / "attr.npy"
    with pytest.raises(GuardError):
        run_stage(out, "attribute", _touch_cmd(res), params={},
                  post_guards=[lambda: (GuardResult("g_shape", False, "rows mismatch"), str(res))])
    assert json.loads((out / "run_status.json").read_text())["status"] == "REFUSED"
