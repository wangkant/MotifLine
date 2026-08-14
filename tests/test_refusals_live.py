"""Each refusal rule fired once against a real (copied) stage layout.

Uses tmp copies of results/k562_cbp_slice500 metadata - never mutates the run
tree itself.
"""
import json, shutil, sys
from pathlib import Path
import pytest
from motiflex.guards import GuardError
from motifline import guards
from motifline.contracts import run_stage

RUN = Path(__file__).resolve().parents[1] / "results/k562_cbp_slice500"
pytestmark = pytest.mark.skipif(not RUN.exists(), reason="slice not yet executed")


def _copy_meta(src, dst):
    dst.mkdir(parents=True)
    for f in ("provenance.json", "params.json", "run_status.json", "guard_outcomes.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)


def _flip_hyp(attr_dir):
    p = json.loads((attr_dir / "params.json").read_text())
    p[-1]["hypothetical"] = False
    (attr_dir / "params.json").write_text(json.dumps(p))


def test_refuses_actual_score_attr_for_hits(tmp_path):
    fake_attr = tmp_path / "attr"; _copy_meta(RUN / "attr", fake_attr)
    _flip_hyp(fake_attr)
    assert guards.attr_is_hypothetical(fake_attr).passed is False


def test_refuses_hyp_act_mismatch(tmp_path):
    a = tmp_path / "a"; _copy_meta(RUN / "attr", a)
    b = tmp_path / "b"; _copy_meta(RUN / "attr", b); _flip_hyp(b)
    assert guards.hyp_flags_match(a, b).passed is False


def test_refuses_finemo_version_drift_against_mmv_verify(tmp_path):
    reg = tmp_path / "reg"; _copy_meta(RUN / "finemo_regions", reg)
    sp041 = "<KANT_ROOT>/envs/mmv_verify/lib/python3.12/site-packages"
    assert guards.finemo_version_matches(reg, sp041).passed is False


def test_refused_stage_writes_no_result(tmp_path):
    fake_attr = tmp_path / "attr"; _copy_meta(RUN / "attr", fake_attr); _flip_hyp(fake_attr)
    out = tmp_path / "hits"; res = out / "hits.tsv"
    with pytest.raises(GuardError):
        run_stage(out, "call_hits", [sys.executable, "-c", f"open({str(res)!r},'w')"],
                  params={}, pre_guards=[lambda: (guards.attr_is_hypothetical(fake_attr), str(fake_attr))])
    assert not res.exists()
    assert json.loads((out / "run_status.json").read_text())["status"] == "REFUSED"
