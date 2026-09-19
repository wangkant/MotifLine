"""Reading the official defaults off the installed tools, and comparing config to them."""
from __future__ import annotations

import os
import shutil

import pytest

from motifline.tools import (PINNED, deviations, parse_help_defaults, pinned_defaults,
                             tool_version)

COMPUTE = os.environ.get("MOTIFLINE_COMPUTE_ENV",
                         "/path/to/envs/compute")
has_compute = pytest.mark.skipif(
    not shutil.which("modisco", path=f"{COMPUTE}/bin"),
    reason="no compute env to read defaults from")

HELP = """usage: finemo call-hits [-h] [-M {pp,hp,ph,hh}] -r REGIONS

options:
  -h, --help            show this help message and exit
  -M {pp,hp,ph,hh}, --mode {pp,hp,ph,hh}
                        The type of attributions to use. (default: pp)
  -r REGIONS, --regions REGIONS
                        A .npz file of input sequences. (default: None)
  -t CWM_TRIM_THRESHOLD, --cwm-trim-threshold CWM_TRIM_THRESHOLD
                        The threshold to determine motif start and end
                        positions within the full CWMs. (default: 0.3)
  -c CONVERGENCE_TOL, --convergence-tol CONVERGENCE_TOL
                        The tolerance for determining convergence. The
                        optimizer exits when the duality gap is less than the
                        tolerance. (default: 0.0005)
  -S MAX_STEPS, --max-steps MAX_STEPS
                        The maximum number of optimization steps. (default:
                        10000)
  -f, --no-post-filter  Do not perform post-hit-calling filtering. (default:
                        False)
"""


# --- parsing ---------------------------------------------------------------

def test_a_default_on_the_same_line_is_read():
    assert parse_help_defaults(HELP)["mode"] == "pp"


def test_a_default_wrapped_onto_the_next_line_is_read():
    """argparse wraps help text, so `(default:` and its value land on separate lines."""
    assert parse_help_defaults(HELP)["max-steps"] == 10000


def test_numbers_come_back_as_numbers_not_strings():
    d = parse_help_defaults(HELP)
    assert d["cwm-trim-threshold"] == 0.3
    assert d["convergence-tol"] == 0.0005


def test_none_and_booleans_are_parsed():
    d = parse_help_defaults(HELP)
    assert d["regions"] is None
    assert d["no-post-filter"] is False


def test_the_help_flag_itself_is_not_a_parameter():
    assert "help" not in parse_help_defaults(HELP)


def test_an_option_with_no_stated_default_is_absent_rather_than_none():
    text = "options:\n  -n MAX, --max_seqlets MAX\n                        Required.\n"
    assert parse_help_defaults(text) == {}


# --- against the tools actually installed -----------------------------------

@has_compute
def test_the_installed_modisco_reports_the_defaults_we_pinned():
    from motifline.tools import live_defaults
    got = live_defaults("modisco-lite", COMPUTE)
    for flag, value in pinned_defaults("modisco-lite").items():
        assert got[flag] == value, f"{flag}: installed {got[flag]!r}, pinned {value!r}"


@has_compute
def test_the_installed_finemo_reports_the_defaults_we_pinned():
    from motifline.tools import live_defaults
    got = live_defaults("finemo", COMPUTE)
    for flag, value in pinned_defaults("finemo").items():
        assert got[flag] == value, f"{flag}: installed {got[flag]!r}, pinned {value!r}"


@has_compute
def test_tool_version_reads_the_env_not_the_orchestrator():
    assert tool_version("finemo", COMPUTE) == "0.30"


# --- deviations -------------------------------------------------------------

def cfg(**over):
    base = {"modisco": {"max_seqlets": 1000000, "window": 400},
            "finemo": {"mode": "pp", "threshold": 0.3, "alpha": 0.7, "batch": 2000}}
    for k, v in over.items():
        section, key = k.split("_", 1)
        base[section][key] = v
    return base


LIVE = {"modisco-lite": {"window": 400, "n_leiden": 2},
        "finemo": {"mode": "pp", "cwm-trim-threshold": 0.3,
                   "global-lambda": 0.7, "batch-size": 2000}}


def test_a_config_on_every_official_default_deviates_in_nothing():
    assert deviations(cfg(), LIVE) == []


def test_a_changed_value_is_reported_with_both_numbers():
    rows = deviations(cfg(modisco_window=500), LIVE)
    assert len(rows) == 1
    r = rows[0]
    assert r.key == "modisco.window" and r.value == 500 and r.official == 400


def test_the_deviation_names_the_flag_the_value_actually_reaches():
    """config.finemo.alpha is passed as --global-lambda; --alpha is a different,
    deprecated flag, and a reader of config alone would guess wrong."""
    r = deviations(cfg(finemo_alpha=0.5), LIVE)[0]
    assert r.flag == "--global-lambda"


def test_a_parameter_with_no_official_default_is_not_a_deviation():
    """modisco -n is required by the CLI; there is no default to deviate from."""
    assert deviations(cfg(modisco_max_seqlets=123), LIVE) == []


def test_a_tool_whose_own_default_moved_since_we_pinned_it_is_reported():
    moved = {**LIVE, "finemo": {**LIVE["finemo"], "global-lambda": 0.9}}
    rows = deviations(cfg(), moved, pinned=PINNED)
    assert any(r.kind == "tool-default-moved" and r.flag == "--global-lambda"
               for r in rows)


def test_config_following_a_moved_tool_default_is_still_reported():
    """The trap: the tool moved its default AND config moved with it, so a live-only
    comparison sees nothing wrong while the configuration is no longer the published one."""
    moved = {**LIVE, "finemo": {**LIVE["finemo"], "global-lambda": 0.9}}
    rows = deviations(cfg(finemo_alpha=0.9), moved, pinned=PINNED)
    assert any(r.kind == "tool-default-moved" for r in rows)


def test_pinned_defaults_cover_every_parameter_the_pipeline_sets():
    from motifline.tools import PARAMETERS
    for p in PARAMETERS:
        if p.official_default_exists:
            assert p.flag.lstrip("-") in pinned_defaults(p.tool), p.key


# --- rendering and probing ---------------------------------------------------

def test_no_deviations_renders_as_a_positive_statement():
    from motifline.tools import render_deviations
    out = render_deviations([])
    assert len(out) == 1 and "official default" in out[0]


def test_a_config_deviation_renders_key_value_and_official_value():
    line = deviations(cfg(modisco_window=500), LIVE)[0].line()
    assert "modisco.window" in line and "500" in line and "400" in line


def test_a_moved_tool_default_renders_the_version_it_was_pinned_at():
    moved = {**LIVE, "finemo": {**LIVE["finemo"], "global-lambda": 0.9}}
    row = [r for r in deviations(cfg(), moved, pinned=PINNED)
           if r.kind == "tool-default-moved"][0]
    assert "0.30" in row.line() and "0.9" in row.line()


@has_compute
def test_a_checkout_that_provides_the_module_probes_importable():
    from motifline.tools import can_import
    ok, detail = can_import("chrombpnet.bpnet",
                            "/path/to/chrombpnet-pytorch",
                            COMPUTE)
    assert ok, detail


@has_compute
def test_a_checkout_that_does_not_provide_it_probes_as_a_reason_not_a_crash(tmp_path):
    from motifline.tools import can_import
    ok, detail = can_import("chrombpnet.bpnet", str(tmp_path), COMPUTE)
    assert not ok and "ModuleNotFoundError" in detail
