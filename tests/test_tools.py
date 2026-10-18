"""Reading the official defaults off the installed tools, and comparing config to them."""
from __future__ import annotations


from conftest import COMPUTE, has_compute
from motifline.tools import deviations, live_defaults, parse_help_defaults

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


@has_compute
def test_the_installed_tools_state_a_default_for_every_parameter_we_compare():
    from motifline.tools import PARAMETERS
    for p in PARAMETERS:
        if p.official_default_exists:
            assert p.flag.lstrip("-") in live_defaults(p.tool, COMPUTE), p.key


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


# --- rendering ---------------------------------------------------

def test_no_deviations_renders_as_a_positive_statement():
    from motifline.tools import render_deviations
    out = render_deviations([])
    assert len(out) == 1 and "official default" in out[0]


def test_a_config_deviation_renders_key_value_and_official_value():
    line = deviations(cfg(modisco_window=500), LIVE)[0].line()
    assert "modisco.window" in line and "500" in line and "400" in line


def _fake_tool(tmp_path, body):
    import os
    exe = tmp_path / "bin" / "finemo"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("#!/bin/sh\n" + body + "\n")
    os.chmod(exe, 0o755)
    return str(tmp_path)


def test_a_failing_help_raises_instead_of_reading_as_no_defaults(tmp_path):
    import subprocess

    import pytest
    env = _fake_tool(tmp_path, "echo broken >&2; exit 3")
    with pytest.raises(subprocess.CalledProcessError):
        live_defaults("finemo", env)


def test_a_help_with_no_defaults_raises(tmp_path):
    import subprocess

    import pytest
    env = _fake_tool(tmp_path, "echo 'usage: finemo call-hits'")
    with pytest.raises(subprocess.SubprocessError):
        live_defaults("finemo", env)
