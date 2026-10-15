"""The command line: what a new user types instead of reading the Snakefile."""
from __future__ import annotations

import sys

import pytest
import yaml

from conftest import COMPUTE, has_compute
from motifline import __version__
from motifline.cli import find_workflow, main, snakemake_argv
from test_config import cfg_cbp


def write_cfg(tmp_path, cfg=None, name="config.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg if cfg is not None else cfg_cbp()))
    return p


# --- version / init --------------------------------------------------------

def test_version_prints_the_package_version(capsys):
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_init_writes_a_config_that_parses_and_names_every_section(tmp_path):
    out = tmp_path / "config" / "default.yaml"
    assert main(["init", "-c", str(out)]) == 0
    parsed = yaml.safe_load(out.read_text())
    assert {"envs", "genome", "modisco", "finemo", "reference_lexicon"} <= set(parsed)


def test_init_refuses_to_clobber_an_existing_config(tmp_path, capsys):
    out = write_cfg(tmp_path)
    before = out.read_text()
    assert main(["init", "-c", str(out)]) != 0
    assert out.read_text() == before
    assert "--force" in capsys.readouterr().err


def test_init_force_overwrites(tmp_path):
    out = write_cfg(tmp_path)
    assert main(["init", "-c", str(out), "--force"]) == 0
    assert "motifline config" in out.read_text()


# --- check -----------------------------------------------------------------

def test_check_passes_a_valid_config_when_paths_are_not_inspected(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["check", "-c", str(cfg), "--no-paths"]) == 0
    assert "OK" in capsys.readouterr().out


def test_check_fails_and_lists_every_problem(tmp_path, capsys):
    broken = cfg_cbp()
    del broken["genome"]
    del broken["runs"]["r1"]["wo_bias"]
    cfg = write_cfg(tmp_path, broken)
    assert main(["check", "-c", str(cfg), "--no-paths"]) != 0
    err = capsys.readouterr().err
    assert "genome" in err and "wo_bias" in err


def test_check_without_no_paths_reports_the_missing_files(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["check", "-c", str(cfg)]) != 0
    assert "does not exist" in capsys.readouterr().err


def test_check_reports_a_config_file_that_is_not_there(tmp_path, capsys):
    assert main(["check", "-c", str(tmp_path / "nope.yaml")]) != 0
    assert "nope.yaml" in capsys.readouterr().err


def test_run_invokes_snakemake_in_the_interpreter_running_the_cli(tmp_path):
    argv = snakemake_argv([], snakefile=tmp_path / "Snakefile")
    assert argv[:3] == [sys.executable, "-m", "snakemake"]


def test_gpu_resource_is_declared_only_when_asked_for(tmp_path):
    sf = tmp_path / "Snakefile"
    assert "--resources" not in snakemake_argv([], snakefile=sf)
    argv = snakemake_argv([], snakefile=sf, gpus=3)
    assert argv[argv.index("--resources") + 1] == "gpu=3"


def test_targets_are_passed_after_a_separator_so_they_are_never_read_as_flags(tmp_path):
    argv = snakemake_argv(["results/r1/hits/hits.tsv"], snakefile=tmp_path / "Snakefile")
    assert argv[-2:] == ["--", "results/r1/hits/hits.tsv"]


def test_extra_snakemake_arguments_are_forwarded(tmp_path):
    argv = snakemake_argv([], snakefile=tmp_path / "Snakefile", extra=["--unlock"])
    assert "--unlock" in argv


def test_dry_run_is_forwarded(tmp_path):
    assert "-n" in snakemake_argv([], snakefile=tmp_path / "Snakefile", dry_run=True)


def test_configfile_is_forwarded_when_given(tmp_path):
    argv = snakemake_argv([], snakefile=tmp_path / "Snakefile", configfile="c.yaml")
    assert argv[argv.index("--configfile") + 1] == "c.yaml"


def test_find_workflow_walks_up_to_the_directory_holding_the_snakefile(tmp_path):
    (tmp_path / "Snakefile").write_text("")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert find_workflow(deep) == tmp_path


def test_find_workflow_returns_none_when_there_is_no_snakefile(tmp_path):
    assert find_workflow(tmp_path) is None


def test_run_outside_a_workflow_fails_with_a_message_naming_the_snakefile(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run"]) != 0
    assert "Snakefile" in capsys.readouterr().err


def test_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        main(["frobnicate"])


# --- the official-configuration report ---------------------------------------

def real_env_cfg(**over):
    c = cfg_cbp()
    c["envs"]["compute"] = COMPUTE
    for k, v in over.items():
        section, key = k.split("_", 1)
        c[section][key] = v
    return c


@has_compute
def test_check_states_that_the_parameters_are_the_tools_own_defaults(tmp_path, capsys):
    cfg = write_cfg(tmp_path, real_env_cfg(modisco_n_leiden=2))
    assert main(["check", "-c", str(cfg), "--no-paths"]) == 0
    assert "official default" in capsys.readouterr().out


@has_compute
def test_a_deviating_parameter_is_reported_but_does_not_fail_the_check(tmp_path, capsys):
    cfg = write_cfg(tmp_path, real_env_cfg(modisco_window=500))
    assert main(["check", "-c", str(cfg), "--no-paths"]) == 0
    out = capsys.readouterr().out
    assert "modisco.window" in out and "500" in out and "400" in out


def test_arguments_after_a_double_dash_reach_snakemake(tmp_path, monkeypatch):
    import motifline.cli as cli
    (tmp_path / "Snakefile").write_text("")
    cfg = write_cfg(tmp_path)
    monkeypatch.chdir(tmp_path)
    seen = []

    class Done:
        returncode = 0

    def fake_run(argv, **kw):
        seen.append(argv)
        return Done()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert main(["run", "-c", str(cfg), "--no-check", "-j", "2", "--", "--rerun-incomplete"]) == 0
    assert "--rerun-incomplete" in seen[-1] and "--" not in seen[-1]


def test_a_double_dash_outside_run_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["check", "-c", str(write_cfg(tmp_path)), "--", "--unlock"])


def test_check_resolves_paths_against_the_checkout_holding_the_config(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / "config").mkdir(parents=True)
    (checkout / "Snakefile").write_text("")
    (checkout / "p.bed").write_text("chr1\t100\t200\tpk\t0\t.\t1.0\t-1\t-1\t50\n")
    cfg = write_cfg(checkout / "config")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    main(["check", "-c", str(cfg)])
    from motifline.config import load, problems
    assert not [p for p in problems(load(cfg), base=checkout) if "p.bed" in p and "exist" in p]
