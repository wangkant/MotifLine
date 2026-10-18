"""The command line: what a new user types instead of reading the Snakefile."""
from __future__ import annotations

import sys

import pytest
import yaml

from conftest import COMPUTE, has_compute
from motifline import __version__
from motifline.cli import ENVS, SNAKEFILE, main, snakemake_argv
from test_config import cfg_cbp


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    # the cwd is the project: init and run must never write into the checkout
    monkeypatch.chdir(tmp_path)


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


def test_init_never_replaces_an_existing_config_but_still_adds_the_envs(tmp_path):
    out = write_cfg(tmp_path)
    before = out.read_text()
    for argv in (["init", "-c", str(out)], ["init", "-c", str(out), "--force"]):
        assert main(argv) == 0
        assert out.read_text() == before
    assert (tmp_path / "envs" / "compute.yaml").is_file()


def test_init_copies_the_env_recipes_into_the_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    for name in ("compute.yaml", "chrombpnet.yaml", "meme.yaml", "orchestration.yaml",
                 "locks/compute.pip.txt", "locks/chrombpnet.pip.txt"):
        assert (tmp_path / "envs" / name).read_bytes() == (ENVS / name).read_bytes()


def test_init_keeps_an_edited_env_recipe_unless_forced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    edited = tmp_path / "envs" / "compute.yaml"
    edited.parent.mkdir()
    edited.write_text("mine")
    assert main(["init"]) == 0
    assert edited.read_text() == "mine"
    assert main(["init", "--force"]) == 0
    assert edited.read_bytes() == (ENVS / "compute.yaml").read_bytes()


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
    argv = snakemake_argv([], snakefile=SNAKEFILE, directory=tmp_path)
    assert argv[:3] == [sys.executable, "-m", "snakemake"]


def test_gpu_resource_is_declared_only_when_asked_for(tmp_path):
    sf = SNAKEFILE
    assert "--resources" not in snakemake_argv([], snakefile=sf, directory=tmp_path)
    argv = snakemake_argv([], snakefile=sf, directory=tmp_path, gpus=3)
    assert argv[argv.index("--resources") + 1] == "gpu=3"


def test_targets_are_passed_after_a_separator_so_they_are_never_read_as_flags(tmp_path):
    argv = snakemake_argv(["results/r1/hits/hits.tsv"], snakefile=SNAKEFILE,
                          directory=tmp_path)
    assert argv[-2:] == ["--", "results/r1/hits/hits.tsv"]


def test_extra_snakemake_arguments_are_forwarded(tmp_path):
    argv = snakemake_argv([], snakefile=SNAKEFILE, directory=tmp_path, extra=["--unlock"])
    assert "--unlock" in argv


def test_dry_run_is_forwarded(tmp_path):
    assert "-n" in snakemake_argv([], snakefile=SNAKEFILE, directory=tmp_path,
                                dry_run=True)


def test_configfile_is_forwarded_when_given(tmp_path):
    argv = snakemake_argv([], snakefile=SNAKEFILE, directory=tmp_path, configfile="c.yaml")
    assert argv[argv.index("--configfile") + 1] == "c.yaml"


def test_the_package_ships_the_snakefile_and_the_env_recipes():
    assert "rule all:" in SNAKEFILE.read_text()
    assert (SNAKEFILE.parent / "steps" / "attr_cbp.py").is_file()
    assert (SNAKEFILE.parent / "steps" / "attr_cbp_tf.py").is_file()
    assert (ENVS / "compute.yaml").is_file() and (ENVS / "locks" / "compute.pip.txt").is_file()
    assert (ENVS / "chrombpnet.yaml").is_file()
    assert (ENVS / "locks" / "chrombpnet.pip.txt").is_file()


def test_run_without_a_config_here_points_at_init(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run"]) != 0
    assert "motifline init" in capsys.readouterr().err


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


def test_run_uses_the_packaged_snakefile_in_the_current_directory(tmp_path, monkeypatch):
    import motifline.cli as cli
    (tmp_path / "config").mkdir()
    write_cfg(tmp_path / "config", name="default.yaml")
    monkeypatch.chdir(tmp_path)
    seen = []

    class Done:
        returncode = 0

    monkeypatch.setattr(cli.subprocess, "run", lambda argv, **kw: seen.append(argv) or Done())
    assert main(["run", "--no-check", "-n"]) == 0
    argv = seen[-1]
    assert argv[argv.index("--snakefile") + 1] == str(SNAKEFILE)
    assert argv[argv.index("--directory") + 1] == str(tmp_path)
    assert argv[argv.index("--configfile") + 1] == "config/default.yaml"


def _other_project(tmp_path):
    other = tmp_path / "other"
    (other / ".snakemake").mkdir(parents=True)
    (other / "config").mkdir()
    cfg = write_cfg(other / "config", name="default.yaml")
    here = tmp_path / "here"
    here.mkdir()
    return cfg, here


def test_run_refuses_a_config_that_belongs_to_another_project(tmp_path, monkeypatch, capsys):
    import motifline.cli as cli
    cfg, here = _other_project(tmp_path)
    monkeypatch.chdir(here)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("snakemake ran"))
    assert main(["run", "-c", str(cfg), "--no-check"]) != 0
    assert "belongs to the project" in capsys.readouterr().err
    assert not (here / "results").exists()


def test_check_refuses_a_config_that_belongs_to_another_project(tmp_path, monkeypatch, capsys):
    cfg, here = _other_project(tmp_path)
    monkeypatch.chdir(here)
    assert main(["check", "-c", str(cfg), "--no-paths"]) != 0
    assert "belongs to the project" in capsys.readouterr().err
    monkeypatch.chdir(tmp_path / "other")
    assert main(["check", "-c", "config/default.yaml", "--no-paths"]) == 0


def test_a_double_dash_outside_run_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["check", "-c", str(write_cfg(tmp_path)), "--", "--unlock"])


def test_check_resolves_relative_paths_against_the_directory_it_runs_in(tmp_path, monkeypatch,
                                                                       capsys):
    path = write_cfg(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    (project / "p.bed").write_text("chr1\t100\t200\tpk\t0\t.\t1.0\t-1\t-1\t50\n")
    monkeypatch.chdir(tmp_path)
    main(["check", "-c", str(path)])
    assert "runs.r1.peaks" in capsys.readouterr().err
    monkeypatch.chdir(project)
    main(["check", "-c", str(path)])
    assert "runs.r1.peaks" not in capsys.readouterr().err


# --- merge-h5 ----------------------------------------------------------------

def test_merge_h5_without_a_usable_env_says_so(tmp_path, capsys):
    rc = main(["merge-h5", "--shard-prefix", "a", "--out-prefix", "o",
               "--compute-env", str(tmp_path / "no_env")])
    assert rc == 1 and "not an executable interpreter" in capsys.readouterr().err


def test_merge_h5_runs_the_step_in_the_given_env(tmp_path):
    pytest.importorskip("tables")
    from test_merge_chrombpnet_h5 import write_shard
    write_shard(tmp_path / "s0", rows=3, first=0)
    write_shard(tmp_path / "s1", rows=2, first=3)
    rc = main(["merge-h5", "--shard-prefix", str(tmp_path / "s0"),
               "--shard-prefix", str(tmp_path / "s1"), "--out-prefix", str(tmp_path / "all"),
               "--compute-env", sys.prefix])
    assert rc == 0
    assert len((tmp_path / "all.interpreted_regions.bed").read_text().splitlines()) == 5
