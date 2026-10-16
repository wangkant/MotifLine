"""motifline init | check | run | smoke | merge-h5. `run` checks the config, then calls snakemake.

The directory motifline runs in is the project: config/default.yaml, results/ and
relative paths in the config are all taken from it. The Snakefile ships with the package.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from motifline import __version__
from motifline.config import ConfigError, check as check_config, load, problems, skeleton
from motifline.tools import TOOLS, deviations, live_defaults, render_deviations

DEFAULT_CONFIG = "config/default.yaml"
PACKAGE = Path(__file__).resolve().parent
SNAKEFILE = PACKAGE / "Snakefile"
ENVS = PACKAGE / "envs"


def snakemake_argv(targets, *, snakefile, directory, jobs=None, gpus=None, dry_run=False,
                   configfile=None, extra=()) -> list[str]:
    argv = [sys.executable, "-m", "snakemake", "--snakefile", str(snakefile),
            "--directory", str(directory)]
    if configfile:
        argv += ["--configfile", str(configfile)]
    if jobs is not None:
        argv += ["-j", str(jobs)]
    if gpus is not None:
        argv += ["--resources", f"gpu={gpus}"]
    if dry_run:
        argv += ["-n"]
    argv += list(extra)
    if targets:
        argv += ["--", *targets]
    return argv


def compute_env(cfg) -> str | None:
    return ((cfg.get("envs") or {}).get("compute")) or None


def deviation_lines(cfg) -> list[str]:
    env = compute_env(cfg)
    if not env or not Path(env, "bin").is_dir():
        return [f"could not read the tools' defaults: envs.compute ({env}) is not a "
                f"usable environment, so the parameters were not checked against them"]
    try:
        live = {name: live_defaults(name, env) for name in TOOLS}
    except (OSError, subprocess.SubprocessError) as e:
        return [f"could not read the tools' defaults from {env}: {type(e).__name__}: {e}"]
    return render_deviations(deviations(cfg, live))


def copy_envs(dest: Path, force: bool) -> list[Path]:
    """The env recipes and locks, so the project keeps the versions it was set up with."""
    written = []
    for src in sorted(ENVS.rglob("*")):
        if src.is_dir() or "__pycache__" in src.parts:
            continue
        out = dest / src.relative_to(ENVS)
        if out.exists() and not force:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        written.append(out)
    return written


def cmd_init(args) -> int:
    out = Path(args.config)
    if out.exists():
        print(f"{out} exists; left as it is")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(skeleton())
        print(f"wrote {out}")
    envs = Path("envs").resolve()
    print(f"wrote {len(copy_envs(envs, args.force))} env file(s) to {envs}")
    print(f"Build the envs, edit the paths, then run: motifline check -c {out}")
    return 0


def owning_project(configfile) -> Path | None:
    """The existing project a config sits in: <dir>/config/x.yaml next to <dir>/.snakemake."""
    d = Path(configfile).resolve().parent
    for p in (d, d.parent):
        if (p / ".snakemake").is_dir():
            return p
    return None


def other_project(configfile) -> str | None:
    owner = owning_project(configfile)
    if owner is None or owner == Path.cwd().resolve():
        return None
    return (f"{configfile} belongs to the project at {owner}, but motifline runs in "
            f"{Path.cwd()}, where results/ would start from scratch; run from {owner}")


def cmd_check(args) -> int:
    try:
        cfg = load(args.config)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1
    if msg := other_project(args.config):
        print(msg, file=sys.stderr)
        return 1
    found = problems(cfg, paths=not args.no_paths, base=Path.cwd())
    if found:
        scope = "structure" if args.no_paths else "structure and paths"
        print(f"{len(found)} problem(s) in {args.config} ({scope}):", file=sys.stderr)
        for p in found:
            print(f"  - {p}", file=sys.stderr)
        return 1
    runs = cfg.get("runs") or {}
    active = [k for k, v in runs.items() if not v.get("cancelled")]
    print(f"OK: {args.config} -- {len(active)} active run(s), "
          f"{len(runs) - len(active)} cancelled")
    for line in deviation_lines(cfg):
        print(line)
    return 0


def cmd_run(args) -> int:
    project = Path.cwd()
    configfile = args.config or DEFAULT_CONFIG
    if not Path(configfile).is_file():
        print(f"{configfile} does not exist; run motifline from the project directory, "
              f"or start one there with `motifline init`", file=sys.stderr)
        return 1
    if msg := other_project(configfile):
        print(msg, file=sys.stderr)
        return 1
    try:
        check_config(load(configfile), paths=not args.no_check, base=project)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        print("\nrefusing to build the DAG on a config that cannot run; "
              "pass --no-check to skip this", file=sys.stderr)
        return 1
    for line in deviation_lines(load(configfile)):
        print(line, file=sys.stderr)
    argv = snakemake_argv(args.targets, snakefile=SNAKEFILE, directory=project,
                          jobs=args.jobs, gpus=args.gpus, dry_run=args.dry_run,
                          configfile=configfile, extra=args.extra)
    print(" ".join(argv), file=sys.stderr)
    return subprocess.run(argv).returncode


def cmd_smoke(args) -> int:
    from motifline.smoke import run_smoke
    return run_smoke(compute_env=args.compute_env, workdir=args.workdir,
                     keep=args.keep)


def cmd_merge_h5(args) -> int:
    env = args.compute_env
    if not env:
        try:
            env = compute_env(load(args.config))
        except ConfigError as e:
            print(f"{e}\n(or pass --compute-env)", file=sys.stderr)
            return 1
    python = Path(env or "", "bin", "python")
    if not env or not os.access(python, os.X_OK):
        print(f"{python} is not an executable interpreter; the compute env holds PyTables",
              file=sys.stderr)
        return 1
    argv = [str(python), str(PACKAGE / "steps" / "merge_chrombpnet_h5.py"),
            "--out_prefix", args.out_prefix]
    for prefix in args.shard_prefix:
        argv += ["--shard_prefix", prefix]
    if args.check_bed:
        argv += ["--check_bed", args.check_bed]
    return subprocess.run(argv).returncode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motifline",
        description="attribution -> MoDISco -> FiNeMo, one line per run")
    p.add_argument("--version", "-V", action="store_true",
                   help="print the version and exit")
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="write a config skeleton and the env recipes")
    init.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    init.add_argument("--force", action="store_true",
                      help="replace existing env files (an existing config is never replaced)")
    init.set_defaults(fn=cmd_init)

    chk = sub.add_parser("check", help="report everything wrong with a config, at once")
    chk.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    chk.add_argument("--no-paths", action="store_true",
                     help="check structure only; do not touch the filesystem")
    chk.set_defaults(fn=cmd_check)

    run = sub.add_parser("run", help="check the config, then run snakemake; arguments after -- go to snakemake")
    run.add_argument("targets", nargs="*", help="snakemake targets (default: rule all)")
    run.add_argument("-c", "--config", default=None)
    run.add_argument("-j", "--jobs", type=int, default=None)
    run.add_argument("--gpus", type=int, default=None,
                     help="concurrent GPU jobs; export CUDA_VISIBLE_DEVICES yourself")
    run.add_argument("-n", "--dry-run", action="store_true")
    run.add_argument("--no-check", action="store_true",
                     help="skip the path checks before building the DAG")
    run.set_defaults(fn=cmd_run)

    smoke = sub.add_parser(
        "smoke", help="plant a motif in synthetic attributions and recover it")
    smoke.add_argument("--compute-env", default=None,
                       help="env prefix holding modisco and finemo "
                            "(default: envs.compute from the config)")
    smoke.add_argument("--workdir", default=None,
                       help="an empty or new directory for the smoke run (default: a temp dir)")
    smoke.add_argument("--keep", action="store_true", help="keep the working directory")
    smoke.set_defaults(fn=cmd_smoke)

    merge = sub.add_parser(
        "merge-h5", help="stack sharded chrombpnet interpret output into one prefix "
                         "for adopt_format: chrombpnet_h5")
    merge.add_argument("--shard-prefix", action="append", required=True,
                       help="<dir>/<prefix> of one shard's .counts_scores.h5 and "
                            ".interpreted_regions.bed; once per shard, in shard order")
    merge.add_argument("--out-prefix", required=True)
    merge.add_argument("--check-bed", default=None,
                       help="the peak bed the shards were cut from; the joined regions must "
                            "equal it row for row")
    merge.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    merge.add_argument("--compute-env", default=None,
                       help="env prefix holding PyTables (default: envs.compute from the config)")
    merge.set_defaults(fn=cmd_merge_h5)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    if {"--version", "-V"} & set(argv):
        print(f"motifline {__version__}")
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if extra and args.cmd != "run":
        parser.error("arguments after -- are only passed on by `motifline run`")
    args.extra = extra
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
