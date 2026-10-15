"""motifline init | check | run | smoke. `run` checks the config, then calls snakemake."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from motifline import __version__
from motifline.config import ConfigError, check as check_config, load, problems, skeleton
from motifline.tools import TOOLS, deviations, live_defaults, render_deviations

DEFAULT_CONFIG = "config/default.yaml"


def find_workflow(start=None) -> Path | None:
    here = Path(start or Path.cwd()).resolve()
    for d in [here, *here.parents]:
        if (d / "Snakefile").is_file():
            return d
    return None


def snakemake_argv(targets, *, snakefile, jobs=None, gpus=None, dry_run=False,
                   configfile=None, extra=()) -> list[str]:
    argv = [sys.executable, "-m", "snakemake", "--snakefile", str(snakefile),
            "--directory", str(Path(snakefile).parent)]
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


def cmd_init(args) -> int:
    out = Path(args.config)
    if out.exists() and not args.force:
        print(f"{out} already exists; pass --force to overwrite it", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(skeleton())
    print(f"wrote {out}\nEdit the paths, then run: motifline check -c {out}")
    return 0


def cmd_check(args) -> int:
    try:
        cfg = load(args.config)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1
    # the checkout holding the config is where it runs
    base = find_workflow(Path(args.config).resolve().parent) or find_workflow() or Path.cwd()
    found = problems(cfg, paths=not args.no_paths, base=base)
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
    workflow = find_workflow()
    if workflow is None:
        print("no Snakefile in this directory or any parent; run motifline from a "
              "checkout of the pipeline", file=sys.stderr)
        return 1
    configfile = args.config or workflow / DEFAULT_CONFIG
    if not Path(configfile).is_file():
        print(f"{configfile} does not exist; start one with `motifline init`",
              file=sys.stderr)
        return 1
    try:
        check_config(load(configfile), paths=not args.no_check, base=workflow)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        print("\nrefusing to build the DAG on a config that cannot run; "
              "pass --no-check to skip this", file=sys.stderr)
        return 1
    for line in deviation_lines(load(configfile)):
        print(line, file=sys.stderr)
    argv = snakemake_argv(args.targets, snakefile=workflow / "Snakefile",
                          jobs=args.jobs, gpus=args.gpus, dry_run=args.dry_run,
                          configfile=configfile, extra=args.extra)
    print(" ".join(argv), file=sys.stderr)
    return subprocess.run(argv).returncode


def cmd_smoke(args) -> int:
    from motifline.smoke import run_smoke
    return run_smoke(compute_env=args.compute_env, workdir=args.workdir,
                     keep=args.keep)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motifline",
        description="attribution -> MoDISco -> FiNeMo, one line per run")
    p.add_argument("--version", "-V", action="store_true",
                   help="print the version and exit")
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="write a config skeleton to fill in")
    init.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
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
    smoke.add_argument("--workdir", default=None)
    smoke.add_argument("--keep", action="store_true", help="keep the working directory")
    smoke.set_defaults(fn=cmd_smoke)
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
