"""``motifline`` -- init, check, run, smoke.

The pipeline is a Snakefile, and ``snakemake`` remains the way to drive it. What this
adds is the part that cannot be left to the caller:

* ``--rerun-triggers mtime params`` is not optional here and is therefore not a flag.
  With ``mtime`` alone -- snakemake's own default -- editing ``finemo.threshold`` or
  ``modisco.window`` leaves every finished stage in place and snakemake reports
  "Nothing to be done"; ``results/`` then holds a mixture of parameter generations that
  nothing on disk distinguishes from a coherent tree. Every rule declares a ``params:``
  signature for exactly this trigger, and forgetting the flag disables all of them.
* a config is checked in one pass, before the DAG, instead of failing inside whichever
  stage first reads the key that is wrong.
* the claim this pipeline is built on -- that its parameters are the tools' own
  official defaults -- is measured against the installed tools rather than asserted in
  prose. Deviations are printed by `check`, by `doctor` and before every `run`, and
  never block: deviating on purpose is ordinary science, deviating without knowing is
  the accident.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from motifline import __version__
from motifline.config import ConfigError, check as check_config, load, problems, skeleton
from motifline.tools import (PARAMETERS, PINNED, TOOLS, can_import, deviations,
                             live_defaults, render_deviations, tool_version)

DEFAULT_CONFIG = "config/default.yaml"

#: Never a flag: see the module docstring.
RERUN_TRIGGERS = ("mtime", "params")


def find_workflow(start=None) -> Path | None:
    """The nearest enclosing directory holding a Snakefile, or None."""
    here = Path(start or Path.cwd()).resolve()
    for d in [here, *here.parents]:
        if (d / "Snakefile").is_file():
            return d
    return None


def snakemake_argv(targets, *, snakefile, jobs=None, gpus=None, dry_run=False,
                   configfile=None, extra=()) -> list[str]:
    """The snakemake command line this pipeline must be driven with."""
    argv = [sys.executable, "-m", "snakemake", "--snakefile", str(snakefile),
            "--directory", str(Path(snakefile).parent),
            "--rerun-triggers", *RERUN_TRIGGERS]
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
        # After `--` so that a target path is never parsed as a snakemake flag.
        argv += ["--", *targets]
    return argv


# --- the official-configuration report -------------------------------------

def compute_env(cfg) -> str | None:
    return ((cfg.get("envs") or {}).get("compute")) or None


def deviation_lines(cfg) -> list[str]:
    """How this config sits against the tools that will actually run it."""
    env = compute_env(cfg)
    if not env or not Path(env, "bin").is_dir():
        return [f"could not read the tools' defaults: envs.compute ({env}) is not a "
                f"usable environment, so the parameters were not checked against them"]
    try:
        live = {name: live_defaults(name, env) for name in TOOLS}
    except (OSError, subprocess.SubprocessError) as e:
        return [f"could not read the tools' defaults from {env}: {type(e).__name__}: {e}"]
    return render_deviations(deviations(cfg, live, pinned=PINNED))


# --- subcommands -----------------------------------------------------------

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
    base = Path(args.config).resolve().parent.parent
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
        check_config(load(configfile), paths=not args.no_check,
                     base=Path(configfile).resolve().parent.parent)
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


def cmd_doctor(args) -> int:
    """One page: is this installation set up to run the official tools correctly?"""
    try:
        cfg = load(args.config)
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 1
    base = Path(args.config).resolve().parent.parent
    print(f"config       {args.config}")
    runs = cfg.get("runs") or {}
    active = [k for k, v in runs.items() if not v.get("cancelled")]
    print(f"             {len(active)} active run(s), {len(runs) - len(active)} cancelled")

    found = problems(cfg, paths=not args.no_paths, base=base)
    if found:
        print(f"\n{len(found)} problem(s):")
        for p in found:
            print(f"  - {p}")
    else:
        print("             no problems")

    env = compute_env(cfg)
    print(f"\ncompute env  {env}")
    for name in TOOLS:
        version = None
        if env and Path(env, "bin").is_dir():
            version = tool_version(name, env)
        print(f"  {name:<14} {version or 'NOT FOUND'}")

    print("\nparameters")
    for line in deviation_lines(cfg):
        print(f"  {line}")
    print("\n  config key            -> tool flag")
    for p in PARAMETERS:
        print(f"  {p.key:<22}-> {p.tool} {p.flag}"
              + (f"   [{p.note}]" if p.note else ""))

    print("\nmodel code (imported the way the attribution steps import it)")
    probes = []
    if cfg.get("chrombpnet_src"):
        probes.append(("chrombpnet.bpnet", cfg["chrombpnet_src"], "chrombpnet_src"))
    for name, r in sorted(runs.items()):
        if isinstance(r, dict) and r.get("ag_src"):
            probes.append(("alphagenome_pytorch", r["ag_src"], f"runs.{name}.ag_src"))
    if not probes:
        print("  nothing to probe: no chrombpnet_src and no run declares ag_src")
    for module, path, label in probes:
        if not env or not Path(env, "bin").is_dir():
            print(f"  {label:<22} could not probe: no usable compute env")
            continue
        ok, detail = can_import(module, str(path), env)
        print(f"  {label:<22} {'OK' if ok else detail}  ({module} from {path})")

    return 1 if found else 0


def cmd_smoke(args) -> int:
    from motifline.smoke import run_smoke
    return run_smoke(compute_env=args.compute_env, workdir=args.workdir,
                     keep=args.keep)


# --- parser ----------------------------------------------------------------

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

    run = sub.add_parser("run", help="drive the pipeline with the required flags")
    run.add_argument("targets", nargs="*", help="snakemake targets (default: rule all)")
    run.add_argument("-c", "--config", default=None)
    run.add_argument("-j", "--jobs", type=int, default=None)
    run.add_argument("--gpus", type=int, default=None,
                     help="concurrent GPU jobs; export CUDA_VISIBLE_DEVICES yourself")
    run.add_argument("-n", "--dry-run", action="store_true")
    run.add_argument("--no-check", action="store_true",
                     help="skip the path checks before building the DAG")
    run.add_argument("extra", nargs=argparse.REMAINDER,
                     help="everything after -- goes to snakemake")
    run.set_defaults(fn=cmd_run)

    doc = sub.add_parser(
        "doctor", help="one page: config, tool versions, parameters, model code")
    doc.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    doc.add_argument("--no-paths", action="store_true")
    doc.set_defaults(fn=cmd_doctor)

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
    # Handled before parsing so that `motifline --version` needs no subcommand.
    if {"--version", "-V"} & set(argv):
        print(f"motifline {__version__}")
        return 0
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
