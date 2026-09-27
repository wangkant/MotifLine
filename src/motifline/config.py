"""What a config has to contain, reported in one pass before the DAG is built.

A misspelled key is the case worth catching: ``n_shard: 4`` is a valid single-process
run. Called by ``motifline check`` (with the filesystem) and by the Snakefile at load
time (``paths=False``, so the DAG still builds where archives have moved). What is
required follows from the runs declared: adopted runs need no genome or model source.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from motifline import naming as nm

MODELS = ("cbp", "ag")

#: Keys any run entry may carry, whatever its model and whoever produced its arrays.
COMMON_KEYS = {"model", "head", "n_shards", "rows", "in_window",
               "cancelled", "cancelled_note", "notes"}

#: Per model: the keys a *computed* run must declare, and the ones it may.
MODEL_KEYS = {
    "cbp": (("wo_bias", "peaks"), {"n_shuffles", "random_state"}),
    "ag": (("ag_src", "weights", "peaks", "track", "chunk"),
           {"crop", "n_shuffles", "random_state"}),
}

#: An adopted run computes nothing, so it declares the archive instead of the model.
ADOPTED_REQUIRED = ("adopt_from", "rows")
ADOPTED_OPTIONAL = {"adopt_peaks", "crop"}

TOP_LEVEL_KEYS = {"envs", "chrombpnet_src", "genome", "attribution",
                  "modisco", "finemo", "reference_lexicon", "runs"}

#: Executables the downstream stages invoke out of ``envs.compute`` by name.
COMPUTE_BINARIES = ("python", "modisco", "finemo")


class ConfigError(ValueError):
    """The config cannot be run as written; the message lists every reason."""


def load(path) -> dict:
    """Parse a config file, naming the file in any parse error."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as e:
        raise ConfigError(f"{path}: {e}") from e
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML -- {e}") from e
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return cfg


def check(cfg: dict, *, paths: bool = True, base=".") -> None:
    """Raise :class:`ConfigError` listing every problem, or return None."""
    found = problems(cfg, paths=paths, base=base)
    if found:
        raise ConfigError("\n".join([f"{len(found)} problem(s) in config:"]
                                    + [f"  - {p}" for p in found]))


def problems(cfg: dict, *, paths: bool = True, base=".") -> list[str]:
    """Every reason this config cannot be run, as one line each.

    ``paths=False`` checks structure alone -- no filesystem is touched. Relative paths
    in the config resolve against ``base``, which is the working directory snakemake
    is invoked from.
    """
    out: list[str] = []
    runs = cfg.get("runs")
    if not isinstance(runs, dict) or not runs:
        out.append("runs: no runs are declared; there is nothing for the pipeline to do")
        runs = {}

    out += _structure(cfg, runs)
    for name in sorted(runs):
        out += _run_structure(name, runs[name])
    out += _lexicon(cfg)
    if paths:
        out += _paths(cfg, runs, Path(base))
    return out


# --- structure -------------------------------------------------------------

def _models_in_use(runs: dict) -> set[str]:
    return {r.get("model") for r in runs.values() if isinstance(r, dict)}


def _computed_models(runs: dict) -> set[str]:
    return {r.get("model") for r in runs.values()
            if isinstance(r, dict) and not r.get("adopt_from")}


def _require(cfg: dict, dotted: str) -> list[str]:
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return [f"{dotted}: required, and missing"]
        node = node[part]
    return []


def _structure(cfg: dict, runs: dict) -> list[str]:
    out = []
    unknown = sorted(set(cfg) - TOP_LEVEL_KEYS)
    if unknown:
        out.append(f"unknown top-level key(s) {unknown}; nothing reads them "
                   f"(expected some of {sorted(TOP_LEVEL_KEYS)})")

    for dotted in ("envs.compute", "modisco.max_seqlets", "modisco.window",
                   "finemo.mode", "finemo.threshold", "finemo.alpha", "finemo.batch"):
        out += _require(cfg, dotted)

    computed = _computed_models(runs)
    if computed:
        out += _require(cfg, "genome.fasta") + _require(cfg, "genome.chrom_sizes")
    if computed & {"cbp", "ag"}:
        out += _require(cfg, "chrombpnet_src")
        out += _require(cfg, "attribution.n_shuffles")
        out += _require(cfg, "attribution.random_state")
    win = (cfg.get("modisco") or {}).get("window")
    if isinstance(win, int) and win <= 0:
        out.append(f"modisco.window: must be a positive number of bases, not {win}")
    return out


def _run_structure(name: str, r) -> list[str]:
    if not isinstance(r, dict):
        return [f"runs.{name}: expected a mapping"]
    out = []
    model = r.get("model")
    if model not in MODELS:
        out.append(f"runs.{name}.model: {model!r} is not one of {list(MODELS)}")
        return out

    adopted = bool(r.get("adopt_from"))
    if adopted:
        required, optional = ADOPTED_REQUIRED, ADOPTED_OPTIONAL
    else:
        required, optional = MODEL_KEYS[model]
    for key in required:
        if key not in r:
            kind = "adopted" if adopted else f"computed {model}"
            out.append(f"runs.{name}.{key}: required for a {kind} run, and missing")

    allowed = COMMON_KEYS | set(required) | set(optional) | {"adopt_from"}
    unknown = sorted(set(r) - allowed)
    if unknown:
        out.append(f"runs.{name}: unknown key(s) {unknown}; nothing reads them, so a "
                   f"misspelling here changes the run silently "
                   f"(expected some of {sorted(allowed)})")

    if r.get("n_shards") and adopted:
        out.append(f"runs.{name}.n_shards: an adopted run computes nothing to shard")
    return out


def _lexicon(cfg: dict) -> list[str]:
    try:
        nm.params(cfg)
    except nm.NamingError as e:
        return [str(e).strip("'")]
    return []


# --- filesystem ------------------------------------------------------------

def _resolve(base: Path, value) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else base / p


def _exists(base: Path, label: str, value, *, kind: str = "file") -> list[str]:
    if value is None:
        return []
    p = _resolve(base, value)
    if not p.exists():
        return [f"{label}: {p} does not exist"]
    if kind == "dir" and not p.is_dir():
        return [f"{label}: {p} is not a directory"]
    if kind == "exe" and not os.access(p, os.X_OK):
        return [f"{label}: {p} is not executable"]
    return []


def _narrowpeak(base: Path, label: str, value) -> list[str]:
    """finemo reads the summit from column 10, so the peaks must carry one."""
    p = _resolve(base, value)
    if not p.exists():
        return [f"{label}: {p} does not exist"]
    for line in p.read_text().splitlines():
        if not line.strip() or line.startswith(("#", "track", "browser")):
            continue
        cols = line.split("\t")
        if len(cols) < 10:
            return [f"{label}: {p} has {len(cols)} columns; finemo reads the summit "
                    f"from column 10, so peaks must be 10/11-column narrowPeak"]
        try:
            int(cols[9])
        except ValueError:
            return [f"{label}: {p} column 10 is {cols[9]!r}, not an integer summit "
                    f"offset (narrowPeak)"]
        return []
    return [f"{label}: {p} has no data lines"]


def _paths(cfg: dict, runs: dict, base: Path) -> list[str]:
    out = []
    envs = cfg.get("envs") or {}
    compute = envs.get("compute")
    out += _exists(base, "envs.compute", compute, kind="dir")
    if compute and _resolve(base, compute).is_dir():
        for binary in COMPUTE_BINARIES:
            out += _exists(base, f"envs.compute/bin/{binary}",
                           _resolve(base, compute) / "bin" / binary, kind="exe")
    out += _exists(base, "chrombpnet_src", cfg.get("chrombpnet_src"), kind="dir")
    genome = cfg.get("genome") or {}
    out += _exists(base, "genome.fasta", genome.get("fasta"))
    out += _exists(base, "genome.chrom_sizes", genome.get("chrom_sizes"))

    lex = cfg.get("reference_lexicon") or {}
    for tag, path in (lex.get("sources") or {}).items():
        out += _exists(base, f"reference_lexicon.sources.{tag}", path)
    out += _exists(base, "reference_lexicon.tomtom", lex.get("tomtom"), kind="exe")
    out += _exists(base, "reference_lexicon.h14_annotation", lex.get("h14_annotation"))
    for name in sorted(runs):
        r = runs[name]
        if not isinstance(r, dict):
            continue
        # A cancelled run's archive may be gone; that is its state, not an error.
        if r.get("cancelled"):
            continue
        for key, kind in (("wo_bias", "file"), ("weights", "file"),
                          ("ag_src", "dir"), ("adopt_from", "dir")):
            if key in r:
                out += _exists(base, f"runs.{name}.{key}", r[key], kind=kind)
        if "peaks" in r:
            out += _narrowpeak(base, f"runs.{name}.peaks", r["peaks"])
    return out


# --- skeleton --------------------------------------------------------------

def skeleton() -> str:
    """The starting config ``motifline init`` writes: every required section, no runs."""
    return Path(__file__).with_name("config_skeleton.yaml").read_text()
