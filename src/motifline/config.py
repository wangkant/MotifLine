"""Config validation, reporting every problem in one pass.

`motifline check` runs it with the filesystem; the Snakefile runs it with paths=False
so the DAG still builds where archives have moved. Relative paths resolve against the
workflow directory, which is where snakemake runs.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from motifline import naming as nm

MODELS = ("cbp", "ag")

COMMON_KEYS = {"model", "head", "n_shards", "rows", "in_window",
               "cancelled", "cancelled_note", "notes"}

# per model: keys a computed run must declare, and keys it may
MODEL_KEYS = {
    "cbp": (("wo_bias", "peaks"), {"n_shuffles", "random_state"}),
    "ag": (("ag_src", "weights", "peaks", "track", "biosample", "chunk"),
           {"crop", "n_shuffles", "random_state"}),
}

ADOPTED_REQUIRED = ("adopt_from", "rows")
ADOPTED_OPTIONAL = {"adopt_peaks", "crop", "adopt_format", "adopt_scores"}

# npy: attr.npy + ohe.npy + a peaks bed; chrombpnet_h5: official ChromBPNet interpret output
ADOPT_FORMATS = ("npy", "chrombpnet_h5")

TOP_LEVEL_KEYS = {"envs", "genome", "attribution",
                  "modisco", "finemo", "reference_lexicon", "runs"}

SECTION_KEYS = {
    "envs": {"compute", "chrombpnet"},
    "genome": {"fasta", "chrom_sizes"},
    "attribution": {"n_shuffles", "random_state"},
    "modisco": {"max_seqlets", "window", "n_leiden"},
    "finemo": {"mode", "threshold", "alpha", "batch"},
    "reference_lexicon": {"sources", "tomtom", "h14_annotation", "qval_threshold"},
}

POSITIVE_INTS = ("head", "n_shards", "rows", "in_window", "crop", "chunk", "n_shuffles")

COMPUTE_BINARIES = ("python", "modisco", "finemo")


class ConfigError(ValueError):
    pass


def load(path) -> dict:
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
    found = problems(cfg, paths=paths, base=base)
    if found:
        raise ConfigError("\n".join([f"{len(found)} problem(s) in config:"]
                                    + [f"  - {p}" for p in found]))


def problems(cfg: dict, *, paths: bool = True, base=".") -> list[str]:
    out: list[str] = []
    runs = cfg.get("runs")
    if not isinstance(runs, dict) or not runs:
        out.append("runs: no runs are declared")
        runs = {}
    bad = [n for n in runs if not _run_name_ok(n)]
    if bad:
        # YAML reads off/no/yes/on as booleans and bare numbers as numbers
        out.append(f"runs: names must be strings without '/', '=' or spaces that do not "
                   f"start with '-', not {bad} (quote YAML booleans and numbers)")
        runs = {k: v for k, v in runs.items() if k not in bad}

    out += _structure(cfg, runs)
    for name in sorted(runs):
        out += _run_structure(name, runs[name])
    out += _lexicon(cfg)
    if paths:
        out += _paths(cfg, runs, Path(base))
    return out


def _run_name_ok(name) -> bool:
    # a path component, and the label in name_patterns' `--h5 <run>=<path>`
    return (isinstance(name, str) and bool(name) and not name.startswith("-")
            and not any(c in name for c in "/=") and not any(c.isspace() for c in name))


def _computed(runs: dict) -> list[dict]:
    return [r for r in runs.values() if isinstance(r, dict) and not r.get("adopt_from")
            and r.get("model") in MODELS]


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
        out.append(f"unknown top-level key(s) {unknown} (expected some of {sorted(TOP_LEVEL_KEYS)})")
    for section, allowed in SECTION_KEYS.items():
        node = cfg.get(section)
        if isinstance(node, dict) and set(node) - allowed:
            out.append(f"{section}: unknown key(s) {sorted(set(node) - allowed)} "
                       f"(expected some of {sorted(allowed)})")

    for dotted in ("envs.compute", "modisco.max_seqlets", "modisco.window", "modisco.n_leiden",
                   "finemo.mode", "finemo.threshold", "finemo.alpha", "finemo.batch"):
        out += _require(cfg, dotted)

    computed = _computed(runs)
    if any(isinstance(r, dict) and r.get("adopt_format") == "chrombpnet_h5"
           for r in runs.values()):
        out += _require(cfg, "genome.fasta")
    if computed:
        out += _require(cfg, "genome.fasta") + _require(cfg, "genome.chrom_sizes")
        out += _require(cfg, "attribution.n_shuffles")
        out += _require(cfg, "attribution.random_state")
    if any(r["model"] == "cbp" for r in computed):
        out += _require(cfg, "envs.chrombpnet")
    win = (cfg.get("modisco") or {}).get("window")
    if win is not None and not (isinstance(win, int) and win > 0):
        out.append(f"modisco.window: must be a positive integer, not {win!r}")
    leiden = (cfg.get("modisco") or {}).get("n_leiden")
    if leiden is not None and not (isinstance(leiden, int) and not isinstance(leiden, bool)
                                   and leiden > 0):
        out.append(f"modisco.n_leiden: must be a positive integer, not {leiden!r}")
    q = (cfg.get("reference_lexicon") or {}).get("qval_threshold")
    if q is not None and not (isinstance(q, (int, float)) and not isinstance(q, bool)
                              and 0 < q <= 1):
        out.append(f"reference_lexicon.qval_threshold: must be in (0, 1], not {q!r}")
    mode = (cfg.get("finemo") or {}).get("mode")
    if mode is not None and mode not in ("pp", "ph", "hp", "hh"):
        out.append(f"finemo.mode: {mode!r} is not one of pp/ph/hp/hh")
    return out


def _run_structure(name: str, r) -> list[str]:
    if not isinstance(r, dict):
        return [f"runs.{name}: expected a mapping"]
    out = []
    model = r.get("model")
    if model not in MODELS:
        return [f"runs.{name}.model: {model!r} is not one of {list(MODELS)}"]

    adopted = bool(r.get("adopt_from"))
    required, optional = (ADOPTED_REQUIRED, ADOPTED_OPTIONAL) if adopted else MODEL_KEYS[model]
    for key in required:
        if key not in r:
            kind = "adopted" if adopted else f"computed {model}"
            out.append(f"runs.{name}.{key}: required for a {kind} run, and missing")

    allowed = COMMON_KEYS | set(required) | set(optional) | {"adopt_from"}
    unknown = sorted(set(r) - allowed)
    if unknown:
        out.append(f"runs.{name}: unknown key(s) {unknown} (expected some of {sorted(allowed)})")

    for key in POSITIVE_INTS:
        v = r.get(key)
        if v is not None and not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
            out.append(f"runs.{name}.{key}: must be a positive integer, not {v!r}")
    track = r.get("track")
    if track is not None and not (isinstance(track, int) and not isinstance(track, bool)
                                  and track >= 0):
        out.append(f"runs.{name}.track: must be a non-negative integer, not {track!r}")
    fmt = r.get("adopt_format", "npy")
    if fmt not in ADOPT_FORMATS:
        out.append(f"runs.{name}.adopt_format: {fmt!r} is not one of {list(ADOPT_FORMATS)}")
    if fmt == "chrombpnet_h5":
        for key in ("adopt_scores", "adopt_peaks"):
            if key not in r:
                out.append(f"runs.{name}.{key}: required with adopt_format: chrombpnet_h5")
        if model != "cbp":
            out.append(f"runs.{name}.model: adopt_format chrombpnet_h5 is a cbp format")
    if r.get("n_shards") and adopted:
        out.append(f"runs.{name}.n_shards: an adopted run computes nothing to shard")
    if "rows" in r and not adopted and not r.get("n_shards"):
        out.append(f"runs.{name}.rows: only checked for sharded and adopted runs")
    if r.get("n_shards") and r.get("head"):
        out.append(f"runs.{name}: head and n_shards cannot be combined")
    if "crop" in r and "in_window" in r and isinstance(r["crop"], int) \
            and isinstance(r["in_window"], int) and r["crop"] > r["in_window"]:
        out.append(f"runs.{name}.crop: {r['crop']} exceeds in_window {r['in_window']}")
    return out


def _lexicon(cfg: dict) -> list[str]:
    try:
        nm.params(cfg)
    except nm.NamingError as e:
        return [str(e).strip("'")]
    return []


def _resolve(base: Path, value) -> Path:
    p = Path(str(value))
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


def _chrom_sizes(base: Path, value) -> dict | None:
    if value is None or not _resolve(base, value).is_file():
        return None
    sizes = {}
    for line in _resolve(base, value).read_text().splitlines():
        if line.strip():
            c, n = line.split()[:2]
            sizes[c] = int(n)
    return sizes


def _narrowpeak(base: Path, label: str, value, sizes: dict | None) -> list[str]:
    """Every line: 10 or 11 columns (11th = True), integer coordinates, summit inside
    the peak, a known chromosome; no two peaks with the same summit (identical windows)."""
    p = _resolve(base, value)
    if not p.exists():
        return [f"{label}: {p} does not exist"]
    bad, seen, dups, n, widths = [], set(), 0, 0, set()
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        n += 1
        cols = line.split("\t")
        widths.add(len(cols))
        if len(cols) not in (10, 11):
            bad.append(f"line {i}: {len(cols)} columns, need 10/11-column narrowPeak")
            continue
        if len(cols) == 11 and cols[10] != "True":
            bad.append(f"line {i}: column 11 must be True (is_peak), not {cols[10]!r}")
        try:
            start, end, summit = int(cols[1]), int(cols[2]), int(cols[9])
        except ValueError:
            bad.append(f"line {i}: start/end/summit are not integers")
            continue
        if not 0 <= summit < end - start:
            bad.append(f"line {i}: summit offset {summit} is outside the peak ({end - start} bp)")
        if sizes is not None and cols[0] not in sizes:
            bad.append(f"line {i}: chromosome {cols[0]!r} is not in genome.chrom_sizes")
        key = (cols[0], start + summit)
        dups += key in seen
        seen.add(key)
    out = [f"{label}: {p} {b}" for b in bad[:5]]
    if len(bad) > 5:
        out.append(f"{label}: {p} has {len(bad) - 5} more bad lines")
    if dups:
        out.append(f"{label}: {p} has {dups} peaks whose summit repeats an earlier peak's")
    if widths >= {10, 11}:
        out.append(f"{label}: {p} mixes 10- and 11-column lines")
    if not n:
        out.append(f"{label}: {p} has no data lines")
    return out


def _paths(cfg: dict, runs: dict, base: Path) -> list[str]:
    out = []
    envs = cfg.get("envs") or {}
    compute = envs.get("compute")
    out += _exists(base, "envs.compute", compute, kind="dir")
    if compute and _resolve(base, compute).is_dir():
        for binary in COMPUTE_BINARIES:
            out += _exists(base, f"envs.compute/bin/{binary}",
                           _resolve(base, compute) / "bin" / binary, kind="exe")
    cbp_env = envs.get("chrombpnet")
    out += _exists(base, "envs.chrombpnet", cbp_env, kind="dir")
    if cbp_env and _resolve(base, cbp_env).is_dir():
        out += _exists(base, "envs.chrombpnet/bin/python",
                       _resolve(base, cbp_env) / "bin" / "python", kind="exe")
    genome = cfg.get("genome") or {}
    out += _exists(base, "genome.fasta", genome.get("fasta"))
    out += _exists(base, "genome.chrom_sizes", genome.get("chrom_sizes"))
    sizes = _chrom_sizes(base, genome.get("chrom_sizes"))

    lex = cfg.get("reference_lexicon") or {}
    for tag, path in (lex.get("sources") or {}).items():
        out += _exists(base, f"reference_lexicon.sources.{tag}", path)
    out += _exists(base, "reference_lexicon.tomtom", lex.get("tomtom"), kind="exe")
    out += _exists(base, "reference_lexicon.h14_annotation", lex.get("h14_annotation"))
    for name in sorted(runs):
        r = runs[name]
        if not isinstance(r, dict) or r.get("cancelled"):
            continue
        for key, kind in (("wo_bias", "file"), ("weights", "file"),
                          ("ag_src", "dir"), ("adopt_from", "dir")):
            if key in r:
                out += _exists(base, f"runs.{name}.{key}", r[key], kind=kind)
        if r.get("model") == "ag" and "ag_src" in r:
            out += _exists(base, f"runs.{name}.ag_src track metadata",
                           _resolve(base, r["ag_src"]) / "alphagenome_pytorch" / "data"
                           / "track_metadata_human.parquet")
        if "peaks" in r:
            out += _narrowpeak(base, f"runs.{name}.peaks", r["peaks"], sizes)
        if r.get("adopt_format") == "chrombpnet_h5" and "adopt_from" in r:
            for key in ("adopt_scores", "adopt_peaks"):
                if key in r:
                    out += _exists(base, f"runs.{name}.{key}",
                                   _resolve(base, r["adopt_from"]) / r[key])
    return out


def attr_inputs(cfg: dict, r: dict) -> list[str]:
    """Files a computed run's attribution reads; the Snakefile declares them as inputs.
    The official ChromBPNet code reads no chrom sizes."""
    model_file = r["wo_bias"] if r["model"] == "cbp" else r["weights"]
    files = [model_file, r["peaks"], cfg["genome"]["fasta"]]
    return files if r["model"] == "cbp" else files + [cfg["genome"]["chrom_sizes"]]


def missing_attr_inputs(cfg: dict, base=".") -> list[str]:
    """`run: path` for each missing input of an active computed run."""
    out = []
    for name, r in sorted((cfg.get("runs") or {}).items()):
        if r.get("adopt_from") or r.get("cancelled"):
            continue
        out += [f"{name}: {p}" for p in attr_inputs(cfg, r)
                if not _resolve(Path(base), p).exists()]
    return out


def skeleton() -> str:
    return Path(__file__).with_name("config_skeleton.yaml").read_text()
