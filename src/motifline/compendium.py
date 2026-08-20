"""Cross-model merge: N MoDISco h5 -> ONE MotifCompendium lexicon.

What a merge is here: two leaves of the SAME region set, one per model (CBP at its
native 2114bp, AG at its native 2048bp), each with its own MoDISco h5, clustered
together so that a cluster containing patterns from both models is one motif the two
models agree on and a cluster containing patterns from one is not. The membership
bookkeeping is the point -- MotifCompendium clusters on CWM similarity alone and is
blind to which file a pattern came from, so "shared / AG-only / CBP-only" has to be
recovered from the ``model`` column afterwards, not read off the clustering.

The line stops at the merged lexicon by decision (2026-08-16): nothing here calls
hits against it. External-lexicon hit calling is a separate, already-existing route
(``lexicons.py``, ``rule call_hits_lexicon``) and pointing it at a compendium would be
a new experiment, not this stage.

A leaf is either a run of this pipeline (``run:``, whose modisco.h5 is then a DAG
input and will be built if missing) or a path to an h5 this pipeline did not produce
(``h5:``, taken as given). Mixing the two in one merge is allowed and is what a
comparison against an archived leaf looks like.
"""
from __future__ import annotations


class MergeError(KeyError):
    """A merge definition cannot be resolved -- refuse rather than guess."""


def merges(cfg: dict) -> dict:
    return (cfg.get("compendium") or {}).get("merges") or {}


def merge_names(cfg: dict) -> list[str]:
    return sorted(merges(cfg))


def merge_def(cfg: dict, name: str) -> dict:
    m = merges(cfg)
    if name not in m:
        raise MergeError(f"unknown merge {name!r}; config defines {sorted(m)}")
    return m[name]


def leaves(cfg: dict, name: str) -> list[tuple[str, str]]:
    """``[(label, modisco_h5_path), ...]`` for a merge, in config order.

    The label becomes MotifCompendium's ``model`` column and lands in every merged
    motif's name, so it is checked here rather than at the far end: MotifCompendium
    parses names on ``-`` and ``.`` (build_from_modisco's own note).
    """
    d = merge_def(cfg, name)
    spec = d.get("leaves")
    if not spec:
        raise MergeError(f"merge {name!r} defines no leaves")
    if len(spec) < 2:
        raise MergeError(f"merge {name!r} has {len(spec)} leaf; a merge needs at least 2")
    out = []
    for label, src in spec.items():
        if set("-.") & set(str(label)):
            raise MergeError(
                f"merge {name!r}: leaf label {label!r} contains '-' or '.', which "
                f"MotifCompendium uses as separators inside motif names")
        if not isinstance(src, dict) or ("run" in src) == ("h5" in src):
            raise MergeError(
                f"merge {name!r}: leaf {label!r} must be exactly one of "
                f"{{run: <run name>}} or {{h5: <path>}}, got {src!r}")
        if "run" in src:
            if src["run"] not in (cfg.get("runs") or {}):
                raise MergeError(
                    f"merge {name!r}: leaf {label!r} names run {src['run']!r}, "
                    f"which is not defined under config.runs")
            out.append((label, f"results/{src['run']}/modisco/modisco.h5"))
        else:
            out.append((label, str(src["h5"])))
    return out


def dag_inputs(cfg: dict, name: str) -> list[str]:
    """The leaf h5 snakemake must build first: the run-backed ones only.

    An ``h5:`` leaf is an input the pipeline did not produce; declaring it as a
    snakemake input would make an absent archive look like a missing rule output.
    """
    spec = merge_def(cfg, name).get("leaves") or {}
    return [f"results/{s['run']}/modisco/modisco.h5"
            for s in spec.values() if isinstance(s, dict) and "run" in s]


def region_width(cfg: dict, name: str) -> int:
    """MoDISco's ``-w`` for these leaves. Only affects avg_dist_from_summit.

    Defaults to this pipeline's canonical ``modisco.window``; a merge over leaves
    somebody else discovered has to declare the width THEY used, because the
    per-motif summit distances would otherwise be off by half the difference.
    """
    d = merge_def(cfg, name)
    if "region_width" in d:
        return int(d["region_width"])
    w = (cfg.get("modisco") or {}).get("window")
    if w is None:
        raise MergeError("config.modisco.window is unset and the merge declares no region_width")
    return int(w)


def params(cfg: dict) -> dict:
    """Clustering parameters, all of them explicit in config."""
    c = cfg.get("compendium") or {}
    missing = [k for k in ("env", "src", "algorithm", "similarity_threshold", "seeds")
               if k not in c]
    if missing:
        raise MergeError(f"config.compendium is missing {missing}")
    return c
