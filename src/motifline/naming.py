"""Where pattern names come from: the source union and TomTom.

    sources (HOCOMOCO H14CORE + JASPAR CORE vert NR)
        -> union MEME, every matrix kept          (steps/lexicon_union.py)
        -> TomTom, name = gene symbol of the first match   (steps/name_patterns.py)

No gate: every pattern is named after its first match, and the q-value travels with the
name. Names are gene symbols; HOCOMOCO ids are mapped through HOCOMOCO's own annotation
(``reference_lexicon.h14_annotation``). The target is the full union rather than a
de-duplicated library, because a pattern's best match is usually not the representative
of its group.
"""
from __future__ import annotations

LEXICON_DIR = "results/lexicon"
UNION = f"{LEXICON_DIR}/lexicon_sources_union.meme"


class NamingError(KeyError):
    """The lexicon is underdeclared."""


def params(cfg: dict) -> dict:
    c = cfg.get("reference_lexicon") or {}
    missing = [k for k in ("sources", "tomtom", "h14_annotation") if k not in c]
    if missing:
        raise NamingError(f"config.reference_lexicon is missing {missing}")
    if not c["sources"]:
        raise NamingError("config.reference_lexicon.sources is empty; the union has no inputs")
    return c


def source_paths(cfg: dict) -> list[str]:
    """The source MEME files, in the order they are concatenated into the union."""
    return list(params(cfg)["sources"].values())


def source_argv(cfg: dict) -> list[str]:
    """``--source tag=path`` pairs, in config order."""
    argv = []
    for tag, path in params(cfg)["sources"].items():
        argv += ["--source", f"{tag}={path}"]
    return argv


def run_names(run: str) -> str:
    return f"results/{run}/naming/pattern_names.tsv"
