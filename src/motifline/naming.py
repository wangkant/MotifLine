"""Reference lexicon config: source MEME files -> union -> TomTom names."""
from __future__ import annotations

LEXICON_DIR = "results/lexicon"
UNION = f"{LEXICON_DIR}/lexicon_sources_union.meme"


class NamingError(KeyError):
    pass


def params(cfg: dict) -> dict:
    c = cfg.get("reference_lexicon") or {}
    missing = [k for k in ("sources", "tomtom", "h14_annotation") if k not in c]
    if missing:
        raise NamingError(f"config.reference_lexicon is missing {missing}")
    if not c["sources"]:
        raise NamingError("config.reference_lexicon.sources is empty; the union has no inputs")
    return c


def source_paths(cfg: dict) -> list[str]:
    return list(params(cfg)["sources"].values())


def source_argv(cfg: dict) -> list[str]:
    argv = []
    for tag, path in params(cfg)["sources"].items():
        argv += ["--source", f"{tag}={path}"]
    return argv


def run_names(run: str) -> str:
    return f"results/{run}/naming/pattern_names.tsv"
