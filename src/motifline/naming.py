"""Where pattern names come from: the reference lexicon and TomTom.

Until 2026-08-24 this pipeline named nothing: patterns travelled as ``pos_patterns.
pattern_3`` and whatever label an upstream report had guessed.

Since 2026-09-27 (user decision) a name is the TF of a pattern's first TomTom match
against the source union, with no gate of any kind:

    sources (HOCOMOCO H14CORE + JASPAR CORE vert NR)
        -> union MEME, every matrix kept          (steps/lexicon_union.py)
        -> TomTom, name = TF of the first match   (steps/name_patterns.py)

The q-value travels with the name; nothing is renamed, withheld or merged on it. The
RSAT clustering, medoid library and cluster-tagged union (rules lexicon_rsat,
lexicon_medoid, lexicon_cltagged) are still buildable, but naming no longer reads
them: the earlier chain named patterns by RSAT cluster and gated them on
``q < 0.05 AND best_ncor >= 0.55``, and both were dropped on 2026-09-27.

The TomTom target is the full union, not the medoid library: matching against
representatives collapses sensitivity, since a pattern's best match is usually some
other member of its cluster.
"""
from __future__ import annotations

#: Everything the lexicon build and the naming stage write.
LEXICON_DIR = "results/lexicon"
UNION = f"{LEXICON_DIR}/lexicon_sources_union.meme"
RSAT_DIR = f"{LEXICON_DIR}/rsat"
CLUSTERS_TAB = f"{RSAT_DIR}/rsat_tables/clusters.tab"
CLUSTERS = f"{LEXICON_DIR}/clusters_rsat.tsv"
METADATA = f"{LEXICON_DIR}/lexicon_rsat.metadata.tsv"
MEDOID_MEME = f"{LEXICON_DIR}/lexicon_rsat.meme"
CLTAGGED = f"{LEXICON_DIR}/union_cltagged.meme"
NCOR = f"{LEXICON_DIR}/ncor.npy"


class NamingError(KeyError):
    """The lexicon is underdeclared -- refuse rather than guess a library."""


def params(cfg: dict) -> dict:
    c = cfg.get("reference_lexicon") or {}
    missing = [k for k in ("sources", "rsat", "tomtom", "w_th") if k not in c]
    if missing:
        raise NamingError(f"config.reference_lexicon is missing {missing}")
    if not c["sources"]:
        raise NamingError("config.reference_lexicon.sources is empty; the union has no inputs")
    return c


def source_paths(cfg: dict) -> list[str]:
    """The source MEME files, in the order they are concatenated into the union."""
    return list(params(cfg)["sources"].values())


def source_argv(cfg: dict) -> list[str]:
    """``--source tag=path`` pairs; order matters and is config order."""
    argv = []
    for tag, path in params(cfg)["sources"].items():
        argv += ["--source", f"{tag}={path}"]
    return argv


def w_th(cfg: dict) -> int:
    """Minimum comparable overlap, in columns, for the all-pairs Ncor that selects each
    cluster's medoid (rule lexicon_medoid). It is NOT ``rsat.lth.w`` -- RSAT discards
    that arm under ``-quick`` -- and it is NOT TomTom's ``-min-overlap``, which stays
    pinned to modiscolite's parameter set. Naming does not read it.
    """
    return int(params(cfg)["w_th"])


def rsat(cfg: dict) -> dict:
    r = params(cfg)["rsat"]
    missing = [k for k in ("bin", "hclust_method", "metric", "lth") if k not in r]
    if missing:
        raise NamingError(f"config.reference_lexicon.rsat is missing {missing}")
    return r


#: Thresholds RSAT keeps under ``-quick``. It rewrites ``%lth`` to exactly these two
#: (``matrix-clustering:2249-2257`) before handing them to the R clustering script, so
#: any other arm passed alongside ``-quick`` is accepted on the command line and then
#: silently dropped. Observed live: a run given ``-lth w 5 -lth cor 0.75 -lth Ncor
#: 0.55 -quick`` reaches R as ``lthsp = 'Ncor_0.55_cor_0.75_'`` -- no ``w``.
QUICK_THRESHOLDS = ("Ncor", "cor")


def rsat_argv(cfg: dict, union: str, out_prefix: str) -> list[str]:
    """``rsat matrix-clustering`` at the declared thresholds.

    The thresholds are passed explicitly because RSAT standalone defaults them to
    empty -- the values that look like defaults are the *web server's*. ``-matrix``
    takes title, file and format: the title becomes the ``union_m<N>_`` id prefix that
    ``lexicon_medoid.py`` strips back off.

    A threshold ``-quick`` would discard is refused rather than passed: config has to
    state the thresholds that are actually in force, or the library's build parameters
    are a fiction. (``lth.w`` is not the Ncor floor used elsewhere in the naming chain
    -- that one is ``reference_lexicon.w_th``, see :func:`w_th`.)
    """
    r = rsat(cfg)
    quick = r.get("quick", True)
    if quick:
        dropped = [k for k in r["lth"] if k not in QUICK_THRESHOLDS]
        if dropped:
            raise NamingError(
                f"config.reference_lexicon.rsat.lth declares {dropped} together with "
                f"quick: true, and RSAT discards every arm but {list(QUICK_THRESHOLDS)} "
                f"under -quick. Remove {dropped} (the clustering never applied them) or "
                f"set quick: false.")
    argv = [r["bin"], "matrix-clustering", "-v", "1",
            "-matrix", "union", union, "meme",
            "-hclust_method", r["hclust_method"],
            "-metric_build_tree", r["metric"]]
    for field, value in r["lth"].items():
        argv += ["-lth", field, str(value)]
    if quick:
        argv += ["-quick"]
    return argv + ["-o", out_prefix]


def naming_inputs(cfg: dict) -> list[str]:
    """What every naming stage reads: the source union, TomTom's target."""
    return [UNION]


def run_names(run: str) -> str:
    return f"results/{run}/naming/pattern_names.tsv"
