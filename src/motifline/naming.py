"""Where pattern names come from: the reference lexicon and the annotation gate.

Until 2026-08-24 this pipeline named nothing: patterns travelled as ``pos_patterns.
pattern_3`` and whatever label an upstream report had guessed.

The naming chain adopted here is one source for every run of the pipeline:

    sources (HOCOMOCO H14CORE + JASPAR CORE vert NR)
        -> union MEME, every matrix kept          (steps/lexicon_union.py)
        -> RSAT matrix-clustering, its partition   (rule lexicon_rsat)
        -> one medoid per cluster                  (steps/lexicon_medoid.py)
        -> the union re-tagged with cluster ids    (steps/lexicon_cltagged.py)
        -> TomTom, gate, cluster-level name        (steps/name_patterns.py)

Two properties of that chain are decisions, not defaults, and both live in config:

* **The TomTom target is the cluster-tagged union, not the medoid library.** Matching
  against cluster representatives collapses sensitivity -- a pattern's best match is
  usually some other member of its cluster -- while matching against the full set and
  reading the cluster id back off the winner's name loses nothing. The medoid library
  is built anyway, as the compact reference to plot and publish from.
* **The gate is two-armed**: ``q < gate.qval`` AND ``best_ncor >= gate.ncor``. The Ncor
  arm is set to the clustering's own merge floor, so "annotated" means the match is both
  better than chance and close enough that the library build would have merged them.
  A pattern that fails either arm keeps its match in the table and is not renamed.

Families -- grouping cluster ids into site types so that two models' patterns can be
paired -- are a human judgement made once per project and are deliberately not derived
here. This module stops at the cluster.
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
    missing = [k for k in ("sources", "rsat", "tomtom", "gate", "w_th") if k not in c]
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
    """Minimum comparable overlap, in columns, for the Ncor arm of the chain.

    Used in two places that must not drift apart: the all-pairs Ncor that selects each
    cluster's medoid, and the IC-core floor below which a pattern has no comparable
    Ncor at all (and therefore fails the gate by construction). It is NOT
    ``rsat.lth.w`` -- RSAT discards that arm under ``-quick`` -- and it is NOT TomTom's
    ``-min-overlap``, which stays pinned to modiscolite's parameter set. Three numbers
    that happen to be 5; only these two are this one.
    """
    return int(params(cfg)["w_th"])


def gate(cfg: dict) -> tuple[float, float]:
    g = params(cfg)["gate"]
    missing = [k for k in ("qval", "ncor") if k not in g]
    if missing:
        raise NamingError(f"config.reference_lexicon.gate is missing {missing}; both arms are required")
    return float(g["qval"]), float(g["ncor"])


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
    """What every naming stage reads: the tagged target plus the untagged sources.

    Both, because the stage measures two different things: TomTom runs against the
    tagged copy, the Ncor arm of the gate runs against the untagged matrices.
    """
    return [CLTAGGED] + source_paths(cfg)


def run_names(run: str) -> str:
    return f"results/{run}/naming/pattern_names.tsv"
