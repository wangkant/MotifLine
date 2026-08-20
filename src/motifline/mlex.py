"""Cross-model merge, motiflex finish: N MoDISco h5 -> annotated merge tree, and,
when the merge declares a cut, a compiled shared lexicon (the end of that line).

Since 2026-08-17 this is how the merge line ends; MotifCompendium
(``compendium.py`` / ``rule merge_compendium``) stays reachable by explicit target
as the comparison arm but is no longer in the default DAG.

The merge definitions are the SAME ``compendium.merges`` entries -- what to merge
is one question and both tools answer it over the same leaves. What differs is the
finish: ``mlex`` builds an average-linkage tree over 1 - ppm_similarity with a
per-pair shuffle null and merges NOTHING until a ``cut_height`` is declared on the
merge; the cut is a hyperparameter and belongs in config, next to the merge it
cuts, not in code. With a cut the group representatives are medoids -- observed
patterns, never cross-model averages -- and ``mlex compile`` writes a lexicon
verified by reading it back with the real finemo loader.

``mlex ingest`` refuses semantics it would have to guess (its BA-11 rule), so a
leaf must supply more than an h5: attributions, one-hot, regions, and the model
metadata. For a ``run:`` leaf all of that is derivable from this pipeline's own
config -- the run def declares the model and geometry, and the run's attr/ stage
wrote the arrays. An ``h5:`` leaf was produced elsewhere, so the extra paths and
the metadata have to be declared on the leaf itself (``attr``/``ohe``/``bed``/
``meta``); refusing an underdeclared leaf here is the same policy ingest applies.
"""
from __future__ import annotations

from motifline import compendium as comp
from motifline.geometry import attr_width, geometry


class MlexError(KeyError):
    """A motiflex merge cannot be resolved from config -- refuse rather than guess."""


#: run-def model tag -> the declared semantics ingest requires.
#: training_celltype None means "the readout celltype" (single-celltype model).
_MODEL_META = {
    "cbp": {"model": "chrombpnet", "training_celltype": None},
    "ag": {"model": "alphagenome", "training_celltype": "MULTI_TRACK_SHARED"},
}

_H5_LEAF_META_KEYS = ("model", "readout", "union_id", "training_celltype",
                      "input_width", "context_window")


def params(cfg: dict) -> dict:
    c = cfg.get("motiflex") or {}
    missing = [k for k in ("mlex", "celltype", "peak_universe_id") if k not in c]
    if missing:
        raise MlexError(f"config.motiflex is missing {missing}")
    return c


def cut_height(cfg: dict, name: str):
    """The declared cut for a merge, or None: tree only, nothing merged."""
    h = comp.merge_def(cfg, name).get("cut_height")
    return None if h is None else float(h)


def db_yaml(cfg: dict, name: str):
    """Optional precomputed database labels for ``mlex annotate``; None skips it."""
    return comp.merge_def(cfg, name).get("db_yaml")


def _run_analysis(cfg: dict, name: str, label: str, run_name: str) -> dict:
    r = (cfg.get("runs") or {}).get(run_name)
    if r is None:
        raise MlexError(f"merge {name!r}: leaf {label!r} names unknown run {run_name!r}")
    meta = _MODEL_META.get(r.get("model"))
    if meta is None:
        raise MlexError(f"merge {name!r}: run {run_name!r} has model {r.get('model')!r}")
    celltype = params(cfg)["celltype"]
    if r["model"] == "ag":
        if "track" not in r:
            raise MlexError(f"merge {name!r}: ag run {run_name!r} declares no track")
        readout = f"atac_track{r['track']}"
    else:
        readout = "counts"
    base = f"results/{run_name}"
    return {
        "id": f"{label}_{name}",
        "model": meta["model"],
        "readout": readout,
        "union_id": label.upper(),
        "context": name,
        "readout_celltype": celltype,
        "training_celltype": meta["training_celltype"] or celltype,
        "input_width": attr_width(cfg, r),
        "context_window": geometry(cfg, r)["in_window"],
        "modisco_h5": f"{base}/modisco/modisco.h5",
        "contributions": f"{base}/attr/attr.npy",
        "one_hot": f"{base}/attr/ohe.npy",
        "regions_bed": f"{base}/attr/regions.bed",
        "search_metaclusters": {"pos_patterns": True, "neg_patterns": True},
    }


def _h5_analysis(cfg: dict, name: str, label: str, src: dict) -> dict:
    missing = [k for k in ("h5", "attr", "ohe", "bed", "meta") if k not in src]
    if missing:
        raise MlexError(
            f"merge {name!r}: h5 leaf {label!r} is missing {missing}; an external "
            f"leaf must declare its arrays and metadata -- nothing is derived from paths")
    meta = src["meta"]
    missing = [k for k in _H5_LEAF_META_KEYS if k not in meta]
    if missing:
        raise MlexError(f"merge {name!r}: leaf {label!r} meta is missing {missing}")
    return {
        "id": f"{label}_{name}",
        "model": meta["model"],
        "readout": meta["readout"],
        "union_id": meta["union_id"],
        "context": name,
        "readout_celltype": params(cfg)["celltype"],
        "training_celltype": meta["training_celltype"],
        "input_width": int(meta["input_width"]),
        "context_window": int(meta["context_window"]),
        "modisco_h5": str(src["h5"]),
        "contributions": str(src["attr"]),
        "one_hot": str(src["ohe"]),
        "regions_bed": str(src["bed"]),
        "search_metaclusters": {"pos_patterns": True, "neg_patterns": True},
    }


def project(cfg: dict, name: str) -> dict:
    """The ``mlex ingest`` project dict for one merge, every semantic field declared."""
    spec = comp.merge_def(cfg, name).get("leaves") or {}
    if len(spec) < 2:
        raise MlexError(f"merge {name!r} has {len(spec)} leaf; a merge needs at least 2")
    analyses = []
    for label, src in spec.items():
        if not isinstance(src, dict) or ("run" in src) == ("h5" in src):
            raise MlexError(
                f"merge {name!r}: leaf {label!r} must be exactly one of "
                f"{{run: ...}} or {{h5: ...}}, got {src!r}")
        if "run" in src:
            analyses.append(_run_analysis(cfg, name, label, src["run"]))
        else:
            analyses.append(_h5_analysis(cfg, name, label, src))
    return {
        "project": f"motifline-{name}",
        "peak_universe_id": params(cfg)["peak_universe_id"],
        "analyses": analyses,
        "shared_attribution_groups": [],
        "provenance": {"random_seed": int(params(cfg).get("seed", 42))},
    }
