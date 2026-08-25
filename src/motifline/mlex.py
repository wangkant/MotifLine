"""Cross-model merge, motiflex finish: N MoDISco h5 -> annotated merge tree, and,
when the merge declares a cut, a compiled shared lexicon (the end of that line).

Since 2026-08-17 this is how the merge line ends; MotifCompendium
(``compendium.py`` / ``rule merge_compendium``) stays reachable by explicit target
as the comparison arm but is no longer in the default DAG.

The merge definitions are the SAME ``compendium.merges`` entries -- what to merge
is one question and both tools answer it over the same leaves.

**2026-08-25.** motiflex removed its ``merge`` subcommand on 2026-08-23 (commit
204cb3f) when it repositioned as an auditor of someone else's consolidation: ``mlex
selectivity`` now scores "any tool's merge" and ``mlex compile --decisions`` reads a
decisions JSON "from any tool". So the merging itself moved here --
``steps/merge_tree.py`` -- while ingest, align, annotate and compile stay with mlex.

The criterion moved with it, and changed. ``mlex merge`` cut an average-linkage tree
over ``1 - ppm_similarity`` at one scalar ``cut_height``. The merge now uses the rule
the established motif-clustering software uses, RSAT ``matrix-clustering``: a node
merges only when the MEAN cross-pair value of EVERY declared arm (Ncor, cor, w, read
at one registration) clears its threshold, and a node that fails breaks the chain for
its ancestors rather than being re-tested. The hyperparameter is therefore a set of
thresholds, declared per merge in config next to the merge it cuts, never defaulted.
Representatives stay medoids -- observed patterns, never cross-model averages -- and
``mlex compile`` still verifies the written lexicon with the real finemo loader.

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


#: The arms of the merge criterion, in report order. Same three RSAT thresholds on.
MERGE_ARMS = ("Ncor", "cor", "w")


def thresholds(cfg: dict, name: str):
    """The merge criterion for a merge, or None: registry and evidence only, no groups.

    Replaces ``cut_height`` (2026-08-25). ``mlex merge``, which cut an average-linkage
    tree over ``1 - ppm_similarity`` at one scalar height, was removed from motiflex on
    2026-08-23; the merging is now done here by ``steps/merge_tree.py`` under the
    criterion RSAT ``matrix-clustering`` uses -- a node merges only if the MEAN
    cross-pair value of EVERY declared arm clears its threshold. So the hyperparameter
    is a set of thresholds, not a height, and a config that still declares ``cut_height``
    is refused rather than reinterpreted: 0.05 on ``1 - ppm_similarity`` has no image
    under the new criterion, and silently mapping it to something would be a made-up
    operating point.
    """
    d = comp.merge_def(cfg, name)
    if "cut_height" in d:
        raise MlexError(
            f"merge {name!r} declares cut_height={d['cut_height']}, which was the "
            f"parameter of the removed `mlex merge` stage. The criterion is now "
            f"thresholds: {{{', '.join(MERGE_ARMS)}}} -- pick an operating point from a "
            f"sweep of merge_nodes.tsv and declare it; it is not derivable from a height.")
    t = d.get("thresholds")
    if t is None:
        return None
    missing = [k for k in MERGE_ARMS if k not in t]
    if missing:
        raise MlexError(
            f"merge {name!r}: thresholds is missing {missing}; every arm is required "
            f"because the criterion is their conjunction -- an absent arm is not 'off', "
            f"it is an undeclared threshold")
    return {k: float(t[k]) for k in MERGE_ARMS}


def db_yaml(cfg: dict, name: str) -> str:
    """The precomputed database ``mlex annotate`` reads for this merge.

    Since 2026-08-25 this defaults to the merge's own naming stage output rather than
    to nothing: the annotate step used to be skipped for every merge, because no merge
    declared a ``db_yaml`` and there was no stage that produced one. It is now built by
    ``rule merge_naming`` from the project's one reference lexicon, so both leaves of a
    merge are labelled from the same library and at the same gate.

    A merge may still override with an explicit path -- an annotation computed
    elsewhere, or a frozen one -- and then that file has to exist; nothing generates it.
    """
    from motifline import naming
    return comp.merge_def(cfg, name).get("db_yaml") or naming.merge_db_yaml(name)


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


def leaf_labels(cfg: dict, name: str) -> list[str]:
    """A merge's leaf labels, in config order (the cross-call fans out over these)."""
    return list(comp.merge_def(cfg, name).get("leaves") or {})


def check_leaf_arrays(name: str, label: str, leaf: dict) -> None:
    """Refuse a declared width the arrays cannot back.

    Called from the stage body, not from ``cross_call_leaf``: config resolution runs at
    DAG-build time and must stay pure, or a dry run would need every archive online.

    ``meta.input_width`` is free text on an ``h5:`` leaf and is handed straight to
    ``finemo extract-regions-modisco-fmt -w``. finemo computes
    ``start = L//2 - w//2`` and slices, so a width larger than the array wraps into a
    short, off-centre region set while ``peak_region_start`` is still computed from the
    declared width -- the scan runs, exits 0, and every recorded coordinate is shifted.
    Same shape of check ``adopt.adopt_attr`` already performs on the adopted path.
    """
    import numpy as np

    attr, ohe, bed, width = leaf["attr"], leaf["ohe"], leaf["bed"], leaf["width"]
    for path in (attr, ohe):
        try:
            arr = np.load(path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise MlexError(f"merge {name!r}: leaf {label!r} cannot read {path}: {exc}") from exc
        if arr.shape[-1] < width:
            raise MlexError(
                f"merge {name!r}: leaf {label!r} declares input_width={width} but "
                f"{path} is {arr.shape[-1]}bp wide; finemo would slice a short, "
                f"off-centre window and still report the declared coordinates")
    rows = np.load(attr, mmap_mode="r").shape[0]
    n_bed = sum(1 for line in open(bed) if line.strip())
    if rows != n_bed:
        raise MlexError(
            f"merge {name!r}: leaf {label!r} has {rows} attribution rows against "
            f"{n_bed} bed lines; rows align by index downstream")


def cross_call_leaf(cfg: dict, name: str, label: str) -> dict:
    """One leaf's arrays for cross-calling: attr/ohe/bed plus the region width.

    Same resolution as ingest: a ``run:`` leaf reads this pipeline's own attr/
    stage at the run's declared width, an ``h5:`` leaf reads exactly what it
    declared. The bed rows are the alignment key across leaves downstream, so
    the bed returned here must be the one the arrays were computed over.
    """
    spec = comp.merge_def(cfg, name).get("leaves") or {}
    if label not in spec:
        raise MlexError(f"merge {name!r} has no leaf {label!r}; it defines {sorted(spec)}")
    src = spec[label]
    if isinstance(src, dict) and "run" in src:
        r = (cfg.get("runs") or {}).get(src["run"])
        if r is None:
            raise MlexError(f"merge {name!r}: leaf {label!r} names unknown run {src['run']!r}")
        base = f"results/{src['run']}/attr"
        return {"attr": f"{base}/attr.npy", "ohe": f"{base}/ohe.npy",
                "bed": f"{base}/regions.bed", "width": attr_width(cfg, r)}
    a = _h5_analysis(cfg, name, label, src)
    return {"attr": a["contributions"], "ohe": a["one_hot"],
            "bed": a["regions_bed"], "width": a["input_width"]}


def cross_call_inputs(cfg: dict, name: str, label: str) -> list[str]:
    """Snakemake inputs for one leaf's cross-call regions: run-backed arrays only.

    Same policy as ``compendium.dag_inputs``: an ``h5:`` leaf's arrays were not
    produced by this pipeline, so declaring them as inputs would make an absent
    archive look like a missing rule output.
    """
    src = (comp.merge_def(cfg, name).get("leaves") or {}).get(label) or {}
    if isinstance(src, dict) and "run" in src:
        base = f"results/{src['run']}/attr"
        return [f"{base}/attr.npy", f"{base}/ohe.npy", f"{base}/regions.bed"]
    return []


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
        # motiflex distinguishes None ("independence unknown") from [] ("checked,
        # nothing shared"). Emitting [] unconditionally asserted a check nobody ran --
        # and two leaves reading one archived attr.npy is constructible from this very
        # config. Declare `shared_attribution: [[a, b]]` next to `leaves:` to state it.
        "shared_attribution_groups": comp.merge_def(cfg, name).get("shared_attribution"),
        "provenance": {"random_seed": int(params(cfg).get("seed", 42))},
    }
