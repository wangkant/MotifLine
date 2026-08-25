# Snakefile
import json
import os
import subprocess
import sys
from pathlib import Path

configfile: "config/default.yaml"

from motifline import compendium as comp
from motifline import mlex as mlx
from motifline import naming as nm
from motifline.geometry import attr_width, geometry
from motifline.lexicons import lexicon_batch, lexicon_path
from motifline.runs import active_runs, cancelled_runs

ENV = config["envs"]["compute"]
PY = f"{ENV}/bin/python"
ENV_ALT = config["envs"]["finemo_alt"]                 # second finemo, comparison only
STEPS = Path(workflow.basedir) / "src/motifline/steps"

def rundef(w):
    return config["runs"][w.run]

def param_sig(**kv):
    """A stable rendering of the scientific parameters a stage runs at.

    Declared as ``params:`` so that ``--rerun-triggers mtime params`` invalidates a
    stage when a value it uses changes in config. Without it, editing
    ``finemo.threshold``, ``modisco.window`` or a run's ``track`` left every existing
    output in place and snakemake reported "Nothing to be done" -- results/ could then
    hold a mixture of parameter generations indistinguishable from a coherent tree.
    The value is never read by a stage body; only its identity matters.
    """
    return json.dumps(kv, sort_keys=True, default=str)


def sig_import_attr(w):
    r = rundef(w)
    return param_sig(src=r["adopt_from"], rows=r["rows"], width=attr_width(config, r),
                     peaks=r.get("adopt_peaks", "peaks.bed"))


def sig_naming(w=None):
    return param_sig(gate=nm.gate(config), tomtom=nm.params(config)["tomtom"],
                     w_th=nm.w_th(config))


def sig_merge_naming(w):
    return param_sig(gate=nm.gate(config), tomtom=nm.params(config)["tomtom"],
                     w_th=nm.w_th(config),
                     leaves=[l for l, _ in nm.merge_leaf_h5(config, w.merge)])


def sig_merge_ingest(w):
    return param_sig(project=mlx.project(config, w.merge),
                     seed=mlx.params(config).get("seed", 42))


def sig_merge_align(w=None):
    c = mlx.params(config)
    return param_sig(workers=c.get("workers", 8), seed=c.get("seed", 42))


def sig_cross_regions(w):
    return param_sig(width=mlx.cross_call_leaf(config, w.merge, w.leaf)["width"])


def sig_compendium(w):
    return param_sig(compendium=comp.params(config),
                     region_width=comp.region_width(config, w.merge))


def run_body(out_dir, argv, env=None):
    """Make the stage directory and run the stage body in it."""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([str(a) for a in argv], check=True,
                   env={**os.environ, **env} if env else None)

# One shard per GPU with 8 threads was the measured optimum for AG (run_ag_par_hyp.sh:14-16);
# without a cap the process oversubscribes the node's cores.
# NUMBA_NUM_THREADS is in the list because MoDISco's hot loops are numba, not BLAS:
# without it numba takes all 80 cores per process and four concurrent discover jobs
# drive the load average past 200.
COMPUTE_THREADS = {k: "8" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS",
                                    "NUMBA_NUM_THREADS")}

def attr_argv(r, out, shard_i=None):
    """Model-dispatched attribution command line; both steps write attr/ohe/regions.bed
    (``attr_<i>.npy``/... when a shard index is given)."""
    g = geometry(config, r)
    att = config["attribution"]
    n_shuffles = str(r.get("n_shuffles", att["n_shuffles"]))
    random_state = str(r.get("random_state", att["random_state"]))
    if r["model"] == "ag":
        argv = [PY, str(STEPS / "attr_ag.py"),
                "--ag_src", r["ag_src"], "--chrombpnet_src", config["chrombpnet_src"],
                "--weights", r["weights"], "--peaks", r["peaks"],
                "--fasta", config["genome"]["fasta"],
                "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out, "--track", str(r["track"]),
                "--in_window", str(g["in_window"]), "--crop", str(g["crop"]),
                "--n_shuffles", n_shuffles, "--chunk", str(r["chunk"]),
                "--random_state", random_state, "--hypothetical"]
    elif r["model"] == "cbp":
        argv = [PY, str(STEPS / "attr_cbp.py"), "--chrombpnet_src", config["chrombpnet_src"],
                "--wo_bias", r["wo_bias"], "--peaks", r["peaks"],
                "--fasta", config["genome"]["fasta"],
                "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out, "--hypothetical", "--in_window", str(g["in_window"]),
                "--n_shuffles", n_shuffles, "--random_state", random_state]
    else:
        raise ValueError(f"unknown model {r['model']!r}; expected 'cbp' or 'ag'")
    if r.get("head"):
        argv += ["--head", str(r["head"])]
    if shard_i is not None:
        argv += ["--shard_i", str(shard_i), "--n_shards", str(r["n_shards"])]
    return argv

ORCH_PY = sys.executable  # the python running snakemake == the motifline orchestration env

# GPU selection is the caller's: export CUDA_VISIBLE_DEVICES=<uuid> before running,
# and bound concurrency with --resources gpu=<G>. House rules stay the same --
# fully-free cards only, pin by UUID (indices shift), leave one working card free.

# An adopted run links an archived attr/ohe pair in instead of computing one; the two
# producers write the same three files, so the run names are partitioned between them
# rather than left to snakemake to disambiguate.
ADOPTED = sorted(k for k, v in config["runs"].items() if v.get("adopt_from"))
COMPUTED = sorted(k for k, v in config["runs"].items() if not v.get("adopt_from"))
# ... and computed runs are split again: a full-scale run attributes in shards
# (one GPU each) and merges, a slice run attributes in one process.
SHARDED = sorted(k for k in COMPUTED if config["runs"][k].get("n_shards"))
SINGLE = sorted(k for k in COMPUTED if not config["runs"][k].get("n_shards"))
# A cancelled run keeps every rule it always had -- it stays resumable by explicit
# target -- but drops out of the default target list, so a bare `snakemake` cannot
# restart work somebody stopped on purpose. See motifline/src/motifline/runs.py and
# the run's own CANCELLED.md (config: cancelled_note).
CANCELLED = cancelled_runs(config)
LEXICON_RUNS = sorted(k for k, v in config["runs"].items() if v.get("lexicon_hits"))
CROSS_VERSION = sorted(k for k, v in config["runs"].items() if v.get("cross_version_hits"))
LEXICON_ROUTES = sorted((config.get("lexicons") or {}).get("routes") or {})
# Cross-model merges: two MoDISco h5 (CBP 2114 + AG 2048) -> one merged lexicon.
# The finish is motiflex (2026-08-17): every merge gets an annotated tree; the ones
# that declare `thresholds` also get a compiled lexicon. MotifCompendium stays as
# the explicit-target comparison arm (rule merge_compendium).
MERGES = comp.merge_names(config)
# A merge that declares `thresholds` gets merged and compiled; one that does not stops
# at the registry and the alignment evidence. (Before 2026-08-25 the hyperparameter was
# `cut_height`, applied by the `mlex merge` stage motiflex has since removed.)
MERGE_GATED = sorted(m for m in MERGES if mlx.thresholds(config, m) is not None)
# A compiled merge is cross-called back over every leaf: the SAME lexicon scanned
# over each model's own regions (rule merge_cross_call), then reduced to per-motif
# occupancy / cross-leaf agreement (rule merge_occupancy).
MERGE_LEAVES = sorted({l for m in MERGE_GATED for l in mlx.leaf_labels(config, m)})


def _alt(names):
    return "|".join(names) if names else "$^"


def shard_dirs(run):
    n = config["runs"][run]["n_shards"]
    return [f"results/{run}/attr_shards/shard_{i}" for i in range(n)]


def shard_outputs(w):
    return [f"{d}/attr_{i}.npy" for i, d in enumerate(shard_dirs(w.run))]

rule all:
    input:
        expand("results/{run}/hits/hits.tsv",
               run=active_runs(config, sorted(config["runs"]))),
        # every run's patterns named from the one reference lexicon (2026-08-25)
        expand("results/{run}/naming/pattern_names.tsv",
               run=active_runs(config, sorted(config["runs"]))),
        # finemo 0.30 vs 0.41 on one and the same regions.npz + modisco.h5
        expand("results/{run}/hits_v041/hits.tsv",
               run=active_runs(config, CROSS_VERSION)),
        # external-lexicon routes
        expand("results/{run}/hits_lexicon_{route}/hits.tsv",
               run=active_runs(config, LEXICON_RUNS), route=LEXICON_ROUTES),
        # cross-model merges: every merge builds its annotated registry; a merge that
        # declares `thresholds` is also merged and compiled into one shared lexicon,
        # which is cross-called over every leaf and reduced to occupancy/agreement.
        expand("results/merges/{merge}/motiflex/evidence/annotation_candidates.parquet",
               merge=MERGES),
        expand("results/merges/{merge}/motiflex/merge/merge_groups.tsv", merge=MERGE_GATED),
        expand("results/merges/{merge}/motiflex/lexicons/core.h5", merge=MERGE_GATED),
        expand("results/merges/{merge}/motiflex/occupancy/agreement.tsv", merge=MERGE_GATED)

rule import_attr:
    """Adopt archived attributions: symlink attr/ohe, normalize peaks -> regions.bed."""
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(ADOPTED)
    params: sig=sig_import_attr
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        src = r["adopt_from"]
        argv = [ORCH_PY, "-m", "motifline.adopt", "--src_dir", src, "--out_dir", out,
                "--expected_rows", str(r["rows"]), "--expected_width", str(attr_width(config, r)),
                "--peaks_name", r.get("adopt_peaks", "peaks.bed")]
        run_body(out, argv)

rule attribute:
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(SINGLE)
    resources: gpu=1
    params: sig=lambda w: param_sig(argv=attr_argv(rundef(w), "<out>"))
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        run_body(out, attr_argv(r, out), env=COMPUTE_THREADS)

rule attribute_sharded:
    """One shard of a full-scale attribution: its own stage dir, its own card."""
    output: "results/{run}/attr_shards/shard_{i}/attr_{i}.npy",
            "results/{run}/attr_shards/shard_{i}/ohe_{i}.npy",
            "results/{run}/attr_shards/shard_{i}/regions_{i}.bed"
    wildcard_constraints: run=_alt(SHARDED), i=r"\d+"
    resources: gpu=1
    params: sig=lambda w: param_sig(argv=attr_argv(rundef(w), "<out>", shard_i=int(w.i)))
    run:
        r = rundef(wildcards)
        i = int(wildcards.i)
        out = f"results/{wildcards.run}/attr_shards/shard_{i}"
        run_body(out, attr_argv(r, out, shard_i=i), env=COMPUTE_THREADS)

rule merge_shards:
    """Concatenate the shards in shard order into the one attr/ohe/regions triple
    the rest of the DAG consumes."""
    input: shard_outputs
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(SHARDED)
    params: sig=lambda w: param_sig(n_shards=rundef(w)["n_shards"], rows=rundef(w).get("rows"))
    run:
        r = rundef(wildcards)
        dirs = shard_dirs(wildcards.run)
        out = f"results/{wildcards.run}/attr"
        argv = [ORCH_PY, "-m", "motifline.merge_shards", "--out_dir", out]
        for d in dirs:
            argv += ["--shard_dir", d]
        if r.get("rows"):
            argv += ["--expected_rows", str(r["rows"])]
        run_body(out, argv)

ATTR_FILES = ["results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy",
              "results/{run}/attr/regions.bed"]

rule discover:
    input: ATTR_FILES
    output: "results/{run}/modisco/modisco.h5"
    params: sig=param_sig(modisco=config["modisco"])
    run:
        attr_dir = f"results/{wildcards.run}/attr"
        out = f"results/{wildcards.run}/modisco"
        argv = [f"{ENV}/bin/modisco", "motifs", "-s", f"{attr_dir}/ohe.npy",
                "-a", f"{attr_dir}/attr.npy", "-n", str(config["modisco"]["max_seqlets"]),
                "-w", str(config["modisco"]["window"]), "-o", f"{out}/modisco.h5"]
        run_body(out, argv, env=COMPUTE_THREADS)

rule extract_regions:
    input: ATTR_FILES
    output: "results/{run}/finemo_regions/regions.npz"
    params: sig=lambda w: param_sig(width=attr_width(config, rundef(w)))
    run:
        r = rundef(wildcards)
        attr_dir = f"results/{wildcards.run}/attr"
        out = f"results/{wildcards.run}/finemo_regions"
        argv = [f"{ENV}/bin/finemo", "extract-regions-modisco-fmt",
                "-s", f"{attr_dir}/ohe.npy", "-a", f"{attr_dir}/attr.npy",
                "-p", f"{attr_dir}/regions.bed", "-o", f"{out}/regions.npz",
                "-w", str(attr_width(config, r))]
        run_body(out, argv)

rule call_hits:
    input: npz=rules.extract_regions.output, lex=rules.discover.output
    output: "results/{run}/hits/hits.tsv"
    resources: gpu=1
    params: sig=param_sig(finemo=config["finemo"], env=ENV)
    run:
        reg_dir = f"results/{wildcards.run}/finemo_regions"
        mod_dir = f"results/{wildcards.run}/modisco"
        out = f"results/{wildcards.run}/hits"
        f = config["finemo"]
        argv = [f"{ENV}/bin/finemo", "call-hits", "-M", f["mode"],
                "-r", f"{reg_dir}/regions.npz", "-m", f"{mod_dir}/modisco.h5",
                "-o", out, "-t", str(f["threshold"]), "-l", str(f["alpha"]),
                "-b", str(f["batch"])]
        run_body(out, argv)

rule call_hits_v041:
    """The same regions.npz and modisco.h5, called by finemo 0.41 instead of 0.30.

    Nothing downstream consumes this directory -- it exists to be compared.
    """
    input: npz=rules.extract_regions.output, lex=rules.discover.output
    output: "results/{run}/hits_v041/hits.tsv"
    wildcard_constraints: run=_alt(CROSS_VERSION)
    resources: gpu=1
    params: sig=param_sig(finemo=config["finemo"], env=ENV_ALT)
    run:
        reg_dir = f"results/{wildcards.run}/finemo_regions"
        mod_dir = f"results/{wildcards.run}/modisco"
        out = f"results/{wildcards.run}/hits_v041"
        f = config["finemo"]
        argv = [f"{ENV_ALT}/bin/finemo", "call-hits", "-M", f["mode"],
                "-r", f"{reg_dir}/regions.npz", "-m", f"{mod_dir}/modisco.h5",
                "-o", out, "-t", str(f["threshold"]), "-l", str(f["alpha"]),
                "-b", str(f["batch"])]
        # Note: this env's torch cannot reach a GPU on this node (torch 2.13.0+cu130
        # vs driver 555.42.06) -- finemo warns and runs on the CPU. Slower, same hits.
        run_body(out, argv)

rule call_hits_lexicon:
    """Hit calling against an EXTERNAL lexicon (run_finemo.sh's gw/mmv routes).

    `-m` is the only thing the route changes, exactly as in the shell original;
    the run's own modisco.h5 is deliberately not an input, so this rule can score
    a context against a vocabulary that context did not discover. `-b` comes from
    config.lexicons, not config.finemo: the union lexicon needs the smaller batch.
    """
    input: npz="results/{run}/finemo_regions/regions.npz",
           lex=lambda w: lexicon_path(config, rundef(w), w.route)
    output: "results/{run}/hits_lexicon_{route}/hits.tsv"
    wildcard_constraints: run=_alt(LEXICON_RUNS), route=_alt(LEXICON_ROUTES)
    resources: gpu=1
    params: sig=lambda w: param_sig(finemo=config["finemo"], batch=lexicon_batch(config))
    run:
        r = rundef(wildcards)
        reg_dir = f"results/{wildcards.run}/finemo_regions"
        out = f"results/{wildcards.run}/hits_lexicon_{wildcards.route}"
        lex = input.lex
        batch = lexicon_batch(config)
        f = config["finemo"]
        argv = [f"{ENV}/bin/finemo", "call-hits", "-M", f["mode"],
                "-r", f"{reg_dir}/regions.npz", "-m", lex,
                "-o", out, "-t", str(f["threshold"]), "-l", str(f["alpha"]),
                "-b", str(batch)]
        run_body(out, argv)

# ---------------------------------------------------------------------------
# Reference lexicon and pattern naming (2026-08-25). One library, built once, for
# every run and every merge in this pipeline; rationale in src/motifline/naming.py.
# The RSAT clustering is the expensive step -- 1h46m on 2614 matrices, measured
# 2026-08-25 -- and it is the only one; everything downstream is minutes and is cached
# by snakemake as usual.
# ---------------------------------------------------------------------------

rule lexicon_union:
    """Source libraries -> one union MEME, matrices untouched (E= 0 added for RSAT)."""
    input: lambda w: nm.source_paths(config)
    output: nm.UNION
    params: sig=param_sig(sources=list(nm.params(config)["sources"]))
    run:
        argv = [PY, str(STEPS / "lexicon_union.py"), *nm.source_argv(config),
                "--out", output[0]]
        run_body(Path(output[0]).parent, argv)

rule lexicon_rsat:
    """RSAT matrix-clustering over the union: THE partition, computed by RSAT itself.

    The thresholds are passed explicitly (naming.rsat_argv explains why) and are the
    web server's documented defaults; standalone leaves them empty, which is not the
    same clustering. Only rsat_tables/clusters.tab is consumed downstream -- the rest
    of RSAT's output tree is kept where it lands, as the provenance for that file.
    """
    input: nm.UNION
    output: nm.CLUSTERS_TAB
    params: sig=param_sig(argv=nm.rsat_argv(config, "<union>", "<out>"))
    run:
        out = nm.RSAT_DIR
        os.makedirs(out, exist_ok=True)
        rsat_bin = Path(nm.rsat(config)["bin"]).parent
        run_body(out, nm.rsat_argv(config, input[0], f"{out}/rsat"),
                 env={"PATH": f"{rsat_bin}:{os.environ['PATH']}"})
        # The launcher cannot report failure (see naming.rsat_argv), and snakemake only
        # checks that clusters.tab exists -- which it would, left over from a previous
        # run at different thresholds. Check the partition covers the union it was given.
        n_union = sum(1 for l in open(input[0]) if l.startswith("MOTIF "))
        n_clustered = sum(len(l.rstrip("\n").split("\t")[1].split(","))
                          for l in open(output[0]))
        if n_clustered != n_union:
            raise ValueError(
                f"RSAT clustered {n_clustered} of {n_union} union motifs; the launcher "
                f"exits 0 whatever happens, so this is the only place a failed or stale "
                f"clustering is caught. See {out}/rsat_log.txt.")

rule lexicon_medoid:
    """One medoid per RSAT cluster -> the compact reference library (+ the Ncor cache)."""
    input: tab=nm.CLUSTERS_TAB, sources=lambda w: nm.source_paths(config)
    output: nm.MEDOID_MEME, nm.METADATA, nm.CLUSTERS, nm.NCOR
    params: sig=param_sig(w_th=nm.w_th(config), sources=list(nm.params(config)["sources"]))  # noqa: E501
    run:
        argv = [PY, str(STEPS / "lexicon_medoid.py"), "--clusters_tab", input.tab,
                *nm.source_argv(config), "--w_th", str(nm.w_th(config)),
                "--out_dir", nm.LEXICON_DIR]
        run_body(nm.LEXICON_DIR, argv, env=COMPUTE_THREADS)

rule lexicon_cltagged:
    """The union with cluster ids prefixed onto the names: the TomTom target."""
    input: union=nm.UNION, clusters=nm.CLUSTERS, metadata=nm.METADATA
    output: nm.CLTAGGED
    run:
        argv = [PY, str(STEPS / "lexicon_cltagged.py"), "--union", input.union,
                "--clusters", input.clusters, "--metadata", input.metadata,
                "--out", output[0]]
        run_body(nm.LEXICON_DIR, argv)

rule name_patterns:
    """One run's MoDISco patterns -> cluster-level names, with the gate applied."""
    input: h5="results/{run}/modisco/modisco.h5",
           lex=lambda w: nm.naming_inputs(config)
    output: nm.run_names("{run}")
    params: sig=sig_naming
    # constrained to declared run names so that a merge's naming directory, which has
    # the same shape, can only ever be built by rule merge_naming
    wildcard_constraints: run=_alt(sorted(config["runs"]))
    run:
        q, ncor = nm.gate(config)
        out = f"results/{wildcards.run}/naming"
        argv = [PY, str(STEPS / "name_patterns.py"),
                "--h5", f"{wildcards.run}={input.h5}",
                "--target", nm.CLTAGGED, *nm.source_argv(config),
                "--tomtom", nm.params(config)["tomtom"],
                "--w_th", str(nm.w_th(config)),
                "--gate_q", str(q), "--gate_ncor", str(ncor), "--out_dir", out]
        run_body(out, argv, env=COMPUTE_THREADS)

rule merge_naming:
    """Every leaf of a merge named from the same library, and the mlex annotate database.

    One stage over all leaves rather than one per leaf: `mlex annotate` reads a single
    database file whose node ids span the merge, and running the leaves together is what
    guarantees they were gated identically.
    """
    input: h5=lambda w: comp.dag_inputs(config, w.merge),
           lex=lambda w: nm.naming_inputs(config)
    output: nm.merge_names("{merge}"), nm.merge_db_yaml("{merge}")
    wildcard_constraints: merge=_alt(MERGES)
    params: sig=sig_merge_naming
    run:
        q, ncor = nm.gate(config)
        merge = wildcards.merge
        out = f"results/merges/{merge}/naming"
        argv = [PY, str(STEPS / "name_patterns.py"),
                "--target", nm.CLTAGGED, *nm.source_argv(config),
                "--tomtom", nm.params(config)["tomtom"],
                "--w_th", str(nm.w_th(config)),
                "--gate_q", str(q), "--gate_ncor", str(ncor),
                "--out_dir", out, "--db_yaml", nm.merge_db_yaml(merge)]
        for label, h5 in nm.merge_leaf_h5(config, merge):
            argv += ["--h5", f"{label}={h5}",
                     "--node_id", f"{label}={nm.node_prefix(merge, label)}"]
        run_body(out, argv, env=COMPUTE_THREADS)

rule merge_compendium:
    """Two or more MoDISco h5 -> ONE MotifCompendium lexicon (comparison arm).

    No longer the end of the line: the DAG finishes merges with motiflex
    (rule merge_motiflex, 2026-08-17). This rule stays reachable by explicit
    target -- `snakemake results/merges/<m>/clusters.tsv` -- for the MC arm.

    The leaves of a merge are the SAME regions under different models, each attributed
    at its own native width (CBP 2114, AG 2048), so the only axis the clustering can
    collapse is the model axis. Which entries the models actually share is not something
    MotifCompendium reports -- it clusters on CWM similarity and never reads the `model`
    column -- so the step recovers it into clusters.tsv.

    Runs in envs/motifcompendium, not the compute env: MotifCompendium pins its own
    numpy/pandas/igraph/leidenalg stack and nothing else in this DAG needs it.
    """
    input: lambda w: comp.dag_inputs(config, w.merge)
    output: "results/merges/{merge}/clusters.tsv", "results/merges/{merge}/motifs.tsv",
            "results/merges/{merge}/merged.meme", "results/merges/{merge}/merged.mc"
    wildcard_constraints: merge=_alt(MERGES)
    params: sig=sig_compendium
    run:
        c = comp.params(config)
        out = f"results/merges/{wildcards.merge}"
        argv = [f"{c['env']}/bin/python", str(STEPS / "merge_compendium.py"),
                "--mc_src", c["src"], "--out_dir", out,
                "--region_width", str(comp.region_width(config, wildcards.merge)),
                "--algorithm", c["algorithm"],
                "--similarity_threshold", str(c["similarity_threshold"]),
                "--seeds", ",".join(str(s) for s in c["seeds"]),
                "--max_cpus", str(c.get("max_cpus", 1))]
        if c.get("use_gpu"):
            argv += ["--use_gpu"]
        if c.get("cluster_within_posneg") is False:
            argv += ["--no_cluster_within_posneg"]
        if c.get("quality_stats") is False:
            argv += ["--no_quality_stats"]
        for label, h5 in comp.leaves(config, wildcards.merge):
            argv += ["--modisco", f"{label}={h5}"]
        run_body(out, argv, env=COMPUTE_THREADS)

MOTIFLEX = "results/merges/{merge}/motiflex"

rule merge_ingest:
    """N MoDISco h5 -> the normalised motif registry (`mlex ingest`).

    Split out of the old single merge rule on 2026-08-25 so that each mlex stage is a
    real snakemake output. The previous body guarded each stage on `.exists()` against
    files snakemake did not track, which meant a re-generated db.yaml or project.yaml
    rebuilt the tree from a stale registry and stale annotations.
    """
    input: h5=lambda w: comp.dag_inputs(config, w.merge)
    output: registry=f"{MOTIFLEX}/registry/registry.json",
            project=f"{MOTIFLEX}/project.yaml"
    wildcard_constraints: merge=_alt(MERGES)
    params: sig=sig_merge_ingest
    run:
        import yaml
        c = mlx.params(config)
        out = Path(MOTIFLEX.format(merge=wildcards.merge))
        out.mkdir(parents=True, exist_ok=True)
        Path(output.project).write_text(
            yaml.safe_dump(mlx.project(config, wildcards.merge), sort_keys=False, width=200))
        run_body(out, [c["mlex"], "ingest", output.project,
                       "--seed", c.get("seed", 42), "--out", out / "registry"])

rule merge_align:
    """Pairwise alignment evidence over the registry (`mlex align`)."""
    input: f"{MOTIFLEX}/registry/registry.json"
    output: f"{MOTIFLEX}/evidence/alignment_edges.parquet"
    wildcard_constraints: merge=_alt(MERGES)
    params: sig=sig_merge_align
    run:
        c = mlx.params(config)
        out = Path(MOTIFLEX.format(merge=wildcards.merge))
        run_body(out, [c["mlex"], "align", out / "registry", "--out", out / "evidence",
                       "--workers", c.get("workers", 8), "--seed", c.get("seed", 42)])

rule merge_annotate:
    """Attach the reference lexicon's labels to the registry (`mlex annotate`).

    The database is this pipeline's own naming stage output (rule merge_naming), so
    both leaves are labelled from one library at one gate.
    """
    input: registry=f"{MOTIFLEX}/registry/registry.json",
           evidence=f"{MOTIFLEX}/evidence/alignment_edges.parquet",
           db=lambda w: mlx.db_yaml(config, w.merge)
    output: f"{MOTIFLEX}/evidence/annotation_candidates.parquet"
    wildcard_constraints: merge=_alt(MERGES)
    run:
        c = mlx.params(config)
        out = Path(MOTIFLEX.format(merge=wildcards.merge))
        run_body(out, [c["mlex"], "annotate", out / "evidence",
                       "--registry", out / "registry",
                       "--tomtom", "--databases", input.db, "--out", out / "evidence"])

rule merge_tree:
    """The merge itself: RSAT matrix-clustering's criterion over the registry.

    This is the stage motiflex removed on 2026-08-23 (commit 204cb3f) when it
    repositioned as an auditor of someone else's consolidation -- `mlex compile
    --decisions` takes a decisions JSON "from any tool", and this is that tool.

    A node merges only when the MEAN cross-pair value of EVERY arm (Ncor, cor, w, all
    read at one registration) clears its threshold, and a failed node breaks the chain
    for its ancestors: `TFBMclust/R/check_alignment.R:80-81,106-109` and
    `align.motifs.R:101`. The thresholds are declared per merge and never defaulted,
    chosen by reading merge/merge_nodes.tsv -- the same policy the old `cut_height` had.
    """
    input: registry=f"{MOTIFLEX}/registry/registry.json",
           annotation=f"{MOTIFLEX}/evidence/annotation_candidates.parquet"
    output: decisions=f"{MOTIFLEX}/merge/merge_decisions.json",
            groups=f"{MOTIFLEX}/merge/merge_groups.tsv",
            nodes=f"{MOTIFLEX}/merge/merge_nodes.tsv"
    wildcard_constraints: merge=_alt(MERGE_GATED)
    params: sig=lambda w: param_sig(thresholds=mlx.thresholds(config, w.merge))
    run:
        out = Path(MOTIFLEX.format(merge=wildcards.merge))
        t = mlx.thresholds(config, wildcards.merge)
        argv = [PY, str(STEPS / "merge_tree.py"), "--registry", out / "registry",
                "--out_dir", out / "merge"]
        for arm, value in t.items():
            argv += [f"--{arm}", str(value)]
        run_body(out / "merge", argv, env=COMPUTE_THREADS)

rule merge_compile:
    """Compile the merged groups into one finemo-callable lexicon (`mlex compile`).

    Group representatives are medoids -- observed patterns, never cross-model averages
    -- and compile reads the written lexicon back with the real finemo loader
    (`--verify-roundtrip auto`).
    """
    input: registry=f"{MOTIFLEX}/registry/registry.json",
           decisions=f"{MOTIFLEX}/merge/merge_decisions.json"
    output: f"{MOTIFLEX}/lexicons/core.h5",
            f"{MOTIFLEX}/lexicons/manifest.tsv"
    wildcard_constraints: merge=_alt(MERGE_GATED)
    params: sig=param_sig(tiers="core", seed=mlx.params(config).get("seed", 42))
    run:
        c = mlx.params(config)
        out = Path(MOTIFLEX.format(merge=wildcards.merge))
        run_body(out, [c["mlex"], "compile", out / "registry",
                       "--decisions", input.decisions,
                       "--tiers", "core", "--verify-roundtrip", "auto",
                       "--seed", c.get("seed", 42), "--out", out / "lexicons"])

rule merge_cross_regions:
    """Package one leaf's arrays as finemo regions for the cross-call.

    The same peak rows under each model, each at its own native width; row index
    is the alignment key downstream (merge_occupancy verifies the beds actually
    agree before relying on it).
    """
    input: lambda w: mlx.cross_call_inputs(config, w.merge, w.leaf)
    output: "results/merges/{merge}/motiflex/cross_call/{leaf}/regions.npz"
    wildcard_constraints: merge=_alt(MERGE_GATED), leaf=_alt(MERGE_LEAVES)
    params: sig=sig_cross_regions
    run:
        l = mlx.cross_call_leaf(config, wildcards.merge, wildcards.leaf)
        mlx.check_leaf_arrays(wildcards.merge, wildcards.leaf, l)
        out = f"results/merges/{wildcards.merge}/motiflex/cross_call/{wildcards.leaf}"
        argv = [f"{ENV}/bin/finemo", "extract-regions-modisco-fmt",
                "-s", l["ohe"], "-a", l["attr"], "-p", l["bed"],
                "-o", f"{out}/regions.npz", "-w", str(l["width"])]
        run_body(out, argv)

rule merge_cross_call:
    """Every leaf scanned with the ONE compiled lexicon (finemo call-hits).

    This is what compiling is for: both models are read through an identical
    vocabulary, so a difference between the hit tables is a model difference,
    not a lexicon difference. `-b` comes from config.lexicons for the same
    reason as rule call_hits_lexicon: batch size must not vary between
    external-lexicon calls.
    """
    input: npz="results/merges/{merge}/motiflex/cross_call/{leaf}/regions.npz",
           lex="results/merges/{merge}/motiflex/lexicons/core.h5"
    output: "results/merges/{merge}/motiflex/cross_call/{leaf}/hits.tsv"
    wildcard_constraints: merge=_alt(MERGE_GATED), leaf=_alt(MERGE_LEAVES)
    resources: gpu=1
    params: sig=param_sig(finemo=config["finemo"], batch=lexicon_batch(config))
    run:
        out = f"results/merges/{wildcards.merge}/motiflex/cross_call/{wildcards.leaf}"
        f = config["finemo"]
        argv = [f"{ENV}/bin/finemo", "call-hits", "-M", f["mode"],
                "-r", input.npz, "-m", input.lex, "-o", out,
                "-t", str(f["threshold"]), "-l", str(f["alpha"]),
                "-b", str(lexicon_batch(config))]
        run_body(out, argv)

rule merge_occupancy:
    """Cross-call hit tables -> per-motif occupancy and cross-leaf agreement.

    Counting caliber lives in steps/merge_occupancy.py's docstring; the step
    refuses if the leaves' beds differ anywhere, because rows align by index.
    """
    input:
        hits=lambda w: [f"results/merges/{w.merge}/motiflex/cross_call/{l}/hits.tsv"
                        for l in mlx.leaf_labels(config, w.merge)],
        manifest="results/merges/{merge}/motiflex/lexicons/manifest.tsv"
    output: "results/merges/{merge}/motiflex/occupancy/occupancy.tsv",
            "results/merges/{merge}/motiflex/occupancy/agreement.tsv"
    wildcard_constraints: merge=_alt(MERGE_GATED)
    run:
        out = f"results/merges/{wildcards.merge}/motiflex/occupancy"
        argv = [PY, str(STEPS / "merge_occupancy.py"),
                "--manifest", input.manifest, "--out_dir", out]
        for label in mlx.leaf_labels(config, wildcards.merge):
            l = mlx.cross_call_leaf(config, wildcards.merge, label)
            argv += ["--hits",
                     f"{label}=results/merges/{wildcards.merge}/motiflex/cross_call/{label}/hits.tsv",
                     "--bed", f"{label}={l['bed']}"]
        run_body(out, argv)
