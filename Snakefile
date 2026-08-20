# Snakefile
import os
import subprocess
import sys
from pathlib import Path

configfile: "config/default.yaml"

from motifline import compendium as comp
from motifline import mlex as mlx
from motifline.geometry import attr_width, geometry
from motifline.lexicons import lexicon_batch, lexicon_path
from motifline.runs import active_runs, cancelled_runs

ENV = config["envs"]["compute"]
PY = f"{ENV}/bin/python"
ENV_ALT = config["envs"]["finemo_alt"]                 # second finemo, comparison only
STEPS = Path(workflow.basedir) / "src/motifline/steps"

def rundef(w):
    return config["runs"][w.run]

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
    if r["model"] == "ag":
        argv = [PY, str(STEPS / "attr_ag.py"),
                "--ag_src", r["ag_src"], "--chrombpnet_src", config["chrombpnet_src"],
                "--weights", r["weights"], "--peaks", r["peaks"],
                "--fasta", config["genome"]["fasta"],
                "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out, "--track", str(r["track"]),
                "--in_window", str(g["in_window"]), "--crop", str(g["crop"]),
                "--n_shuffles", str(r["n_shuffles"]), "--chunk", str(r["chunk"]),
                "--random_state", "42", "--hypothetical"]
    elif r["model"] == "cbp":
        argv = [PY, str(STEPS / "attr_cbp.py"), "--chrombpnet_src", config["chrombpnet_src"],
                "--wo_bias", r["wo_bias"], "--peaks", r["peaks"],
                "--fasta", config["genome"]["fasta"],
                "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out, "--hypothetical", "--in_window", str(g["in_window"])]
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
# that declare a cut_height also get a compiled lexicon. MotifCompendium stays as
# the explicit-target comparison arm (rule merge_compendium).
MERGES = comp.merge_names(config)
MERGE_CUT = sorted(m for m in MERGES if mlx.cut_height(config, m) is not None)
# A compiled merge is cross-called back over every leaf: the SAME lexicon scanned
# over each model's own regions (rule merge_cross_call), then reduced to per-motif
# occupancy / cross-leaf agreement (rule merge_occupancy).
MERGE_LEAVES = sorted({l for m in MERGE_CUT for l in mlx.leaf_labels(config, m)})


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
        # finemo 0.30 vs 0.41 on one and the same regions.npz + modisco.h5
        expand("results/{run}/hits_v041/hits.tsv",
               run=active_runs(config, CROSS_VERSION)),
        # external-lexicon routes
        expand("results/{run}/hits_lexicon_{route}/hits.tsv",
               run=active_runs(config, LEXICON_RUNS), route=LEXICON_ROUTES),
        # cross-model merges: every merge builds its motiflex tree; a merge with
        # cut_height also compiles a shared lexicon, which is then cross-called
        # over every leaf and reduced to occupancy/agreement tables.
        expand("results/merges/{merge}/motiflex/merge/tree.txt", merge=MERGES),
        expand("results/merges/{merge}/motiflex/lexicons/core.h5", merge=MERGE_CUT),
        expand("results/merges/{merge}/motiflex/occupancy/agreement.tsv", merge=MERGE_CUT)

rule import_attr:
    """Adopt archived attributions: symlink attr/ohe, normalize peaks -> regions.bed."""
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(ADOPTED)
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
    input: npz="results/{run}/finemo_regions/regions.npz"
    output: "results/{run}/hits_lexicon_{route}/hits.tsv"
    wildcard_constraints: run=_alt(LEXICON_RUNS), route=_alt(LEXICON_ROUTES)
    resources: gpu=1
    run:
        r = rundef(wildcards)
        reg_dir = f"results/{wildcards.run}/finemo_regions"
        out = f"results/{wildcards.run}/hits_lexicon_{wildcards.route}"
        lex = lexicon_path(config, r, wildcards.route)
        batch = lexicon_batch(config)
        f = config["finemo"]
        argv = [f"{ENV}/bin/finemo", "call-hits", "-M", f["mode"],
                "-r", f"{reg_dir}/regions.npz", "-m", lex,
                "-o", out, "-t", str(f["threshold"]), "-l", str(f["alpha"]),
                "-b", str(batch)]
        run_body(out, argv)

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

rule merge_motiflex:
    """N MoDISco h5 -> motiflex registry / alignment evidence / annotated merge TREE.

    Nothing merges here by design: `mlex merge` without a cut writes the
    average-linkage tree (1 - ppm_similarity, per-pair shuffle null) with the
    statistics at every node. The cut is a hyperparameter, declared per merge as
    `cut_height` and applied by rule merge_motiflex_cut; it is chosen by reading
    merge/merge_nodes.csv and tree.txt, never defaulted.

    Runs the mlex CLI from config.motiflex.mlex (envs/mmv_verify); CPU only.
    Stage bodies skip existing output, so an interrupted build resumes.
    """
    input: lambda w: comp.dag_inputs(config, w.merge)
    output: "results/merges/{merge}/motiflex/merge/tree.txt",
            "results/merges/{merge}/motiflex/merge/merge_nodes.csv"
    wildcard_constraints: merge=_alt(MERGES)
    run:
        import yaml
        c = mlx.params(config)
        out = Path(f"results/merges/{wildcards.merge}/motiflex")
        out.mkdir(parents=True, exist_ok=True)
        (out / "project.yaml").write_text(
            yaml.safe_dump(mlx.project(config, wildcards.merge), sort_keys=False, width=200))
        seed = c.get("seed", 42)
        if not (out / "registry/registry.json").exists():
            run_body(out, [c["mlex"], "ingest", out / "project.yaml",
                           "--seed", seed, "--out", out / "registry"])
        if not (out / "evidence/alignment_edges.parquet").exists():
            run_body(out, [c["mlex"], "align", out / "registry", "--out", out / "evidence",
                           "--workers", c.get("workers", 8), "--seed", seed])
        db = mlx.db_yaml(config, wildcards.merge)
        if db and not (out / "evidence/annotation_candidates.parquet").exists():
            run_body(out, [c["mlex"], "annotate", out / "evidence",
                           "--registry", out / "registry",
                           "--tomtom", "--databases", db, "--out", out / "evidence"])
        run_body(out, [c["mlex"], "merge", out / "evidence",
                       "--alignment", out / "evidence",
                       "--registry", out / "registry", "--out", out / "merge"])

rule merge_motiflex_cut:
    """Apply the merge's declared cut_height; compile the shared lexicon.

    Group representatives are medoids -- observed patterns, never cross-model
    averages -- and compile reads the written lexicon back with the real finemo
    loader (--verify-roundtrip auto). merge_cut/merge_decisions.json records each
    group's members, representative and evidence.
    """
    input: "results/merges/{merge}/motiflex/merge/tree.txt"
    output: "results/merges/{merge}/motiflex/lexicons/core.h5",
            "results/merges/{merge}/motiflex/lexicons/manifest.tsv",
            "results/merges/{merge}/motiflex/merge_cut/merge_decisions.json"
    wildcard_constraints: merge=_alt(MERGE_CUT)
    run:
        c = mlx.params(config)
        out = Path(f"results/merges/{wildcards.merge}/motiflex")
        run_body(out, [c["mlex"], "merge", out / "evidence",
                       "--alignment", out / "evidence",
                       "--registry", out / "registry",
                       "--cut-height", mlx.cut_height(config, wildcards.merge),
                       "--out", out / "merge_cut"])
        run_body(out, [c["mlex"], "compile", out / "registry",
                       "--decisions", out / "merge_cut/merge_decisions.json",
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
    wildcard_constraints: merge=_alt(MERGE_CUT), leaf=_alt(MERGE_LEAVES)
    run:
        l = mlx.cross_call_leaf(config, wildcards.merge, wildcards.leaf)
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
    wildcard_constraints: merge=_alt(MERGE_CUT), leaf=_alt(MERGE_LEAVES)
    resources: gpu=1
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
    wildcard_constraints: merge=_alt(MERGE_CUT)
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
