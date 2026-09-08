# Snakefile -- one line per run:
#
#     attribute (cbp | ag | star)  ->  discover  ->  extract_regions  ->  call_hits
#                                            \->  name_patterns
#     built once:  lexicon_union -> lexicon_rsat -> lexicon_medoid -> lexicon_cltagged
#
# The cross-model merge (motiflex / MotifCompendium), the external-lexicon routes and
# the finemo 0.30-vs-0.41 arm were removed on 2026-09-07 (branch `linear`); git history
# before that commit has them. `geometry.py` went with them: an attribution's width is
# read off the array (attr_width below), not derived from config.
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

configfile: "config/default.yaml"

from motifline import naming as nm
from motifline.runs import active_runs, cancelled_runs

ENV = config["envs"]["compute"]
PY = f"{ENV}/bin/python"
# GPN-Star runs in its own env: its torch (2.6.0+cu124) is not the compute env's.
PY_STAR = f"{config['envs']['star']}/bin/python"
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
    # `width` is the run's declared width, kept in the signature so that finished
    # adopted runs are not re-imported (and their hours of MoDISco re-run) just because
    # geometry.py went away; nothing reads it.
    r = rundef(w)
    return param_sig(src=r["adopt_from"], rows=r["rows"],
                     width=int(r.get("crop", r.get("in_window"))),
                     peaks=r.get("adopt_peaks", "peaks.bed"))


def sig_naming(w=None):
    return param_sig(gate=nm.gate(config), tomtom=nm.params(config)["tomtom"],
                     w_th=nm.w_th(config))


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


def attr_width(run):
    """Width of a run's saved attributions, read off the array itself.

    Replaces geometry.py (2026-09-07): the width used to be a config-derived number
    that had to be kept equal to what the arrays actually were. Reading it from
    attr.npy cannot drift. It exists only once the attr stage does, so it is called
    inside rule bodies, never in ``input:``/``params:``.
    """
    return int(np.load(f"results/{run}/attr/attr.npy", mmap_mode="r").shape[-1])


def attr_argv(r, out, shard_i=None):
    """Model-dispatched attribution command line; every step writes attr/ohe/regions.bed
    (``attr_<i>.npy``/... when a shard index is given).

    Window sizes are passed only when the run entry declares them; otherwise the step's
    own defaults apply (cbp 2114; ag 2048 in / 2048 stored; star 1536).
    """
    model = r["model"]
    if model in ("cbp", "ag"):
        att = config["attribution"]
        n_shuffles = str(r.get("n_shuffles", att["n_shuffles"]))
        random_state = str(r.get("random_state", att["random_state"]))
    if model == "ag":
        argv = [PY, str(STEPS / "attr_ag.py"),
                "--ag_src", r["ag_src"], "--chrombpnet_src", config["chrombpnet_src"],
                "--weights", r["weights"], "--peaks", r["peaks"],
                "--fasta", config["genome"]["fasta"],
                "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out, "--track", str(r["track"]),
                "--n_shuffles", n_shuffles, "--chunk", str(r["chunk"]),
                "--random_state", random_state, "--hypothetical"]
        for k in ("in_window", "crop"):
            if k in r:
                argv += [f"--{k}", str(r[k])]
    elif model == "cbp":
        argv = [PY, str(STEPS / "attr_cbp.py"), "--chrombpnet_src", config["chrombpnet_src"],
                "--wo_bias", r["wo_bias"], "--peaks", r["peaks"],
                "--fasta", config["genome"]["fasta"],
                "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out, "--hypothetical",
                "--n_shuffles", n_shuffles, "--random_state", random_state]
        if "in_window" in r:
            argv += ["--in_window", str(r["in_window"])]
    elif model == "star":
        s = config["star"]
        argv = [PY_STAR, str(STEPS / "attr_star.py"),
                "--peaks", r["peaks"], "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out,
                "--msa_path", r.get("msa_path", s["msa_path"]),
                "--repo", r.get("repo", s["repo"]),
                "--stride", str(r["stride"]), "--source", r["source"],
                "--chunk", str(r.get("chunk", s.get("chunk", 100)))]
        if "in_window" in r:
            argv += ["--in_window", str(r["in_window"])]
    else:
        raise ValueError(f"unknown model {model!r}; expected 'cbp', 'ag' or 'star'")
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
# restart work somebody stopped on purpose (or re-import an archive that is gone).
# See src/motifline/runs.py and the run's own CANCELLED.md (config: cancelled_note).
CANCELLED = cancelled_runs(config)


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
               run=active_runs(config, sorted(config["runs"])))

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
                "--expected_rows", str(r["rows"]),
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
    """Package attr/ohe/regions for finemo at the width the array actually is."""
    input: ATTR_FILES
    output: "results/{run}/finemo_regions/regions.npz"
    run:
        attr_dir = f"results/{wildcards.run}/attr"
        out = f"results/{wildcards.run}/finemo_regions"
        argv = [f"{ENV}/bin/finemo", "extract-regions-modisco-fmt",
                "-s", f"{attr_dir}/ohe.npy", "-a", f"{attr_dir}/attr.npy",
                "-p", f"{attr_dir}/regions.bed", "-o", f"{out}/regions.npz",
                "-w", str(attr_width(wildcards.run))]
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

# ---------------------------------------------------------------------------
# Reference lexicon and pattern naming (2026-08-25). One library, built once, for
# every run in this pipeline; rationale in src/motifline/naming.py.
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
