# Snakefile -- one line per run:
#
#     attribute (cbp | ag)  ->  discover  ->  extract_regions  ->  call_hits  ->  finemo_report
#                                        \->  name_patterns  (TomTom vs lexicon_union)
#                                        \->  modisco_report
import json
import os
import sys
from pathlib import Path

import numpy as np

configfile: "config/default.yaml"

from motifline import naming as nm
from motifline.attribution import hyp_check
from motifline.config import check as check_config
from motifline.runs import active_runs
from motifline.stages import run_body as _run_body

# Structure only: the DAG has to build on a machine where the archives have moved.
# `motifline check` is the same validator with the filesystem included.
check_config(config, paths=False)

ENV = config["envs"]["compute"]
PY = f"{ENV}/bin/python"
STEPS = Path(workflow.basedir) / "src/motifline/steps"

def rundef(w):
    return config["runs"][w.run]

def param_sig(**kv):
    """The scientific parameters a stage runs at, declared as ``params:`` so that
    snakemake reruns the stage when one of them changes in config."""
    return json.dumps(kv, sort_keys=True, default=str)


def sig_import_attr(w):
    # `width` is only here so the signature of the finished adopted runs stays what it
    # was; nothing reads it.
    r = rundef(w)
    return param_sig(src=r["adopt_from"], rows=r["rows"],
                     width=int(r.get("crop", r.get("in_window"))),
                     peaks=r.get("adopt_peaks", "peaks.bed"))


def sig_naming(w=None):
    lex = nm.params(config)
    return param_sig(target=nm.UNION, tomtom=lex["tomtom"],
                     h14_annotation=lex["h14_annotation"])


def run_body(out_dir, argv, stage, env=None, prefix=None, sig=None, packages=None,
             checks=()):
    """motifline.stages.run_body; ``prefix`` defaults to the compute env."""
    kw = {} if packages is None else {"packages": packages}
    return _run_body(out_dir, argv, stage=stage, param_sig=sig, env=env,
                     env_prefix=ENV if prefix is None else prefix, checks=checks, **kw)

def thread_env(n):
    """Cap every threading runtime a stage body uses. NUMBA is in the list because
    MoDISco's hot loops are numba: uncapped, each discover job takes all 80 cores."""
    return {k: str(n) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS",
                                "NUMBA_NUM_THREADS")}


# Attribution: one shard per GPU at 8 threads was the measured optimum for AG.
ATTR_THREADS = thread_env(8)
# MoDISco (rule discover; the name is kept because it is in that rule's code). 32 gives a
# byte-identical modisco.h5 to 8 (new_experiment/modisco_threads_2026-09-27).
MODISCO_THREADS = 32
COMPUTE_THREADS = thread_env(MODISCO_THREADS)


def attr_width(run):
    """Width of a run's saved attributions, read off attr.npy. It exists only once the
    attr stage does, so it is called inside rule bodies, never in ``input:``/``params:``."""
    return int(np.load(f"results/{run}/attr/attr.npy", mmap_mode="r").shape[-1])


def attr_argv(r, out, shard_i=None):
    """Model-dispatched attribution command line; every step writes attr/ohe/regions.bed
    (``attr_<i>.npy``/... when a shard index is given).

    Window sizes are passed only when the run entry declares them; otherwise the step's
    own defaults apply (cbp 2114; ag 2048 in / 2048 stored).
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
    else:
        raise ValueError(f"unknown model {model!r}; expected 'cbp' or 'ag'")
    if r.get("head"):
        argv += ["--head", str(r["head"])]
    if shard_i is not None:
        argv += ["--shard_i", str(shard_i), "--n_shards", str(r["n_shards"])]
    return argv

ORCH_PY = sys.executable  # the python running snakemake == the motifline orchestration env
ORCH_PREFIX = sys.prefix

# GPU selection is the caller's: pin the card by UUID in the environment and bound
# concurrency with --resources gpu=<G>.

# Three producers write attr/{attr,ohe}.npy + regions.bed: adopt an archive, attribute
# in shards and merge, or attribute in one process. Run names are partitioned between them.
ADOPTED = sorted(k for k, v in config["runs"].items() if v.get("adopt_from"))
COMPUTED = sorted(k for k, v in config["runs"].items() if not v.get("adopt_from"))
SHARDED = sorted(k for k in COMPUTED if config["runs"][k].get("n_shards"))
SINGLE = sorted(k for k in COMPUTED if not config["runs"][k].get("n_shards"))


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
        expand("results/{run}/naming/pattern_names.tsv",
               run=active_runs(config, sorted(config["runs"]))),
        expand("results/{run}/modisco_report/motifs.html",
               run=active_runs(config, sorted(config["runs"]))),
        expand("results/{run}/finemo_report/report.html",
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
        run_body(out, argv, "import_attr", prefix=ORCH_PREFIX, sig=params.sig,
                 checks=(hyp_check(),))

rule attribute:
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(SINGLE)
    resources: gpu=1
    params: sig=lambda w: param_sig(argv=attr_argv(rundef(w), "<out>"))
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        run_body(out, attr_argv(r, out), "attribute", env=ATTR_THREADS,
                 sig=params.sig, checks=(hyp_check(),))

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
        run_body(out, attr_argv(r, out, shard_i=i), "attribute_sharded",
                 env=ATTR_THREADS, sig=params.sig, checks=(hyp_check(f"attr_{i}.npy"),))

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
        run_body(out, argv, "merge_shards", prefix=ORCH_PREFIX, sig=params.sig,
                 checks=(hyp_check(),))

ATTR_FILES = ["results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy",
              "results/{run}/attr/regions.bed"]

rule discover:
    input: ATTR_FILES
    output: "results/{run}/modisco/modisco.h5"
    threads: MODISCO_THREADS
    params: sig=param_sig(modisco=config["modisco"])
    run:
        attr_dir = f"results/{wildcards.run}/attr"
        out = f"results/{wildcards.run}/modisco"
        argv = [f"{ENV}/bin/modisco", "motifs", "-s", f"{attr_dir}/ohe.npy",
                "-a", f"{attr_dir}/attr.npy", "-n", str(config["modisco"]["max_seqlets"]),
                "-w", str(config["modisco"]["window"]), "-o", f"{out}/modisco.h5"]
        if "n_leiden" in config["modisco"]:
            argv += ["-l", str(config["modisco"]["n_leiden"])]
        run_body(out, argv, "discover", env=COMPUTE_THREADS, sig=params.sig)

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
        run_body(out, argv, "extract_regions")

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
        run_body(out, argv, "call_hits", sig=params.sig)

rule modisco_report:
    """modiscolite's own HTML report, matched against the union the names come from.
    Its TomTom line is the one name_patterns copies, so the matches are the same; the
    report shows library ids, the gene symbols are in naming/pattern_names.tsv.
    -n (3 matches) and -l (off) are left at the CLI defaults."""
    input: h5=rules.discover.output, lex=nm.UNION
    output: "results/{run}/modisco_report/motifs.html"
    params: sig=param_sig(target=nm.UNION)
    run:
        out = f"results/{wildcards.run}/modisco_report"
        argv = [f"{ENV}/bin/modisco", "report", "-i", str(input.h5), "-o", out,
                "-m", str(input.lex)]
        # modiscolite shells out to `tomtom` by name; it lives in the meme env
        tomtom_bin = Path(nm.params(config)["tomtom"]).parent
        run_body(out, argv, "modisco_report", env={"PATH": f"{tomtom_bin}:{os.environ['PATH']}"},
                 sig=params.sig)

rule finemo_report:
    """FiNeMo's own HTML report on the run's hits, with seqlet recall against modisco.h5.
    -W must be the window MoDISco ran on, so it is read from modisco.window."""
    input: npz=rules.extract_regions.output, hits=rules.call_hits.output, h5=rules.discover.output
    output: "results/{run}/finemo_report/report.html"
    params: sig=param_sig(window=config["modisco"]["window"])
    run:
        out = f"results/{wildcards.run}/finemo_report"
        argv = [f"{ENV}/bin/finemo", "report",
                "-r", f"results/{wildcards.run}/finemo_regions/regions.npz",
                "-H", f"results/{wildcards.run}/hits",
                "-m", f"results/{wildcards.run}/modisco/modisco.h5",
                "-W", str(config["modisco"]["window"]), "-o", out]
        run_body(out, argv, "finemo_report", sig=params.sig)

rule lexicon_union:
    """Source libraries -> one union MEME, matrices untouched; built once, shared by every run."""
    input: lambda w: nm.source_paths(config)
    output: nm.UNION
    params: sig=param_sig(sources=list(nm.params(config)["sources"]))
    run:
        argv = [PY, str(STEPS / "lexicon_union.py"), *nm.source_argv(config),
                "--out", output[0]]
        run_body(Path(output[0]).parent, argv, "lexicon_union", sig=params.sig)

rule name_patterns:
    """One run's MoDISco patterns -> TF of the first TomTom match against the union."""
    input: h5="results/{run}/modisco/modisco.h5", lex=nm.UNION,
           h14=lambda w: nm.params(config)["h14_annotation"]
    output: nm.run_names("{run}")
    params: sig=sig_naming
    wildcard_constraints: run=_alt(sorted(config["runs"]))
    run:
        out = f"results/{wildcards.run}/naming"
        argv = [PY, str(STEPS / "name_patterns.py"),
                "--h5", f"{wildcards.run}={input.h5}", "--target", input.lex,
                "--h14_annotation", input.h14,
                "--tomtom", nm.params(config)["tomtom"], "--out_dir", out]
        run_body(out, argv, "name_patterns", sig=params.sig)
