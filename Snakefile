# attribute (cbp | ag) -> discover -> extract_regions -> call_hits -> finemo_report
#                                  \-> name_patterns (TomTom vs lexicon_union)
#                                  \-> modisco_report
import json
import os
import sys
from pathlib import Path

import numpy as np
from snakemake.exceptions import WorkflowError

from motifline import naming as nm
from motifline.attribution import hyp_check
from motifline.config import check as check_config
from motifline.runs import active_runs
from motifline.stages import run_body as _run_body

if not config:
    raise WorkflowError("no config: run through `motifline run`, or pass --configfile")
check_config(config, paths=False)

ENV = config["envs"]["compute"]
PY = f"{ENV}/bin/python"
STEPS = Path(workflow.basedir) / "src/motifline/steps"


def rundef(w):
    return config["runs"][w.run]


def param_sig(**kv):
    """Declared as params: so snakemake reruns a stage when one of these changes."""
    return json.dumps(kv, sort_keys=True, default=str)


def sig_import_attr(w):
    r = rundef(w)
    return param_sig(src=r["adopt_from"], rows=r["rows"], width=adopt_width(r),
                     peaks=r.get("adopt_peaks", "peaks.bed"))


def sig_naming(w=None):
    lex = nm.params(config)
    return param_sig(target=nm.UNION, tomtom=lex["tomtom"],
                     h14_annotation=lex["h14_annotation"])


def run_body(out_dir, argv, stage, env=None, prefix=None, sig=None, checks=(), code=None):
    code = {"motifline": workflow.basedir, **(code or {})}
    return _run_body(out_dir, argv, stage=stage, param_sig=sig, env=env,
                     env_prefix=ENV if prefix is None else prefix, checks=checks, code=code)


def thread_env(n):
    return {k: str(n) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS",
                                "NUMBA_NUM_THREADS")}


ATTR_THREADS = 8
MODISCO_THREADS = 32


def attr_width(run):
    """Width of the saved attributions; only valid inside a rule body."""
    return int(np.load(f"results/{run}/attr/attr.npy", mmap_mode="r").shape[-1])


def adopt_width(r):
    # stored width: AG keeps the central crop, CBP its whole input window
    w = r.get("crop") if r["model"] == "ag" else r.get("in_window")
    return None if w is None else int(w)


def attr_input_paths(r):
    model_file = r["wo_bias"] if r["model"] == "cbp" else r["weights"]
    return [model_file, r["peaks"], config["genome"]["fasta"], config["genome"]["chrom_sizes"]]


def attr_inputs(w):
    return attr_input_paths(rundef(w))


def attr_code(r):
    code = {"chrombpnet_src": config["chrombpnet_src"]}
    if r["model"] == "ag":
        code["ag_src"] = r["ag_src"]
    return code


def attr_argv(r, out, shard_i=None):
    """Attribution command line. Window sizes are passed only when the run declares
    them; otherwise the step's defaults apply (cbp 2114; ag 2048 in, 2048 kept)."""
    att = config["attribution"]
    n_shuffles = str(r.get("n_shuffles", att["n_shuffles"]))
    random_state = str(r.get("random_state", att["random_state"]))
    if r["model"] == "ag":
        argv = [PY, str(STEPS / "attr_ag.py"),
                "--ag_src", r["ag_src"], "--chrombpnet_src", config["chrombpnet_src"],
                "--weights", r["weights"], "--peaks", r["peaks"],
                "--fasta", config["genome"]["fasta"],
                "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out, "--track", str(r["track"]), "--biosample", r["biosample"],
                "--n_shuffles", n_shuffles, "--chunk", str(r["chunk"]),
                "--random_state", random_state, "--hypothetical"]
        for k in ("in_window", "crop"):
            if k in r:
                argv += [f"--{k}", str(r[k])]
    elif r["model"] == "cbp":
        argv = [PY, str(STEPS / "attr_cbp.py"), "--chrombpnet_src", config["chrombpnet_src"],
                "--wo_bias", r["wo_bias"], "--peaks", r["peaks"],
                "--fasta", config["genome"]["fasta"],
                "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out, "--hypothetical",
                "--n_shuffles", n_shuffles, "--random_state", random_state]
        if "in_window" in r:
            argv += ["--in_window", str(r["in_window"])]
    else:
        raise ValueError(f"unknown model {r['model']!r}")
    if r.get("head"):
        argv += ["--head", str(r["head"])]
    if shard_i is not None:
        argv += ["--shard_i", str(shard_i), "--n_shards", str(r["n_shards"])]
    return argv


def sig_attr(w, shard_i=None):
    # interpreter and script paths left out: moving the checkout or env is not a new result
    return param_sig(argv=attr_argv(rundef(w), "<out>", shard_i=shard_i)[2:])


ORCH_PY = sys.executable
ORCH_PREFIX = sys.prefix

# attr/ has three producers; run names are partitioned between them.
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


ACTIVE = active_runs(config, sorted(config["runs"]))

# A missing input would let snakemake keep an existing attr.npy without rerunning it.
_missing = [f"{n}: {p}" for n in ACTIVE if not config["runs"][n].get("adopt_from")
            for p in attr_input_paths(config["runs"][n]) if not Path(p).exists()]
if _missing:
    raise WorkflowError("attribution inputs missing (cancel the run to freeze it):\n  "
                        + "\n  ".join(_missing))

rule all:
    input:
        expand("results/{run}/hits/hits.tsv", run=ACTIVE),
        expand("results/{run}/naming/pattern_names.tsv", run=ACTIVE),
        expand("results/{run}/modisco_report/motifs.html", run=ACTIVE),
        expand("results/{run}/finemo_report/report.html", run=ACTIVE)

rule import_attr:
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(ADOPTED)
    params: sig=sig_import_attr
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        argv = [ORCH_PY, "-m", "motifline.adopt", "--src_dir", r["adopt_from"], "--out_dir", out,
                "--expected_rows", str(r["rows"]),
                "--peaks_name", r.get("adopt_peaks", "peaks.bed")]
        if adopt_width(r) is not None:
            argv += ["--expected_width", str(adopt_width(r))]
        run_body(out, argv, "import_attr", prefix=ORCH_PREFIX, sig=params.sig,
                 checks=(hyp_check(),))

rule attribute:
    input: attr_inputs
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(SINGLE)
    resources: gpu=1
    params: sig=sig_attr
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        run_body(out, attr_argv(r, out), "attribute", env=thread_env(ATTR_THREADS),
                 sig=params.sig, checks=(hyp_check(),), code=attr_code(r))

rule attribute_sharded:
    input: attr_inputs
    output: "results/{run}/attr_shards/shard_{i}/attr_{i}.npy",
            "results/{run}/attr_shards/shard_{i}/ohe_{i}.npy",
            "results/{run}/attr_shards/shard_{i}/regions_{i}.bed"
    wildcard_constraints: run=_alt(SHARDED), i=r"\d+"
    resources: gpu=1
    params: sig=lambda w: sig_attr(w, shard_i=int(w.i))
    run:
        r = rundef(wildcards)
        i = int(wildcards.i)
        out = f"results/{wildcards.run}/attr_shards/shard_{i}"
        run_body(out, attr_argv(r, out, shard_i=i), "attribute_sharded",
                 env=thread_env(ATTR_THREADS), sig=params.sig,
                 checks=(hyp_check(f"attr_{i}.npy"),), code=attr_code(r))

rule merge_shards:
    input: shard_outputs
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(SHARDED)
    params: sig=lambda w: param_sig(n_shards=rundef(w)["n_shards"], rows=rundef(w).get("rows"))
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        argv = [ORCH_PY, "-m", "motifline.merge_shards", "--out_dir", out]
        for d in shard_dirs(wildcards.run):
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
        run_body(out, argv, "discover", env=thread_env(threads), sig=params.sig)

rule extract_regions:
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
    """modiscolite's own report, matched against the same union. Its TomTom query
    differs slightly from name_patterns' (nsites), so near-tied matches can differ;
    pattern_names.tsv is the reference for names."""
    input: h5=rules.discover.output, lex=nm.UNION
    output: "results/{run}/modisco_report/motifs.html"
    params: sig=param_sig(target=nm.UNION)
    run:
        out = f"results/{wildcards.run}/modisco_report"
        argv = [f"{ENV}/bin/modisco", "report", "-i", str(input.h5), "-o", out,
                "-m", str(input.lex)]
        tomtom_bin = Path(nm.params(config)["tomtom"]).parent
        run_body(out, argv, "modisco_report", env={"PATH": f"{tomtom_bin}:{os.environ['PATH']}"},
                 sig=params.sig)

rule finemo_report:
    """-W is the window MoDISco ran on; seqlet recall is wrong at any other width."""
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
    input: lambda w: nm.source_paths(config)
    output: nm.UNION
    params: sig=param_sig(sources=list(nm.params(config)["sources"]))
    run:
        argv = [PY, str(STEPS / "lexicon_union.py"), *nm.source_argv(config),
                "--out", output[0]]
        run_body(Path(output[0]).parent, argv, "lexicon_union", sig=params.sig)

rule name_patterns:
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
