# Snakefile
import json
import os
import sys
from pathlib import Path

configfile: "config/default.yaml"

from motifline import contracts, guards
from motifline.gpu import free_gpu_uuids, gpu_lease
from motifline.versions import dist_info_version

ENV = config["envs"]["compute"]
PY = f"{ENV}/bin/python"
SP = config["compute_site_packages"]
STEPS = Path(workflow.basedir) / "src/motifline/steps"

def rundef(w):
    return config["runs"][w.run]

def compute_versions():
    return {p: dist_info_version(SP, p)
            for p in ("tangermeme", "modisco_lite", "finemo")}

def attr_width(r):
    """Width of the saved attributions: AG attributes 16384bp but stores the central crop."""
    return r["crop"] if r["model"] == "ag" else r["in_window"]

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
    if r["model"] == "ag":
        argv = [PY, str(STEPS / "attr_ag.py"),
                "--ag_src", r["ag_src"], "--chrombpnet_src", config["chrombpnet_src"],
                "--weights", r["weights"], "--peaks", r["peaks"],
                "--fasta", config["genome"]["fasta"],
                "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out, "--track", str(r["track"]),
                "--in_window", str(r["in_window"]), "--crop", str(r["crop"]),
                "--n_shuffles", str(r["n_shuffles"]), "--chunk", str(r["chunk"]),
                "--random_state", "42", "--hypothetical"]
    elif r["model"] == "cbp":
        argv = [PY, str(STEPS / "attr_cbp.py"), "--chrombpnet_src", config["chrombpnet_src"],
                "--wo_bias", r["wo_bias"], "--peaks", r["peaks"],
                "--fasta", config["genome"]["fasta"],
                "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out, "--hypothetical", "--in_window", str(r["in_window"])]
    else:
        raise ValueError(f"unknown model {r['model']!r}; expected 'cbp' or 'ag'")
    if r.get("head"):
        argv += ["--head", str(r["head"])]
    if shard_i is not None:
        argv += ["--shard_i", str(shard_i), "--n_shards", str(r["n_shards"])]
    return argv

def attr_params(r, shard_i=None, gpu_uuid=None):
    if r["model"] == "ag":
        extra = {"track": r["track"], "crop": r["crop"], "n_shuffles": r["n_shuffles"],
                 "batch_size": 1, "weights": r["weights"], "ag_src": r["ag_src"],
                 "chunk": r["chunk"]}
    else:
        extra = {"task": "counts", "batch_size": 32, "wo_bias": r["wo_bias"]}
    if shard_i is not None:
        extra.update({"shard_i": shard_i, "n_shards": r["n_shards"]})
    if gpu_uuid is not None:
        extra["gpu_uuid"] = gpu_uuid          # which card produced these rows
    return {"hypothetical": True, "model": r["model"], "in_window": r["in_window"],
            "head": r.get("head"), "random_state": 42, "software": compute_versions(), **extra}

ORCH_PY = sys.executable  # the python running snakemake == the motifline orchestration env

# Concurrent GPU stages pick their card through this lock directory, not from
# nvidia-smi alone: two jobs starting together both see the same idle card.
GPU_LOCKS = str(Path(workflow.basedir) / ".gpu_locks")
GPU_WAIT_S = float(config.get("gpu_wait_s", 3600))

# An adopted run links an archived attr/ohe pair in instead of computing one; the two
# producers write the same three files, so the run names are partitioned between them
# rather than left to snakemake to disambiguate.
ADOPTED = sorted(k for k, v in config["runs"].items() if v.get("adopt_from"))
COMPUTED = sorted(k for k, v in config["runs"].items() if not v.get("adopt_from"))
# ... and computed runs are split again: a full-scale run attributes in shards
# (one GPU each) and merges, a slice run attributes in one process.
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
        expand("results/{run}/equivalence/attr_report.json",
               run=["k562_cbp_slice500", "k562_ag_slice200", "k562_cbp_full"]),
        expand("results/{run}/hits/hits.tsv", run=["k562_cbp_slice500"]),
        expand("results/{run}/equivalence/{report}_report.json",
               run=ADOPTED + ["k562_cbp_full"], report=["modisco", "hits"])

rule import_attr:
    """Adopt archived attributions: symlink attr/ohe, normalize peaks -> regions.bed."""
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(ADOPTED)
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        src = r["adopt_from"]
        w = attr_width(r)
        peaks_src = os.path.join(src, r.get("adopt_peaks", "peaks.bed"))
        argv = [ORCH_PY, "-m", "motifline.adopt", "--src_dir", src, "--out_dir", out,
                "--expected_rows", str(r["rows"]), "--expected_width", str(w),
                "--peaks_name", r.get("adopt_peaks", "peaks.bed")]
        contracts.run_stage(
            out, "import_attr", argv,
            params={"hypothetical": True, "model": r["model"], "in_window": r["in_window"],
                    "crop": r.get("crop"), "head": None, "adopted_from": src,
                    "adopted_peaks": peaks_src, "expected_rows": r["rows"],
                    "attr_width": w, "software": compute_versions()},
            inputs=[os.path.join(src, "attr.npy"), os.path.join(src, "ohe.npy"), peaks_src],
            post_guards=[lambda: (guards.rows_and_width_match(
                f"{out}/attr.npy", f"{out}/ohe.npy", f"{out}/regions.bed", w), out)])

rule attribute:
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(SINGLE)
    resources: gpu=1
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        w = attr_width(r)
        with gpu_lease(GPU_LOCKS, leave_free=1, wait_s=GPU_WAIT_S) as uuid:
            contracts.run_stage(
                out, "attribute", attr_argv(r, out), params=attr_params(r, gpu_uuid=uuid),
                inputs=[r["peaks"], r["weights"] if r["model"] == "ag" else r["wo_bias"]], seed=42,
                env={"CUDA_VISIBLE_DEVICES": uuid, **COMPUTE_THREADS},
                post_guards=[lambda: (guards.rows_and_width_match(
                    f"{out}/attr.npy", f"{out}/ohe.npy", f"{out}/regions.bed", w), out)])

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
        w = attr_width(r)
        with gpu_lease(GPU_LOCKS, leave_free=1, wait_s=GPU_WAIT_S) as uuid:
            contracts.run_stage(
                out, "attribute", attr_argv(r, out, shard_i=i),
                params=attr_params(r, shard_i=i, gpu_uuid=uuid),
                inputs=[r["peaks"], r["weights"] if r["model"] == "ag" else r["wo_bias"]], seed=42,
                env={"CUDA_VISIBLE_DEVICES": uuid, **COMPUTE_THREADS},
                post_guards=[lambda: (guards.rows_and_width_match(
                    f"{out}/attr_{i}.npy", f"{out}/ohe_{i}.npy",
                    f"{out}/regions_{i}.bed", w), out)])

rule merge_shards:
    """Concatenate the shards in shard order into the one attr/ohe/regions triple
    the rest of the DAG consumes. Every shard must have finished SUCCESSFULLY:
    a merge over a REFUSED shard would be a short matrix that still looks valid."""
    input: shard_outputs
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(SHARDED)
    run:
        r = rundef(wildcards)
        dirs = shard_dirs(wildcards.run)
        out = f"results/{wildcards.run}/attr"
        w = attr_width(r)
        argv = [ORCH_PY, "-m", "motifline.merge_shards", "--out_dir", out]
        for d in dirs:
            argv += ["--shard_dir", d]
        if r.get("rows"):
            argv += ["--expected_rows", str(r["rows"])]
        contracts.run_stage(
            out, "merge_shards", argv,
            params={"hypothetical": True, "model": r["model"], "in_window": r["in_window"],
                    "crop": r.get("crop"), "head": r.get("head"), "n_shards": r["n_shards"],
                    "shard_dirs": dirs, "expected_rows": r.get("rows"),
                    "attr_width": w, "software": compute_versions()},
            inputs=[f"{d}/attr_{i}.npy" for i, d in enumerate(dirs)],
            pre_guards=[(lambda d=d: (guards.upstream_success(d), d)) for d in dirs],
            post_guards=[lambda: (guards.rows_and_width_match(
                f"{out}/attr.npy", f"{out}/ohe.npy", f"{out}/regions.bed", w), out)])

ATTR_FILES = ["results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy",
              "results/{run}/attr/regions.bed"]

rule discover:
    input: ATTR_FILES
    output: "results/{run}/modisco/modisco.h5"
    run:
        r = rundef(wildcards)
        attr_dir = f"results/{wildcards.run}/attr"
        out = f"results/{wildcards.run}/modisco"
        argv = [f"{ENV}/bin/modisco", "motifs", "-s", f"{attr_dir}/ohe.npy",
                "-a", f"{attr_dir}/attr.npy", "-n", str(config["modisco"]["max_seqlets"]),
                "-w", str(config["modisco"]["window"]), "-o", f"{out}/modisco.h5"]
        contracts.run_stage(
            out, "discover", argv,
            params={"attr_dir": attr_dir, "attr_hypothetical": True,
                    **config["modisco"], "software": compute_versions()},
            inputs=[f"{attr_dir}/attr.npy", f"{attr_dir}/ohe.npy"],
            env=COMPUTE_THREADS,
            pre_guards=[
                lambda: (guards.upstream_success(attr_dir), attr_dir),
                lambda: (guards.attr_is_hypothetical(attr_dir), attr_dir),
                lambda: (guards.rows_and_width_match(f"{attr_dir}/attr.npy", f"{attr_dir}/ohe.npy",
                                                     f"{attr_dir}/regions.bed", attr_width(r)), attr_dir)])

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
                "-w", str(attr_width(r))]
        contracts.run_stage(
            out, "extract_regions", argv,
            params={"attr_dir": attr_dir, "width": attr_width(r),
                    "software": compute_versions()},
            inputs=[f"{attr_dir}/attr.npy", f"{attr_dir}/ohe.npy"],
            pre_guards=[
                lambda: (guards.upstream_success(attr_dir), attr_dir),
                lambda: (guards.attr_is_hypothetical(attr_dir), attr_dir)])

rule call_hits:
    input: npz=rules.extract_regions.output, lex=rules.discover.output
    output: "results/{run}/hits/hits.tsv"
    resources: gpu=1
    run:
        attr_dir = f"results/{wildcards.run}/attr"
        reg_dir = f"results/{wildcards.run}/finemo_regions"
        mod_dir = f"results/{wildcards.run}/modisco"
        out = f"results/{wildcards.run}/hits"
        f = config["finemo"]
        argv = [f"{ENV}/bin/finemo", "call-hits", "-M", f["mode"],
                "-r", f"{reg_dir}/regions.npz", "-m", f"{mod_dir}/modisco.h5",
                "-o", out, "-t", str(f["threshold"]), "-l", str(f["alpha"]),
                "-b", str(f["batch"])]
        with gpu_lease(GPU_LOCKS, leave_free=1, wait_s=GPU_WAIT_S) as uuid:
            contracts.run_stage(
                out, "call_hits", argv,
                params={"attr_dir": attr_dir, "lexicon_dir": mod_dir, **f,
                        "gpu_uuid": uuid, "software": compute_versions()},
                inputs=[f"{reg_dir}/regions.npz", f"{mod_dir}/modisco.h5"],
                env={"CUDA_VISIBLE_DEVICES": uuid},
                pre_guards=[
                    lambda: (guards.upstream_success(reg_dir), reg_dir),
                    lambda: (guards.upstream_success(mod_dir), mod_dir),
                    lambda: (guards.attr_is_hypothetical(attr_dir), attr_dir),
                    lambda: (guards.hyp_flags_match(attr_dir, attr_dir), attr_dir),
                    lambda: (guards.finemo_version_matches(reg_dir, SP), SP)])

rule equivalence_attr:
    input: ATTR_FILES
    output: "results/{run}/equivalence/attr_report.json"
    wildcard_constraints: run="|".join(COMPUTED) if COMPUTED else "$^"
    run:
        from motifline.equivalence import compare_attr
        r = rundef(wildcards)
        os.makedirs(f"results/{wildcards.run}/equivalence", exist_ok=True)
        rep = compare_attr(f"results/{wildcards.run}/attr", r["old_attr"],
                           r["old_regions"], r["head"], output[0])
        print(json.dumps(rep, indent=2))

rule equivalence_modisco:
    """Level-2 consistency report: new lexicon vs the archived modisco.h5 (no gate)."""
    input: new="results/{run}/modisco/modisco.h5"
    output: "results/{run}/equivalence/modisco_report.json"
    run:
        from motifline.equivalence import compare_modisco
        r = rundef(wildcards)
        os.makedirs(f"results/{wildcards.run}/equivalence", exist_ok=True)
        rep = compare_modisco(input.new, r["old_modisco"], output[0])
        print(json.dumps({k: v for k, v in rep.items() if k != "per_old_pattern"}, indent=2))

rule equivalence_hits:
    """Level-3 consistency report: new hits vs the archived finemo hits (no gate).

    Joined on CWM identity, not on motif name: MoDISco names patterns by rank, so
    two lexicons of different sizes reuse the same names for different motifs.
    """
    input: new="results/{run}/hits/hits.tsv",
           modisco_report="results/{run}/equivalence/modisco_report.json"
    output: "results/{run}/equivalence/hits_report.json"
    run:
        from motifline.equivalence import compare_hits, identity_map_from_modisco_report
        r = rundef(wildcards)
        os.makedirs(f"results/{wildcards.run}/equivalence", exist_ok=True)
        imap = identity_map_from_modisco_report(input.modisco_report)
        rep = compare_hits(input.new, r["old_hits"], output[0], identity_map=imap)
        print(json.dumps({k: v for k, v in rep.items() if k != "per_motif_counts"}, indent=2))
