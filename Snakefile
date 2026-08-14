# Snakefile
import json
import os
import sys
from pathlib import Path

configfile: "config/default.yaml"

from motifline import contracts, guards
from motifline.gpu import free_gpu_uuids
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

def attr_argv(r, out):
    """Model-dispatched attribution command line; both steps write attr/ohe/regions.bed."""
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
    return argv

def attr_params(r):
    if r["model"] == "ag":
        extra = {"track": r["track"], "crop": r["crop"], "n_shuffles": r["n_shuffles"],
                 "batch_size": 1, "weights": r["weights"], "ag_src": r["ag_src"]}
    else:
        extra = {"task": "counts", "batch_size": 32, "wo_bias": r["wo_bias"]}
    return {"hypothetical": True, "model": r["model"], "in_window": r["in_window"],
            "head": r.get("head"), "random_state": 42, "software": compute_versions(), **extra}

ORCH_PY = sys.executable  # the python running snakemake == the motifline orchestration env

# An adopted run links an archived attr/ohe pair in instead of computing one; the two
# producers write the same three files, so the run names are partitioned between them
# rather than left to snakemake to disambiguate.
ADOPTED = sorted(k for k, v in config["runs"].items() if v.get("adopt_from"))
COMPUTED = sorted(k for k, v in config["runs"].items() if not v.get("adopt_from"))

rule all:
    input:
        expand("results/{run}/equivalence/attr_report.json",
               run=["k562_cbp_slice500", "k562_ag_slice200"]),
        expand("results/{run}/hits/hits.tsv", run=["k562_cbp_slice500"]),
        expand("results/{run}/equivalence/{report}_report.json",
               run=ADOPTED, report=["modisco", "hits"])

rule import_attr:
    """Adopt archived attributions: symlink attr/ohe, normalize peaks -> regions.bed."""
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run="|".join(ADOPTED) if ADOPTED else "$^"
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
    wildcard_constraints: run="|".join(COMPUTED) if COMPUTED else "$^"
    resources: gpu=1
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        free = free_gpu_uuids(leave_free=1)
        if not free:
            raise RuntimeError("no fully-free GPU while leaving one free; not sharing occupied cards")
        w = attr_width(r)
        contracts.run_stage(
            out, "attribute", attr_argv(r, out), params=attr_params(r),
            inputs=[r["peaks"], r["weights"] if r["model"] == "ag" else r["wo_bias"]], seed=42,
            env={"CUDA_VISIBLE_DEVICES": free[0], **COMPUTE_THREADS},
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
        free = free_gpu_uuids(leave_free=1)
        if not free:
            raise RuntimeError("no fully-free GPU while leaving one free")
        f = config["finemo"]
        argv = [f"{ENV}/bin/finemo", "call-hits", "-M", f["mode"],
                "-r", f"{reg_dir}/regions.npz", "-m", f"{mod_dir}/modisco.h5",
                "-o", out, "-t", str(f["threshold"]), "-l", str(f["alpha"]),
                "-b", str(f["batch"])]
        contracts.run_stage(
            out, "call_hits", argv,
            params={"attr_dir": attr_dir, "lexicon_dir": mod_dir, **f,
                    "software": compute_versions()},
            inputs=[f"{reg_dir}/regions.npz", f"{mod_dir}/modisco.h5"],
            env={"CUDA_VISIBLE_DEVICES": free[0]},
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
