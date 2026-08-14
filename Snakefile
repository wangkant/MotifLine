# Snakefile
import json
import os
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

rule all:
    input:
        expand("results/{run}/equivalence/attr_report.json", run=["k562_cbp_slice500"]),
        expand("results/{run}/hits/hits.tsv", run=["k562_cbp_slice500"])

rule attribute:
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    resources: gpu=1
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        free = free_gpu_uuids(leave_free=1)
        if not free:
            raise RuntimeError("no fully-free GPU while leaving one free; not sharing occupied cards")
        argv = [PY, str(STEPS / "attr_cbp.py"), "--chrombpnet_src", config["chrombpnet_src"],
                "--wo_bias", r["wo_bias"], "--peaks", r["peaks"],
                "--fasta", config["genome"]["fasta"], "--chrom_sizes", config["genome"]["chrom_sizes"],
                "--out_dir", out, "--hypothetical", "--head", str(r["head"]),
                "--in_window", str(r["in_window"])]
        contracts.run_stage(
            out, "attribute", argv,
            params={"hypothetical": True, "in_window": r["in_window"], "head": r["head"],
                    "task": "counts", "batch_size": 32, "random_state": 42,
                    "software": compute_versions()},
            inputs=[r["peaks"], r["wo_bias"]], seed=42,
            env={"CUDA_VISIBLE_DEVICES": free[0]},
            post_guards=[lambda: (guards.rows_and_width_match(
                f"{out}/attr.npy", f"{out}/ohe.npy", f"{out}/regions.bed", r["in_window"]), out)])

rule discover:
    input: rules.attribute.output
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
            pre_guards=[
                lambda: (guards.upstream_success(attr_dir), attr_dir),
                lambda: (guards.attr_is_hypothetical(attr_dir), attr_dir),
                lambda: (guards.rows_and_width_match(f"{attr_dir}/attr.npy", f"{attr_dir}/ohe.npy",
                                                     f"{attr_dir}/regions.bed", r["in_window"]), attr_dir)])

rule extract_regions:
    input: rules.attribute.output
    output: "results/{run}/finemo_regions/regions.npz"
    run:
        r = rundef(wildcards)
        attr_dir = f"results/{wildcards.run}/attr"
        out = f"results/{wildcards.run}/finemo_regions"
        argv = [f"{ENV}/bin/finemo", "extract-regions-modisco-fmt",
                "-s", f"{attr_dir}/ohe.npy", "-a", f"{attr_dir}/attr.npy",
                "-p", f"{attr_dir}/regions.bed", "-o", f"{out}/regions.npz",
                "-w", str(r["in_window"])]
        contracts.run_stage(
            out, "extract_regions", argv,
            params={"attr_dir": attr_dir, "width": r["in_window"],
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
    input: rules.attribute.output
    output: "results/{run}/equivalence/attr_report.json"
    run:
        from motifline.equivalence import compare_attr
        r = rundef(wildcards)
        os.makedirs(f"results/{wildcards.run}/equivalence", exist_ok=True)
        rep = compare_attr(f"results/{wildcards.run}/attr", r["old_attr"],
                           r["old_regions"], r["head"], output[0])
        print(json.dumps(rep, indent=2))
