# Snakefile
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

configfile: "config/default.yaml"

from motifline.gpu import env_sees_cuda, free_gpu_uuids, gpu_lease
from motifline.lexicons import attr_width, lexicon_batch, lexicon_path
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
# Runs that have an archived counterpart to report against, and runs that carry
# one of the two extra hit-calling stages. Kept as explicit lists so that asking
# for a report a run has no archive for is an unmatched target, not a KeyError
# raised halfway through the DAG.
ARCHIVE_COMPARED = sorted(k for k, v in config["runs"].items() if v.get("old_modisco"))
# A cancelled run keeps every rule it always had -- it stays resumable by explicit
# target -- but drops out of the default target list, so a bare `snakemake` cannot
# restart work somebody stopped on purpose. See motifline/src/motifline/runs.py and
# the run's own CANCELLED.md (config: cancelled_note).
CANCELLED = cancelled_runs(config)
LEXICON_RUNS = sorted(k for k, v in config["runs"].items() if v.get("lexicon_hits"))
CROSS_VERSION = sorted(k for k, v in config["runs"].items() if v.get("cross_version_hits"))
LEXICON_ROUTES = sorted((config.get("lexicons") or {}).get("routes") or {})


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
               run=active_runs(config, ["k562_cbp_slice500", "k562_ag_slice200",
                                        "k562_cbp_full"])),
        expand("results/{run}/hits/hits.tsv",
               run=active_runs(config, ["k562_cbp_slice500"])),
        expand("results/{run}/equivalence/{report}_report.json",
               run=active_runs(config, ARCHIVE_COMPARED), report=["modisco", "hits"]),
        # finemo 0.30 vs 0.41 on one and the same regions.npz + modisco.h5
        expand("results/{run}/equivalence/hits_v041_report.json",
               run=active_runs(config, CROSS_VERSION)),
        # external-lexicon routes vs their archived per-context hits
        expand("results/{run}/equivalence/hits_lexicon_{route}_report.json",
               run=active_runs(config, LEXICON_RUNS), route=LEXICON_ROUTES)

rule import_attr:
    """Adopt archived attributions: symlink attr/ohe, normalize peaks -> regions.bed."""
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(ADOPTED)
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        src = r["adopt_from"]
        argv = [ORCH_PY, "-m", "motifline.adopt", "--src_dir", src, "--out_dir", out,
                "--expected_rows", str(r["rows"]), "--expected_width", str(attr_width(r)),
                "--peaks_name", r.get("adopt_peaks", "peaks.bed")]
        run_body(out, argv)

rule attribute:
    output: "results/{run}/attr/attr.npy", "results/{run}/attr/ohe.npy", "results/{run}/attr/regions.bed"
    wildcard_constraints: run=_alt(SINGLE)
    resources: gpu=1
    run:
        r = rundef(wildcards)
        out = f"results/{wildcards.run}/attr"
        with gpu_lease(GPU_LOCKS, leave_free=1, wait_s=GPU_WAIT_S) as uuid:
            run_body(out, attr_argv(r, out),
                     env={"CUDA_VISIBLE_DEVICES": uuid, **COMPUTE_THREADS})

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
        with gpu_lease(GPU_LOCKS, leave_free=1, wait_s=GPU_WAIT_S) as uuid:
            run_body(out, attr_argv(r, out, shard_i=i),
                     env={"CUDA_VISIBLE_DEVICES": uuid, **COMPUTE_THREADS})

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
                "-w", str(attr_width(r))]
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
        with gpu_lease(GPU_LOCKS, leave_free=1, wait_s=GPU_WAIT_S) as uuid:
            run_body(out, argv, env={"CUDA_VISIBLE_DEVICES": uuid})

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
        # Lease a card only if this finemo's torch can reach one. Measured on this
        # node it cannot (torch 2.13.0+cu130 vs driver 555.42.06), and leasing
        # anyway squatted a GPU for 65 min of pure CPU work while an attribution
        # shard sat stopped behind it. The runtime fact is recorded either way.
        sees_cuda = env_sees_cuda(f"{ENV_ALT}/bin/python")
        lease = (gpu_lease(GPU_LOCKS, leave_free=1, wait_s=GPU_WAIT_S) if sees_cuda
                 else contextlib.nullcontext(None))
        with lease as uuid:
            run_body(out, argv,
                     env={"CUDA_VISIBLE_DEVICES": uuid} if uuid else None)

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
        with gpu_lease(GPU_LOCKS, leave_free=1, wait_s=GPU_WAIT_S) as uuid:
            run_body(out, argv, env={"CUDA_VISIBLE_DEVICES": uuid})

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
    wildcard_constraints: run=_alt(ARCHIVE_COMPARED)
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
    wildcard_constraints: run=_alt(ARCHIVE_COMPARED)
    run:
        from motifline.equivalence import compare_hits, identity_map_from_modisco_report
        r = rundef(wildcards)
        os.makedirs(f"results/{wildcards.run}/equivalence", exist_ok=True)
        imap = identity_map_from_modisco_report(input.modisco_report)
        rep = compare_hits(input.new, r["old_hits"], output[0], identity_map=imap)
        print(json.dumps({k: v for k, v in rep.items() if k != "per_motif_counts"}, indent=2))

rule equivalence_hits_v041:
    """0.41's hits against 0.30's, same regions and same lexicon.

    Joined on motif_name and NOT through a CWM identity map: both sides scored
    the identical modisco.h5, so the names denote the same motifs by construction
    and any remapping here would only be able to hide a difference.
    """
    input: new="results/{run}/hits_v041/hits.tsv", old="results/{run}/hits/hits.tsv"
    output: "results/{run}/equivalence/hits_v041_report.json"
    wildcard_constraints: run=_alt(CROSS_VERSION)
    run:
        from motifline.equivalence import compare_hits
        os.makedirs(f"results/{wildcards.run}/equivalence", exist_ok=True)
        rep = compare_hits(input.new, input.old, output[0])
        print(json.dumps({k: v for k, v in rep.items() if k != "per_motif_counts"}, indent=2))

rule equivalence_hits_lexicon:
    """External-lexicon hits against the archived hits for the same context+route.

    Also a motif_name join, for the same reason: `-m` pointed both runs at one
    lexicon file, so the pattern names are shared rather than rank-local.
    """
    input: new="results/{run}/hits_lexicon_{route}/hits.tsv"
    output: "results/{run}/equivalence/hits_lexicon_{route}_report.json"
    wildcard_constraints: run=_alt(LEXICON_RUNS), route=_alt(LEXICON_ROUTES)
    run:
        from motifline.equivalence import compare_hits
        r = rundef(wildcards)
        os.makedirs(f"results/{wildcards.run}/equivalence", exist_ok=True)
        rep = compare_hits(input.new, r["lexicon_hits"][wildcards.route], output[0])
        print(json.dumps({k: v for k, v in rep.items() if k != "per_motif_counts"}, indent=2))
