configfile: "config/config.yaml"

from pathlib import Path

S = Path(workflow.basedir) / "scripts"
ENV = config["envs"]["compute"]
PY = f"{ENV}/bin/python"
G = config["genome"]
RUNS = config["runs"]
GPUS = config["gpus"]
LEX = config["reference_lexicon"]
UNION = "results/lexicon/union.meme"

wildcard_constraints:
    run="[^/]+",
    i=r"\d+"


def shards(run):
    return range(RUNS[run].get("n_shards", 1))


def width(run):
    # the attribution step stores the whole input: ChromBPNet's 2114 bp, AlphaGenome's in_window
    return RUNS[run].get("in_window", 2048) if RUNS[run]["model"] == "ag" else 2114


def attr_cmd(w):
    r = RUNS[w.run]
    a = config["attribution"]
    common = (f"--peaks {r['peaks']} --fasta {G['fasta']} --chrom_sizes {G['chrom_sizes']} "
              f"--out_dir results/{w.run}/shards/{w.i} "
              f"--shard_i {w.i} --n_shards {len(shards(w.run))} "
              f"--n_shuffles {a['n_shuffles']} --random_state {a['random_state']}")
    if r["model"] == "ag":
        return (f"{PY} {S}/attr_ag.py --weights {r['weights']} "
                f"--track {r['track']} --in_window {r.get('in_window', 2048)} {common}")
    if r.get("backend") == "tf":
        return (f"TF_FORCE_GPU_ALLOW_GROWTH=true {config['envs']['chrombpnet']}/bin/python "
                f"{S}/attr_cbp_tf.py --model_h5 {r['wo_bias']} {common}")
    return (f"{PY} {S}/attr_cbp.py --wo_bias {r['wo_bias']} {common}")


rule all:
    input:
        expand("results/{run}/hits/hits.tsv", run=RUNS),
        expand("results/{run}/naming/pattern_names.tsv", run=RUNS),
        expand("results/{run}/modisco_report/motifs.html", run=RUNS),
        expand("results/{run}/finemo_report/report.html", run=RUNS),
        expand("results/plots/{plot}.png", plot=config["plots"])


rule attribute:
    input:
        lambda w: [RUNS[w.run]["peaks"], G["fasta"],
                   RUNS[w.run].get("wo_bias") or RUNS[w.run]["weights"]]
    output:
        "results/{run}/shards/{i}/attr.npy",
        "results/{run}/shards/{i}/ohe.npy",
        "results/{run}/shards/{i}/regions.bed"
    params: cmd=attr_cmd
    resources: gpu=1
    log: "results/{run}/shards/{i}/attr.log"
    run:
        gpu = GPUS[int(wildcards.i) % len(GPUS)]
        shell("CUDA_VISIBLE_DEVICES={gpu} {params.cmd} > {log} 2>&1")


rule merge_shards:
    input:
        lambda w: expand("results/{run}/shards/{i}/{f}", run=w.run, i=shards(w.run),
                         f=("attr.npy", "ohe.npy", "regions.bed"))
    output:
        "results/{run}/attr/attr.npy",
        "results/{run}/attr/ohe.npy",
        "results/{run}/attr/regions.bed"
    params: dirs=lambda w: [f"results/{w.run}/shards/{i}" for i in shards(w.run)]
    shell: "{PY} {S}/merge_shards.py --out_dir results/{wildcards.run}/attr {params.dirs}"


rule discover:
    input: rules.merge_shards.output
    output: "results/{run}/modisco/modisco.h5"
    params: m=config["modisco"]
    threads: 32
    log: "results/{run}/modisco/modisco.log"
    shell:
        "OMP_NUM_THREADS={threads} NUMBA_NUM_THREADS={threads} {ENV}/bin/modisco motifs -s {input[1]} -a {input[0]} -n {params.m[max_seqlets]} "
        "-w {params.m[window]} -l {params.m[n_leiden]} -o {output} > {log} 2>&1"


rule extract_regions:
    input: rules.merge_shards.output
    output: "results/{run}/finemo_regions/regions.npz"
    params: w=lambda w: width(w.run)
    shell:
        "{ENV}/bin/finemo extract-regions-modisco-fmt -s {input[1]} -a {input[0]} "
        "-p {input[2]} -o {output} -w {params.w}"


rule call_hits:
    input: npz=rules.extract_regions.output, h5=rules.discover.output
    output: "results/{run}/hits/hits.tsv"
    params: f=config["finemo"], out="results/{run}/hits"
    resources: gpu=1
    shell:
        "CUDA_VISIBLE_DEVICES={GPUS[0]} {ENV}/bin/finemo call-hits -M {params.f[mode]} "
        "-r {input.npz} -m {input.h5} -o {params.out} -t {params.f[threshold]} "
        "-l {params.f[alpha]} -b {params.f[batch]}"


rule modisco_report:
    input: h5=rules.discover.output, lex=UNION
    output: "results/{run}/modisco_report/motifs.html"
    params: out="results/{run}/modisco_report", tomtom=Path(LEX["tomtom"]).parent
    shell: "PATH={params.tomtom}:$PATH {ENV}/bin/modisco report -i {input.h5} -o {params.out} -m {input.lex}"


rule finemo_report:
    input: npz=rules.extract_regions.output, hits=rules.call_hits.output, h5=rules.discover.output
    output: "results/{run}/finemo_report/report.html"
    params: hits="results/{run}/hits", out="results/{run}/finemo_report"
    shell:
        "{ENV}/bin/finemo report -r {input.npz} -H {params.hits} -m {input.h5} "
        "-W {config[modisco][window]} -o {params.out}"


rule lexicon_union:
    input: LEX["sources"]
    output: UNION
    shell: "{PY} {S}/lexicon_union.py --out {output} {input}"


rule name_patterns:
    input: h5=rules.discover.output, lex=UNION, h14=LEX["h14_annotation"]
    output: "results/{run}/naming/pattern_names.tsv"
    params: out="results/{run}/naming"
    shell:
        "{PY} {S}/name_patterns.py --h5 {input.h5} --target {input.lex} "
        "--h14_annotation {input.h14} --tomtom {LEX[tomtom]} "
        "--qval_threshold {LEX[qval_threshold]} --out_dir {params.out}"


MODEL = {"cbp": "ChromBPNet", "ag": "AlphaGenome"}


rule plot_region:
    input:
        lambda w: expand("results/{run}/{f}", run=config["plots"][w.plot]["runs"],
                         f=("attr/attr.npy", "hits/hits.tsv", "naming/pattern_names.tsv"))
    output: "results/plots/{plot}.png"
    params:
        region=lambda w: config["plots"][w.plot]["region"],
        runs=lambda w: " ".join(f"--run '{MODEL[RUNS[r]['model']]} ({r})=results/{r}'"
                                for r in config["plots"][w.plot]["runs"])
    shell: "{PY} {S}/plot_region.py --region {params.region} {params.runs} --out {output}"
