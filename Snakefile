configfile: "config/config.yaml"

from pathlib import Path

S = Path(workflow.basedir) / "scripts"
ENV = config["envs"]["compute"]
PY = f"{ENV}/bin/python"
G = config["genome"]
RUNS = config["runs"]

wildcard_constraints:
    run="[^/]+"


def width(run):
    # what the attribution step stores: ChromBPNet's 2114 bp input, AlphaGenome's 2048
    return 2048 if RUNS[run]["model"] == "ag" else 2114


def attr_cmd(w):
    r = RUNS[w.run]
    a = config["attribution"]
    common = (f"--peaks {r['peaks']} --fasta {G['fasta']} --chrom_sizes {G['chrom_sizes']} "
              f"--out_dir results/{w.run}/attr "
              f"--n_shuffles {a['n_shuffles']} --random_state {a['random_state']}")
    if r["model"] == "ag":
        return (f"{PY} {S}/attr_ag.py --ag_src {r['ag_src']} --weights {r['weights']} "
                f"--track {r['track']} {common}")
    return (f"{PY} {S}/attr_cbp.py "
            f"--wo_bias {r['wo_bias']} {common}")


rule all:
    input:
        expand("results/{run}/hits/hits.tsv", run=RUNS)


rule attribute:
    output:
        "results/{run}/attr/attr.npy",
        "results/{run}/attr/ohe.npy",
        "results/{run}/attr/regions.bed"
    params: cmd=attr_cmd
    resources: gpu=1
    log: "results/{run}/attr/attr.log"
    shell: "{params.cmd} > {log} 2>&1"


rule discover:
    input: rules.attribute.output
    output: "results/{run}/modisco/modisco.h5"
    params: m=config["modisco"]
    log: "results/{run}/modisco/modisco.log"
    shell:
        "{ENV}/bin/modisco motifs -s {input[1]} -a {input[0]} -n {params.m[max_seqlets]} "
        "-w {params.m[window]} -o {output} > {log} 2>&1"


rule extract_regions:
    input: rules.attribute.output
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
        "{ENV}/bin/finemo call-hits -M {params.f[mode]} -r {input.npz} -m {input.h5} "
        "-o {params.out} -t {params.f[threshold]} -l {params.f[alpha]} -b {params.f[batch]}"
