# motifline

A Snakemake pipeline from model attributions to MoDISco motifs to FiNeMo hits,
for ChromBPNet and AlphaGenome, on the same `tangermeme.deep_lift_shap` code
path. No bespoke algorithms: attribution is tangermeme 1.0.4, discovery is
modisco-lite 2.4.0, hit calling is finemo 0.30 (pinned). Parameters are the
tools' official defaults; the one deviation (`lexicons.batch: 1000`) is a
measured OOM limit on the 289-motif union lexicon.

## Layout

```
Snakefile            the DAG: attr -> discover -> extract_regions -> call_hits
config/default.yaml  runs + shared canonical parameters + site paths
src/motifline/       adopt / merge_shards / lexicons / runs helpers
src/motifline/steps/ attr_cbp.py, attr_ag.py (+ ag_support.py) -- compute-env scripts
skill/SKILL.md       how to drive the pipeline
```

## Running

```bash
envs/motifline/bin/snakemake --rerun-triggers mtime -j <N> --resources gpu=<G> -- <target>
```

`--rerun-triggers mtime` is the standard invocation: the `run:` blocks only
assemble argv, so snakemake's code trigger would otherwise propose recomputing
finished stages after any cosmetic edit.

GPU selection is the caller's: `export CUDA_VISIBLE_DEVICES=<uuid>` (fully-free
cards only, pin by UUID -- indices shift on this node -- and leave one working
card free), and bound concurrency with `--resources gpu=<G>`.

## Environments

| env | role |
|---|---|
| `envs/motifline` (py3.12) | snakemake + this package |
| `envs/chrombpnet_local` (py3.10) | all compute: torch, chrombpnet, tangermeme, modisco-lite, finemo 0.30 |
| `envs/mmv_verify` (py3.12) | finemo 0.41, `call_hits_v041` comparison only (its torch runs CPU on this node) |

Step scripts never import this package; every site path arrives as an argument
(AlphaGenome in particular: pass `--ag_src`, the package is not importable from
the compute env).

## House rules

- Attributions are HYPOTHETICAL (`--hypothetical`), always.
- hits are produced by finemo 0.30; note the version next to any hits you keep.
- `new_experiment/region_v3|v4` archives are read-only.
- peaks fed to finemo must be 10/11-column narrowPeak (summit in column 10);
  the attribution steps normalize this themselves.
