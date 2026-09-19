# Installing motifline

Four conda environments, two model source checkouts, three reference files, one
config. `motifline check` tells you what is still missing, and `motifline smoke`
tells you whether the half that does not need a model works.

The pipeline itself is pure orchestration: it assembles command lines and hands them
to interpreters in other environments. Nothing here is imported into the environment
that runs snakemake, which is why they can hold incompatible versions of torch.

---

## 1. The package

```bash
git clone https://github.com/wangkant/motifline.git
cd motifline
```

## 2. The environments

`envs/*.yaml` are readable recipes pinned at the versions these runs were produced
with; `envs/locks/` holds the exact package trees of the machine that produced them,
for reproducing published numbers to the digit. Build the prefixes somewhere outside
the repository — anywhere you like, config points at them by path — so that a clone
never starts tracking a conda tree.

```bash
ENVS=$HOME/motifline-envs                  # or wherever
conda env create -p $ENVS/orchestration -f envs/orchestration.yaml
conda env create -p $ENVS/compute       -f envs/compute.yaml
conda env create -p $ENVS/meme          -f envs/meme.yaml      # TomTom
conda env create -p $ENVS/rsat          -f envs/rsat.yaml      # matrix-clustering
$ENVS/orchestration/bin/pip install -e .
```

| env | what runs in it | needed for |
|---|---|---|
| orchestration | snakemake, `motifline` | always |
| compute | the attribution steps, MoDISco, FiNeMo | always |
| meme | `tomtom` | naming patterns |
| rsat | `matrix-clustering` | building the reference lexicon (once) |

`mamba` works in place of `conda` and is much faster on the rsat env (733 packages).

**CUDA.** `envs/compute.yaml` pins `torch==2.3.1+cu121`. If your driver needs a
different CUDA build, change that line and the `--extra-index-url` above it. The
DeepLIFT/SHAP references are seeded (`attribution.random_state`), so a torch change
can move the last digits of an attribution — state the version next to anything you
keep.

## 3. The model code

Neither model is a pip package here. The attribution steps put a checkout on
`sys.path` themselves, so you clone the repository and point config at it.

```bash
git clone https://github.com/jsxlei/chrombpnet-pytorch.git         # -> chrombpnet_src
git clone https://github.com/genomicsxai/alphagenome-pytorch.git   # -> a run's ag_src
```

The runs in this repository used `chrombpnet-pytorch@01ad10b` and
`alphagenome-pytorch@64e9b75`. `ag_src` is the `src/` directory inside that checkout,
and it is declared **per run**, not at the top level — `attr_argv` reads
`r["ag_src"]`, so a top-level key of that name is never read by anything.

Model weights are yours to supply: a ChromBPNet run needs a bias-corrected checkpoint
(`wo_bias`), an AlphaGenome run needs `weights`.

## 4. Reference data

| what | used by | where it came from |
|---|---|---|
| `hg38.fa` + `hg38.chrom.sizes` | every computed run | UCSC |
| HOCOMOCO H14 CORE, MEME format (1,595 matrices) | `reference_lexicon.sources.H14` | hocomoco.autosome.org |
| JASPAR CORE vertebrates non-redundant, MEME format (1,019) | `reference_lexicon.sources.JAS` | jaspar.elixir.no |

Both motif libraries are the publishers' own non-redundant releases and are used
without further de-duplication on purpose; `src/motifline/naming.py` says why the
TomTom target has to be the full set rather than the cluster representatives.

Peaks must be 10- or 11-column narrowPeak with the summit offset in column 10:
`finemo extract-regions-modisco-fmt -p` names the first ten columns positionally and
reads the summit from there. `motifline check` verifies this. A 33,917-peak K562 list
in that form ships at `config/peaks/k562_allpeaks_33917.bed`.

## 5. The config

```bash
motifline init                      # writes config/default.yaml
$EDITOR config/default.yaml         # fill in the paths and declare your runs
motifline check                     # every problem at once, before anything runs
```

`config/example.yaml` is a worked config with entries of each kind: computed,
sharded, adopted, and cancelled. `motifline check` reports missing keys, misspelled
keys (`n_shard` would otherwise silently make a sharded run single-process), paths
that do not exist and peaks files without a summit column — all of them in one pass.

## 6. Is it set up correctly?

```bash
motifline doctor
```

One page: the config and its problems, the tool versions the compute env actually
reports, whether every parameter is on its tool's official default, the config-key →
tool-flag mapping, and whether the model checkouts import the way the attribution steps
import them. A deviating parameter is reported, never blocked.

`doctor` exits non-zero on a config problem, never on a deviation.

## 7. Does it work?

```bash
motifline smoke
```

This plants AP-1 (`TGACTCA`) in 180 of 300 synthetic hypothetical attributions, runs
MoDISco and FiNeMo over them, and checks that the motif comes back. It takes about a
minute on CPU and needs no genome, no model weights and no GPU.

It covers the compute environment and the downstream half of the pipeline. It does
**not** cover the attribution steps (they need real model weights) or the naming chain
(it needs the RSAT clustering, which is 1h46m).

## 8. Running

```bash
motifline run -j 8 --gpus 2
motifline run -n                                        # dry run
motifline run results/<run>/hits/hits.tsv -j 4          # one target
```

`motifline run` is `snakemake` with `--rerun-triggers mtime params` baked in. That is
not a convenience: with snakemake's default `mtime` alone, editing `finemo.threshold`
or `modisco.window` leaves every finished stage in place and snakemake reports
"Nothing to be done", so `results/` ends up holding a mixture of parameter generations
that nothing on disk distinguishes from a coherent tree.

Every `run` prints the parameter deviation report before it starts.

GPU selection is the caller's. Export `CUDA_VISIBLE_DEVICES` to fully-free cards,
pinned by UUID rather than index, and bound concurrency with `--gpus`.

## Upgrading a tool

`src/motifline/tool_defaults.yaml` pins what each tool's defaults were at the version
this pipeline's parameters were chosen against. After a deliberate upgrade, run
`motifline doctor`: any default the new version moved is listed. Decide for each one
whether it is a change you meant to adopt, then edit the `version` and the value in
that file. Re-pinning is how an upgrade gets acknowledged; it is not a formality, and
it is the only thing that keeps "these are the official defaults" from quietly
becoming "these were the official defaults in 2026".

## Reproducing the exact environment

```bash
conda create -p $ENVS/compute --file envs/locks/compute.conda.txt
$ENVS/compute/bin/pip install -r envs/locks/compute.pip.txt
```

The locks were frozen on 2026-09-19 from the machine the runs in this repository were
produced on. They exclude two editable installs that pointed at directories which no
longer exist and which nothing imports — the pipeline reaches model code through
`chrombpnet_src` / `ag_src`, never through an installed package.
