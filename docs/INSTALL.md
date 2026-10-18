# Installing motifline

The motifline package, three more conda environments, the model source checkouts
your runs need, reference files and a config. `motifline check` reports what is still missing;
`motifline smoke` tests the downstream half without a model.

motifline only assembles command lines: each stage runs in its own environment, so
the orchestration and compute environments can hold incompatible packages.

## 1. The package and a project

```bash
ENVS=$HOME/motifline-envs
conda create -p $ENVS/orchestration python=3.12        # or any Python >= 3.11 env
$ENVS/orchestration/bin/pip install "git+https://github.com/wangkant/motifline.git"
conda activate $ENVS/orchestration
mkdir my-analysis && cd my-analysis
motifline init
```

The environment motifline is installed in is the orchestration environment: it runs
snakemake (installed with motifline) and the `motifline` command. For the exact
versions motifline was tested with, install `envs/locks/orchestration.pip.txt` into it
after `init`.

The project directory is where motifline runs: `init` writes `config/default.yaml`
and `envs/` there, and `run` writes `results/` and snakemake's `.snakemake/` there.
Relative paths in the config are resolved against it. A config that sits in another
project (next to its `.snakemake/`) is refused, so a run started from the wrong
directory does not begin again from nothing. In a checkout of this repository the
checkout root is the project, as before.

## 2. The environments

`envs/*.yaml` pin the versions motifline is tested with; `envs/locks/` holds full
package lists for reproducing results exactly. `motifline init` copies both from the
package into the project and leaves existing env files alone (`--force` replaces
them); it never replaces an existing config. Build the prefixes outside the project
and point the config at them.

```bash
conda env create -p $ENVS/compute    -f envs/compute.yaml
conda env create -p $ENVS/chrombpnet -f envs/chrombpnet.yaml
conda env create -p $ENVS/meme       -f envs/meme.yaml
```

| env | config key | runs | needed for |
|---|---|---|---|
| orchestration | — | snakemake, `motifline` | always |
| compute | `envs.compute` | MoDISco, FiNeMo, AlphaGenome and PyTorch-port attribution | always |
| chrombpnet | `envs.chrombpnet` | official ChromBPNet attribution (TensorFlow) | computed ChromBPNet runs |
| meme | `reference_lexicon.tomtom` | `tomtom` | naming patterns |

`envs/chrombpnet.yaml` installs chrombpnet 1.0.1 from PyPI, which pins TensorFlow
2.8.0, kundajelab-shap, deeplift and deepdish, with CUDA 11.2 and cuDNN 8.1 from
conda-forge. The step sets `TF_FORCE_GPU_ALLOW_GROWTH`, so TensorFlow takes only the
memory it needs instead of the whole card. It runs chrombpnet's own interpret code
with one change: the official code draws its dinucleotide-shuffled references
unseeded, and motifline seeds shuffle k with `random_state + k`, as tangermeme does on
the other paths. GPU arithmetic can still change the last float16 digit of a score
between runs.

`envs/compute.yaml` pins `torch==2.3.1+cu121`; for another CUDA build change that
line and the index URL. The DeepLIFT/SHAP references are seeded, but a different
torch build can change the last digits of an attribution, so record the version
with any attributions you keep (`run_info.json` does).

With `backend: pytorch`, importing the port's `chrombpnet.interpret` downloads a motif
file and genome chrom sizes through `pooch`. On a GPU node without network access, run the import once on a
connected node so the files are in pooch's cache.

## 3. The model code

The official ChromBPNet is a package in its env. The ChromBPNet PyTorch port and
AlphaGenome are not: the attribution steps put their checkouts on `sys.path`, and the
config points at them. `chrombpnet_src` is needed only for `backend: pytorch` runs, the
fallback when the official TensorFlow code cannot be used.

```bash
git clone https://github.com/jsxlei/chrombpnet-pytorch.git          # chrombpnet_src
git -C chrombpnet-pytorch checkout 01ad10b
git clone -b deeplift-support https://github.com/rastogiruchir/alphagenome-pytorch.git
git -C alphagenome-pytorch checkout 64e9b75                       # a run's ag_src = .../src
```

The AlphaGenome path needs the `deeplift-support` branch of Ruchir Rastogi's fork:
its `attribution/deeplift.py` is not in `genomicsxai/alphagenome-pytorch`. A branch on
a personal fork can move or disappear, so keep your own copy if the AG path matters.

`ag_src` is set per run, not at the top level. Each AG run also names its `track` (an
index into the ATAC head) and the `biosample` of that track; the step looks the track
up in `alphagenome_pytorch/data/track_metadata_human.parquet` and refuses a mismatch
(K562 is track 60).

Model weights: a ChromBPNet run needs, as `wo_bias`, the Keras model chrombpnet trains
without the bias (`chrombpnet_nobias.h5`). With `backend: pytorch` it needs instead a
bias-corrected PyTorch state dict for `BPNet(512 filters, 8 layers)`. An AlphaGenome
run needs `weights` loadable by `AlphaGenome.from_pretrained`.

Official ChromBPNet scores computed elsewhere can be imported instead: point an
adopted run at the directory holding `<prefix>.counts_scores.h5` and
`<prefix>.interpreted_regions.bed` from chrombpnet's interpret step, with
`adopt_format: chrombpnet_h5` (see `config/example.yaml` in the repository). The import needs
`genome.fasta`, the same genome chrombpnet read. An interpret run done in shards is
stacked into one prefix first, in envs.compute:

```bash
motifline merge-h5 --shard-prefix run/shard_0/k562 --shard-prefix run/shard_1/k562 \
    --out-prefix merged/k562 --check-bed peaks.bed
```

## 4. Reference data

| what | config key | source |
|---|---|---|
| `hg38.fa`, `hg38.chrom.sizes` | `genome.fasta`, `genome.chrom_sizes` | UCSC |
| HOCOMOCO H14 CORE, MEME format (1,595 matrices) | `reference_lexicon.sources.H14` | hocomoco14.autosome.org |
| HOCOMOCO `H14CORE_annotation.jsonl` | `reference_lexicon.h14_annotation` | hocomoco14.autosome.org, same release |
| JASPAR CORE vertebrates non-redundant, MEME format (1,019) | `reference_lexicon.sources.JAS` | jaspar.elixir.no |

TomTom matches against the union of the two libraries as published, without further
de-duplication.

Peaks must be 10- or 11-column narrowPeak (11th column `True`) with the summit
offset in column 10, no header lines, every chromosome in `chrom_sizes`, every summit inside its peak and no
two peaks with the same summit. `motifline check` verifies all of this.

The repository's `config/peaks/k562_allpeaks_33917.bed` is an example K562 peak set
covering 15 chromosomes; it is not part of the package. For your own data, point a
run's `peaks:` at your narrowPeak.

## 5. The config

```bash
$EDITOR config/default.yaml
motifline check
```

The repository's `config/example.yaml` has a sharded official ChromBPNet run, a
PyTorch-port run, an AG run, an imported official ChromBPNet run and a cancelled
adopted run.
`motifline check` reports, in one pass: missing or misspelled keys (also inside
`modisco:`, `finemo:` and the other sections), values of the wrong type, missing
paths and invalid peak files. Relative paths resolve against the project directory.

## 6. Does it work?

```bash
motifline smoke
```

Plants AP-1 (`TGACTCA`) in 180 of 300 synthetic hypothetical attributions, runs
MoDISco and FiNeMo, and checks that the motif is discovered and that the hits land
on the planted positions. Needs no GPU (it uses one only if `CUDA_VISIBLE_DEVICES` is
set), no genome and no weights. It works in a temporary directory, or in an empty
`--workdir`, and deletes it after a pass. Attribution and naming are
not covered.

## 7. Running

```bash
motifline run -n
motifline run -j 8
motifline run results/<run>/hits/hits.tsv -j 4
motifline run -j 8 -- --rerun-incomplete      # anything after -- goes to snakemake
```

A stage reruns when its declared inputs (for attribution: checkpoint, peaks, genome)
or its parameters change, or when its rule body in the Snakefile changes. Edits to
the step scripts (`motifline/steps/*.py`) or to the model checkouts do not trigger
reruns, including when they arrive with a motifline upgrade; after either, force the
affected stages with `-R <rule>`. Each `run_info.json` records the motifline version and the
code it ran: the commit of a git checkout, or a digest of the installed package's
code; attribution stages also record the model checkouts, and the official ChromBPNet
stage the chrombpnet, tensorflow, kundajelab-shap and deeplift versions of its env.

`motifline run` fixes the run's cards once: `--devices 0,2` (ids or UUIDs) if given,
else `CUDA_VISIBLE_DEVICES` if set, else every card `nvidia-smi` lists. Each GPU step
(attribution, its shards, FiNeMo call-hits) takes a free card from that set under a file
lock in `.motifline/gpu_locks/` and runs with `CUDA_VISIBLE_DEVICES` set to it alone,
so concurrent jobs use different cards and shards run in parallel. `--gpus N` caps
concurrent GPU jobs (default: the number of cards); a job that finds every card taken
waits. The card a stage ran on is recorded as `device` in its `run_info.json`. Running
snakemake directly, without `motifline run`, leaves the device choice to the environment.

## Reproducing the exact environment

```bash
conda create -p $ENVS/compute --file envs/locks/compute.conda.txt
$ENVS/compute/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r envs/locks/compute.pip.txt
$ENVS/compute/bin/pip install pyarrow==17.0.0

conda create -p $ENVS/chrombpnet -c conda-forge -c bioconda --file envs/locks/chrombpnet.conda.txt
$ENVS/chrombpnet/bin/pip install -r envs/locks/chrombpnet.pip.txt
```

The locks predate the AlphaGenome track check, which needs `pyarrow`.
