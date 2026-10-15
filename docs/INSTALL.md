# Installing motifline

Three conda environments, two model source checkouts, reference files and a config.
`motifline check` reports what is still missing; `motifline smoke` tests the
downstream half without a model.

motifline only assembles command lines: each stage runs in its own environment, so
the orchestration and compute environments can hold incompatible packages.

## 1. The package

```bash
git clone https://github.com/wangkant/motifline.git
cd motifline
```

## 2. The environments

`envs/*.yaml` pin the versions motifline is tested with; `envs/locks/` holds full
package lists for reproducing results exactly. Build the prefixes outside the
repository and point the config at them.

```bash
ENVS=$HOME/motifline-envs
conda env create -p $ENVS/orchestration -f envs/orchestration.yaml
conda env create -p $ENVS/compute       -f envs/compute.yaml
conda env create -p $ENVS/meme          -f envs/meme.yaml
$ENVS/orchestration/bin/pip install -e .
```

| env | runs | needed for |
|---|---|---|
| orchestration | snakemake, `motifline` | always |
| compute | attribution steps, MoDISco, FiNeMo | always |
| meme | `tomtom` | naming patterns |

`envs/compute.yaml` pins `torch==2.3.1+cu121`; for another CUDA build change that
line and the index URL. The DeepLIFT/SHAP references are seeded, but a different
torch build can change the last digits of an attribution, so record the version
with any attributions you keep (`run_info.json` does).

Importing `chrombpnet.interpret` downloads a motif file and genome chrom sizes
through `pooch`. On a GPU node without network access, run the import once on a
connected node so the files are in pooch's cache.

## 3. The model code

Neither model is installed as a package. The attribution steps put the checkouts on
`sys.path`; the config points at them.

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

Model weights: a ChromBPNet run needs a bias-corrected PyTorch state dict for
`BPNet(512 filters, 8 layers)` (`wo_bias`); Keras `.h5` models are not read. An
AlphaGenome run needs `weights` loadable by `AlphaGenome.from_pretrained`.
Official TensorFlow ChromBPNet results are imported rather than computed: point an
adopted run at the directory holding `<prefix>.counts_scores.h5` and
`<prefix>.interpreted_regions.bed` from chrombpnet's interpret step, with
`adopt_format: chrombpnet_h5` (see `config/example.yaml`). The import needs
`genome.fasta`, the same genome chrombpnet read.

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

`config/peaks/k562_allpeaks_33917.bed` is an example K562 peak set covering 15
chromosomes. For your own data, point a run's `peaks:` at your narrowPeak.

## 5. The config

```bash
motifline init
$EDITOR config/default.yaml
motifline check
```

`config/example.yaml` has one computed CBP run, one AG run and one adopted run.
`motifline check` reports, in one pass: missing or misspelled keys (also inside
`modisco:`, `finemo:` and the other sections), values of the wrong type, missing
paths and invalid peak files. Relative paths resolve against the motifline checkout.

## 6. Does it work?

```bash
motifline smoke
```

Plants AP-1 (`TGACTCA`) in 180 of 300 synthetic hypothetical attributions, runs
MoDISco and FiNeMo, and checks that the motif is discovered and that the hits land
on the planted positions. Needs no GPU (it uses one only if `CUDA_VISIBLE_DEVICES` is
set), no genome and no weights. Attribution and naming are
not covered.

## 7. Running

```bash
motifline run -n
motifline run -j 8 --gpus 1
motifline run results/<run>/hits/hits.tsv -j 4
motifline run -j 8 -- --rerun-incomplete      # anything after -- goes to snakemake
```

A stage reruns when its declared inputs (for attribution: checkpoint, peaks, genome)
or its parameters change, or when its rule body in the Snakefile changes. Edits to
`src/motifline/steps/*.py` or to the model checkouts do not trigger reruns; after
updating either, force the affected stages with `-R <rule>`. Each `run_info.json`
records the motifline commit, and attribution stages also the model checkouts'.

Every GPU step uses the first visible device. `--gpus N` only limits how many GPU
jobs run at once, so with N > 1 they share that device. Set `CUDA_VISIBLE_DEVICES`
to the card you want.

## Reproducing the exact environment

```bash
conda create -p $ENVS/compute --file envs/locks/compute.conda.txt
$ENVS/compute/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r envs/locks/compute.pip.txt
$ENVS/compute/bin/pip install pyarrow==17.0.0
```

The locks predate the AlphaGenome track check, which needs `pyarrow`.
