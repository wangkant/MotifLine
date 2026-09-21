# motifline

**A reproducible Snakemake workflow for regulatory motif discovery, hit calling,
and annotation from ChromBPNet and AlphaGenome attributions.**

Starting from genomic peaks and a trained sequence model, motifline computes
sequence attributions, discovers recurring motifs, identifies their occurrences,
and assigns reference-based annotations. A single configuration file connects
these stages, with a shared downstream workflow for both supported models.

[Installation](docs/INSTALL.md) · [Example configuration](config/example.yaml) ·
[Output reference](docs/ARTIFACTS.md)

## What it does

- **Discover motifs:** compute DeepLIFT/SHAP attributions with tangermeme and
  identify recurring patterns with TF-MoDISco through modisco-lite.
- **Locate motif occurrences:** use FiNeMo to call motif hits across the analyzed regions.
- **Annotate patterns:** match motifs with TomTom against an RSAT-clustered
  HOCOMOCO and JASPAR reference lexicon, with significance and similarity gates.
- **Keep analyses traceable:** record stage commands, parameters, environments,
  and tool versions in `run_info.json`; track parameter changes for reruns.
- **Support different run modes:** compute new attributions in one process or
  across GPU shards, or reuse archived attributions for downstream analysis.

motifline integrates established methods; it does not introduce a new motif
analysis algorithm. Its contribution is a consistent, configurable workflow from
attribution to annotated motif hits. ChromBPNet and AlphaGenome share every stage
after `attr/`, making the downstream analysis consistent across models. Model
windows and retained regions can still differ; see the [output reference](docs/ARTIFACTS.md).

## Workflow

```text
Genomic peaks + model weights + reference genome
                     |
                     v
       Attribute: ChromBPNet or AlphaGenome
            tangermeme DeepLIFT/SHAP
                     |
          attr.npy / ohe.npy / regions.bed
                     |
             +-------+--------------------+
             |                            |
             v                            v
     Discover: TF-MoDISco          Package regions: FiNeMo
         modisco.h5                    regions.npz
             |                            |
             +-------------+--------------+
             |             |
             |             v
             |     Call motif hits: FiNeMo
             |          hits.tsv
             v
     Annotate: TomTom + annotation gates
         pattern_names.tsv
             ^
             |
     Shared reference lexicon
     HOCOMOCO H14 CORE + JASPAR CORE vertebrates NR
     -> RSAT clustering -> medoids + cluster-tagged union
```

The reference lexicon is built once and shared across runs. The reference set
used in this repository yields 821 clusters, each represented by one medoid;
TomTom matches against the cluster-tagged union of reference motifs.

## Installation

Setup uses four conda environments, model source checkouts, model weights, and
reference data. Follow the [installation guide](docs/INSTALL.md) for the complete
setup, including CUDA requirements and reference file formats.

```bash
git clone https://github.com/wangkant/motifline.git
cd motifline

ENVS=$HOME/motifline-envs
conda env create -p "$ENVS/orchestration" -f envs/orchestration.yaml
conda env create -p "$ENVS/compute"       -f envs/compute.yaml
conda env create -p "$ENVS/meme"          -f envs/meme.yaml
conda env create -p "$ENVS/rsat"          -f envs/rsat.yaml
"$ENVS/orchestration/bin/pip" install -e .
conda activate "$ENVS/orchestration"
```

**AlphaGenome requires the `deeplift-support` branch of Ruchir Rastogi's
`alphagenome-pytorch` fork.** The attribution rules used here are not included in
the base PyTorch port described in the installation guide. Set `ag_src` to that
checkout's `src/` directory **inside each AlphaGenome run entry**.

For computed runs, provide a reference genome and a 10- or 11-column narrowPeak
file with summit offsets in column 10, plus the model checkpoint. The repository
includes a selected 33,917-peak K562 set for the example analyses.

## Quick start

After completing the installation and reference setup, run these commands from
the repository directory with the orchestration environment active:

```bash
motifline init                 # create config/default.yaml
$EDITOR config/default.yaml    # set paths, model weights, and runs
motifline check                # validate configuration and input paths
motifline doctor               # inspect versions, defaults, and model imports
motifline smoke                # check motif discovery and hit calling on CPU
motifline run -n               # preview the workflow
motifline run -j 8 --gpus 2     # execute with up to 8 jobs and 2 GPU resources
```

Choose available devices with `CUDA_VISIBLE_DEVICES`; `--gpus` limits GPU
concurrency. You can also request a specific output:

```bash
motifline run results/<run>/hits/hits.tsv -j 4
```

### Configure runs

Each entry under `runs:` defines one analysis. The
[example configuration](config/example.yaml) includes computed, sharded, adopted,
and cancelled runs, along with shared parameters and reference settings.

| Setting | Behavior |
|---|---|
| `model: cbp` or `model: ag` | Select ChromBPNet or AlphaGenome. |
| `adopt_from` | Reuse archived attributions and run downstream stages. |
| `n_shards` | Split attribution into GPU shards, then merge the results. |
| Neither `adopt_from` nor `n_shards` | Compute attributions in one process. |
| `cancelled: true` | Exclude the run from default targets; explicit targets remain available. |

The configuration follows upstream tool defaults where applicable and documents
explicit choices where required. `motifline doctor` reports deviations from
installed defaults and changes relative to the pinned defaults, so intentional
parameter choices remain visible.

### Validate the setup

| Command | What it checks |
|---|---|
| `motifline check` | Missing or misspelled keys, missing paths, and peak files without the required summit column, reported together. |
| `motifline doctor` | Configuration, installed tool versions and defaults, parameter deviations, and model imports. Parameter deviations alone do not block a run. |
| `motifline smoke` | Recovery of an AP-1 motif planted in 180 of 300 synthetic attributions using MoDISco and FiNeMo. Runs on CPU in about a minute without model weights. |

The smoke test covers motif discovery and hit calling. It does **not** validate
model attribution or the reference naming chain.

## Outputs

Results are organized by run under `results/<run>/`. Key files include:

| File | Contents |
|---|---|
| `attr.npy` | Hypothetical contribution scores for each analyzed sequence. |
| `ohe.npy` | One-hot encoded sequences aligned to the attributions. |
| `regions.bed` | Regions corresponding to the attribution rows. |
| `modisco.h5` | Discovered motif patterns. |
| `regions.npz` | Region data packaged for FiNeMo. |
| `hits.tsv` | Called motif occurrences. |
| `pattern_names.tsv` | Reference-based pattern annotations. |
| `run_info.json` | Per-stage execution metadata and provenance. |

Use `regions.bed` as the row key for downstream analysis: filtering during
attribution means the retained rows may differ from the original peak file.
See [docs/ARTIFACTS.md](docs/ARTIFACTS.md) for stage directories, array shapes,
checkpoint behavior, and model-specific details.

## Repository layout

```text
Snakefile             Snakemake workflow
config/default.yaml   Local configuration generated by motifline init (not tracked)
config/example.yaml   Example runs and shared settings
config/peaks/         Selected K562 peak set used by the example analyses
envs/                 Environment recipes and exact package locks
src/motifline/        CLI, validation, attribution, naming, and orchestration
src/motifline/steps/  Scripts executed in the compute environment
docs/                 Installation and output documentation
tests/                Tests runnable without a GPU
```

Implementation rationale lives alongside the relevant code: `naming.py` covers
the reference lexicon and annotation gates, `tools.py` covers parameter checks,
`attribution.py` covers hypothetical attribution measurements, and
`steps/attr_ag.py` / `steps/ag_support.py` cover AlphaGenome attribution.

## Citation and attribution

If you use motifline, cite the pipeline using [CITATION.cff](CITATION.cff) and
the underlying tools and methods relevant to your analysis.

| tool | version here | where |
|---|---|---|
| ChromBPNet (PyTorch port) | `jsxlei/chrombpnet-pytorch@01ad10b` | github.com/kundajelab/chrombpnet |
| AlphaGenome (the model) | — | github.com/google-deepmind/alphagenome |
| AlphaGenome PyTorch port | `genomicsxai/alphagenome-pytorch` | github.com/genomicsxai/alphagenome-pytorch |
| AlphaGenome DeepLIFT rules | `rastogiruchir/alphagenome-pytorch@64e9b75`, branch `deeplift-support` | Ruchir Rastogi — github.com/rastogiruchir/alphagenome-pytorch |
| tangermeme | 1.0.4 | github.com/jmschrei/tangermeme |
| modisco-lite | 2.4.0 | github.com/jmschrei/tfmodisco-lite |
| FiNeMo | 0.30 (`austintwang/finemo_gpu@5c6f521`) | github.com/austintwang/finemo_gpu |
| HOCOMOCO | H14 CORE | hocomoco.autosome.org |
| JASPAR | CORE vertebrates non-redundant | jaspar.elixir.no |

The AG DeepLIFT rules are Ruchir Rastogi's; cite them from his repository rather than from
the port or from DeepMind.

Published references for the methods: DeepLIFT — Shrikumar, Greenside & Kundaje (2017),
ICML, arXiv:1704.02685 · TF-MoDISco — Shrikumar et al. (2018), arXiv:1811.00416 · RSAT
matrix-clustering — Castro-Mondragon et al. (2017), *NAR* 45:e119, doi:10.1093/nar/gkx314
· TomTom — Gupta et al. (2007), *Genome Biology* 8:R24, doi:10.1186/gb-2007-8-2-r24 ·
MEME Suite — Bailey et al. (2015), *NAR* 43:W39, doi:10.1093/nar/gkv416 · Snakemake —
Mölder et al. (2021), *F1000Research* 10:33, doi:10.12688/f1000research.29032.2

`CITATION.cff` covers this pipeline itself.

## License

MIT — see [LICENSE](LICENSE).
