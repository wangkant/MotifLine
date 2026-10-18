# motifline

**A reproducible Snakemake workflow for regulatory motif discovery, hit calling,
and annotation from ChromBPNet and AlphaGenome attributions.**

Starting from genomic peaks and a trained sequence model, motifline computes
sequence attributions, discovers recurring motifs, identifies their occurrences,
and assigns reference-based annotations. A single configuration file connects
these stages, with a shared downstream workflow for both supported models.

[Installation](https://github.com/wangkant/motifline/blob/main/docs/INSTALL.md) · [Example configuration](https://github.com/wangkant/motifline/blob/main/config/example.yaml) ·
[Output reference](https://github.com/wangkant/motifline/blob/main/docs/ARTIFACTS.md)

## What it does

- **Discover motifs:** compute DeepLIFT/SHAP attributions (ChromBPNet: the official
  TensorFlow DeepSHAP from chrombpnet's interpret step; AlphaGenome: tangermeme) and
  identify recurring patterns with TF-MoDISco through modisco-lite.
- **Locate motif occurrences:** use FiNeMo to call motif hits across the analyzed regions.
- **Annotate patterns:** match motifs with TomTom against the union of the
  HOCOMOCO and JASPAR reference motifs; the best match names the pattern. Its
  q-value is reported, and a `confident` column flags matches with q at or below
  `qval_threshold` without dropping the rest.
- **Keep analyses traceable:** record stage commands, parameters, environments,
  and tool versions in `run_info.json`; track parameter changes for reruns.
- **Support different run modes:** compute new attributions in one process or
  across GPU shards, or reuse archived attributions for downstream analysis.

motifline integrates established methods; it does not introduce a new motif
analysis algorithm. Its contribution is a consistent, configurable workflow from
attribution to annotated motif hits. ChromBPNet and AlphaGenome share every stage
after `attr/`, making the downstream analysis consistent across models. Model
windows and retained regions can still differ; see the [output reference](https://github.com/wangkant/motifline/blob/main/docs/ARTIFACTS.md).

## Workflow

```text
Genomic peaks + model weights + reference genome
                     |
                     v
   Attribute: ChromBPNet (official TensorFlow DeepSHAP)
          or AlphaGenome (tangermeme DeepLIFT/SHAP)
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
     Annotate: TomTom, best match + q-value
         pattern_names.tsv
             ^
             |
     Shared reference lexicon
     HOCOMOCO H14 CORE + JASPAR CORE vertebrates NR
     -> their union
```

The reference lexicon is built once and shared across runs: the union of the
HOCOMOCO H14 CORE and JASPAR CORE vertebrates non-redundant motifs. TomTom
matches each pattern against it, and the best match gives the name.

## Installation

motifline is a Python package; the workflow, the steps and the environment recipes
ship inside it. The analysis itself needs three more conda environments, model source
checkouts, model weights and reference data. Follow the [installation guide](https://github.com/wangkant/motifline/blob/main/docs/INSTALL.md) for the complete
setup, including CUDA requirements and reference file formats.

```bash
python -m pip install "git+https://github.com/wangkant/motifline.git"   # Python >= 3.11

mkdir my-analysis && cd my-analysis
motifline init                 # config/default.yaml, plus envs/ with recipes and locks

ENVS=$HOME/motifline-envs
conda env create -p "$ENVS/compute"    -f envs/compute.yaml
conda env create -p "$ENVS/chrombpnet" -f envs/chrombpnet.yaml
conda env create -p "$ENVS/meme"       -f envs/meme.yaml
```

The environment motifline is installed in runs snakemake. ChromBPNet attribution runs
in `chrombpnet` (the official TensorFlow code), the other steps in `compute`, and
TomTom in `meme`; their prefixes go in the config.

**AlphaGenome requires the `deeplift-support` branch of Ruchir Rastogi's
`alphagenome-pytorch` fork.** The attribution rules used here are not included in
the base PyTorch port described in the installation guide. Set `ag_src` to that
checkout's `src/` directory **inside each AlphaGenome run entry**, together with the
ATAC `track` and its `biosample`; the step checks the pair against AlphaGenome's track
metadata.

For computed runs, provide a reference genome and a 10- or 11-column narrowPeak
file with summit offsets in column 10, plus the model: for ChromBPNet, the Keras
model chrombpnet trains without the bias (`chrombpnet_nobias.h5`). The repository
includes an example K562 peak set.

## Quick start

After completing the installation and reference setup, run these commands from
the project directory. It holds `config/default.yaml` and `results/`, and relative
paths in the config are resolved against it.

```bash
$EDITOR config/default.yaml    # set paths, model weights, and runs
motifline check                # validate configuration and input paths
motifline smoke                # check motif discovery and hit calling on CPU
motifline run -n               # preview the workflow
motifline run -j 8               # up to 8 jobs, one GPU job per visible card
```

Each GPU step claims one free card and sees only that card. The cards are
`--devices` if given, else `CUDA_VISIBLE_DEVICES`, else every card `nvidia-smi` lists.
`--gpus` caps concurrent GPU jobs (default: one per card); jobs beyond the free cards
wait. Arguments after `--` go to snakemake.
You can also request a specific output:

```bash
motifline run results/<run>/hits/hits.tsv -j 4
```

### Configure runs

Each entry under `runs:` defines one analysis. The
[example configuration](https://github.com/wangkant/motifline/blob/main/config/example.yaml) includes computed, sharded, adopted,
and cancelled runs, along with shared parameters and reference settings.

| Setting | Behavior |
|---|---|
| `model: cbp` or `model: ag` | Select ChromBPNet or AlphaGenome. |
| `track` + `biosample` (AG) | ATAC head index and the biosample it must be. |
| `adopt_from` | Reuse attributions computed elsewhere (`attr.npy` + `ohe.npy`) and run downstream stages. |
| `adopt_format: chrombpnet_h5` | Import official ChromBPNet `counts_scores.h5` + `interpreted_regions.bed` computed elsewhere, checked row by row against the genome. `motifline merge-h5` stacks a sharded interpret run into one. |
| `n_shards` | Split attribution into GPU shards, then merge the results. |
| Neither `adopt_from` nor `n_shards` | Compute attributions in one process. |
| `cancelled: true` | Exclude the run from default targets and freeze its `attr/`: it is never recomputed or re-adopted, and downstream stages can still be rebuilt from it by explicit target. |

The configuration follows upstream tool defaults where applicable and documents
explicit choices where required. `motifline check` reports where parameters deviate
from the installed tools' defaults, so intentional parameter choices remain visible.

### Validate the setup

| Command | What it checks |
|---|---|
| `motifline check` | Missing or misspelled keys, wrong value types, missing paths, and invalid peak files (columns, summit inside the peak, known chromosomes, no repeated summits), reported together. |
| `motifline smoke` | An AP-1 motif planted in 180 of 300 synthetic attributions is discovered by MoDISco and its FiNeMo hits fall on the planted positions. No GPU or model weights needed. |

The smoke test covers motif discovery and hit calling. It does **not** validate
model attribution or naming.

## Outputs

Results are organized by run under `results/<run>/` in the project directory. Key
files include:

| File | Contents |
|---|---|
| `attr.npy` | Hypothetical contribution scores for each analyzed sequence. |
| `ohe.npy` | One-hot encoded sequences aligned to the attributions. |
| `regions.bed` | Regions corresponding to the attribution rows. |
| `modisco.h5` | Discovered motif patterns. |
| `regions.npz` | Region data packaged for FiNeMo. |
| `hits.tsv` | Called motif occurrences, per region; overlapping windows repeat a hit. |
| `hits_unique.tsv` | The same hits, once per genomic position. |
| `pattern_names.tsv` | Reference-based pattern annotations. |
| `run_info.json` | Per-stage execution metadata and provenance. |

Use `regions.bed` as the row key for downstream analysis: peaks whose window runs
off the chromosome are dropped, and so are peaks with a non-ACGT base except on the
official ChromBPNet path, which keeps them as chrombpnet does. Rows can therefore
differ from the peak file and between models.
See [docs/ARTIFACTS.md](https://github.com/wangkant/motifline/blob/main/docs/ARTIFACTS.md) for stage directories, array shapes,
checkpoint behavior, and model-specific details.

## Repository layout

```text
src/motifline/            CLI, validation, attribution, naming, and orchestration
src/motifline/Snakefile   Snakemake workflow
src/motifline/steps/      Scripts executed in the compute and chrombpnet environments
src/motifline/envs/       Environment recipes and exact package locks
config/example.yaml       Example runs and shared settings
config/peaks/             Example K562 peak set
docs/                     Installation and output documentation
tests/                    Tests runnable without a GPU
```

To work on motifline itself: `pip install -e ".[dev]"` in a checkout, then `pytest`.

## Citation and attribution

If you use motifline, cite the pipeline using [CITATION.cff](https://github.com/wangkant/motifline/blob/main/CITATION.cff) and
the underlying tools and methods relevant to your analysis.

| tool | version here | where |
|---|---|---|
| ChromBPNet (official, TensorFlow; the `attribute` rule for cbp runs, and `adopt_format: chrombpnet_h5`) | 1.0.1 | github.com/kundajelab/chrombpnet |
| DeepSHAP for TensorFlow, as ChromBPNet uses it | `kundajelab-shap` 1 | github.com/kundajelab/shap |
| DeepLIFT (dinucleotide shuffles for the ChromBPNet references) | 0.6.13.0 | github.com/kundajelab/deeplift |
| AlphaGenome (the model) | — | github.com/google-deepmind/alphagenome |
| AlphaGenome PyTorch port | `genomicsxai/alphagenome-pytorch` | github.com/genomicsxai/alphagenome-pytorch |
| AlphaGenome DeepLIFT rules | `rastogiruchir/alphagenome-pytorch@64e9b75`, branch `deeplift-support` | Ruchir Rastogi — github.com/rastogiruchir/alphagenome-pytorch |
| tangermeme (AlphaGenome) | 1.0.4 | github.com/jmschrei/tangermeme |
| modisco-lite | 2.4.0 | github.com/jmschrei/tfmodisco-lite |
| FiNeMo | 0.30 (`austintwang/finemo_gpu@5c6f521`) | github.com/austintwang/finemo_gpu |
| HOCOMOCO | H14 CORE | hocomoco.autosome.org |
| JASPAR | CORE vertebrates non-redundant | jaspar.elixir.no |

The AG DeepLIFT rules are Ruchir Rastogi's; cite them from his repository rather than from
the port or from DeepMind.

Published references for the methods: DeepLIFT — Shrikumar, Greenside & Kundaje (2017),
ICML, arXiv:1704.02685 · TF-MoDISco — Shrikumar et al. (2018), arXiv:1811.00416
· TomTom — Gupta et al. (2007), *Genome Biology* 8:R24, doi:10.1186/gb-2007-8-2-r24 ·
MEME Suite — Bailey et al. (2015), *NAR* 43:W39, doi:10.1093/nar/gkv416 · Snakemake —
Mölder et al. (2021), *F1000Research* 10:33, doi:10.12688/f1000research.29032.2

`CITATION.cff` covers this pipeline itself.

## License

MIT — see [LICENSE](https://github.com/wangkant/motifline/blob/main/LICENSE).
