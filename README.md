# motifline

A Snakemake workflow for motif discovery, hit calling and annotation from ChromBPNet and
AlphaGenome attributions. Peaks and a trained model go in; DeepLIFT/SHAP attributions,
TF-MoDISco patterns, FiNeMo hits and TomTom names come out. Both models share every stage
after attribution.

## Workflow

```text
peaks + model + genome
  -> attribute    ChromBPNet: counts head, 2,114 bp input,
                  dinucleotide-shuffled references
                  AlphaGenome: log1p of the ATAC track summed over the central 1,000 bp,
                  2,048 bp input, local dinucleotide-shuffled references
                  20 references per peak, seeded; peaks split into GPU shards
  -> discover     TF-MoDISco                          modisco/modisco.h5
  -> call hits    FiNeMo                              hits/hits.tsv
  -> name         TomTom vs HOCOMOCO H14 + JASPAR     naming/pattern_names.tsv
```

## Setup

```bash
git clone https://github.com/wangkant/motifline.git && cd motifline
conda env create -f envs/orchestration.yaml
conda env create -f envs/compute.yaml
conda env create -f envs/meme.yaml

git clone -b deeplift-support https://github.com/rastogiruchir/alphagenome-pytorch.git
git -C alphagenome-pytorch checkout 64e9b75
```

`envs/locks/` holds the full package lists of the environments the runs used.

Reference data:

| file | config key | source |
|---|---|---|
| `hg38.fa`, `hg38.chrom.sizes` | `genome` | UCSC |
| HOCOMOCO H14 CORE (MEME) and `H14CORE_annotation.jsonl` | `reference_lexicon` | hocomoco14.autosome.org |
| JASPAR 2026 CORE vertebrates non-redundant (MEME) | `reference_lexicon` | jaspar.elixir.no |

Model weights: for ChromBPNet, the Keras h5 of the model without its bias, 512 filters
and 8 layers (`wo_bias`); for AlphaGenome, a checkpoint loadable by
`AlphaGenome.from_pretrained` (`weights`).

## Running

Fill in the paths in `config/config.yaml`, then

```bash
conda activate motifline-orchestration
snakemake -j 32 --resources gpu=4
```

Shard *i* of a run uses `gpus[i % len(gpus)]`. Peaks need the summit offset in column 10
(narrowPeak).

## Outputs

Under `results/<run>/`:

| path | contents |
|---|---|
| `attr/attr.npy` | `(N, 4, W)` float32 hypothetical contribution scores; `W` = 2,114 (ChromBPNet) or 2,048 (AlphaGenome) |
| `attr/ohe.npy` | `(N, 4, W)` int8 one-hot input |
| `attr/regions.bed` | the `N` peaks, narrowPeak, in row order |
| `modisco/modisco.h5` | TF-MoDISco patterns |
| `hits/` | FiNeMo hits: `hits.tsv` per region window, `hits_unique.tsv` per genomic position |
| `naming/pattern_names.tsv` | per pattern: top three TomTom matches, `name`, `confident` |
| `modisco_report/`, `finemo_report/` | the tools' HTML reports |

Peaks whose window runs off the chromosome or contains a non-ACGT base are dropped, so
`N` can be smaller than the peak count.

## Example

![Per-base contributions and motif hits in one K562 region](docs/figures/example_region.png)

One K562 accessible region from the example peak set: per-base contributions from
ChromBPNet and AlphaGenome (letter height = contribution), with the FiNeMo hits each
model's run called there, labelled by motif family.

## Software

| tool | version | source |
|---|---|---|
| AlphaGenome model code with DeepLIFT rules | `64e9b75`, branch `deeplift-support` | [rastogiruchir/alphagenome-pytorch](https://github.com/rastogiruchir/alphagenome-pytorch) |
| tangermeme | 1.0.4 | [jmschrei/tangermeme](https://github.com/jmschrei/tangermeme) |
| modisco-lite | 2.4.0 | [jmschrei/tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite) |
| FiNeMo | 0.30 (`5c6f521`) | [austintwang/finemo_gpu](https://github.com/austintwang/finemo_gpu) |
| MEME Suite (TomTom) | 5.5.9 | [meme-suite.org](https://meme-suite.org) |
| Snakemake | 9.25.1 | [snakemake/snakemake](https://github.com/snakemake/snakemake) |

## License

MIT; see [LICENSE](LICENSE). `scripts/ag_deeplift.py` is adapted from Apache-2.0 code;
see [NOTICE](NOTICE).
