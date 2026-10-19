# motifline

**A Snakemake workflow for motif discovery, hit calling and annotation from
ChromBPNet and AlphaGenome attributions.**

Peaks and a trained model go in; attributions, TF-MoDISco patterns, FiNeMo hits and
TomTom names come out. Both models share every stage after attribution. motifline
connects established tools; it adds no new motif algorithm.

[Installation](https://github.com/wangkant/motifline/blob/main/docs/INSTALL.md) · [Example configuration](https://github.com/wangkant/motifline/blob/main/config/example.yaml) ·
[Output reference](https://github.com/wangkant/motifline/blob/main/docs/ARTIFACTS.md)

## Workflow

```text
peaks + model + genome
  -> attribute    ChromBPNet: official DeepSHAP, counts head
                  ChromBPNet, backend: pytorch: the port's DeepLIFT/SHAP, counts head
                  AlphaGenome: tangermeme DeepLIFT/SHAP (2,048 bp input by default),
                  log1p of the ATAC track summed over the central 1,000 bp
  -> discover     TF-MoDISco             modisco.h5
  -> call hits    FiNeMo                 hits.tsv
  -> name         TomTom vs HOCOMOCO H14 + JASPAR; every pattern takes its best
                  match, `confident` = q <= qval_threshold
                                         pattern_names.tsv
```

Every stage records its command, parameters, environment and tool versions in
`run_info.json`.

## Install and run

```bash
python -m pip install "git+https://github.com/wangkant/motifline.git"   # Python >= 3.11
mkdir my-analysis && cd my-analysis
motifline init                 # config/default.yaml and envs/
# create the compute, chrombpnet and meme envs from envs/ (see the installation guide)

$EDITOR config/default.yaml    # paths, weights, runs
motifline check                # validate the config and inputs
motifline smoke                # motif discovery and hit calling on synthetic data; needs no model or GPU
motifline run -j 8
```

AlphaGenome needs the `deeplift-support` branch of Ruchir Rastogi's
`alphagenome-pytorch` fork. Model code, weights, reference data and GPU settings are
in the [installation guide](https://github.com/wangkant/motifline/blob/main/docs/INSTALL.md).

To work on motifline itself: `pip install -e ".[dev]"` in a checkout, then `pytest`
(no GPU needed).

## Outputs

Under `results/<run>/`, one directory per stage: `attr/` (`attr.npy`, hypothetical
contribution scores; `ohe.npy`; `regions.bed`, the row key), `modisco/modisco.h5`,
`hits/` (`hits.tsv` per region window, `hits_unique.tsv` per genomic position),
`naming/pattern_names.tsv`, and the TF-MoDISco and FiNeMo HTML reports
(`modisco_report/`, `finemo_report/`). Each stage directory has its own
`run_info.json`. Details are in the
[output reference](https://github.com/wangkant/motifline/blob/main/docs/ARTIFACTS.md).

## Example results

![Per-base contributions and motif hits in one K562 region](https://raw.githubusercontent.com/wangkant/motifline/main/docs/figures/example_region.png)

One K562 accessible region from the example peak set: per-base contributions from
ChromBPNet and AlphaGenome (letter height = contribution), with the FiNeMo motif hits
each model's run called there, labelled by motif family (curated from TomTom matches).

## Citation

Cite motifline with [CITATION.cff](https://github.com/wangkant/motifline/blob/main/CITATION.cff), and the tools your analysis used:

| tool | version here | where |
|---|---|---|
| ChromBPNet (official, TensorFlow; also `adopt_format: chrombpnet_h5`) | 1.0.1 | [kundajelab/chrombpnet](https://github.com/kundajelab/chrombpnet) |
| DeepSHAP for TensorFlow, as ChromBPNet uses it | `kundajelab-shap` 1 | [kundajelab/shap](https://github.com/kundajelab/shap) |
| DeepLIFT (dinucleotide shuffles for the ChromBPNet references) | 0.6.13.0 | [kundajelab/deeplift](https://github.com/kundajelab/deeplift) |
| ChromBPNet (PyTorch port; `backend: pytorch`) | `jsxlei/chrombpnet-pytorch@01ad10b` | [jsxlei/chrombpnet-pytorch](https://github.com/jsxlei/chrombpnet-pytorch) |
| AlphaGenome (the model) | — | [google-deepmind/alphagenome](https://github.com/google-deepmind/alphagenome) |
| AlphaGenome PyTorch port | `genomicsxai/alphagenome-pytorch` | [genomicsxai/alphagenome-pytorch](https://github.com/genomicsxai/alphagenome-pytorch) |
| AlphaGenome DeepLIFT rules | `rastogiruchir/alphagenome-pytorch@64e9b75`, branch `deeplift-support` | Ruchir Rastogi — [rastogiruchir/alphagenome-pytorch](https://github.com/rastogiruchir/alphagenome-pytorch) |
| tangermeme (AlphaGenome and the ChromBPNet PyTorch port) | 1.0.4 | [jmschrei/tangermeme](https://github.com/jmschrei/tangermeme) |
| modisco-lite | 2.4.0 | [jmschrei/tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite) |
| FiNeMo | 0.30 (`austintwang/finemo_gpu@5c6f521`) | [austintwang/finemo_gpu](https://github.com/austintwang/finemo_gpu) |
| MEME Suite (TomTom) | 5.5.9 | [meme-suite.org](https://meme-suite.org) |
| Snakemake | 9.25.1 | [snakemake/snakemake](https://github.com/snakemake/snakemake) |
| HOCOMOCO | H14 CORE | [hocomoco14.autosome.org](https://hocomoco14.autosome.org) |
| JASPAR | 2026 CORE vertebrates non-redundant | [jaspar.elixir.no](https://jaspar.elixir.no) |

The AlphaGenome DeepLIFT rules are Ruchir Rastogi's; cite them from his repository rather
than from genomicsxai/alphagenome-pytorch or from DeepMind.

Published references: ChromBPNet — Pampari et al. (2024), bioRxiv,
doi:10.1101/2024.12.25.630221 · AlphaGenome — Avsec et al. (2026), *Nature*,
doi:10.1038/s41586-025-10014-0 · DeepLIFT — Shrikumar, Greenside & Kundaje (2017), ICML,
arXiv:1704.02685 · tangermeme — Schreiber (2026), *Nature Methods*,
doi:10.1038/s41592-026-03254-z · TF-MoDISco — Shrikumar et al. (2018), arXiv:1811.00416 ·
TomTom — Gupta et al. (2007), *Genome Biology* 8:R24, doi:10.1186/gb-2007-8-2-r24 ·
MEME Suite — Bailey et al. (2015), *NAR* 43:W39, doi:10.1093/nar/gkv416 · HOCOMOCO —
Vorontsov et al. (2024), *NAR*, doi:10.1093/nar/gkad1077 · JASPAR 2026 — Ovek Baydar et
al. (2026), *NAR*, doi:10.1093/nar/gkaf1209 · Snakemake — Mölder et al. (2021),
*F1000Research* 10:33, doi:10.12688/f1000research.29032.2

## License

MIT — see [LICENSE](https://github.com/wangkant/motifline/blob/main/LICENSE).
