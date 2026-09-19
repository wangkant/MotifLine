# motifline

Give it a set of peaks and a trained sequence model. It gives you back the motifs that
model learned, where they occur, and what they are called.

```
   peaks.bed  +  model weights
        │
        ▼
   attribute  (ChromBPNet | AlphaGenome)      what the model reads at each base
        │
        │   attr.npy   ohe.npy   regions.bed
        │
        ├────────────────────────────────┐
        ▼                                ▼
   discover                         extract_regions
   MoDISco                          FiNeMo packaging
        │  modisco.h5                    │  regions.npz
        │                                │
        ├────────────────────────────────┴──►  call_hits  ──►  hits.tsv
        │                                      FiNeMo             where each motif occurs
        ▼
   name_patterns  ──►  pattern_names.tsv       what each motif is
   TomTom + gate
        ▲
        │  the reference lexicon, built once and shared by every run:
        │  HOCOMOCO H14CORE + JASPAR CORE vert NR → RSAT matrix-clustering
        │  → 821 clusters → one medoid each → the cluster-tagged union
```

It is a Snakemake pipeline around the standard tools and contributes no algorithms of
its own: attributions are DeepLIFT/SHAP through `tangermeme.deep_lift_shap`, discovery is
modisco-lite 2.4.0, hit calling is FiNeMo 0.30, naming is TomTom against an RSAT-clustered
lexicon. What it contributes is that the whole line runs end to end from one config file,
at the tools' own default parameters, with every output labelled by what produced it —
and that two models, ChromBPNet and AlphaGenome, share the entire path after `attr/`, so
what differs between two runs is the model rather than the analysis.

## Install

Four conda environments, two model source checkouts, three reference files, one config.
**[docs/INSTALL.md](docs/INSTALL.md)** has all of it, including which checkout the
AlphaGenome path needs — it is not the one you would guess.

```bash
git clone https://github.com/wangkant/motifline.git && cd motifline
ENVS=$HOME/motifline-envs
conda env create -p $ENVS/orchestration -f envs/orchestration.yaml
conda env create -p $ENVS/compute       -f envs/compute.yaml
conda env create -p $ENVS/meme          -f envs/meme.yaml
conda env create -p $ENVS/rsat          -f envs/rsat.yaml
$ENVS/orchestration/bin/pip install -e .
```

## Using it

```bash
motifline init            # write config/default.yaml
$EDITOR config/default.yaml
motifline check           # every problem in the config at once, before anything runs
motifline doctor          # tool versions, parameters vs the tools' own defaults, model code
motifline smoke           # does this installation work? ~1 min, CPU, no model needed
motifline run -j 8 --gpus 2
```

One run is one entry under `runs:` in the config. What rules it gets follows from what
the entry contains: `adopt_from` → adopt an archived attribution; `n_shards` → attribute
in shards, one GPU each, then merge; neither → attribute in one process; `cancelled: true`
→ out of the default target list but still reachable by explicit target. `model:` picks
`cbp` or `ag`. `config/example.yaml` has a worked entry of each kind.

**`motifline check`** reports missing keys, misspelled keys, paths that do not exist and
peaks files without a summit column — all in one pass, rather than failing inside
whichever stage first reads the key that is wrong. A misspelling is not cosmetic here:
`n_shard: 4` is a perfectly valid single-process run.

**`motifline doctor`** reads the defaults the installed tools print in `--help` and says
where your config differs from them, and where the tools have moved a default since this
pipeline pinned them. It reports and never blocks; deviating on purpose is ordinary
science, deviating without knowing is the accident.

**`motifline smoke`** plants AP-1 (`TGACTCA`) in 180 of 300 synthetic attributions and
checks MoDISco and FiNeMo give it back. It covers the compute environment and everything
after `attr/`, not attribution itself and not the naming chain.

## Layout

```
Snakefile             the DAG
config/default.yaml   runs + shared parameters + site paths (yours; not in git)
config/example.yaml   a worked config with an entry of each kind
config/peaks/         the fixed 33,917-peak K562 set these runs share
envs/                 environment recipes; envs/locks/ the exact trees we ran
src/motifline/        cli, config validation, tool-default checking, provenance,
                      adopt, attribution, merge_shards, naming, runs, smoke, stages
src/motifline/steps/  compute-env scripts; they never import motifline
docs/                 INSTALL.md, ARTIFACTS.md
tests/                the load-bearing claims, runnable without a GPU
```

Why each decision is what it is lives next to the code it governs: `naming.py` for the
lexicon and the gate, `tools.py` for the parameter checking, `attribution.py` for the
hypothetical measurement, `steps/attr_ag.py` and `steps/ag_support.py` for the AlphaGenome
path.

## Citing the tools this pipeline runs

motifline is orchestration. The science is in the tools below, and each should be cited
from its own project.

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
