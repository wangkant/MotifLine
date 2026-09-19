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

## What you can ask for

```bash
motifline run results/<run>/hits/hits.tsv -j 4        # one target
motifline run -n                                      # dry run
```

| you want | rule | target |
|---|---|---|
| attribute a slice or small run in one process | `attribute` | `results/<run>/attr/attr.npy` |
| attribute full-scale, one shard per GPU | `attribute_sharded` | `results/<run>/attr_shards/shard_<i>/attr_<i>.npy` |
| glue shards into one matrix | `merge_shards` | `results/<run>/attr/attr.npy` |
| use an archived `interpret_*_par_hyp` attribution | `import_attr` | `results/<run>/attr/attr.npy` |
| discover motifs (MoDISco) | `discover` | `results/<run>/modisco/modisco.h5` |
| build the reference lexicon (once, ~1h45m) | `lexicon_rsat` → `lexicon_cltagged` | `results/lexicon/union_cltagged.meme` |
| name a run's patterns | `name_patterns` | `results/<run>/naming/pattern_names.tsv` |
| prepare regions for FiNeMo | `extract_regions` | `results/<run>/finemo_regions/regions.npz` |
| call hits against the run's own motifs | `call_hits` | `results/<run>/hits/hits.tsv` |

Every stage also writes a `run_info.json` beside its output once it has succeeded: the
argv, the parameters it reran on, the environment and the tool versions that environment
reports, and anything it measured about what it produced. Its presence means the stage
finished. `docs/ARTIFACTS.md` says what each stage writes.

## Four things that will bite you

**`--rerun-triggers mtime params` is not optional.** With snakemake's default `mtime`
alone, editing `finemo.threshold` or `modisco.window` leaves every finished stage in place
and snakemake reports "Nothing to be done", so `results/` ends up holding a mixture of
parameter generations that nothing on disk distinguishes from a coherent tree.
`motifline run` bakes it in; driving `snakemake` directly, pass it yourself.

**Attributions are hypothetical, always.** `--hypothetical` decides whether a score is
stored for every base or only the observed one, and the two arrays are identical in shape,
dtype and filename — MoDISco consumes either without failing. Both attribution steps
refuse to run without the flag, and every stage that produces or adopts an attribution
measures what it actually got and records it.

**The AlphaGenome path needs a specific checkout.** The DeepLIFT rules it needs are Ruchir
Rastogi's, on the `deeplift-support` branch of `rastogiruchir/alphagenome-pytorch`, and
they are not merged into the port — cloning the port gives you an AG path that cannot run.

**FiNeMo 0.30 is pinned and 0.41 is not a drop-in.** Note the version next to any hits you
keep; `run_info.json` records it for you.

## Hit calling and naming are different questions

`call_hits` scores a run's **own** MoDISco patterns against its own regions, so its hits
are self-contained. The reference lexicon never enters hit calling — it only supplies
names, and a name is only assigned when the TomTom match passes both arms of the gate
(`q < gate.qval` **and** `best_ncor >= gate.ncor`). A pattern that fails either arm keeps
its match in the table marked `annotated=False` and is not renamed. The chain stops at the
cluster: grouping clusters into families is a human judgement, not derivable from the
matrices.

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

## GPUs

GPU selection is the caller's: `export CUDA_VISIBLE_DEVICES=<uuid>` (fully-free cards
only, pinned by UUID — indices shift — and leave one working card free), and bound
concurrency with `--gpus <G>`. The pipeline never picks a card for you.

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

MIT — see [LICENSE](LICENSE), which also names the Apache-2.0 code `steps/ag_support.py`
derives from. It covers this pipeline's code only; the tools it invokes are separate works
under their own licenses and are not redistributed here.
