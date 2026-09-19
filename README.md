# motifline

A Snakemake pipeline from model attributions to MoDISco motifs to FiNeMo hits, one line
per run, for **ChromBPNet** and **AlphaGenome** on the same downstream code path.

No bespoke algorithms: attribution is DeepLIFT/SHAP through `tangermeme.deep_lift_shap`,
discovery is modisco-lite 2.4.0, hit calling is FiNeMo 0.30 (pinned), and pattern names
come from an RSAT `matrix-clustering` lexicon over HOCOMOCO and JASPAR, matched with
TomTom. Parameters are the tools' own defaults, and `motifline doctor` checks that
against the installed binaries rather than asserting it here.

```
attribute (cbp | ag)  ->  discover  ->  extract_regions  ->  call_hits
                                 \->  name_patterns
built once:  lexicon_union -> lexicon_rsat -> lexicon_medoid -> lexicon_cltagged
```

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

## Quick start

```bash
motifline init            # write config/default.yaml
$EDITOR config/default.yaml
motifline check           # every problem in the config at once, before anything runs
motifline doctor          # tool versions, parameters vs their defaults, model code
motifline smoke           # does this installation work? ~1 min, CPU, no model needed
motifline run -j 8 --gpus 2
```

`motifline check` reports missing keys, misspelled keys, paths that do not exist and
peaks files without a summit column — all in one pass, rather than failing inside
whichever stage first reads the key that is wrong. A misspelling is not cosmetic here:
`n_shard: 4` is a perfectly valid single-process run.

`motifline smoke` plants AP-1 (`TGACTCA`) in 180 of 300 synthetic hypothetical
attributions and checks MoDISco and FiNeMo give it back. It covers the compute
environment and the downstream half of the pipeline, not attribution and not naming.

## Targets

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
| call hits against the run's own lexicon | `call_hits` | `results/<run>/hits/hits.tsv` |

Which rules a run gets is decided by what its config entry contains: `adopt_from`
→ adopted; `n_shards` → sharded + merge; neither → single-process; `cancelled: true`
→ out of `rule all` but still resumable by explicit target. `model:` picks the step
(`cbp`, `ag`).

## Four things that will bite you

**`--rerun-triggers mtime params` is not optional.** With snakemake's default `mtime`
alone, editing `finemo.threshold` or `modisco.window` leaves every finished stage in
place and snakemake reports "Nothing to be done", so `results/` ends up holding a
mixture of parameter generations that nothing on disk distinguishes from a coherent
tree. `motifline run` bakes it in; driving `snakemake` directly, pass it yourself.

**Attributions are hypothetical, always.** `--hypothetical` decides whether a score is
stored for every base or only the observed one, and the two arrays are identical in
shape, dtype and filename — MoDISco consumes either without failing. Both steps refuse
to run without the flag, and every stage that produces or adopts an attribution measures
what it got (`frac(|attr| > 1e-9)`: ~1.0 hypothetical, ~0.25 projected) and records it.

**The AlphaGenome path needs a specific checkout.** tangermeme's nonlinearity table
covers CNN-shaped models, which is all ChromBPNet needs; AlphaGenome needs six more op
types, and three of those rules are Ruchir Rastogi's, on the `deeplift-support` branch
of `rastogiruchir/alphagenome-pytorch`. They are not merged into the port, so cloning
the port gives you an AG path that cannot run. `steps/ag_support.py` has the details.

**FiNeMo 0.30 is pinned and 0.41 is not a drop-in.** Note the version next to any hits
you keep — `run_info.json` records it for you.

## Is this the official configuration?

`motifline check`, `motifline doctor` and every `motifline run` compare twice: config
against the defaults the installed tools print in `--help`, and those installed defaults
against `src/motifline/tool_defaults.yaml`, which records what they were at the pinned
version. The second comparison catches a tool that moves a default when config moves
with it. Both report and neither blocks — deviating on purpose is ordinary science,
deviating without knowing is the accident.

`doctor` also prints the config-key → tool-flag mapping, which is not guessable:
`finemo.threshold` is `--cwm-trim-threshold` (a CWM trim, not a hit score) and
`finemo.alpha` is `--global-lambda`, while FiNeMo *also* has a deprecated `--alpha`
meaning something else. Rationale in `src/motifline/tools.py`.

## What a finished stage records

Every stage writes `run_info.json` next to its output once its body has succeeded: the
stage, the exact argv, the parameter signature it reran on, the environment prefix and
the versions that environment reports, host, timestamps, and any checks it measured on
its own output. Written after success and never before, so its presence means the stage
finished; nothing reads it back. `docs/ARTIFACTS.md` has the rest of what each stage
writes.

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

## Attribution geometry

Each model attributes its own native window and stores all of it: **ChromBPNet 2114 bp**,
**AlphaGenome 2048 bp** (AG's input floor, 16 × 128 bp trunk bins). A run may declare
`in_window` (and, for AG, `crop`) to override the step's default. Every downstream stage
reads the width off `attr.npy` itself rather than from a config value that would have to
be kept equal to what the arrays are.

## Naming patterns

A MoDISco pattern arrives as `pos_patterns.pattern_3`. Naming it is a separate chain
with its own library, built once and shared by every run:

```
HOCOMOCO H14CORE (1595) + JASPAR CORE vert NR (1019)
   -> lexicon_sources_union.meme (2614, every matrix kept, "E= 0" added)
   -> RSAT matrix-clustering            -> 821 clusters
   -> lexicon_rsat.meme                 -> one MEDOID per cluster (never an average)
   -> union_cltagged.meme               -> the union, names prefixed with cluster ids
   -> TomTom + gate                     -> results/<run>/naming/pattern_names.tsv
```

Two decisions in that chain look like details and are not — the TomTom target is the
cluster-tagged union rather than the medoid library, and the gate has two arms
(`q < gate.qval` **and** `best_ncor >= gate.ncor`) because q alone passes matches that
are significant and wrong. Both are argued in `src/motifline/naming.py`.

The chain stops at the cluster: grouping cluster ids into families is a human judgement
made once per project and is not derivable from the matrices. `rule lexicon_rsat` is the
one expensive step, 1h46m over 2,614 matrices.

## Running

```bash
motifline run [TARGET ...] -j <N> --gpus <G>
```

GPU selection is the caller's: `export CUDA_VISIBLE_DEVICES=<uuid>` (fully-free cards
only, pinned by UUID — indices shift — and leave one working card free), and bound
concurrency with `--gpus <G>`.

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

The AG DeepLIFT rules are Ruchir Rastogi's; cite them from his repository rather than
from the port or from DeepMind.

Published references for the methods: DeepLIFT — Shrikumar, Greenside & Kundaje (2017),
ICML, arXiv:1704.02685 · TF-MoDISco — Shrikumar et al. (2018), arXiv:1811.00416 · RSAT
matrix-clustering — Castro-Mondragon et al. (2017), *NAR* 45:e119,
doi:10.1093/nar/gkx314 · TomTom — Gupta et al. (2007), *Genome Biology* 8:R24,
doi:10.1186/gb-2007-8-2-r24 · MEME Suite — Bailey et al. (2015), *NAR* 43:W39,
doi:10.1093/nar/gkv416 · Snakemake — Mölder et al. (2021), *F1000Research* 10:33,
doi:10.12688/f1000research.29032.2

`CITATION.cff` covers this pipeline itself.

## License

MIT — see [LICENSE](LICENSE), which also names the Apache-2.0 code `steps/ag_support.py`
derives from. It covers this pipeline's code only; the tools it invokes are separate
works under their own licenses and are not redistributed here.
