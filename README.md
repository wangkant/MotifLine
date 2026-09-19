# motifline

A Snakemake pipeline from model attributions to MoDISco motifs to FiNeMo hits, one
line per run, for **ChromBPNet** and **AlphaGenome** on the same downstream code path.
Both models are attributed with DeepLIFT/SHAP hypothetical contributions through
`tangermeme.deep_lift_shap`, so what differs between two runs is the model, not the
attribution method.

No bespoke algorithms. Discovery is modisco-lite 2.4.0, hit calling is FiNeMo 0.30
(pinned), pattern names come from an RSAT `matrix-clustering` lexicon over HOCOMOCO
and JASPAR, matched with TomTom. Parameters are the tools' official defaults; the
deviations are stated, with the reason, where they are set.

```
attribute (cbp | ag)  ->  discover  ->  extract_regions  ->  call_hits
                                 \->  name_patterns
built once:  lexicon_union -> lexicon_rsat -> lexicon_medoid -> lexicon_cltagged
```

## Install

Four conda environments, two model source checkouts, three reference files, one
config — **[docs/INSTALL.md](docs/INSTALL.md)** has the whole of it. The short version:

```bash
git clone https://github.com/wangkant/motifline.git && cd motifline
ENVS=$HOME/motifline-envs                  # the recipes are in envs/; the prefixes go elsewhere
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
motifline doctor          # one page: tool versions, parameters, model code
motifline smoke           # does this installation work? ~1 min, CPU, no model needed
motifline run -j 8 --gpus 2
```

`motifline smoke` plants AP-1 (`TGACTCA`) in 180 of 300 synthetic hypothetical
attributions, runs MoDISco and FiNeMo over them and checks the motif comes back. It
covers the compute environment and the downstream half of the pipeline; it does not
cover the attribution steps (they need model weights) or the naming chain (it needs
the RSAT clustering, which is 1h46m).

`motifline check` reports missing keys, misspelled keys, paths that do not exist and
peaks files without a summit column — all of them in one pass, rather than failing
inside whichever stage first reads the key that is wrong. A misspelling is not a
cosmetic problem here: `n_shard: 4` is a perfectly valid single-process run.

## Is this the official configuration?

That this pipeline runs MoDISco and FiNeMo is not the point — anyone can run those.
The point is that it runs them at the tools' own defaults, with the handful of
deviations stated and argued. A claim like that is true on the day it is written and
unchecked ever after, so here it is measured instead, against the binaries that are
actually going to run:

```
$ motifline doctor
compute env  /…/envs/compute
  modisco-lite   2.4.0
  finemo         0.30

parameters
  every parameter this pipeline sets is on its tool's official default

  config key            -> tool flag
  modisco.max_seqlets   -> modisco-lite --max_seqlets   [-n is required by the CLI; there is no default to adopt]
  modisco.window        -> modisco-lite --window
  finemo.mode           -> finemo --mode
  finemo.threshold      -> finemo --cwm-trim-threshold   [-t trims the CWM; it is not a hit score threshold]
  finemo.alpha          -> finemo --global-lambda   [passed as -l. finemo also has a DEPRECATED --alpha, which this is not]
  finemo.batch          -> finemo --batch-size
```

Two comparisons, because one is not enough:

- **config against the installed tool.** Both tools print `(default: X)` in `--help`,
  so the defaults are read off the binary rather than from documentation.
- **the installed tool against `src/motifline/tool_defaults.yaml`**, which records what
  each default *was* at the pinned version. Without this second comparison, a tool that
  moves a default is invisible whenever config moves with it: config would still match
  the installed tool while no longer being the configuration the published runs used.

Neither blocks. Deviating on purpose is ordinary science; deviating without knowing is
the accident, and that is what these report. The mapping table is printed because it is
not guessable: `finemo.threshold` is a CWM trim threshold and not a hit score, and
`finemo.alpha` reaches `--global-lambda` while finemo *also* has a deprecated `--alpha`
that means something else.

## Was the attribution hypothetical?

`--hypothetical` decides whether a score is stored for every base at every position or
only for the base that is there. The two come out as arrays of the same shape, the same
dtype and the same filename, and MoDISco consumes either without failing. Nothing in a
run log tells them apart. The array does:

```
frac = (|attr| > 1e-9).mean()     ~1.00 hypothetical,  ~0.25 projected
```

Both attribution steps now refuse to run without the flag, and every stage that
produces or adopts an attribution measures the array it ended up with and records the
result in its `run_info.json`. An array that does not look hypothetical is reported
loudly and not deleted — an attribution that cost GPU-days is not thrown away by its
own label.

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

## Layout

```
Snakefile             the DAG
config/default.yaml   runs + shared parameters + site paths (yours; not in git)
config/example.yaml   a worked config with an entry of each kind
config/peaks/         a 33,917-peak K562 list in the 11-column narrowPeak form
envs/                 environment recipes; envs/locks/ the exact trees we ran
src/motifline/        cli, config validation, tool-default checking, provenance,
                      adopt, attribution, merge_shards, naming, runs, smoke, stages
                      (orchestration env)
  tool_defaults.yaml  what each tool's defaults were at the version pinned here
src/motifline/steps/  attr_cbp.py, attr_ag.py (+ ag_support.py), lexicon_*.py,
                      name_patterns.py (+ motif_matrix.py)
                      -- compute-env scripts; they never import motifline
docs/INSTALL.md       installing it
docs/ARTIFACTS.md     what each stage writes
tests/                the load-bearing claims, runnable without a GPU
```

## Attribution geometry

Each model attributes its own native window and stores all of it: **ChromBPNet
2114 bp**, **AlphaGenome 2048 bp** (AG's input floor, 16 × 128 bp trunk bins). A run
may declare `in_window` (and, for AG, `crop`) to override the step's default; the
adopted AG runs do, because their archived arrays were computed at 16384 bp cropped to
2114. Every downstream stage reads the width off `attr.npy` itself rather than from a
config value that would have to be kept equal to what the arrays are.

Motif discovery is robust to windows that differ by tens or hundreds of bp; per-position
profiles are not.

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

Two decisions are worth stating because both look like details:

- **The TomTom target is the cluster-tagged union, not the medoid library.** A
  pattern's best library match is usually *not* its cluster's medoid, so matching
  against representatives alone loses the member that actually fits and inflates the
  q-value of a correct assignment. Matching against the full set and reading the
  cluster id back off the winner's name costs nothing — the medoid library is built
  anyway, as the compact reference to plot and publish from.
- **The gate has two arms**: `q < gate.qval` **and** `best_ncor >= gate.ncor`, where
  `best_ncor` is measured on the pattern's IC core against the untagged matrices with
  the same Ncor the clustering was built on, and `gate.ncor` is set to the clustering's
  own merge floor. q alone passes matches that are significant and wrong — a repeat
  element whose core contains a TF half-site is the standard case. A pattern failing
  either arm keeps its match in the table with `annotated=False` and is not renamed.

Nothing here assigns **families**. Grouping cluster ids into site types so that two
models' patterns can be paired is a human judgement made once per project and is not
derivable from the matrices. The chain stops at the cluster.

`rule lexicon_rsat` is the expensive step — **1h46m** over 2,614 matrices, measured
2026-08-25 — and the only one; everything after it is minutes. The build was verified
against the artifacts it replaced: union, medoid MEME and cluster-tagged union come out
byte-identical, the Ncor matrix agrees to 1.1e-15.

## Running

```bash
motifline run [TARGET ...] -j <N> --gpus <G>
```

`motifline run` is `snakemake` with `--rerun-triggers mtime params` baked in. That is
not a convenience. With snakemake's default `mtime` alone, editing `finemo.threshold`
or `modisco.window` leaves every finished stage in place and snakemake reports
"Nothing to be done" — `results/` then holds a mixture of parameter generations that
nothing on disk distinguishes from a coherent tree. Every rule declares a `params:`
signature of the values it actually runs at, for exactly this trigger. The `code`
trigger stays off on purpose: the `run:` blocks only assemble argv.

Driving `snakemake` directly still works; pass the triggers yourself.

GPU selection is the caller's: `export CUDA_VISIBLE_DEVICES=<uuid>` (fully-free cards
only, pinned by UUID — indices shift — and leave one working card free), and bound
concurrency with `--gpus <G>`.

## What a finished stage records

Every stage writes a `run_info.json` next to its output once its body has succeeded:
the stage name, the exact argv, the parameter signature it reran on, the environment
prefix and the versions that environment reports for FiNeMo, modisco-lite, tangermeme,
torch and numpy, plus host, timestamps, and any `checks` that stage measured on its own
output — for the attribution stages, the hypothetical fraction above.

It is written after the body succeeds and never before, so a `run_info.json` means the
stage finished. Nothing in the pipeline reads it back and no stage refuses to run
because of it — it is a label on the output, not a gate on the input.

## Environments

| env | role |
|---|---|
| `envs/orchestration.yaml` (py3.12) | snakemake + this package |
| `envs/compute.yaml` (py3.10) | attribution and everything downstream: torch, tangermeme, modisco-lite, FiNeMo 0.30 |
| `envs/rsat.yaml` (rsat-core 2025.04.04) | `matrix-clustering`, invoked through its own `bin/rsat` launcher; `rule lexicon_rsat` only |
| `envs/meme.yaml` (meme 5.5.9) | `tomtom`, for `name_patterns` |

Step scripts never import this package, and no model code is installed into any
environment: every site path arrives as an argument, and `attr_cbp.py` / `attr_ag.py`
put the checkout named by `chrombpnet_src` / a run's `ag_src` on `sys.path` themselves.

## House rules

- Attributions are HYPOTHETICAL (`--hypothetical`), always. Both steps refuse without
  it, and the array is measured afterwards rather than taken on trust.
- Hits are produced by FiNeMo 0.30; the version is not interchangeable with 0.41. Note
  it next to any hits you keep — `run_info.json` records it for you.
- Peaks fed to FiNeMo must be 10/11-column narrowPeak with the summit in column 10;
  the attribution steps normalize this themselves and `motifline check` verifies it.

## Citing the tools this pipeline runs

motifline is orchestration. The science is in the tools below, and each should be cited
from its own project — this pipeline is not a substitute for any of them.

| tool | version here | where |
|---|---|---|
| ChromBPNet (PyTorch port) | `jsxlei/chrombpnet-pytorch@01ad10b` | github.com/kundajelab/chrombpnet |
| AlphaGenome (PyTorch port) | `genomicsxai/alphagenome-pytorch@64e9b75` | github.com/google-deepmind/alphagenome |
| tangermeme | 1.0.4 | github.com/jmschrei/tangermeme |
| modisco-lite | 2.4.0 | github.com/jmschrei/tfmodisco-lite |
| FiNeMo | 0.30 (`austintwang/finemo_gpu@5c6f521`) | github.com/austintwang/finemo_gpu |
| HOCOMOCO | H14 CORE | hocomoco.autosome.org |
| JASPAR | CORE vertebrates non-redundant | jaspar.elixir.no |

Published references for the methods:

- DeepLIFT — Shrikumar, Greenside & Kundaje (2017), *Learning important features
  through propagating activation differences*, ICML. arXiv:1704.02685
- TF-MoDISco — Shrikumar et al. (2018), *Technical note on Transcription Factor Motif
  Discovery from Importance Scores*. arXiv:1811.00416
- RSAT matrix-clustering — Castro-Mondragon et al. (2017), *Nucleic Acids Research*
  45:e119. doi:10.1093/nar/gkx314
- TomTom — Gupta, Stamatoyannopoulos, Bailey & Noble (2007), *Genome Biology* 8:R24.
  doi:10.1186/gb-2007-8-2-r24
- MEME Suite — Bailey et al. (2015), *Nucleic Acids Research* 43:W39–W49.
  doi:10.1093/nar/gkv416
- Snakemake — Mölder et al. (2021), *F1000Research* 10:33.
  doi:10.12688/f1000research.29032.2

ChromBPNet, AlphaGenome, tangermeme, FiNeMo, HOCOMOCO and JASPAR should be cited from
their current project pages; take the reference from the source rather than from here.

`CITATION.cff` covers this pipeline itself.

## License

MIT — see [LICENSE](LICENSE). It covers this pipeline's code only; the tools it invokes
are separate works under their own licenses and are not redistributed here.
