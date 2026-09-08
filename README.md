# motifline

A Snakemake pipeline from model attributions to MoDISco motifs to FiNeMo hits, one
line per run, for three models on the same downstream code path:

* **ChromBPNet** and **AlphaGenome** — DeepLIFT/SHAP hypothetical attributions via
  `tangermeme.deep_lift_shap` (tangermeme 1.0.4);
* **GPN-Star** (since 2026-09-07) — masked-language-model contributions,
  `softmax(logits) − 1/4` per position, songlab's own MoDISco recipe.

No bespoke algorithms: discovery is modisco-lite 2.4.0, hit calling is finemo 0.30
(pinned), and patterns are named against a reference lexicon built by RSAT
`matrix-clustering` and matched with TomTom. Parameters are the tools' official
defaults; the deviations are the RSAT thresholds and `-quick` (standalone leaves the
thresholds empty, so they are stated explicitly), and the TomTom command line, which is
modiscolite's parameter set — `-dist pearson -min-overlap 5 -evalue -thresh 10.0
-no-ssc` — minus `-mi 1` (below).

**What was removed on 2026-09-07 (branch `linear`).** The cross-model merge
(motiflex stages, MotifCompendium comparison arm, cross-call and occupancy), the
external-lexicon hit-calling routes and the finemo 0.30-vs-0.41 arm. Cross-model work
is done outside this pipeline now, on each run's own `modisco.h5` / `hits.tsv`. The git
history before that commit has all of it. `geometry.py` went with them: a run's
attribution width is read off `attr.npy`, not derived from config.

## Layout

```
Snakefile            the DAG: attr -> discover -> extract_regions -> call_hits
                                        \-> name_patterns
                     built once:  lexicon_union -> lexicon_rsat ->
                                  lexicon_medoid -> lexicon_cltagged
config/default.yaml  runs + shared canonical parameters + site paths
config/peaks/        the K562 33,917-peak list (11-column narrowPeak) the runs share
src/motifline/       adopt / merge_shards / naming / runs   (orchestration env)
src/motifline/steps/ attr_cbp.py, attr_ag.py (+ ag_support.py), attr_star.py,
                     lexicon_*.py, name_patterns.py (+ motif_matrix.py)
                     -- compute-env scripts; they never import motifline
docs/ARTIFACTS.md    what each stage writes
tests/               the load-bearing claims, runnable without a GPU
```

## Attribution geometry

Each model attributes its own native window and stores all of it: **ChromBPNet
2114bp, AlphaGenome 2048bp** (AG's input floor, 16 × 128bp trunk bins), **GPN-Star
1536bp** (its `max_position_embeddings`). A run may declare `in_window` (and, for AG,
`crop`) to override the step's default; the adopted AG runs do, because their archived
arrays were computed at 16384bp cropped to 2114. Motif discovery is robust to windows
that differ by tens or hundreds of bp; per-position profiles are not.

## GPN-Star (`steps/attr_star.py`)

Per region: a 1536bp window centred on the summit is read from the multiz100way
alignment, all species. The human column is masked in groups — every position `p`
with `p % stride == r` in one forward pass, `stride` passes per strand — and the
logits at the masked positions are read out. Both strands are scored and averaged in
probability space. Grouped masking is what makes the full set affordable: one position
per forward (upstream's `LogitsInference`) is ~2,900 GPU-h for 33,917 × 1536 positions;
at stride 8 it is ~16 GPU-h, and the two agree at Pearson ≥ 0.9987 on the contributions
(measured 2026-09-07, `new_experiment/star/results/stride_fidelity.tsv`). The model was
trained with 15% of positions masked, so stride ≥ 7 stays inside the training density.

A star run declares two things: `stride`, and `source` — `none` scores with the
alignment (the published use), `mask` blanks the 99 non-human columns so only the
human sequence is seen. They are different readouts; do not pool them.

Rows whose human column holds anything but A/C/G/T inside the window are dropped and
listed in `n_dropped.tsv` (ChromBPNet's policy, not AG's fill). The step runs in
`envs/gpn_star` (its own torch); everything downstream of `attr/` runs in the compute env.

## Naming patterns (2026-08-25)

A MoDISco pattern arrives as `pos_patterns.pattern_3`. Naming it is a separate chain
with its own library, run once and shared by every run:

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
  against representatives alone loses the member that actually fits and inflates
  the q-value of a correct assignment. Matching against the full set and reading
  the cluster id back off the winner's name costs nothing — the medoid library is
  built anyway, as the compact reference to plot and publish from.
- **The gate has two arms**: `q < gate.qval` **and** `best_ncor >= gate.ncor`,
  where `best_ncor` is measured on the pattern's IC core against the untagged
  matrices with the same Ncor the clustering was built on, and `gate.ncor` is set
  to the clustering's own merge floor. q alone passes matches that are significant
  and wrong — a repeat element whose core contains a TF half-site is the standard
  case. A pattern failing either arm keeps its match in the table with
  `annotated=False` and is not renamed.

Nothing here assigns **families** — grouping cluster ids into site types so two
models' patterns can be paired is a human judgement made once per project, and it
is not derivable from the matrices. The chain stops at the cluster.

`rule lexicon_rsat` is the expensive step — **1h46m** over 2614 matrices, measured
2026-08-25 — and the only one; everything after it is minutes. The build is verified
against the artifacts it replaced: union, medoid MEME and cluster-tagged union come out
byte-identical, the Ncor matrix agrees to 1.1e-15.

## Running

```bash
envs/motifline/bin/snakemake --rerun-triggers mtime params -j <N> --resources gpu=<G> -- <target>
```

`--rerun-triggers mtime params` is the standard invocation. `mtime` alone silently
disabled the parameter trigger: editing `finemo.threshold` or `modisco.window` left
every finished stage in place and snakemake reported "Nothing to be done". Every rule
declares a `params:` signature of the values it actually runs at (`param_sig`). The
`code` trigger stays off on purpose: the `run:` blocks only assemble argv.

| you want | rule | target |
|---|---|---|
| attribute a slice/small run in one process | `attribute` | `results/<run>/attr/attr.npy` |
| attribute full-scale, one shard per GPU | `attribute_sharded` | `results/<run>/attr_shards/shard_<i>/attr_<i>.npy` |
| glue shards into one matrix | `merge_shards` | `results/<run>/attr/attr.npy` |
| use an archived `interpret_*_par_hyp` attribution | `import_attr` | `results/<run>/attr/attr.npy` |
| discover motifs (MoDISco) | `discover` | `results/<run>/modisco/modisco.h5` |
| build the reference lexicon (once, ~1h45m) | `lexicon_rsat` → `lexicon_cltagged` | `results/lexicon/union_cltagged.meme` |
| name a run's patterns | `name_patterns` | `results/<run>/naming/pattern_names.tsv` |
| prepare regions for FiNeMo | `extract_regions` | `results/<run>/finemo_regions/regions.npz` |
| call hits against the run's own lexicon | `call_hits` | `results/<run>/hits/hits.tsv` |

Which rules a run gets is decided by what its config entry contains: `adopt_from`
→ adopted; `n_shards` → sharded+merge; neither → single-process; `cancelled: true`
→ out of `rule all` but still resumable by explicit target. `model:` picks the step
(`cbp`, `ag`, `star`).

GPU selection is the caller's: `export CUDA_VISIBLE_DEVICES=<uuid>` (fully-free
cards only, pin by UUID — indices shift on this node — and leave one working card
free), and bound concurrency with `--resources gpu=<G>`.

## Environments

| env | role |
|---|---|
| `envs/motifline` (py3.12) | snakemake + this package |
| `envs/chrombpnet_local` (py3.10) | compute for cbp/ag and everything downstream: torch, chrombpnet, tangermeme, modisco-lite, finemo 0.30 |
| `envs/gpn_star` (py3.13) | `attr_star.py` only: torch 2.6.0+cu124, gpn 0.9.0, zarr |
| `envs/rsat` (rsat-core 2025.04.04) | `matrix-clustering`, invoked through its own `bin/rsat` launcher; `rule lexicon_rsat` only |
| `envs/meme` | `tomtom`, for `name_patterns` |

Step scripts never import this package; every site path arrives as an argument
(AlphaGenome in particular: pass `--ag_src`, the package is not importable from
the compute env).

## House rules

- Attributions are HYPOTHETICAL (`--hypothetical` for cbp/ag; star's recipe is
  hypothetical by construction), always.
- hits are produced by finemo 0.30; note the version next to any hits you keep.
- `new_experiment/region_v3|v4` archives are read-only.
- peaks fed to finemo must be 10/11-column narrowPeak (summit in column 10);
  the attribution steps normalize this themselves.
