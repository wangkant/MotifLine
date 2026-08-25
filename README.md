# motifline

A Snakemake pipeline from model attributions to MoDISco motifs to FiNeMo hits,
for ChromBPNet and AlphaGenome, on the same `tangermeme.deep_lift_shap` code
path. No bespoke algorithms: attribution is tangermeme 1.0.4, discovery is
modisco-lite 2.4.0, hit calling is finemo 0.30 (pinned), and cross-model motif
merging finishes with motiflex (`mlex`, since 2026-08-17; MotifCompendium 1.0.18
stays as the explicit-target comparison arm). Patterns are named against a
reference lexicon built by RSAT `matrix-clustering` and matched with TomTom
(since 2026-08-25). Parameters are the tools' official defaults; the deviations
are `lexicons.batch: 1000` (a measured OOM limit on the 289-motif union lexicon),
`compendium.cluster_within_posneg`, the RSAT thresholds and `-quick` (standalone
leaves the thresholds empty, so they are stated explicitly), and the TomTom command
line, which is modiscolite's parameter set — `-dist pearson -min-overlap 5 -evalue
-thresh 10.0 -no-ssc`, five deviations from tomtom's own defaults — minus `-mi 1`
(all below).

**Attribution geometry.** Each model attributes its own native window and stores
all of it: **ChromBPNet 2114bp, AlphaGenome 2048bp** (AG's input floor, 16 x 128bp
trunk bins). Neither is cropped to meet the other, so MoDISco sees windows 66bp
apart — motif discovery is robust to that, per-position profiles are not. The
canonical pair is `config/default.yaml: geometry`; a run may override it, and the
adopted runs do, because their archived arrays are AG 16384bp cropped to 2114.

## Layout

```
Snakefile            the DAG: attr -> discover -> extract_regions -> call_hits
                                        \-> name_patterns
                     off to the side:   discover x2 -> merge_naming -> merge_ingest
                                        -> merge_align -> merge_annotate
                     (-> merge_tree -> merge_compile -> cross-call -> occupancy
                      when the merge declares thresholds)
                     built once:        lexicon_union -> lexicon_rsat ->
                                        lexicon_medoid -> lexicon_cltagged
config/default.yaml  runs + shared canonical parameters + site paths
                     (not tracked: cp config/example.yaml config/default.yaml
                      and fill in the site paths)
src/motifline/       adopt / merge_shards / lexicons / naming / geometry /
                     compendium / mlex / runs
src/motifline/steps/ attr_cbp.py, attr_ag.py (+ ag_support.py), merge_compendium.py,
                     lexicon_*.py, name_patterns.py (+ motif_matrix.py)
                     -- compute-env scripts
docs/ARTIFACTS.md    what each stage writes
```

## Naming patterns (2026-08-25)

A MoDISco pattern arrives as `pos_patterns.pattern_3`. Naming it is a separate
chain with its own library, run once and shared by every run and every merge:

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

`rule merge_naming` does the same for a merge's leaves and writes the `db.yaml`
that `mlex annotate` reads, so both models' patterns are labelled from one library
at one gate. Before this, no merge declared a `db_yaml` and annotate was skipped
outright; it now runs by default (a merge may still override `db_yaml` with a
frozen annotation, which then has to exist).

Nothing here assigns **families** — grouping cluster ids into site types so two
models' patterns can be paired is a human judgement made once per project, and it
is not derivable from the matrices. The chain stops at the cluster.

The build is verified against the artifacts it replaces: the union, the medoid
MEME and the cluster-tagged union come out byte-identical, and the Ncor matrix
agrees to 1.1e-15, and RSAT's own `clusters.tab` comes back byte-identical
(821 clusters / 2614 motifs). `rule lexicon_rsat` is the expensive step -- **1h46m**
over 2614 matrices, measured 2026-08-25 -- and the only one; everything after it is
minutes.

## Merging two models' motifs

The MoDISco h5 of the same regions under both models go through
`merge_ingest -> merge_align -> merge_annotate -> merge_tree -> merge_compile`. Four
of those five stages are [motiflex](https://github.com/wangkant/motiflex) (`mlex`);
the merging itself is this pipeline's, because motiflex removed its `merge`
subcommand on 2026-08-23 (commit `204cb3f`) when it repositioned as an auditor of
*someone else's* consolidation — `mlex selectivity` scores "any tool's merge" and
`mlex compile --decisions` reads a decisions JSON "from any tool". This is that tool.

**The criterion is RSAT's, not a height** (2026-08-25). The removed stage cut an
average-linkage tree over `1 - ppm_similarity` at one scalar `cut_height`. A single
height on a single score is not how the established motif-clustering software decides
a merge, and this project already runs the alternative: RSAT `matrix-clustering`
builds the reference lexicon. Its rule, applied here by `rule merge_tree`:

- the scores are read at **one registration** per pair — `Ncor`, `cor`, and `w` (the
  number of aligned columns) — not at each score's own optimum;
- under average linkage a node's score is the **mean over all cross-pairs** between
  the two child clusters (`TFBMclust/R/check_alignment.R:80-81`);
- the node merges only if **every** declared arm clears its threshold —
  `if(sum(unlist(th)) == length(thresholds))` (`check_alignment.R:106-109`);
- a node that fails **breaks the chain**: its ancestors inherit the failure instead of
  being re-tested (`align.motifs.R:101`, `find_clusters.R`'s bottom-up walk).

So a group is a maximal subtree in which every internal node passed every arm. The
thresholds are declared per merge and never defaulted — the same policy `cut_height`
had, and RSAT standalone's own (it defaults them to empty). A config that still
declares `cut_height` is **refused**, not reinterpreted: 0.05 on `1 - ppm_similarity`
has no image under this criterion. Representatives stay medoids — observed patterns,
never cross-model averages — with ties broken on seqlet count and **flagged**
(`representative_tied`), because for a two-member group the medoid is undefined.
`mlex compile` then writes the lexicon and reads it back with the real finemo loader
(`--verify-roundtrip auto`).

Merge definitions stay in `config.compendium.merges`; the mlex-specific fields sit on
the merge and the tool block is `config.motiflex`:

```yaml
compendium:
  merges:
    islandA:
      thresholds:             # the merge criterion; omit -> registry + evidence only
        Ncor: 0.55            # RSAT's web defaults, the same ones the reference
        cor: 0.75             # lexicon is built at, so the cross-model merge and
        w: 5                  # the library are cut by one rule
      # db_yaml: /path/db.yaml  # optional: freeze annotate's labels. Default =
      #                          # this merge's own naming stage output.
      leaves:
        cbp:    {run: <a cbp run>}   # this pipeline builds its modisco.h5 + arrays
        ag2048: {h5: ..., attr: ..., ohe: ..., bed: ...,   # external leaf: declared,
                 meta: {model: alphagenome, readout: atac_track60,   # never derived
                        union_id: AG2048, training_celltype: MULTI_TRACK_SHARED,
                        input_width: 2048, context_window: 2048}}
motiflex:
  mlex: /path/to/envs/mmv_verify/bin/mlex
  celltype: K562
  peak_universe_id: <declared universe id>
```

```bash
envs/motifline/bin/snakemake --rerun-triggers mtime -j 1 -- \
    results/merges/islandA/motiflex/lexicons/core.h5   # or .../merge/merge_nodes.tsv
```

`merge/merge_nodes.tsv` carries every internal node with its mean `Ncor`/`cor`/`w`,
the verdict, and **which arm failed** — thresholds are chosen by reading it, not
defaulted. `merge/merge_groups.tsv` gives each group its members, representative,
`representative_tied` flag and `shared`/`<label>_only` membership;
`merge/merge_decisions.json` is what `mlex compile` consumes. `mlex ingest` refuses
semantics it would have to guess, so an `h5:` leaf must declare its arrays and
metadata (`attr`/`ohe`/`bed`/`meta`); a `run:` leaf gets all of that from the run's
own config and outputs.

**The cross-call (2026-08-20).** A compiled merge no longer stops at the lexicon:
every leaf's own regions are scanned with the ONE compiled lexicon
(`rule merge_cross_regions` -> `rule merge_cross_call`, finemo `call-hits` with
`lexicons.batch`), so a difference between the hit tables is a model difference,
not a lexicon difference. `rule merge_occupancy` reduces the hit tables to
`occupancy/occupancy.tsv` (per motif x leaf: n_hits, n_peaks with >=1 hit,
frac_peaks), `agreement.tsv` (per motif x leaf pair: both / only-a / only-b /
neither on the binarised peak x motif matrices, jaccard), and one
`peak_by_motif_<leaf>.tsv.gz` count matrix per leaf. The leaves of a merge are
the same peak rows, so alignment is by row index — the step refuses to run if
the leaves' beds differ anywhere. A motif one model never calls keeps its zero
row: those zeros are the comparison's point.

`rule merge_compendium` (MotifCompendium 1.0.18, the tool that built the ENCODE
CWM motif sets) remains the comparison arm, reachable only by explicit target
`results/merges/<merge>/clusters.tsv`. Its `clusters.tsv` recovers the
`shared`/`<label>_only` bookkeeping MC does not do, and positive/negative patterns
are kept apart (`cluster_within_posneg`) because MC's similarity is sign-blind —
motiflex records metacluster mixing per node instead (`cross_meta` in the tree).

## Running

```bash
envs/motifline/bin/snakemake --rerun-triggers mtime -j <N> --resources gpu=<G> -- <target>
```

`--rerun-triggers mtime` is the standard invocation: the `run:` blocks only
assemble argv, so snakemake's code trigger would otherwise propose recomputing
finished stages after any cosmetic edit.

| you want | rule | target |
|---|---|---|
| attribute a slice/small run in one process | `attribute` | `results/<run>/attr/attr.npy` |
| attribute full-scale, one shard per GPU | `attribute_sharded` | `results/<run>/attr_shards/shard_<i>/attr_<i>.npy` |
| glue shards into one matrix | `merge_shards` | `results/<run>/attr/attr.npy` |
| use an archived `interpret_*_par_hyp` attribution | `import_attr` | `results/<run>/attr/attr.npy` |
| discover motifs (MoDISco) | `discover` | `results/<run>/modisco/modisco.h5` |
| build the reference lexicon (once, ~1h45m) | `lexicon_rsat` → `lexicon_cltagged` | `results/lexicon/union_cltagged.meme` |
| name a run's patterns | `name_patterns` | `results/<run>/naming/pattern_names.tsv` |
| name a merge's leaves + mlex annotate db | `merge_naming` | `results/merges/<merge>/naming/db.yaml` |
| prepare regions for FiNeMo | `extract_regions` | `results/<run>/finemo_regions/regions.npz` |
| call hits against the run's own lexicon | `call_hits` | `results/<run>/hits/hits.tsv` |
| call hits against an external lexicon (gw/mmv) | `call_hits_lexicon` | `results/<run>/hits_lexicon_<route>/hits.tsv` |
| same regions+lexicon with finemo 0.41 (comparison) | `call_hits_v041` | `results/<run>/hits_v041/hits.tsv` |
| registry + alignment + labels for a merge | `merge_ingest` → `merge_annotate` | `results/merges/<merge>/motiflex/evidence/annotation_candidates.parquet` |
| merge under the declared thresholds | `merge_tree` | `results/merges/<merge>/motiflex/merge/merge_groups.tsv` |
| compile the merged groups into a lexicon | `merge_compile` | `results/merges/<merge>/motiflex/lexicons/core.h5` |
| scan a leaf's regions with the compiled lexicon | `merge_cross_call` | `results/merges/<merge>/motiflex/cross_call/<leaf>/hits.tsv` |
| occupancy + cross-leaf agreement tables | `merge_occupancy` | `results/merges/<merge>/motiflex/occupancy/agreement.tsv` |
| MotifCompendium comparison arm (explicit target only) | `merge_compendium` | `results/merges/<merge>/clusters.tsv` |

Which rules a run gets is decided by what its config entry contains: `adopt_from`
→ adopted; `n_shards` → sharded+merge; neither → single-process;
`lexicon_hits`+`lexicon_model` → external-lexicon targets; `cross_version_hits`
→ the 0.41 target; `cancelled: true` → out of `rule all` but still resumable by
explicit target. Attribution geometry comes from `config.geometry` unless the run
overrides it — omit `in_window`/`crop` for new runs.


GPU selection is the caller's: `export CUDA_VISIBLE_DEVICES=<uuid>` (fully-free
cards only, pin by UUID -- indices shift on this node -- and leave one working
card free), and bound concurrency with `--resources gpu=<G>`.

## Environments

| env | role |
|---|---|
| `envs/motifline` (py3.12) | snakemake + this package |
| `envs/chrombpnet_local` (py3.10) | all compute: torch, chrombpnet, tangermeme, modisco-lite, finemo 0.30 |
| `envs/mmv_verify` (py3.12) | `mlex` (motiflex merge finish) + finemo 0.41 for the `call_hits_v041` comparison (its torch runs CPU on this node) |
| `envs/motifcompendium` (py3.11) | MotifCompendium 1.0.18, `merge_compendium` comparison arm only (editable install of `models/MotifCompendium`; no cupy on this node, so similarity runs on CPU) |
| `envs/rsat` (rsat-core 2025.04.04) | `matrix-clustering`, invoked through its own `bin/rsat` launcher; `rule lexicon_rsat` only |
| `envs/meme` | `tomtom`, for the naming stages |

Step scripts never import this package; every site path arrives as an argument
(AlphaGenome in particular: pass `--ag_src`, the package is not importable from
the compute env).

## House rules

- Attributions are HYPOTHETICAL (`--hypothetical`), always.
- hits are produced by finemo 0.30; note the version next to any hits you keep.
- `new_experiment/region_v3|v4` archives are read-only.
- peaks fed to finemo must be 10/11-column narrowPeak (summit in column 10);
  the attribution steps normalize this themselves.
