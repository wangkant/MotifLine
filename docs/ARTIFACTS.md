# What each stage writes

A stage directory holds only the files that stage produces — the ones listed
below. A stage that crashed part way through can leave partial output whose
names are the same as a finished run's, so treat a directory as finished only
after its target reruns clean.

Shapes and dtypes below were read off `results/k562_cbp_full` and
`results/k562_ag_adopted` (2026-08-14), not from the code that intends them.

Stage directories produced before 2026-08-14 also carry `params.json`,
`provenance.json`, `run_status.json` and `guard_outcomes.json`. Those are
historical records of the audit layer this pipeline used to run; the pipeline no
longer writes or reads them.

---

## `attribute` / `attribute_sharded` → `attr/` (or `attr_shards/shard_<i>/`)

| file | shape / dtype |
|---|---|
| `attr.npy` | `(N, 4, W)` `float32` — hypothetical contribution scores |
| `ohe.npy` | `(N, 4, W)` `int8` — the one-hot input the scores belong to |
| `regions.bed` | `N` lines, 11 tab-separated columns |

`W` is the run's own attribution width, `motifline.geometry.attr_width`. Canonical
since 2026-08-16: **CBP 2114, AG 2048** — each model attributes its native window
and stores all of it, so the two differ by 66bp and neither is cropped to meet the
other. Runs that declare the older AG geometry (16384bp window, central crop to
2114) keep it, because that is what their archived arrays are.
Sharded runs write `attr_<i>.npy` / `ohe_<i>.npy` /
`regions_<i>.bed` — the index is in the filename so `merge_shards` can check
that shard *i* really is the *i*-th directory it was given.

`regions.bed` is narrowPeak-shaped because the summit must survive to FiNeMo:
`finemo extract-regions-modisco-fmt -p` names the first ten columns positionally
and reads the **summit from column 10** (`finemo/data_io.py:36-47`). The CBP
step passes through the archive's 11-column `load_region_df` form; the AG step
expands its 4-column `(chr,start,end,summit)` frame to the same 11 columns
(`attr_ag.narrowpeak_text`, byte-identical to `adopt.normalize_peaks`).

**`N` is not necessarily `len(peaks.bed)`.** Both steps drop regions whose
window runs off the chromosome (`summit ± in_window/2`), and the two models then
handle ambiguous bases differently: CBP additionally drops any row containing an
`0.25` (N) column or an all-zero column (`attr_cbp.py:129-134`), while AG fills
`N → A` and keeps the row, so that the AG region set stays equal to CBP's. This is
why `regions.bed` — not `peaks.bed` — is the row key for everything downstream.

That fill used to be free: at the 16384 → 2114 geometry the N's were in flanks the
step cropped away. At the native 2048 geometry there is no flank, so a filled base
is a base MoDISco reads as an `A` that is not one. The step therefore writes
`n_filled.tsv` (`chr, start, summit, n_filled_window, n_filled_kept`) listing the
regions whose fill landed inside the kept window — empty file = no such region. It
covers only the chunks *that process* computed; a resumed run cannot recount the
chunks it skipped.

### AG chunk checkpoints — `_chunks_<i>/`

`attr_<c>.npy` / `ohe_<c>.npy` per 100 regions. These are **not** stage outputs;
they exist so a kill costs at most one chunk. Resume is by file existence
(`chunk <c>/<n> exists -> resume-skip`), and they live per shard so one shard
cannot resume-skip another's chunk. A directory holding only chunks is a run in
progress, not a result.

## `import_attr` → `attr/`

Same three files, one difference: `attr.npy` and `ohe.npy` are **absolute
symlinks** into the archived `interpret_*_par_hyp/` tree (the old tree is
read-only and these cost GPU-days), and only `regions.bed` is newly written.

`adopt.adopt_attr` checks the archive's rows and width against the run's
`rows:`/`crop:` before it links anything, and raises `ValueError` on a drift, so
a failed adoption leaves an empty directory rather than a half-linked one.

## `discover` → `modisco/`

| file | contents |
|---|---|
| `modisco.h5` | MoDISco-lite report: `pos_patterns/` and `neg_patterns/`, each pattern with `contrib_scores` (the CWM, `(L,4)`), `sequence`, `hypothetical_contribs`, `seqlets/` |

Patterns are named **by rank** (`pattern_0`, `pattern_1`, …). Two lexicons of
different sizes therefore reuse the same names for different motifs — which is
why every level-3 comparison against an archive joins on CWM identity, not name.

## `extract_regions` → `finemo_regions/`

| file | contents |
|---|---|
| `regions.npz` | `sequences (N,4,W) int8`, `contributions (N,4,W) float16`, `chr <U`, `chr_id int64`, `start int32`, `peak_id uint32`, `peak_name <U` |

**The contributions are stored `float16`.** Everything downstream of this stage
sees a quantized copy of the attributions, and differences at ~1e-4 between two
hit tables are at that floor, not above it.

## `call_hits` → `hits/` (also `hits_lexicon_<route>/`, `hits_v041/`)

| file | contents |
|---|---|
| `hits.tsv` | one row per hit: `chr, start, end, start_untrimmed, end_untrimmed, motif_name, hit_coefficient, hit_coefficient_global, hit_similarity, hit_correlation, hit_importance, hit_importance_sq, strand, peak_name, peak_id` |
| `hits_unique.tsv` | the same, one row per (region, motif) after de-duplication |
| `hits.bed` | the hits as a bed |
| `motif_data.tsv` | per motif: the trim window actually used, both strands |
| `motif_cwms.npy` | the CWMs as scored |
| `peaks_qc.tsv` | per region QC |
| `parameters.json` | finemo's own resolved parameter dump |

`start`/`end` are the **trimmed** coordinates: `call-hits` re-derives each motif's
extent by trimming its CWM at `-t`, so two runs whose CWMs differ slightly emit
the same hits at coordinates shifted by a base or two — compare hit sets with a
±2bp tolerance in mind, not by exact coordinates.

`hits_v041/` and `hits_lexicon_<route>/` are **leaves**: nothing in the DAG
consumes them.

## `lexicon_*` → `results/lexicon/`

The reference lexicon, built once and shared by every naming stage (2026-08-25).
Verified against the artifacts it replaces: `lexicon_sources_union.meme`,
`lexicon_rsat.meme` and `union_cltagged.meme` are byte-identical to the
hand-built originals, `ncor.npy` agrees to 1.1e-15.

| file | contents |
|---|---|
| `lexicon_sources_union.meme` | the source libraries concatenated, every matrix kept; JASPAR ids joined (`MA0004.1_Arnt`), `E= 0` added where absent |
| `rsat/` | RSAT `matrix-clustering`'s full output tree; only `rsat/rsat_tables/clusters.tab` is consumed, the rest is that file's provenance |
| `clusters_rsat.tsv` | flat `cluster, motif, source, is_medoid` |
| `lexicon_rsat.meme` | one **medoid** PPM per cluster (never an average) |
| `lexicon_rsat.metadata.tsv` | per cluster: `lexicon_id`, `size`, `medoid`, `width`, per-source counts, `min_ncor`, `tf_symbols`, `members` |
| `union_cltagged.meme` | the union with `{lexicon_id}__` prefixed onto each name — the TomTom target |
| `ncor.npy` | `(M, M) float64` all-pairs Ncor over the union, in source order |

`E= 0` is not cosmetic: RSAT's MEME reader requires the field and drops records
without it **silently**, so a union missing it clusters one library and looks
like it worked.

## `name_patterns` / `merge_naming` → `<run>/naming/`, `merges/<name>/naming/`

| file | contents |
|---|---|
| `<label>_query.meme` | the patterns as queried: PPM cut to modiscolite's `|CWM|` 30%-of-max span |
| `tomtom_<label>/` | TomTom's own output (`-mi 1` deliberately absent — against a batch query it would score only the first pattern) |
| `pattern_names.tsv` | one row per pattern: `label, key, group, pattern, n_seqlets, trim_*, ic_core_len`, `match0..2`/`cluster0..2`/`qval0..2`/`overlap0..2`, `assigned_cluster`, `cluster_stable`, `best_ncor`, `best_ncor_match`, `annotated` |
| `db.yaml` | (merge stages only) the gated matches as an `mlex annotate` database |

`assigned_cluster` is read off the winning target's name — the cluster id is
already there, which is what the tagged union is for. `cluster_stable` says the
top three matches all landed in one cluster; false means the pattern itself
straddles clusters and the assignment deserves a look.

`annotated` is the two-armed gate, `qval0 < gate.qval` **and**
`best_ncor >= gate.ncor`. `best_ncor` is a second, independent measurement: Ncor
of the pattern's IC>0.3 core against the untagged matrices, i.e. the same
similarity the clustering was built on, on a different trim than the TomTom
query. Rows failing either arm are kept with `annotated=False`, are not renamed,
and do not enter `db.yaml`. A pattern whose IC core is shorter than 5 columns has
no comparable Ncor (`best_ncor` empty) and fails the gate by construction.

Families are **not** here. Grouping cluster ids into site types is a human
judgement made once per project; the chain stops at the cluster.

## `merge_ingest` / `merge_align` / `merge_annotate` / `merge_tree` / `merge_compile` → `merges/<name>/motiflex/`

Four motiflex stages plus the merge itself, which is this pipeline's since motiflex
removed its `merge` subcommand on 2026-08-23 (commit `204cb3f`). Each stage is its own
rule with its own declared output — before 2026-08-25 they were one rule that skipped
stages on `.exists()` against files snakemake did not track, so a regenerated db.yaml
rebuilt the tree from a stale registry and stale annotations.

| file | contents |
|---|---|
| `project.yaml` | the ingest declaration this rule wrote from config — every semantic field (model, readout, union_id, widths, celltypes) explicit |
| `registry/registry.json` + `registry/arrays.h5` | normalised motif registry: CWM, hypothetical CWM, PPM, seqlet count, trimmed core. Node ids are `<analysis id>::<group>_patterns.<pattern_N>` |
| `evidence/alignment_edges.parquet` | per registered pair: similarity at the winning registration, per-pair shuffle-null empirical p |
| `evidence/alignment_run_summary.json` | node/pair counts and exclusion reasons (`no_offset_meets_overlap_floor` marks short-core patterns that can never merge) |
| `evidence/annotation_candidates.parquet` | the labels `merge_naming`'s db.yaml proposed, joined onto the registry |
| `merge/merge_nodes.tsv` | every internal node: `height`, `n_left`/`n_right`, `mean_Ncor`/`mean_cor`/`mean_w`, `failed_arms`, `inherited_break`, `merged`. **This is what a threshold is chosen by reading.** |
| `merge/merge_groups.tsv` | per group: `size`, `representative`, `representative_tied`, `membership` (`shared` / `<label>_only`), `min_ncor`, `n_seqlets`, `members` |
| `merge/merge_decisions.json` | `{"merges": [{"members": [...], "representative": ...}]}` — what `mlex compile` reads. Singletons are absent: they survive as themselves |
| `lexicons/core.h5` + `manifest.tsv` | the compiled lexicon, content-addressed, read back with the real finemo loader |

The criterion is RSAT `matrix-clustering`'s, not a dendrogram height: a node merges
only when the **mean cross-pair** value of **every** arm (`Ncor`, `cor`, `w`, all read
at one registration) clears its threshold, and a node that fails breaks the chain for
its ancestors rather than being re-tested (`TFBMclust/R/check_alignment.R:80-81,
106-109`; `align.motifs.R:101`). A group is therefore a maximal subtree in which every
internal node passed every arm.

`representative_tied` is not decoration. The representative is the medoid by mean
within-group distance, but for a two-member group both members have the same mean
distance by construction, so the medoid carries no information and the tie-break
(seqlet count, then node id) decides. A tied representative must not be read as "the
most typical member".

The thresholds are a hyperparameter: declared per merge in config after reading
`merge_nodes.tsv`, never defaulted. Without them the line stops at the annotated
registry. A merge that still declares the old `cut_height` is refused — it was the
parameter of a stage that no longer exists and does not map onto this criterion.

## `merge_cross_regions` / `merge_cross_call` / `merge_occupancy` → `merges/<name>/motiflex/{cross_call,occupancy}/`

The cross-call (2026-08-20): each leaf's own regions scanned with the ONE
compiled lexicon, then reduced to occupancy and cross-leaf agreement. Alignment
across leaves is by bed row index; `merge_occupancy` refuses if the beds differ.

| file | contents |
|---|---|
| `cross_call/<leaf>/regions.npz` | the leaf's arrays packaged for finemo at the leaf's native width |
| `cross_call/<leaf>/hits.tsv` | finemo `call-hits` of the compiled lexicon over that leaf (`-b` = `lexicons.batch`) |
| `occupancy/peak_by_motif_<leaf>.tsv.gz` | hit **counts** per (peak row, lexicon entry); columns named by `variant_id` |
| `occupancy/occupancy.tsv` | per (motif, leaf): `n_hits`, `n_peaks` (peaks with >=1 hit), `frac_peaks` |
| `occupancy/agreement.tsv` | per (motif, leaf pair): `n_both`/`n_only_a`/`n_only_b`/`n_neither` on the binarised matrices, `jaccard` (empty when no peak carries the motif in either leaf) |

The motif universe is the compiled lexicon's manifest, not the union of motifs
seen in hits — an entry one model never calls keeps its zero row.

## `merge_compendium` → `merges/<name>/`

Comparison arm (explicit target only; the DAG finish is motiflex above). Two or
more MoDISco h5 clustered into one lexicon by MotifCompendium 1.0.18.

| file | contents |
|---|---|
| `clusters.tsv` | one row per cluster: `name` (`cluster#<i>`), `source_cluster`, `num_motifs`, `num_seqlets`, `leaves` (per-leaf pattern counts), `posneg`, `avg_dist_from_summit`, MotifCompendium's quality columns (`best_match_similarity`, `lowest_constituent_cluster_similarity`, `lowest_internal_similarity`, `highest_external_similarity`), `n_<label>` per leaf, `membership` |
| `motifs.tsv` | one row per **input** pattern: `name` (`<label>-<pos\|neg>.pattern_<i>`), `num_seqlets`, `model` (= leaf label), `posneg`, `avg_dist_from_summit`, `cluster`, `pattern` |
| `merged.meme` | the merged lexicon — one cluster-average PPM per cluster, zero-flanks trimmed |
| `compendium.mc` | every input pattern with its cluster assignment |
| `merged.mc` | the cluster averages (`merged.meme` was written from this) |

`membership` is `shared` when every leaf contributed a pattern to the cluster,
`<label>_only` when one did, and the joined labels when some did — it is computed
here, not by MotifCompendium, which clusters on CWM similarity and never reads the
`model` column.

Positive and negative patterns cannot land in the same cluster
(`compendium.cluster_within_posneg`): similarity is computed on IC-scaled PPMs and
is sign-blind, so without the constraint an activating and a repressive pattern of
the same shape would merge.

Nothing downstream consumes this directory — the merged lexicon is the deliverable.

## `merge_shards` → `attr/`

The same three files as `attribute`, concatenated in shard order — shard order
*is* row order, because the bounds are contiguous slices of the peak list.

`merge_shards` checks that shard *i* is the *i*-th directory it was given and
that each shard's attr/ohe/regions agree on row count, and the rule passes
`--expected_rows` so a merge over an unfinished shard is a hard error rather
than a short matrix that still looks valid.

(The `equivalence_*` comparison rules and their reports were removed on
2026-08-15 together with the audit layer; pre-existing `equivalence/` result
directories, if any survive elsewhere, are historical.)
