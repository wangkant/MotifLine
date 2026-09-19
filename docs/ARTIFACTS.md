# What each stage writes

A stage directory holds only the files that stage produces — the ones listed
below. A stage that crashed part way through can leave partial output whose
names are the same as a finished run's, so treat a directory as finished only
after its target reruns clean.

Shapes and dtypes below were read off `results/k562_cbp_full` and
`results/k562_ag_adopted` (2026-08-14), not from the code that intends them.

Every stage directory also carries a **`run_info.json`** (since 2026-09-19),
written after the body succeeded and never before: `stage`, the exact `argv`, the
`param_sig` the rule reruns on, the `env` prefix and the `tools` versions that
environment reports, plus `host` and `started`/`finished`. Nothing reads it back
and no stage refuses to run because of it — its presence means the stage finished.

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

`W` is whatever width the step stored, and every downstream stage reads it off
`attr.npy` (the Snakefile's `attr_width`; `geometry.py` was removed 2026-09-07). The
steps' own defaults: **CBP 2114, AG 2048** — each model attributes its
native window and stores all of it. Runs that declare the older AG geometry (16384bp
window, central crop to 2114) keep it, because that is what their archived arrays are.
Sharded runs write `attr_<i>.npy` / `ohe_<i>.npy` /
`regions_<i>.bed` — the index is in the filename so `merge_shards` can check
that shard *i* really is the *i*-th directory it was given.

`regions.bed` is narrowPeak-shaped because the summit must survive to FiNeMo:
`finemo extract-regions-modisco-fmt -p` names the first ten columns positionally
and reads the **summit from column 10** (`finemo/data_io.py:36-47`). The CBP
step passes through the archive's 11-column `load_region_df` form; the AG step
expands its 4-column `(chr,start,end,summit)` frame to the same 11 columns
(`attr_ag.narrowpeak_text`, byte-identical to `adopt.normalize_peaks`).

**`N` is not necessarily `len(peaks.bed)`.** Every step drops regions whose
window runs off the chromosome (`summit ± in_window/2`), and the models then
handle ambiguous bases differently: CBP additionally drops any row containing an
`0.25` (N) column or an all-zero column (`attr_cbp.py:129-134`), while AG fills
`N → A` and keeps the row, so that the AG region set stays equal to CBP's. This is
why `regions.bed` — not `peaks.bed` — is the row key for everything downstream.

That fill used to be free: at the 16384 → 2114 geometry the N's were in flanks the
step cropped away. At the native 2048 geometry there is no flank, so a filled base
is a base MoDISco reads as an `A` that is not one. The step therefore writes
`n_filled.tsv` (`chr, start, summit, n_filled_window, n_filled_kept`) listing the
regions whose fill landed inside the kept window — **empty file = no such region**,
and since 2026-08-25 that is true: the record is written per chunk
(`_chunks_<i>/n_filled_<c>.tsv`) and concatenated at the end, so it covers the whole
run including the chunks a resumed process skipped. Before that it was accumulated in
memory and rewritten from whatever the last process happened to compute, so resuming
silently replaced the record with a shorter one.

### AG chunk checkpoints — `_chunks_<i>/`

`attr_<c>.npy` / `ohe_<c>.npy` / `n_filled_<c>.tsv` per 100 regions, plus one
`chunk_identity.json`. These are **not** stage outputs; they exist so a kill costs at
most one chunk. Resume is by file existence (`chunk <c>/<n> exists -> resume-skip`),
and they live per shard so one shard cannot resume-skip another's chunk. A directory
holding only chunks is a run in progress, not a result.

`chunk_identity.json` records the arguments that decide what a chunk *contains*
(weights, peaks, track, head, in_window, crop, n_shuffles, random_state, chunk,
shard index). Resuming onto checkpoints computed under different values is **refused**:
the resume test was previously the chunk index alone, so changing a run's `track` or
`n_shuffles` and re-running produced an `attr.npy` concatenated from two parameter
generations, silently.

## `import_attr` → `attr/`

Same three files, one difference: `attr.npy` and `ohe.npy` are **absolute
symlinks** into the archived `interpret_*_par_hyp/` tree (the old tree is
read-only and these cost GPU-days), and only `regions.bed` is newly written.

`adopt.adopt_attr` checks the archive's rows against the run's `rows:` (and that
attr and ohe agree on width) before it links anything, and raises `ValueError` on a
drift, so a failed adoption leaves an empty directory rather than a half-linked one.

## `discover` → `modisco/`

| file | contents |
|---|---|
| `modisco.h5` | MoDISco-lite report: `pos_patterns/` and `neg_patterns/`, each pattern with `contrib_scores` (the CWM, `(L,4)`), `sequence`, `hypothetical_contribs`, `seqlets/` |

Patterns are named **by rank** (`pattern_0`, `pattern_1`, …). Two lexicons of
different sizes therefore reuse the same names for different motifs — compare
patterns across runs on CWM identity, not name.

## `extract_regions` → `finemo_regions/`

| file | contents |
|---|---|
| `regions.npz` | `sequences (N,4,W) int8`, `contributions (N,4,W) float16`, `chr <U`, `chr_id int64`, `start int32`, `peak_id uint32`, `peak_name <U` |

**The contributions are stored `float16`.** Everything downstream of this stage
sees a quantized copy of the attributions, and differences at ~1e-4 between two
hit tables are at that floor, not above it.

## `call_hits` → `hits/`

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

## `lexicon_*` → `results/lexicon/`

The reference lexicon, built once and shared by every run's naming stage (2026-08-25).
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
| `ncor.npy` | `(M, M) float64` all-pairs Ncor over the union, in source order. Evidence, **not a cache**: it is the similarity every cluster was judged on and nothing else on disk records it, but no stage reads it back |

`E= 0` is not cosmetic: RSAT's MEME reader requires the field and drops records
without it **silently**, so a union missing it clusters one library and looks
like it worked.

## `name_patterns` → `<run>/naming/`

| file | contents |
|---|---|
| `<label>_query.meme` | the patterns as queried: PPM cut to modiscolite's `|CWM|` 30%-of-max span |
| `tomtom_<label>/` | TomTom's own output (`-mi 1` deliberately absent — against a batch query it would score only the first pattern) |
| `pattern_names.tsv` | one row per pattern: `label, key, group, pattern, n_seqlets, trim_*, ic_core_len`, `match0..2`/`cluster0..2`/`qval0..2`/`overlap0..2`, `assigned_cluster`, `cluster_stable`, `best_ncor`, `best_ncor_match`, `annotated`, plus the provenance the names depend on — `target`, `target_sha256`, `gate_q`, `gate_ncor`, `w_th` |
| `tomtom_<label>/target.sha256` | digest of the library that produced that TomTom run. The run is reused only when the digest matches AND the TSV ends with TomTom's closing comment block; a rebuilt lexicon or a truncated file forces a re-run instead of being silently accepted |

`assigned_cluster` is read off the winning target's name — the cluster id is
already there, which is what the tagged union is for. `cluster_stable` says the
top three matches all landed in one cluster; false means the pattern itself
straddles clusters and the assignment deserves a look.

`annotated` is the two-armed gate, `qval0 < gate.qval` **and**
`best_ncor >= gate.ncor`. `best_ncor` is a second, independent measurement: Ncor
of the pattern's IC>0.3 core against the untagged matrices, i.e. the same
similarity the clustering was built on, on a different trim than the TomTom
query. Rows failing either arm are kept with `annotated=False` and are not renamed. A pattern whose IC core is shorter than 5 columns has
no comparable Ncor (`best_ncor` empty) and fails the gate by construction.

Families are **not** here. Grouping cluster ids into site types is a human
judgement made once per project; the chain stops at the cluster.

## `merge_shards` → `attr/`

The same three files as `attribute`, concatenated in shard order — shard order
*is* row order, because the bounds are contiguous slices of the peak list.

`merge_shards` checks that shard *i* is the *i*-th directory it was given and
that each shard's attr/ohe/regions agree on row count, and the rule passes
`--expected_rows` so a merge over an unfinished shard is a hard error rather
than a short matrix that still looks valid.

(The `equivalence_*` comparison rules were removed on 2026-08-15 with the audit
layer; the cross-model merge, the external-lexicon routes and the finemo 0.41 arm on
2026-09-07. `results/merges/`, `hits_lexicon_*/` and `hits_v041/` directories, where
they survive, are historical.)
