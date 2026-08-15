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

`W` is 2114 for both models: ChromBPNet's input window, and AG's central crop of
its 16384bp window. Sharded runs write `attr_<i>.npy` / `ohe_<i>.npy` /
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
`N → A` inside the flanks it is about to crop away and keeps the row
(`attr_ag.py:144-147`), so that the AG region set stays equal to CBP's. This is
why `regions.bed` — not `peaks.bed` — is the row key for everything downstream.

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
