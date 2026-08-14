# What each stage writes

Every stage writes its outputs plus `params.json`, `provenance.json`,
`run_status.json` and `guard_outcomes.json`. The last two are how you tell a
finished directory from a failed run's leftovers, which otherwise look
identical: the files a crashed attribution leaves behind have the same names as
the files a successful one leaves behind.

Shapes and dtypes below were read off `results/k562_cbp_full` and
`results/k562_ag_adopted` (2026-08-14), not from the code that intends them.

## The record every stage carries

| file | shape | contents |
|---|---|---|
| `params.json` | **list**, one entry appended per attempt, written *before* the body | what this attempt was asked to do |
| `provenance.json` | list of records | the exact argv, sha256 of every declared input, seed, software versions |
| `run_status.json` | object | the **last attempt's** outcome, written *last* |
| `guard_outcomes.json` | list | every guard that ran: `guard_id`, `subject`, `passed`, `detail`, `recorded_utc` |

`run_status.json`:

```json
{"status": "SUCCESS" | "REFUSED" | "CRASHED",
 "command": "...", "exit_code": 0, "detail": null,
 "finished_utc": "2026-08-14T12:46:45Z", "provenance_records": 1,
 "artifacts_are_from": { ...the last SUCCESSFUL run's whole record... }}
```

Three semantics that are easy to get wrong:

* **`status` is about the attempt, `artifacts_are_from` is about the files.** A
  REFUSED or CRASHED stage can still hold artifacts from an earlier successful
  run, and `artifacts_are_from` is that run's record. The string
  `NO_SUCCESSFUL_RUN_RECORDED` means nothing in the directory is trustworthy.
* **REFUSED means the body never ran.** A guard failed, its outcome was written
  to `guard_outcomes.json`, and the stage raised before any subprocess started.
  So a REFUSED directory contains no file this attempt produced.
* **`params.json` is joined, not indexed from the end.** The entry belonging to
  the artifacts is `params[artifacts_are_from.provenance_records - 1]` —
  `contracts.params_of_success()`. `params[-1]` is the last *attempt*, which may
  be the one that failed.

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
why `rows_and_width_match` compares the three output files to each other rather
than to the input peak count, and why `regions.bed` — not `peaks.bed` — is the
row key for everything downstream.

Guards: `rows_and_width_match` (post).

### AG chunk checkpoints — `_chunks_<i>/`

`attr_<c>.npy` / `ohe_<c>.npy` per 100 regions. These are **not** stage outputs
and carry no record; they exist so a kill costs at most one chunk. Resume is by
file existence (`chunk <c>/<n> exists -> resume-skip`), and they live per shard
so one shard cannot resume-skip another's chunk. A directory holding only chunks
and a CRASHED `run_status.json` is a run in progress, not a result.

## `import_attr` → `attr/`

Same three files, same contract, one difference: `attr.npy` and `ohe.npy` are
**absolute symlinks** into the archived `interpret_*_par_hyp/` tree (the old tree
is read-only and these cost GPU-days), and only `regions.bed` is newly written.
Plus `adopt_info.json`: rows, width, the peak-normalization outcome and the
three resolved source paths.

Nothing is linked until every shape check passes, so a refused adoption leaves an
empty directory rather than a half-linked one. `params.json` records
`adopted_from`, `adopted_peaks` and `expected_rows`, and `provenance.json`
carries the sha256 of the archived files themselves.

Guards: `rows_and_width_match` (post); `expected_rows`/`expected_width` are
enforced inside `adopt.adopt_attr`, so a row-count drift in the archive refuses
the stage rather than propagating.

## `discover` → `modisco/`

| file | contents |
|---|---|
| `modisco.h5` | MoDISco-lite report: `pos_patterns/` and `neg_patterns/`, each pattern with `contrib_scores` (the CWM, `(L,4)`), `sequence`, `hypothetical_contribs`, `seqlets/` |

Patterns are named **by rank** (`pattern_0`, `pattern_1`, …). Two lexicons of
different sizes therefore reuse the same names for different motifs — which is
why every level-3 comparison against an archive joins on CWM identity, not name.

Guards (pre): `upstream_success`, `attr_is_hypothetical`, `rows_and_width_match`.

## `extract_regions` → `finemo_regions/`

| file | contents |
|---|---|
| `regions.npz` | `sequences (N,4,W) int8`, `contributions (N,4,W) float16`, `chr <U`, `chr_id int64`, `start int32`, `peak_id uint32`, `peak_name <U` |

**The contributions are stored `float16`.** Everything downstream of this stage
sees a quantized copy of the attributions, and differences at ~1e-4 between two
hit tables are at that floor, not above it.

Guards (pre): `upstream_success`, `attr_is_hypothetical`.

## `call_hits` → `hits/` (also `hits_lexicon_<route>/`, `hits_v041/`)

| file | contents |
|---|---|
| `hits.tsv` | one row per hit: `chr, start, end, start_untrimmed, end_untrimmed, motif_name, hit_coefficient, hit_coefficient_global, hit_similarity, hit_correlation, hit_importance, hit_importance_sq, strand, peak_name, peak_id` |
| `hits_unique.tsv` | the same, one row per (region, motif) after de-duplication |
| `hits.bed` | the hits as a bed |
| `motif_data.tsv` | per motif: the trim window actually used, both strands |
| `motif_cwms.npy` | the CWMs as scored |
| `peaks_qc.tsv` | per region QC |
| `parameters.json` | **finemo's own** resolved parameter dump — independent of `params.json` |

`start`/`end` are the **trimmed** coordinates: `call-hits` re-derives each motif's
extent by trimming its CWM at `-t`, so two runs whose CWMs differ slightly emit
the same hits at coordinates shifted by a base or two. That is why level-3
reports carry both an exact and a ±2bp-tolerant Jaccard, and why an exact
Jaccard far below the tolerant one is a coordinate effect rather than a
motif-identity one.

Guards (pre): `upstream_success` ×2, `attr_is_hypothetical`, `hyp_flags_match`,
`finemo_version_matches`. The route variant swaps in `lexicon_exists`; the
cross-version variant replaces `finemo_version_matches` with
`recorded_finemo_version_is` + `finemo_version_declared` — replaced, never
dropped, because a version mismatch is the thing that stage measures.

`hits_v041/` and `hits_lexicon_<route>/` are **leaves**: nothing in the DAG
consumes them.

## `merge_shards` → `attr/`

The same three files as `attribute`, concatenated in shard order — shard order
*is* row order, because the bounds are contiguous slices of the peak list.

Guards: `upstream_success` on **every** shard (pre), `rows_and_width_match`
(post), plus `--expected_rows`. A merge over a REFUSED or unfinished shard would
be a short matrix that still looks valid, which is the failure this rule exists
to prevent.

## `equivalence_*` → `equivalence/`

Reports, not results. They are written by the orchestration env and carry no
guard record of their own; their inputs' records are the provenance.

| file | level | gate? |
|---|---|---|
| `attr_report.json` | 1 — new attributions vs archived, row by row | **yes**: `passed = regions_aligned AND min_row_pearson >= 0.999` |
| `modisco_report.json` | 2 — new lexicon vs archived lexicon | no |
| `hits_report.json` | 3 — new hits vs archived hits | no |
| `hits_v041_report.json` | 3 — finemo 0.41's hits vs 0.30's, same inputs | no |
| `hits_lexicon_<route>_report.json` | 3 — external-lexicon hits vs the archive's | no |

`attr_report.json`: `n`, `regions_aligned`, `bitwise_equal_rows`,
`max_abs_diff`, `min_row_pearson`, `mean_row_pearson`, `row_pearson_gate`,
`passed`.

`modisco_report.json`: `n_patterns_new/old`, `pos_/neg_new/old`,
`per_old_pattern` (each archived pattern → its best new match and that cosine),
`min_/mean_best_cosine`, `n_old_below_cosine_tol`, `flags`, `within_tolerance`.

`hits_report.json`: `join` (`cwm_identity` or `motif_name`), `n_new`, `n_old`,
`ratio_total_new_over_old`, `exact_jaccard`, `tolerant_jaccard`, `tol_bp`,
`n_motifs_new/old`, `old_motifs_without_cwm_counterpart`, `per_motif_counts`,
`flags`, `within_tolerance`.

**`within_tolerance` is a summary of a soft band, not a verdict**, and it never
suppresses a number. Level 2 and 3 compare runs at canonical (official)
parameters against an archive built at other parameters, so an out-of-band value
is evidence about a parameter — to be attributed, in the report, to the
parameter it belongs to. Only level 1 gates, and it gates on a mathematical
identity: the same model, the same input, the same code path.

## Absence is not zero

A motif **not discovered**, one **discovered and not hit**, one **the lexicon
does not contain**, and one whose count is a **measured zero** are four
different facts. `compare_hits` keeps them apart: a motif absent from the archive
is a flag naming it, an archived motif with no new CWM counterpart is listed in
`old_motifs_without_cwm_counterpart` and its hits are counted under its own name
rather than being quietly dropped into someone else's.
