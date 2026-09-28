# What each stage writes

A stage directory holds only the files that stage produces — the ones listed
below. A stage that crashed part way through can leave partial output whose
names are the same as a finished run's, so treat a directory as finished only
after its target reruns clean.

Every stage directory also carries a **`run_info.json`**,
written after the body succeeded and never before: `stage`, the exact `argv`, the
`param_sig` the rule reruns on, the `env` prefix and the `tools` versions that
environment reports, plus `host` and `started`/`finished`. Nothing reads it back
and no stage refuses to run because of it — its presence means the stage finished.

Stages that produce or adopt an attribution (`attribute`, `attribute_sharded`,
`merge_shards`, `import_attr`) also carry `checks`, measured on the array they
wrote: `hypothetical_fraction` = `frac(|attr| > 1e-9)` over the first 64 rows,
`looks_hypothetical` = whether that is ≥ 0.99, and a `warning` when it is not.
~1.0 is hypothetical, ~0.25 is projected; the two are otherwise identical on disk.

---

## `attribute` / `attribute_sharded` → `attr/` (or `attr_shards/shard_<i>/`)

| file | shape / dtype |
|---|---|
| `attr.npy` | `(N, 4, W)` `float32` — hypothetical contribution scores |
| `ohe.npy` | `(N, 4, W)` `int8` — the one-hot input the scores belong to |
| `regions.bed` | `N` lines, 11 tab-separated columns |

`W` is whatever width the step stored, and every downstream stage reads it off
`attr.npy` (the Snakefile's `attr_width`). The
steps' own defaults: **CBP 2114, AG 2048** — each model attributes its
native window and stores all of it.
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

A filled base is a base MoDISco reads as an `A` that is not one, so the step writes
`n_filled.tsv` (`chr, start, summit, n_filled_window, n_filled_kept`) listing the
regions whose fill landed inside the kept window — **empty file = no such region**.
It is written per chunk (`_chunks_<i>/n_filled_<c>.tsv`) and concatenated at the end,
so it covers chunks a resumed process skipped.

### AG chunk checkpoints — `_chunks_<i>/`

`attr_<c>.npy` / `ohe_<c>.npy` / `n_filled_<c>.tsv` per 100 regions, plus one
`chunk_identity.json`. These are **not** stage outputs; they exist so a kill costs at
most one chunk. Resume is by file existence (`chunk <c>/<n> exists -> resume-skip`),
and they live per shard so one shard cannot resume-skip another's chunk. A directory
holding only chunks is a run in progress, not a result.

`chunk_identity.json` records the arguments that decide what a chunk *contains*
(weights, peaks, track, head, in_window, crop, n_shuffles, random_state, chunk,
shard index). Resuming onto checkpoints computed under different values is refused,
so one `attr.npy` cannot mix two parameter generations.

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

## `modisco_report` → `modisco_report/`

modiscolite's own `modisco report` on `modisco.h5`, matched against
`lexicon_sources_union.meme` (the naming target), `-n`/`-l` at the CLI defaults.
`motifs.html` plus the logo images it links to. Matches are the same TomTom line
as `name_patterns`; the report shows library ids, the gene symbols are in
`naming/pattern_names.tsv`.

## `finemo_report` → `finemo_report/`

`finemo report` on `finemo_regions/regions.npz`, `hits/` and `modisco.h5`, with
`-W` = `modisco.window` (the window MoDISco ran on; the seqlet recall is wrong at
any other width). `report.html` plus the tables and figures it links to.

## `lexicon_union` → `results/lexicon/`

| file | contents |
|---|---|
| `lexicon_sources_union.meme` | the source libraries concatenated, every matrix kept; JASPAR ids joined (`MA0004.1_Arnt`), `E= 0` added where absent |

## `name_patterns` → `<run>/naming/`

| file | contents |
|---|---|
| `<label>_query.meme` | the patterns as queried: PPM cut to modiscolite's `|CWM|` 30%-of-max span |
| `tomtom_<label>/` | TomTom's own output (`-mi 1` deliberately absent — against a batch query it would score only the first pattern) |
| `pattern_names.tsv` | one row per pattern: `label, key, group, pattern, n_seqlets, trim_*`, `match0..2`/`tf0..2`/`pval0..2`/`qval0..2`/`overlap0..2`, `name`, plus the provenance the names depend on — `target`, `target_sha256`, `h14_annotation_sha256` |
| `tomtom_<label>/target.sha256` | digest of the library that produced that TomTom run. The run is reused only when the digest matches AND the TSV ends with TomTom's closing comment block; a rebuilt lexicon or a truncated file forces a re-run instead of being silently accepted |

`name` is `tf0`, the gene symbol of the first TomTom match against the source
union (`lexicon_sources_union.meme`), for every pattern whatever its q-value.
`match0..2` keep the library ids. A HOCOMOCO id's prefix is the UniProt entry name
(`TYY1` for YY1), so `tf` for a HOCOMOCO match is read from
`reference_lexicon.h14_annotation` (HOCOMOCO's `H14CORE_annotation.jsonl`, HUMAN
`gene_symbol`); an id missing from it stops the stage. A JASPAR `tf` is the symbol
after `_`, upper-cased (`MA2503.1_Banp` → `BANP`).
`qval0` is reported beside the name and nothing is decided on it.

## `merge_shards` → `attr/`

The same three files as `attribute`, concatenated in shard order — shard order
*is* row order, because the bounds are contiguous slices of the peak list.

`merge_shards` checks that shard *i* is the *i*-th directory it was given and
that each shard's attr/ohe/regions agree on row count, and the rule passes
`--expected_rows` so a merge over an unfinished shard is a hard error rather
than a short matrix that still looks valid.

