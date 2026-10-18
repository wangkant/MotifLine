# What each stage writes

A stage directory holds the files listed below plus `run_info.json` and
`stage.log` (everything the body printed, stderr included). A stage that
crashed part way can leave partial files with the final names; treat a directory as
finished only once its target has rerun cleanly.

`run_info.json` is deleted when a stage starts and written only after it succeeds:
`stage`, the exact `argv`, the `param_sig` the rule reruns on, the `env` prefix and the
`tools` versions that env reports, `host`, `started`/`finished`, the `motifline`
version, and `code`: for motifline (plus `chrombpnet_src` for PyTorch-port and `ag_src` for
AlphaGenome attribution stages) the git commit, with a digest appended when the checkout has uncommitted
changes, or, for code that is not in git (an installed package), `commit: null` and a
`sha256` of its `.py` files and Snakefile.

Stages that produce or adopt an attribution (`attribute`, `attribute_sharded`,
`merge_shards`, `import_attr`) record `checks` measured on the written array:
`hypothetical_fraction` = `frac(|attr| > 1e-9)` over the first 64 rows,
`looks_hypothetical` = whether that is ≥ 0.99, and a `warning` when it is not.
Hypothetical scores give ~1.0, projected ones ~0.25; the files are otherwise identical.

---

## `attribute` / `attribute_sharded` → `attr/` (or `attr_shards/shard_<i>/`)

| file | shape / dtype |
|---|---|
| `attr.npy` | `(N, 4, W)` `float32`, hypothetical contribution scores |
| `ohe.npy` | `(N, 4, W)` `int8`, the one-hot input the scores belong to |
| `regions.bed` | `N` lines, 10 (official ChromBPNet) or 11 tab-separated columns |

`W` is the stored width; downstream stages read it from `attr.npy`. Defaults: CBP
2114 bp (the official ChromBPNet: the model's input length); AG 2048 bp in and 2048 bp kept. 2048 bp is this pipeline's choice for AG,
not the model's training context (~1 Mb).

Sharded runs write `attr_<i>.npy` / `ohe_<i>.npy` / `regions_<i>.bed`.

`regions.bed` is narrowPeak-shaped because finemo reads the summit from column 10.
The official ChromBPNet writes chrombpnet's `interpreted_regions.bed`; the PyTorch port
writes its `load_region_df` frame; AG writes the same 11 columns
(`attr_ag.narrowpeak_text`, identical to `adopt.normalize_peaks`).

**`N` can be smaller than the peak count.** Every path drops peaks whose window runs
off the chromosome (`summit ± in_window/2`). The PyTorch port and AG also drop peaks
whose analysed window contains a non-ACGT base (CBP: the 2114 bp input; AG: the
central `crop` bp); the official ChromBPNet keeps them, with all-zero one-hot columns,
as chrombpnet does.

Peak files with two peaks sharing a summit, or with chromosomes absent from
`chrom_sizes`, are refused. With different windows (2114 vs 2048) the two models
can keep slightly different peaks, so join on `regions.bed`, never on the peak file.

When AG's input is wider than `crop`, non-ACGT bases in the flanks are set to `A` so
the input is valid one-hot. `n_filled.tsv` (`chr, start, summit, n_filled`) lists
those peaks; an empty table means none.

### The official ChromBPNet

cbp runs without `backend: pytorch` run chrombpnet's own interpret entry point
(`chrombpnet.evaluation.interpret.interpret.main`) on the run's peaks, counts head only,
and convert what it writes as `import_attr` does below: `attr.npy` is `/shap/seq`
(float16 in chrombpnet's file, float32 here), `ohe.npy` is `/raw/seq`, and nothing is
written unless the scores are hypothetical, `/projected_shap` equals `raw * shap`, and
every row equals the genome. chrombpnet's files stay in the directory:
`chrombpnet.counts_scores.h5`, `chrombpnet.interpreted_regions.bed`,
`chrombpnet.interpret.args.json`, and `input_peaks.bed`, the peaks it was given (10
columns); sharded runs add `_<i>`.

The references are `n_shuffles` dinucleotide shuffles from deeplift, shuffle k drawn
from `RandomState(random_state + k)`; the official code leaves them unseeded. The stage
fails unless chrombpnet asked for references once per sequence, so the seeding cannot
be bypassed silently. There is no `convergence.json`: the official DeepSHAP has no
such check.

### tangermeme paths

`convergence.json` (AlphaGenome and the ChromBPNet PyTorch port) records tangermeme's DeepLIFT check: for each
(sequence, reference) pair, |f(x) − f(ref) − Σ attributions|. It holds the threshold
(1e-3, tangermeme's default), the number of pairs, how many exceeded it and the largest
delta. Nothing is decided on it; a large share over the threshold means the
attribution rules do not satisfy summation-to-delta for this model.

Sharded runs write `n_filled_<i>.tsv` / `convergence_<i>.json` per shard;
`merge_shards` combines them into `attr/`.

### AG chunk checkpoints: `_chunks/` (`_chunks_<i>/` when sharded)

`attr_<c>.npy` / `ohe_<c>.npy` / `n_filled_<c>.tsv` per `chunk` regions, plus
`chunk_identity.json`. They are not stage outputs; they let a killed job resume.
`chunk_identity.json` holds the parameters, a sha256 over every kept region and its
sequence, the sha256 of the weights, the `ag_src` commit and the sha256 of
`attr_ag.py` / `ag_support.py`. Resuming onto chunks
with a different identity is refused; delete the directory to recompute. Chunks are
not removed after success, so a rerun with changed inputs also stops here until
they are deleted.

## `import_attr` → `attr/`

With `adopt_format: chrombpnet_h5`, the three files are written, not linked:
`attr.npy` is `/shap/seq` of chrombpnet's `counts_scores.h5` (hypothetical, float16
in the file, float32 here), `ohe.npy` is `/raw/seq`, and `regions.bed` is
`interpreted_regions.bed` (10 columns). Nothing is written unless `/shap` is nonzero
off the observed base (hypothetical, not projected), `/projected_shap` equals
`raw * shap`, and every row's one-hot equals `genome.fasta` at `start + summit ± W/2`. chrombpnet keeps peaks with non-ACGT bases (all-zero
columns), so these arrays can contain them; the count is printed in the stage log.

Otherwise (`npy`):

The same three files. `attr.npy` and `ohe.npy` are absolute symlinks to the adopted
arrays; only `regions.bed` is written. Rows are checked against the run's `rows:`,
attr and ohe widths against each other and against the declared width (`crop` for
AG, `in_window` for CBP) when there is one. On any mismatch nothing is linked.

## `merge_shards` → `attr/`

The shard triples concatenated in shard order (shard order is row order). Shard
*i* must be the *i*-th directory; row counts must agree within each shard; with
`rows:` set, a total that differs is refused before anything is written. `rows:`
counts rows after the filters above, not peaks. A single-process run cannot set it.

## `discover` → `modisco/`

| file | contents |
|---|---|
| `modisco.h5` | `pos_patterns/` and `neg_patterns/`; each pattern has `contrib_scores` (CWM, `(L,4)`), `sequence`, `hypothetical_contribs`, `seqlets/` |

Patterns are named by rank (`pattern_0`, …), so the same name in two runs can be
different motifs. Compare patterns across runs by CWM, not name.

## `extract_regions` → `finemo_regions/`

| file | contents |
|---|---|
| `regions.npz` | `sequences (N,4,W) int8`, `contributions (N,4,W) float16`, `chr`, `chr_id`, `start`, `peak_id`, `peak_name` |

Contributions are stored as `float16`; differences near 1e-4 between two hit tables
are at that precision.

## `call_hits` → `hits/`

| file | contents |
|---|---|
| `hits.tsv` | one row per hit per region: `chr, start, end, start_untrimmed, end_untrimmed, motif_name, hit_coefficient, hit_coefficient_global, hit_similarity, hit_correlation, hit_importance, hit_importance_sq, strand, peak_name, peak_id` |
| `hits_unique.tsv` | `hits.tsv` de-duplicated on genomic position (`chr, start, motif_name, strand`) |
| `hits.bed` | `hits_unique.tsv` as a bed |
| `motif_data.tsv` | per motif: the trim window used, both strands |
| `motif_cwms.npy` | the CWMs as scored |
| `peaks_qc.tsv` | per-region QC |
| `parameters.json` | finemo's resolved parameters |

Windows of nearby peaks overlap, so one genomic hit can appear once per region in
`hits.tsv`. Count genomic occurrences from `hits_unique.tsv`.

`start`/`end` are trimmed coordinates (CWM trimmed at `-t`); hits from slightly
different CWMs can shift by a base or two.

## `modisco_report` → `modisco_report/`

modiscolite's `modisco report` against `lexicon_sources_union.meme`, CLI defaults
otherwise. Its TomTom query is written slightly differently from `name_patterns`'
(`nsites`), so near-tied matches can be ordered differently; `pattern_names.tsv` is
the reference for names.

## `finemo_report` → `finemo_report/`

`finemo report` with `-W` = `modisco.window`; seqlet recall is wrong at any other width.

## `lexicon_union` → `results/lexicon/`

| file | contents |
|---|---|
| `lexicon_sources_union.meme` | the sources concatenated, every matrix kept; JASPAR ids joined (`MA0004.1_Arnt`), `E= 0` added where absent |

A MOTIF line that is not `id` or `id name`, or an id repeated across sources, stops
the stage.

## `name_patterns` → `<run>/naming/`

| file | contents |
|---|---|
| `<label>_query.meme` | patterns as queried: PPM cut to modiscolite's `|CWM|` ≥ 30%-of-max span |
| `tomtom_<label>/` | TomTom output, regenerated on every run of the stage |
| `pattern_names.tsv` | per pattern: `label, key, group, pattern, n_seqlets, trim_*`, `match0..2`, `tf0..2`, `pval0..2`, `qval0..2`, `overlap0..2`, `name`, `confident`, `qval_threshold`, `target`, `target_sha256`, `h14_annotation_sha256` |

`key` (`pos_patterns.pattern_0`) is the `motif_name` used in `hits.tsv`.

`name` is `tf0`: the gene symbol of the best TomTom match (lowest q, then p) at
E ≤ 10, whatever its q-value. `confident` is true when `qval0` ≤
`reference_lexicon.qval_threshold` (default 0.05 in the example config). It is a flag
for filtering, not a filter: no pattern is renamed or dropped, and a `name` with
`confident` false is a candidate annotation, not an identified TF. A pattern with no match at E ≤ 10 has an empty name. HOCOMOCO symbols come
from `H14CORE_annotation.jsonl` (HUMAN `gene_symbol`); an id missing there stops the
stage. JASPAR symbols are the name after `_`, upper-cased (`MA2503.1_Banp` → `BANP`);
for non-human entries that is not an HGNC symbol, and dimers stay `A::B`.
