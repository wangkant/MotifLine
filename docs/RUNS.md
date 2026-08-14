# RUNS — what has actually been executed, and what came out

Every number here is read back from a file under `results/`; that file is named
next to it. Stages are trusted only where `run_status.json` says SUCCESS.
`results/` is not tracked by git, so this file is the in-repo record of it.

## Batch 3 (2026-08-14): full-scale canonical production

### `k562_cbp_full` — ChromBPNet, 33917 peaks, end to end (COMPLETE)

Attributions recomputed from the model (not adopted), in 4 shards, then merged
and taken through modisco and finemo at the canonical official parameters.

| stage | status | UTC start → end | GPU |
|---|---|---|---|
| `attr_shards/shard_0` | SUCCESS | 11:29:21 → 11:49:23 | `GPU-A` |
| `attr_shards/shard_2` | SUCCESS | 11:29:21 → 11:49:28 | `GPU-B` |
| `attr_shards/shard_1` | SUCCESS | 11:49:25 → 12:09:25 | `GPU-A` |
| `attr_shards/shard_3` | SUCCESS | 11:49:29 → 12:09:25 | `GPU-B` |
| `attr` (merge_shards) | SUCCESS | 12:09:26 → 12:09:31 | — |
| `finemo_regions` | SUCCESS | 12:11:24 → 12:12:35 | — |
| `modisco` | SUCCESS | 12:11:24 → 12:44:37 | — |
| `hits` | SUCCESS | 12:44:39 → 12:46:45 | `GPU-B` |

Wall clock 11:29:21 → 12:46:45 = **77 min**, two fully-free cards (three were
free; one is always left free), 8 threads per process.

**Level 1** (`equivalence/attr_report.json`), all 33917 rows against
`interpret_cbp_par_hyp/attr.npy`:

```
regions_aligned    true          min_row_pearson   0.9999999999969946
n                  33917         mean_row_pearson  0.9999999999999979
bitwise_equal_rows 0             max_abs_diff      1.9371509552001953e-07
passed             true          (gate: aligned and min row pearson >= 0.999)
```

**Levels 2 and 3** against the archive (`equivalence/{modisco,hits}_report.json`),
next to the adopted run that shares the same canonical parameters but uses the
*archived* attributions (`results/k562_cbp_adopted/equivalence/…`):

| | pos new/old | neg new/old | mean CWM cos | min cos | #<0.85 | hits new/old | ratio | exact J | tol J | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| `k562_cbp_adopted` | 20/22 | 10/11 | 0.959407 | 0.701762 | 4 | 756333/748010 | 1.01113 | 0.7383 | 0.8216 | 14 |
| `k562_cbp_full` | 20/22 | 10/11 | 0.959407 | 0.701762 | 4 | 756278/748010 | 1.01105 | 0.7385 | 0.8219 | 14 |

The two rows agree to six decimals, which is the point: re-attributing changes
nothing that the archive comparison measures, so **all 14 flags are the
`modisco -w 500 → 400` change** already documented in `PARAMS_AUDIT.md` §9,
not an artefact of recomputation.

Measured directly (`equivalence/vs_adopted_{modisco,hits}_report.json`,
full vs adopted, only the attributions differ):

```
patterns 30 vs 30   mean CWM cosine 0.9992915   min 0.9787500   flags: none
hits 756278 vs 756333 (ratio 0.99993)   exact Jaccard 0.99926   tolerant 0.99934
one flagged motif: pos_patterns.pattern_18, 49 vs 222
```

That single motif is a **rank swap between two near-duplicate low-rank
patterns**: old `pattern_18` ↔ new `pattern_19` match at cosine 1.0000 and old
`pattern_19` ↔ new `pattern_18` at 0.9788, and the pair carries 869 hits new vs
1038 old out of 756278 (0.02%). A ~1e-7 attribution difference is enough to move
seqlets between two patterns that were already almost the same pattern.

## Batch 4 (2026-08-14): finemo 0.30 vs 0.41, and the external-lexicon routes

### `k562_cbp_full/hits_v041` — the same inputs, called by finemo 0.41

Same `finemo_regions/regions.npz` (extracted by 0.30) and same
`modisco/modisco.h5`, same `-M pp -t 0.3 -l 0.7 -b 2000`. **This table is data
for a version decision, not a verdict; nothing here picks a version.**

| | 0.30 (`hits/`) | 0.41 (`hits_v041/`) |
|---|---|---|
| binary | `envs/chrombpnet_local/bin/finemo` | `envs/mmv_verify/bin/finemo` |
| torch | 2.3.1+cu121 | 2.13.0+cu130 |
| device **on this node** | GPU `GPU-B` | **CPU** — driver 555.42.06 is older than cu130 needs, torch reports no GPU |
| wall | 12:44:39 → 12:46:45 = **2 min 06 s** | 14:44:06 → 15:58:47 = **74 min 41 s** |
| hits | 756278 | 756232 |
| motifs | 30 | 30 |

**The runtimes are not a version comparison.** 0.41 could not reach a GPU here,
so 74 min vs 2 min is CPU vs GPU. Whether 0.41 is faster or slower than 0.30 on
equal hardware is **not measured** and cannot be read off this row.

Agreement (`equivalence/hits_v041_report.json`, joined on `motif_name` — both
sides scored the identical modisco.h5, so the names denote the same motifs):

```
ratio_total_new_over_old  0.999939     exact_jaccard     0.999020
n_motifs 30 / 30                       tolerant_jaccard  0.999063  (tol 2bp)
flags                     0            within_tolerance  true
```

Keyed on `(chr,start,end,motif_name,strand,peak_id)` — the finest key the file
supports, and unique on both sides — 755859 hits are shared, **373 are 0.41-only
and 419 are 0.30-only** (Jaccard 0.998953). On the shared hits the scores agree
to the level of a different BLAS:

| column | max abs diff | median abs diff | pearson |
|---|---|---|---|
| `hit_similarity` | 4.649e-04 | 6.110e-05 | 0.999998739 |
| `hit_importance` | 2.175e-04 | 1.489e-06 | 0.999999996 |
| `hit_coefficient` | 7.144e+01 | 9.949e-04 | 0.999930344 |

Per-motif counts, all 30 motifs, ratio 0.41/0.30 (largest first):

| motif | 0.41 | 0.30 | 0.41/0.30 | 0.30/0.41 |
|---|---|---|---|---|
| neg_patterns.pattern_0 | 320513 | 320526 | 0.99996 | 1.00004 |
| neg_patterns.pattern_1 | 73215 | 73225 | 0.99986 | 1.00014 |
| pos_patterns.pattern_2 | 63124 | 63126 | 0.99997 | 1.00003 |
| pos_patterns.pattern_1 | 43387 | 43384 | 1.00007 | 0.99993 |
| pos_patterns.pattern_4 | 37929 | 37935 | 0.99984 | 1.00016 |
| neg_patterns.pattern_2 | 30912 | 30910 | 1.00006 | 0.99994 |
| neg_patterns.pattern_4 | 27434 | 27440 | 0.99978 | 1.00022 |
| pos_patterns.pattern_10 | 24027 | 24039 | 0.99950 | 1.00050 |
| neg_patterns.pattern_6 | 23732 | 23728 | 1.00017 | 0.99983 |
| pos_patterns.pattern_7 | 22372 | 22371 | 1.00004 | 0.99996 |
| pos_patterns.pattern_14 | 19983 | 19995 | 0.99940 | 1.00060 |
| pos_patterns.pattern_6 | 17821 | 17821 | 1.00000 | 1.00000 |
| pos_patterns.pattern_0 | 12576 | 12578 | 0.99984 | 1.00016 |
| pos_patterns.pattern_3 | 11839 | 11833 | 1.00051 | 0.99949 |
| pos_patterns.pattern_5 | 5191 | 5192 | 0.99981 | 1.00019 |
| neg_patterns.pattern_8 | 3970 | 3969 | 1.00025 | 0.99975 |
| pos_patterns.pattern_11 | 3808 | 3810 | 0.99948 | 1.00053 |
| pos_patterns.pattern_8 | 3677 | 3677 | 1.00000 | 1.00000 |
| pos_patterns.pattern_12 | 2603 | 2603 | 1.00000 | 1.00000 |
| pos_patterns.pattern_15 | 2251 | 2250 | 1.00044 | 0.99956 |
| neg_patterns.pattern_5 | 1539 | 1537 | 1.00130 | 0.99870 |
| pos_patterns.pattern_9 | 1122 | 1122 | 1.00000 | 1.00000 |
| pos_patterns.pattern_16 | 931 | 932 | 0.99893 | 1.00107 |
| pos_patterns.pattern_19 | 821 | 820 | 1.00122 | 0.99878 |
| neg_patterns.pattern_7 | 584 | 583 | 1.00172 | 0.99829 |
| pos_patterns.pattern_17 | 487 | 488 | 0.99795 | 1.00205 |
| neg_patterns.pattern_3 | 188 | 188 | 1.00000 | 1.00000 |
| neg_patterns.pattern_9 | 122 | 122 | 1.00000 | 1.00000 |
| pos_patterns.pattern_18 | 49 | 49 | 1.00000 | 1.00000 |
| pos_patterns.pattern_13 | 25 | 25 | 1.00000 | 1.00000 |
| **total** | **756232** | **756278** | 0.99994 | 1.00006 |

Every ratio is in [0.998, 1.002]; the pre-registered band is [0.8, 1.25].

#### `hitcaller.py` 0.30 vs 0.41 — what actually differs

Compared as normalized ASTs (docstrings, comments and annotations stripped):
243 → 249 statement lines. **No algorithmic change in `hitcaller.py`.**

| # | Change | Kind |
|---|---|---|
| 1 | `l,n,b,m,w` → `L,N,B,M,W`; `cwms`→`cwms_tensor` etc. so the numpy arg and its tensor stop sharing a name | surface |
| 2 | `BatchLoaderBase` becomes an `ABC` with `@abstractmethod load_batch` (was `raise NotImplementedError`) | surface |
| 3 | `fit_contribs` signature gains explicit defaults (`step_size_max=3.0`, `batch_size=2000`, …) — the same values `main.py` already passed | surface |
| 4 | `load_batch(int(load_start), int(load_end))` casts | surface |
| 5 | jaxtyping / typing imports | surface |

The two changes that could move a number are **not** in `hitcaller.py`:

| # | Where | Change | Effect |
|---|---|---|---|
| 6 | `data_io.load_modisco_motifs` | reverse-strand trim was `trim_motif(cwm_rev, t)`, is now `(len-end_fwd, len-start_fwd)` | **algorithmic in form, inert in fact**: `trim_motif` scans `sum(abs(cwm),axis=0)` for first/last position ≥ `t·max`, and that profile for `cwm[::-1,::-1]` is the forward profile reversed. Checked empirically: `motif_data.tsv` trim windows (60 rows = 30 motifs × 2 strands) are **identical** between the two runs. |
| 7 | `data_io._motif_name_sort_key` | `pattern_N` → `(N,)`, else `(-1,name)`; now tries `_`, then `#`, then `(1,name)` | same order for MoDISco's `pattern_N` names; only reaches new branches for lexicons named otherwise |
| 8 | `data_io.write_hits` | `data_unique` now filters null `chr` first | affects `hits_unique.tsv`/`hits.bed` only, not `hits.tsv` |
| 9 | `main.call_hits` | new `-T/--cwm-trim-thresholds`, `-R/--cwm-trim-coords` (unset here); explicit invalid-`mode` error; `torch.device(device)` wrap | inert at these settings |

So the ~0.1% of hits that differ is **not attributable to a code change we can
point at**: identical trim windows, identical motif set, score differences at
1e-4 on shared hits. It has the shape of floating-point non-determinism in the
proximal-gradient optimizer's convergence test — which on this node is also a
CPU-vs-GPU difference, since the two runs did not use the same backend.
`hits_v041/` is a leaf: nothing downstream reads it.

### External-lexicon routes

### `k562_cbp_cl1_adopted` — gw and mmv routes vs the region_v4 archive

**Why this context.** The archive's 15 contexts × 2 models all score
attributions that this pipeline would have to produce first. Exactly one family
of them is already in the tree: `prep_allclusters.py` built each CBP cluster
leaf by row-subsetting `interpret_cbp_par_hyp/{attr,ohe}.npy` — the same archive
`k562_cbp_adopted` links — on an explicit `(chrom, summit_abs)` join. cl1
(n=2399) is the smallest such leaf with hits on **both** routes.
`tests/test_lexicon_context_is_subset.py` checks that claim as byte equality
over all 2399×4×2114 attr and ohe values (passes).

The `ag2048` half of the route table is **not** reproducible from this tree:
its attributions come from `region_v4/02_motif_grammar/interpret_ag2048_par_hyp`
at width **2048**, a different run from the adopted AG archive (width 2114).
The route entry is ported and configured; only the CBP side is validated.

| stage | status | UTC start → end | wall | note |
|---|---|---|---|---|
| `attr` (import_attr) | SUCCESS | 14:45:06 → 14:45:06 | <1 s | symlinks + `peaks_np.bed` passthrough, 2399 rows |
| `finemo_regions` | SUCCESS | 14:45:08 → 14:45:16 | 8 s | `-w 2114` |
| `hits_lexicon_mmv` | SUCCESS | 14:45:20 → 14:53:43 | **503 s** | 289-pattern union lexicon, `-b 1000` |
| `hits_lexicon_gw` | SUCCESS | 14:53:51 → 14:54:15 | **24 s** | 33-pattern genome-wide lexicon, `-b 1000` |

Both on `GPU-C`, finemo 0.30, `-M pp -t 0.3 -l 0.7`.

**Level 3 vs the archive** (`equivalence/hits_lexicon_{gw,mmv}_report.json`,
joined on `motif_name` — legitimate here because `-m` pointed both runs at the
*same lexicon file*, so the names denote the same motifs):

| route | archive hits | new hits | ratio | exact Jaccard | tolerant J | motifs new/old | flags |
|---|---|---|---|---|---|---|---|
| gw | 54810 | 54810 | 1.000000 | **1.000000** | 1.000000 | 32 / 32 | 0 |
| mmv | 98775 | 98776 | 1.000010 | 0.999989 | 0.999989 | 211 / 211 | 0 |

The gw route reproduces the archive **exactly** — every one of 54810 hits, at
identical coordinates, on identical motifs. The mmv route differs by a single
hit: `chr3:16386807-16386815 pos_patterns.pattern_57 +`, present in the new run
and absent from the archive, with **nothing lost in the other direction**
(486 vs 485 for that motif; every other one of the 211 motifs matches exactly).
That is 1 call in 98775 = 0.001%, inside the 0–0.016% run-to-run jitter the
archive's own script measured for this hit caller
(`run_finemo.sh:34-36`) — a coefficient at the convergence boundary, not a
routing or parameter difference.

This is the strongest reproduction in the project so far, and it is expected to
be: unlike the adopted `discover → call_hits` chains, nothing here is
recomputed except the hit calling itself. Same attributions (symlinked), same
lexicon file, same `-M/-t/-l/-b`, same finemo 0.30.

### `k562_ag_full` — AlphaGenome, 33917 peaks (**CANCELLED 2026-08-15**)

**Stopped on purpose at 63.4% of the attribution, by the user's decision.** Not
a failure, and not a result: nothing downstream of it was ever computed, so
nothing is retracted. The canonical AG side of this project is
`k562_ag_adopted`. Full record, per-shard checkpoint state and the exact resume
commands: `results/k562_ag_full/CANCELLED.md` (that tree is gitignored; this
section is the in-repo copy of the decision).

Grounds, both already on disk: the AG *code path* is proven by
`k562_ag_slice200`'s level-1 report (`min_row_pearson 0.9999999994123285`,
`results/k562_ag_slice200/equivalence/attr_report.json`), and the AG
*downstream* by `k562_ag_adopted` running all 33917 AG rows through
discover/extract/call_hits (§9 of `PARAMS_AUDIT.md`). Attribution is per-row, so
rows 201..33917 exercise no further code path.

State kept, resumable: 215 of 342 chunks on disk (21500/33917 peaks, 868 MB),
shard 5 never launched. All five shard `run_status.json` files read
`CRASHED / "body exited -15"` — the SIGTERM — and are **left as they are**;
`merge_shards` refuses over a non-SUCCESS shard, which is the point. `rule all`
no longer asks for this run (`config: cancelled: true`,
`src/motifline/runs.py`); every explicit target still resolves.

5 of 6 shards had been launched, one card left free; see
`results/k562_ag_full/LAUNCH.md` for PIDs, cards, logs, the first-chunk
timestamps and the verified kill/resume evidence.

**Shard 4 was stopped twice in batch 4** to free the only card batch 4 could
get (`LAUNCH.md` records the restart command; the orchestrator authorised this
for the highest-numbered shard). Ledger:

| event | UTC | chunks on disk |
|---|---|---|
| stopped (`SIGTERM`, `CRASHED: body exited -15`) | 13:37:07 | 6 |
| restarted, `chunk 0..5 -> resume-skip` in `logs/ag_full_shard4.log` | 14:42:21 | 6 |
| stopped again (still inside the resume-skip phase) | 14:44:37 | 6 |
| restarted, resume-skip verified again | 14:54:30 | 6 |

**Total stopped: 75 min 07 s.** Chunk count never moved, so the only lost work
is the chunk in flight at 13:37 (≤1 chunk ≈ 13 min); no completed chunk was
discarded and no output was corrupted. GPU `GPU-C` in both cases.
