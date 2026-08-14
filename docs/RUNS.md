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

### `k562_ag_full` — AlphaGenome, 33917 peaks (RUNNING, batch 5 collects)

5 of 6 shards launched, one card left free; see
`results/k562_ag_full/LAUNCH.md` for PIDs, cards, logs, the first-chunk
timestamps and the verified kill/resume evidence.
