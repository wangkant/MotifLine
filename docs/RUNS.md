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

## Batch 4 (2026-08-14): external-lexicon routes

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

### `k562_ag_full` — AlphaGenome, 33917 peaks (RUNNING, batch 5 collects)

5 of 6 shards launched, one card left free; see
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
