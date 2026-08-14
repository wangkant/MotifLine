# k562_ag_full — CANCELLED 2026-08-15

This run was stopped on purpose before it finished. Nothing here failed.
`LAUNCH.md` (written when it started) describes a batch-5 collection that will
not happen; **this file supersedes `LAUNCH.md`'s "What batch 5 has to do"**.

The canonical AlphaGenome side of this project is **`k562_ag_adopted`** — the
archived AG attributions, adopted into the contract tree and taken through
MoDISco and FiNeMo at canonical parameters.

## Why

The user's decision, on the ground that the remaining verification value was
zero. Both halves of what a full AG re-attribution would have shown are already
on disk:

| What full-scale AG would have demonstrated | Already demonstrated by | Evidence |
|---|---|---|
| motifline's AG attribution code path reproduces the archived AG math | `k562_ag_slice200`, level-1 attr comparison on 200 peaks | `results/k562_ag_slice200/equivalence/attr_report.json`: `regions_aligned true`, `min_row_pearson 0.9999999994123285`, `mean_row_pearson 0.9999999998822585`, `max_abs_diff 2.42e-06`, `passed true` |
| the AG downstream (discover → extract_regions → call_hits) runs end to end on 33917 AG rows | `k562_ag_adopted`, all four stages SUCCESS on the full 33917 rows | `results/k562_ag_adopted/*/run_status.json`; consistency vs the archive in `results/k562_ag_adopted/equivalence/{modisco,hits}_report.json` (28/28 pos patterns, mean CWM cosine 0.978 — `docs/PARAMS_AUDIT.md` §9) |

Attribution is a per-row computation: rows 201..33917 exercise no code path that
rows 1..200 did not. So the only thing the remaining ~12 GPU-hours × 6 cards
would have produced is a second copy of numbers that already exist, at the cost
of six cards for another night.

**This is a scope decision, not a result.** No claim in this project is made or
withdrawn because of it, and nothing downstream of `k562_ag_full` was ever
computed, so nothing has to be retracted.

## State at the moment of cancellation

Five of six shards were running; shard 5 was never launched (only five cards
were free and the project rule holds one card back — `LAUNCH.md`). The
orchestrator sent `SIGTERM` to all five compute processes at
**2026-08-14T22:56:35–36Z**.

Each shard's `run_status.json` therefore says:

```
"status": "CRASHED", "detail": "body exited -15",
"artifacts_are_from": "NO_SUCCESSFUL_RUN_RECORDED"
```

**That record is correct and is deliberately left as it is.** `-15` is the
SIGTERM. The stage bodies did not finish, so no shard may claim SUCCESS, and a
`merge_shards` over them would refuse at `upstream_success` — which is the
intended behaviour. Editing these files to SUCCESS would be falsifying a record
in the one direction the guards exist to prevent.

Chunk checkpoints on disk (counted 2026-08-14T19:00Z; each chunk file is a
complete `(100, 4, 2114) float32` array, indices contiguous from 0, all
3382528 bytes):

| shard | rows assigned | chunks done / 57 | rows attributed | % |
|---|---|---|---|---|
| 0 | `[0:5652]` = 5652 | 47 (`attr_0..attr_46`) | 4700 | 83.2 |
| 1 | `[5652:11305]` = 5653 | 44 (`attr_0..attr_43`) | 4400 | 77.8 |
| 2 | `[11305:16958]` = 5653 | 44 (`attr_0..attr_43`) | 4400 | 77.8 |
| 3 | `[16958:22611]` = 5653 | 43 (`attr_0..attr_42`) | 4300 | 76.1 |
| 4 | `[22611:28264]` = 5653 | 37 (`attr_0..attr_36`) | 3700 | 65.5 |
| 5 | `[28264:33917]` = 5653 | **never launched** | 0 | 0 |
| **total** | 33917 | 215 | **21500** | **63.4** |

Shard bounds are `np.linspace(0, 33917, 7).astype(int)` =
`[0, 5652, 11305, 16958, 22611, 28264, 33917]` (`docs/PARAMS_AUDIT.md` §10).

## The checkpoints are kept, and they are resumable

868 MB under `results/k562_ag_full/` (`du -sh`, 2026-08-14). Nothing was deleted.

Resume is by chunk-existence, not by any bookkeeping file: on relaunch the step
logs `chunk <i>/57 exists -> resume-skip` and continues at the first missing
index. This was verified live during batch 3 by deliberately killing shard 0
after one chunk and relaunching it — see `LAUNCH.md` §"Resume behaviour
(verified live, not asserted)", `logs/ag_full_shard0.log:110`. Chunk files live
in `_chunks_<i>/` per shard, so no shard can resume-skip another shard's chunk.
A CRASHED shard is restarted by re-requesting the same target; restarting is
always safe.

Left-over GPU locks: **none**. `.gpu_locks/` was empty after the kills (the
lease releases in a `finally`), and the five compute PIDs recorded in
`LAUNCH.md` (1042244, 1078462, 1078849, 1079366, 1079822) were all confirmed
dead with `kill -0` before this file was written.

## If this is ever resumed

Nothing needs to be repaired first. The exact commands:

1. **Un-cancel the run.** In `config/default.yaml` delete `cancelled: true` and
   `cancelled_note:` from `runs.k562_ag_full`. Until then the run is out of
   `rule all` (`src/motifline/runs.py`), though every explicit target below
   still resolves.

2. **Restart the five stopped shards and start shard 5.** One `setsid`
   snakemake per shard, exactly the form in `LAUNCH.md` §"Not launched", with
   `i` in `0..5` — and only as many at once as there are fully free cards minus
   one:

   ```bash
   cd <KANT_ROOT>/motifline
   setsid nohup <KANT_ROOT>/envs/motifline/bin/snakemake \
     -j 1 --resources gpu=1 --rerun-triggers mtime --nolock -- \
     results/k562_ag_full/attr_shards/shard_<i>/attr_<i>.npy \
     > logs/ag_full_shard<i>.log 2>&1 < /dev/null &
   ```

   Remaining work at the state above: 5652+5653+5653+5653+5653+5653 − 21500 =
   **12417 peaks**, ≈13 min per 100-peak chunk with five shards resident
   (`LAUNCH.md`), i.e. ≈27 h of single-card time spread over however many cards
   are free.

3. **Merge and compare**, once and only once all six shards report SUCCESS
   (`merge_shards` pre-guards `upstream_success` on each shard directory, so a
   five-shard merge cannot happen by accident):

   ```bash
   <KANT_ROOT>/envs/motifline/bin/snakemake \
     --rerun-triggers mtime -j 1 --resources gpu=1 -- \
     results/k562_ag_full/equivalence/attr_report.json \
     results/k562_ag_full/equivalence/modisco_report.json \
     results/k562_ag_full/equivalence/hits_report.json
   ```

   Pre-registered expectation for the level-1 report, unchanged: the archived AG
   chunks were stored as **fp16** (`ag_shap_shard.py` docstring), so
   `max_abs_diff` ~1e-3 is quantization, not disagreement;
   `min_row_pearson >= 0.999` remains the gate.

---
**2026-08-15 用户最终裁定：本 run 永久关闭，不再续跑。** 上文的续跑命令仅作
历史记录保留；AG 侧正典为 `results/k562_ag_adopted`（见 PARAMS_AUDIT §14）。
