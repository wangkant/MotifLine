# motifline — final report (buildout batches 1–5, closed 2026-08-15)

Scope as delivered: **attribution → MoDISco → FiNeMo**, for ChromBPNet and
AlphaGenome, on K562 and GM12878, with archived attributions adoptable into the
same contract tree, external-lexicon (gw/mmv) hit calling, and a finemo
0.30-vs-0.41 comparison. `insilico` and `syntax` were cut by the user and are
not migrated; motiflex's `occupancy`/`instances` segments are not wired.

Every number below is read back from a named file. Nothing is quoted from
recollection, and nothing that was not computed is reported as a value.

## 1. Deliverables

### Vertical slice (pre-batch-1, context)
`contracts.run_stage` (params+provenance before the body, run_status last),
the guard set, GPU leasing, the ChromBPNet attribution step, level-1
`compare_attr`, the Snakemake DAG, and the `k562_cbp_slice500` run.

### Batch 1 — AG attribution step
* `src/motifline/steps/ag_support.py` — `AlphaGenomeWrapper`,
  `_ag_nonlinear_ops`, `ag_seqs_centered`, `load_peaks`, copied verbatim from the
  archived validation tree with the source and date in the file header.
* `src/motifline/steps/attr_ag.py` — the same `deep_lift_shap` call the archive
  made, `--hypothetical` mandatory, chunked checkpoints, shard support.
* `rule attribute` dispatches on `runs.<run>.model`.
* Run `k562_ag_slice200` + its level-1 report.

### Batch 2 — adoption, level-2/3 comparators, adopted downstream
* `src/motifline/adopt.py` — symlink an archived `interpret_*_par_hyp` into a
  stage tree, normalize both archived peak formats, refuse on row/width drift.
* `equivalence.compare_modisco` / `compare_hits` /
  `identity_map_from_modisco_report` — level 3 joins on **CWM identity**, not on
  MoDISco's rank-derived pattern names.
* `rule import_attr`, `equivalence_modisco`, `equivalence_hits`.
* Four adopted runs taken through discover → extract_regions → call_hits:
  `k562_cbp_adopted`, `k562_ag_adopted`, `gm12878_cbp_adopted`,
  `gm12878_ag_adopted`.
* `docs/PARAMS_AUDIT.md` — every parameter against its official source.

### Batch 3 — full-scale canonical production
* `steps/attr_cbp.py` shard support + `src/motifline/merge_shards.py` +
  `rule attribute_sharded` / `merge_shards` (shard order is row order, checked).
* Run `k562_cbp_full`: 33917 peaks re-attributed in 4 shards, end to end.
* `k562_ag_full` launched in the background (later cancelled — §4).

### Batch 4 — version comparison and external lexicons
* `src/motifline/lexicons.py` + `rule call_hits_lexicon` — the archive's
  `lexicon_for`/`width_for` routing as config, gw and mmv.
* `rule call_hits_v041` + `guards.finemo_version_declared` /
  `recorded_finemo_version_is` — the cross-version stage replaces the version
  guard rather than dropping it.
* `gpu.env_sees_cuda` — a runtime that cannot reach a GPU does not lease one.
* Runs: `k562_cbp_full/hits_v041`, `k562_cbp_cl1_adopted` gw + mmv.
* `docs/RUNS.md`.

### Batch 5 — close-out
* `results/k562_ag_full/CANCELLED.md` + `src/motifline/runs.py` (a cancelled run
  leaves `rule all` but stays resumable by explicit target).
* `skill/SKILL.md` (L3), `docs/ARTIFACTS.md`, `PARAMS_AUDIT` §13 (canonical
  values on one page), rewritten `README.md`, this report.
* One defect found and fixed by the audit (§5.1).

## 2. All runs and their verdicts

Level 1 is the only gate. Levels 2 and 3 are consistency reports with soft
bands; out-of-band is a **flag with an attribution**, never an automatic verdict.

### Level 1 — attributions vs the archive (gate: aligned ∧ min row Pearson ≥ 0.999)

| run | n | aligned | min row Pearson | max abs diff | passed |
|---|---|---|---|---|---|
| `k562_cbp_slice500` | 500 | true | 0.9999999999938575 | 3.376e-07 | **yes** |
| `k562_ag_slice200` | 200 | true | 0.9999999994123285 | 2.421e-06 | **yes** |
| `k562_cbp_full` | 33917 | true | 0.9999999999969946 | 1.937e-07 | **yes** |

Files: `results/<run>/equivalence/attr_report.json`.

Both code paths are therefore proven to reproduce the archived math: ChromBPNet
on all 33917 rows, AlphaGenome on a 200-row slice. Attribution is a per-row
computation, so an AG slice exercises every code path a full AG run would.

### Levels 2 and 3 — downstream vs the archive

| run | pos new/old | neg new/old | mean CWM cos | min cos | #<0.85 | hits new/old | ratio | exact J | tol J | L3 flags |
|---|---|---|---|---|---|---|---|---|---|---|
| `k562_cbp_adopted` | 20/22 | 10/11 | 0.9594 | 0.7018 | 4 | 756333/748010 | 1.0111 | 0.7383 | 0.8216 | 14 |
| `k562_cbp_full` | 20/22 | 10/11 | 0.9594 | 0.7018 | 4 | 756278/748010 | 1.0110 | 0.7385 | 0.8219 | 14 |
| `k562_ag_adopted` | 28/28 | 0/0 | 0.9780 | 0.8438 | 1 | 144717/151126 | 0.9576 | 0.2261 | 0.5865 | 13 |
| `gm12878_cbp_adopted` | 19/23 | 0/0 | 0.9595 | 0.5949 | 2 | 230992/236176 | 0.9781 | 0.7982 | 0.9179 | 5 |
| `gm12878_ag_adopted` | 33/34 | 0/0 | 0.9524 | 0.6141 | 5 | 153263/140278 | 1.0926 | 0.6617 | 0.7307 | 18 |

Files: `results/<run>/equivalence/{modisco,hits}_report.json`.

### Cross-checks that isolate one variable

| comparison | result | file |
|---|---|---|
| `k562_cbp_full` vs `k562_cbp_adopted` (**only the attributions differ**) | 30 vs 30 patterns, mean CWM cos 0.99929, min 0.97875, 0 L2 flags; hits 756278 vs 756333 (ratio 0.99993), exact J 0.99926, 1 L3 flag | `k562_cbp_full/equivalence/vs_adopted_*.json` |
| finemo 0.41 vs 0.30, same regions.npz + modisco.h5 | 756232 vs 756278 (ratio 0.99994), exact J 0.99902, 30/30 motifs, **0 flags**, within tolerance | `k562_cbp_full/equivalence/hits_v041_report.json` |
| gw route vs the region_v4 archive (cl1) | 54810 vs 54810, **exact Jaccard 1.000000**, 32/32 motifs, 0 flags | `k562_cbp_cl1_adopted/equivalence/hits_lexicon_gw_report.json` |
| mmv route vs the region_v4 archive (cl1) | 98776 vs 98775 (one extra call), exact J 0.999989, 211/211 motifs, 0 flags | `…/hits_lexicon_mmv_report.json` |

### Stage status

All 34 executed stages across the 8 live runs report `SUCCESS`
(`results/*/*/run_status.json`). The only non-SUCCESS records in the tree are the
five `k562_ag_full` attribution shards, which read `CRASHED / "body exited -15"`
because they were deliberately stopped — see §4.

## 3. Yellow flags, each with its attribution

**None of these is unexplained, and none is a silent difference.**

1. **All four adopted runs breach the level-2/3 soft bands** (5–18 flags each,
   `within_tolerance: false`). Attributed to **`modisco -w 500 → 400`**, the
   single canonical≠archive parameter in these chains (`PARAMS_AUDIT.md` §1, §7,
   §9). The evidence that it is the cause rather than a guess: the attributions
   in these runs are *byte-identical* to the archive's (they are the archive,
   symlinked), so nothing else could have changed; and the affected patterns are
   consistently the highest-numbered, lowest-seqlet ones (`pattern_17..21` in
   K562-CBP, `pattern_26..33` in GM12878-AG) — a narrower discovery window
   admits fewer seqlets per region, and rare motifs lose the support that kept
   them separable.
2. **`gm12878_cbp_adopted` pattern count −4 pos**, outside the ±2 band (the only
   run of the four to breach the *count* tolerance). Same attribution as (1).
3. **`k562_ag_adopted` exact Jaccard 0.2261 against tolerant Jaccard 0.5865.**
   A coordinate-offset effect, not a motif-identity one: this run's lexicon
   matches the archive best of all four (28/28 patterns, mean cosine 0.978), but
   `call-hits` re-derives each hit's start/end by trimming the CWM at `-t 0.3`,
   so a slightly different CWM shifts every hit of that motif by a base or two.
4. **`k562_cbp_full` vs `k562_cbp_adopted`, one flagged motif**
   (`pos_patterns.pattern_18`, 49 vs 222). A **rank swap between two
   near-duplicate low-rank patterns**: old `pattern_18` ↔ new `pattern_19` match
   at cosine 1.0000 and old `pattern_19` ↔ new `pattern_18` at 0.9788; the pair
   carries 869 vs 1038 hits out of 756278 (0.02%). A ~1e-7 attribution
   difference moves seqlets between two patterns that were already nearly the
   same pattern.
5. **finemo 0.41 vs 0.30: 373 hits 0.41-only, 419 0.30-only** out of ~756k
   (0.1%). Not attributable to a code change: the `hitcaller.py` AST diff is
   surface-only, the trim windows in `motif_data.tsv` are identical on both
   sides, and scores on shared hits agree to ~1e-4 (`RUNS.md`). It has the shape
   of floating-point non-determinism in the proximal-gradient convergence test —
   which on this node is *also* a CPU-vs-GPU difference (see 6).
6. **The 0.30-vs-0.41 runtimes (2 min vs 75 min) are NOT a version
   comparison.** `envs/mmv_verify` ships torch 2.13.0+cu130 against driver
   555.42.06, so finemo 0.41 ran on the CPU here. Whether 0.41 is faster or
   slower than 0.30 on equal hardware is **not measured**.
7. **mmv route, one extra hit in 98775** (`chr3:16386807-16386815
   pos_patterns.pattern_57 +`, nothing lost in the other direction). 0.001%,
   inside the 0–0.016% run-to-run jitter the archive's own script measured for
   this hit caller, and the batch size the two runs shared was measured to cost
   4 calls in 54810 on the gw route.

## 4. `k562_ag_full` — cancelled, not failed

Stopped by the user's decision at **63.4%** of the attribution (21500 of 33917
peaks; 215 chunks across five shards; shard 5 never launched), because the
verification value of finishing had gone to zero: the AG code path is proven by
`k562_ag_slice200`'s level-1 report and the AG downstream by `k562_ag_adopted`
running all 33917 AG rows through discover/extract/call_hits.

**The canonical AlphaGenome side of this project is `k562_ag_adopted`.**

Nothing downstream of `k562_ag_full` was ever computed, so nothing is retracted.
The five shard records keep `CRASHED / "body exited -15"` — the SIGTERM — and
were **not** edited to SUCCESS; `merge_shards` refusing over a non-SUCCESS shard
is the behaviour being relied on. Checkpoints (868 MB) are kept and resume by
chunk existence, verified live in batch 3. Reason, per-shard state and the exact
resume commands: `results/k562_ag_full/CANCELLED.md`.

## 5. Close-out audit

| check | result |
|---|---|
| `pytest -q` | **86 passed**, 1 warning, across 14 test files |
| `snakemake -n --rerun-triggers mtime` | **`Nothing to be done (all requested files are present and up to date).`** |
| `snakemake -n` (bare) | 16 jobs proposed — see below |
| `steps/` imports motiflex/motifline | **none** (the names appear only in docstrings stating the prohibition) |
| hardcoded site paths under `src/` | **one**, `steps/ag_support.py:4` — the copied-from provenance comment, explicitly exempt |
| `config/default.yaml` | site paths, as designed |

**The bare `snakemake -n` is not clean, and `--rerun-triggers mtime` is the
documented standard invocation.** The 16 proposed jobs come from snakemake's
*code* trigger (`Rules with provenance triggered jobs: attribute call_hits`):
those rules' `run:` blocks changed in later batches — `attribute` gained shard
arguments, `call_hits` gained the GPU lease — and neither change alters an
artifact. The alternative, `snakemake --cleanup-metadata <outfile>`, was
rejected: it suppresses the trigger by deleting metadata **per output file** and
has to be repeated after every edit, whereas the flag is one decision written
down once. The reproducibility record that matters is each stage's own
`params.json` + `provenance.json` — the exact argv plus the sha256 of every
declared input — not snakemake's bookkeeping. Recorded in `README.md`,
`skill/SKILL.md` and `PARAMS_AUDIT.md` §13.

Two test files carry absolute archive paths (`test_lexicon_context_is_subset.py`,
`test_refusals_live.py`). Both are deliberate — they assert byte-level facts
*about* the archive — and both are `pytest.mark.skipif`-guarded on the path
existing, so the suite still runs where the archive is not mounted.

### 5.1 Defect found and fixed by this audit

`steps/attr_ag.py` wrote AG's native **4-column** `(chr,start,end,summit)`
`regions.bed`, but `finemo extract-regions-modisco-fmt -p` names the first ten
bed columns positionally and reads the summit from **column 10**
(`finemo/data_io.py:36-47`). Handed four columns it raises polars `ShapeError` —
verified against the installed finemo 0.30, not inferred. The break was latent
(no computed AG run had ever reached `extract_regions`: slice200 stops at level
1 and `k562_ag_full` is cancelled) and would have blocked the first downstream
stage of any resumed AG production. Fixed: the step now writes the same
11-column expansion `adopt.normalize_peaks` writes — which is what the adopted
AG runs use and what reproduces the archive's own `peaks_np.bed` — with a test
pinning the two implementations together. Commit `203751e`.

### 5.2 Counts

* **29 commits** (this report included), none with a `Co-Authored-By` trailer
  (`git log --format=%B | grep -ci co-authored-by` = 0).
* **86 tests** in 14 files.
* **9 runs** defined in config; 8 live, 1 cancelled.
* **34 executed stages**, all SUCCESS, plus 5 deliberately stopped AG shards.
* **18 equivalence reports** on disk.

## 6. Known residuals

Open items, none of them a silent failure:

1. **`k562_ag_full` is 63.4% complete and cancelled** (§4). Resumable; the
   commands are in `CANCELLED.md`. A user decision, not a defect.
2. **The computed-AG downstream has never been executed end to end.** Batch
   5's fix (§5.1) makes `regions.bed` parseable and is unit-tested against
   `adopt.normalize_peaks`, but no computed AG attribution has actually been run
   through `extract_regions → discover → call_hits`. The adopted AG runs cover
   those stages on AG data; what is untested is the *junction* between the AG
   step's own output and them.
3. **The `ag2048` half of the lexicon route table is configured but not
   validated.** Its attributions live at width 2048 in
   `region_v4/02_motif_grammar/interpret_ag2048_par_hyp`, a different run from
   the adopted AG archive (width 2114), and this tree cannot produce them. Only
   the CBP side of gw/mmv is reproduced.
4. **`results/k562_cbp_slice500/hits/` ran `-b 1000`**, the vertical slice's
   pre-audit value, not the canonical 2000. Deliberately not recomputed;
   `PARAMS_AUDIT.md` §3 argues why that is harmless, but the slice's hits are
   therefore not at canonical parameters.
5. **`results/k562_ag_slice200/attr/regions.bed` is the pre-fix 4-column form.**
   Its level-1 report reads only columns 1–3, so the report is unaffected; a
   rerun would now write 11 columns.
6. **Bare `snakemake -n` is not clean** (§5). By decision, not by oversight.
7. **GM12878's two archives differ by one region** (CBP 21084 vs AG 21085 peaks).
   An archive property, recorded in `PARAMS_AUDIT.md` §6; it means the two
   GM12878 hit sets are not row-identical by construction.
8. **`modisco -n 1000000` has no official default to adopt** — the CLI marks the
   flag required. Kept at the project's saturating value; if the user wants the
   library's 20000 instead, every downstream number changes. Flagged ⚠ in
   `PARAMS_AUDIT.md` §1 as the user's to overturn.

## 7. What this project does and does not license

**Does:** claims about *pipeline* behaviour — that motifline's attribution
reproduces the archived attributions to r ≥ 0.9999999994; that re-attributing
changes nothing the archive comparison measures (§2, cross-checks); that the
external-lexicon gw route reproduces the archive exactly; that finemo 0.30 and
0.41 agree to 0.1% of hits and ~1e-4 in score on identical inputs.

**Does not:** any claim about which *model* is better, about biology, or about
which finemo version to prefer. The 0.30-vs-0.41 table is decision material for
the user and picks no side, and its runtime row is a CPU-vs-GPU artefact of this
node. Level-2/3 differences are evidence about `modisco -w`, not about
AlphaGenome versus ChromBPNet.
