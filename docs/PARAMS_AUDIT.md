# PARAMS_AUDIT — every motifline parameter against its official source

Written 2026-08-14 (buildout batch 2). Scope: **retroactive** — it covers the
vertical slice and batch 1 as well as the parameters this batch introduces.

**Rule applied** (user instruction, 2026-08-14): where the tool publishes an
official default, canonical takes the official value. A deviation is kept only
for a *technical* reason (OOM, hardware, format, or "the flag has no official
default at all") and the reason is written in the table. Historical value alone
is never a reason to keep a value.

**Judgement calls are marked ⚠ and are the user's to overturn.**

## Sources cited in the tables

| Tag | What it is |
|---|---|
| `help:modisco` | `envs/chrombpnet_local/bin/modisco motifs --help` (modisco-lite 2.4.0), run 2026-08-14 |
| `help:finemo-extract` | `envs/chrombpnet_local/bin/finemo extract-regions-modisco-fmt --help` (finemo 0.30) |
| `help:finemo-hits` | `envs/chrombpnet_local/bin/finemo call-hits --help` (finemo 0.30) |
| `readme:finemo` | `envs/chrombpnet_local/lib/python3.10/site-packages/finemo-0.30.dist-info/METADATA` (the project README; lines cited) |
| `api:modiscolite` | `inspect.signature(modiscolite.tfmodisco.TFMoDISco)` — the *library* defaults behind the CLI |
| `api:tangermeme` | `inspect.signature(tangermeme.deep_lift_shap.deep_lift_shap)` (tangermeme 1.0.4) |
| `api:chrombpnet` | `inspect.signature(chrombpnet.interpret._deep_lift_shap)` (`models/chrombpnet-pytorch/chrombpnet/interpret.py`) |
| `hist:*` | on-disk archaeology of the region_v3/v4 runs, file:line given |

`modisco-lite`'s METADATA carries no long description (30 lines, dependency
metadata only), so `--help` and the library signature are its only official
sources. finemo 0.30's METADATA *is* the README.

---

## 1. MoDISco — `modisco motifs`

| Parameter | Historical value (source) | Official value (source) | Canonical | Reason |
|---|---|---|---|---|
| `-n/--max_seqlets` | `1000000` (`region_v3/k562/merge_modisco.py:21,55`; vertical slice `results/k562_cbp_slice500/modisco/params.json`) | **REQUIRED, no CLI default** (`help:modisco`: "-n MAX_SEQLETS ... (default: None)", `required=True` at `envs/chrombpnet_local/bin/modisco:33`). Library default `max_seqlets_per_metacluster=20000` (`api:modiscolite`) | `1000000` | ⚠ The CLI deliberately publishes **no** default for this flag, so there is no official value to adopt; the library's 20000 belongs to a different interface. Kept at the project's saturating value (= effectively "no cap"), which is also what makes the archive comparison interpretable. **If the user wants strict library-default behaviour, this becomes 20000 and every downstream number changes.** |
| `-w/--window` | `500` (`merge_modisco.py:22,55`; slice `params.json`) | `400` (`help:modisco`) | **`400`** | Official default adopted. **This is the one canonical≠historical parameter in the adopted-run chains** — any level-2/3 tolerance breach is attributed here first. |
| `-l/--n_leiden` | not set → 2 | `2` (`help:modisco`) | `2` (unset) | official |
| `-z/--size` | not set → 20 | `20` (`help:modisco`) | `20` (unset) | official |
| `-t/--trim_size` | not set → 30 | `30` (`help:modisco`) | `30` (unset) | official |
| `-f/--seqlet_flank_size` | not set → 5 | `5` (`help:modisco`) | `5` (unset) | official |
| `-g/--initial_flank_to_add` | not set → 10 | `10` (`help:modisco`) | `10` (unset) | official |
| `-j/--final_flank_to_add` | not set → 0 | `0` (`help:modisco`) | `0` (unset) | official |
| `-s`, `-a` | ohe.npy / attr.npy | no default | stage inputs | data paths |

Note: `api:modiscolite` shows the CLI itself already deviates from the library
in places (`n_leiden_runs=50` in the library vs `-l 2` on the CLI,
`sliding_window_size=21` vs `-z 20`). motifline follows the **CLI**, which is
the interface it actually calls.

## 2. FiNeMo — `finemo extract-regions-modisco-fmt`

| Parameter | Historical value (source) | Official value (source) | Canonical | Reason |
|---|---|---|---|---|
| `-w/--region-width` | `2114` (`region_v3/gm12878/run_interpret_cbp.sh:52-53`, `run_interpret_ag.sh:53-54`; `interpret_ag_par_hyp/fix_ag_finemo_2114.log:3`; archived `finemo_hits/parameters.json:"region_width": 2114`) | `1000` (`help:finemo-extract`; `readme:finemo:134`) | `2114` | **Technical.** The stored attributions are 2114bp wide (CBP input window; AG is cropped to it). `-w 1000` would silently crop 1114bp of every region away. The width is data, not a preference. |
| `-p/--peaks` | CBP: `peaks.bed`; AG: `peaks_np.bed` (`run_interpret_ag.sh:50-54`) | optional, `None` | stage `regions.bed` | Normalized in-stage; `-p` omitted would drop genomic coordinates from all downstream hits. |
| `-C/--chrom-order` | not set | `None` | unset | official |

## 3. FiNeMo — `finemo call-hits`

Archaeology source is stronger than a log here: finemo writes its full resolved
parameter set to `finemo_hits/parameters.json` in every archived run.

| Parameter | Historical value (source) | Official value (source) | Canonical | Reason |
|---|---|---|---|---|
| `-M/--mode` | `pp` (`run_interpret_cbp.sh:54`; all four archived `parameters.json`) | `pp` (`help:finemo-hits`) | `pp` | official = historical |
| `-t/--cwm-trim-threshold` | `0.3` (same) | `0.3` (`help:finemo-hits`) | `0.3` | official = historical |
| `-l/--global-lambda` | `0.7` (same) | `0.7` (`help:finemo-hits`) | `0.7` | official = historical |
| `-b/--batch-size` | region_v3: **`2000`** (`run_interpret_cbp.sh:55`, `run_interpret_ag.sh:56`, all four archived `parameters.json`). region_v4 lexicon runs: `1000` (`region_v4/05_allcluster_lexicon/run_finemo.sh:31`). Vertical slice ran `1000` (`results/k562_cbp_slice500/hits/params.json`) | `2000` (`help:finemo-hits`; `readme:finemo:187`) | **`2000`** | Official default adopted. The `1000` in memory belongs to the 331-motif *union* lexicon of region_v4, where `readme:finemo:251` ("if you encounter GPU OOM, lower this") applied; the per-run modisco lexicons here are 33–68 motifs and do not need it. The vertical slice's already-accepted hits stay at `1000` and were **not** recomputed. |
| `-s/--step-size-max` | `3.0` (archived `parameters.json`) | `3.0` | unset | official |
| `-i/--step-size-min` | `0.08` | `0.08` | unset | official |
| `-j/--step-adjust` | `0.7` | `0.7` | unset | official |
| `-c/--convergence-tol` | `5e-4` | `5e-4` | unset | official |
| `-S/--max-steps` | `10000` | `10000` | unset | official |
| `-f/--no-post-filter` | false | `False` | unset | official |
| `-q/--sqrt-transform` | false | `False` | unset | official |
| `-J/--compile` | false | `False` | unset | official |
| `-d/--device` | `cuda` in region_v3 (`run_interpret_cbp.sh:54`) — **deprecated**, emits a UserWarning (`fix_ag_finemo_2114.log:5-6`) | deprecated; use `CUDA_VISIBLE_DEVICES` (`help:finemo-hits`) | **not passed**; GPU pinned by `CUDA_VISIBLE_DEVICES` UUID | Official guidance adopted; also the project's GPU rule. |

## 4. ChromBPNet attribution (`steps/attr_cbp.py` → `chrombpnet.interpret._deep_lift_shap`)

| Parameter | Historical value (source) | Official value (source) | Canonical | Reason |
|---|---|---|---|---|
| `in_window` | `2114` (`region_v3/k562/shap_shard.py:32`) | no default (model property) | `2114` | **Technical.** ChromBPNet's trained input width. |
| `task` | `counts` (`shap_shard.py:29`) | no default | `counts` | Project canon: counts head = accessibility level; the archive is counts, so anything else is not comparable. |
| `batch_size` | `32` (`shap_shard.py:31`) | `1024` (`api:chrombpnet`); `32` (`api:tangermeme`) | `32` | **Technical (memory).** Each item expands to `n_shuffles=20` references of 4×2114; 1024×20 does not fit. tangermeme's own default is 32. Batching does not change the per-item math. |
| `n_shuffles` | 20 (default, not overridden) | `20` (`api:chrombpnet`, `api:tangermeme`) | `20` | official |
| `references` | `dinucleotide_shuffle` (default) | same | default | official |
| `target` | `0` (default) | `0` | default | official |
| `hypothetical` | `True` (`--hypothetical`) | `False` | **`True`** | **Standing project rule**: DeepLIFT→MoDISco/FiNeMo consume hypothetical scores; `attr_is_hypothetical` refuses anything else. |
| `random_state` | `42` (`shap_shard.py:35`) | `None` (`api:chrombpnet`) | `42` | **Technical (reproducibility).** `None` re-randomizes the 20 reference shuffles, so no run is reproducible and no equivalence report is meaningful. |
| `warning_threshold` | `1e8` (`shap_shard.py:75`) | `0.001` (`api:chrombpnet`) | `1e8` | **Technical.** The convergence-delta check aborts the run on ChromBPNet's counts head at the official value; `1e8` disables an abort, not a computation. Numerically inert. |
| `verbose` | `True` (`shap_shard.py:74`) | `False` | `True` | Logging only. |
| `device` | `cuda` | `cuda` | `cuda` | official |

## 5. AlphaGenome attribution (`steps/attr_ag.py` → `tangermeme.deep_lift_shap`)

| Parameter | Historical value (source) | Official value (source) | Canonical | Reason |
|---|---|---|---|---|
| `in_window` | `16384` (`region_v3/k562/ag_cbp_deeplift_validation/ag_shap_shard.py:39`) | no default | `16384` | **Technical.** AG's validated DeepLIFT window; AG's floor is 2048bp and inputs must be 128bp-binned powers of two. |
| `crop` | `2114` (`ag_shap_shard.py:40`) | no default | `2114` | **Technical.** Central crop so AG rows are the same width as CBP rows. |
| `track` | `60` K562 / `63` GM12878 (`ag_shap_shard.py:41`; `region_v3/gm12878/run_interpret_ag.sh:3`) | no default | per cell line | Site data, not a tunable. |
| `batch_size` | `1` (`ag_shap_shard.py:43`) | `32` (`api:tangermeme`) | `1` | **Technical (memory).** 16384bp × 20 references at batch 32 does not fit a 49GiB card. |
| `references` | `deeplift.local_dinucleotide_shuffle` (`ag_shap_shard.py:109`) | `dinucleotide_shuffle` (`api:tangermeme`) | `local_dinucleotide_shuffle` | **Technical.** Shipped by the AG DeepLIFT support package (`alphagenome-pytorch deeplift-support`) for its long inputs; a global dinucleotide shuffle over 16kb destroys the local composition the 2114bp crop depends on. |
| `n_shuffles` | `20` (`ag_shap_shard.py:42`) | `20` (`api:tangermeme`) | `20` | official |
| `target` | `0` (`ag_shap_shard.py:108`) | `0` | `0` | official |
| `additional_nonlinear_ops` | AG op table, softmax log-ratio on / layernorm log-ratio off (`ag_shap_shard.py:110`) | `None` (`api:tangermeme`) | AG op table | **Technical.** tangermeme has no rule for AG's softmax/layernorm; `None` gives wrong gradients, not a different preference. |
| `hypothetical` | `True` (`ag_shap_shard.py:111`) | `False` | `True` | standing project rule (as §4) |
| `warning_threshold` | `1e8` (`ag_shap_shard.py:110`) | `0.001` | `1e8` | as §4 |
| `random_state` | `42` (`ag_shap_shard.py:45`) | `None` | `42` | as §4 |
| `chunk` | `300` archive / `100` motifline (`ag_shap_shard.py:44`) | n/a | `100` | Checkpoint granularity only; no effect on output. |

## 6. Adoption (`import_attr`, this batch)

| Parameter | Value | Source / reason |
|---|---|---|
| `expected_rows` | 33917 K562 / 21084 GM12878-CBP / **21085** GM12878-AG | Archive row counts; a drift refuses the stage. The GM12878 sides genuinely differ by one region (`interpret_cbp_par_hyp/peaks.bed` 21084 vs `interpret_ag_par_hyp/peaks.bed` 21085), so their hit sets are not row-identical by construction. |
| `expected_width` | 2114 | All four archives. |
| peaks normalization | CBP `peaks.bed` passed through (11 col); AG 4-col `peaks.bed` expanded to `chr,start,end,.,.,.,.,.,.,summit,True` | The expansion reproduces the archive's own `peaks_np.bed` **byte-for-byte** on both AG runs (verified 2026-08-14). |

## 7. What canonical changed relative to the archived runs

Exactly two values differ from the archived region_v3 downstream, and only the
first can move a number:

1. **`modisco -w 500 → 400`** (official default). Expect fewer seqlets per
   region and therefore a different pattern count and different per-motif hit
   counts. **Any level-2/3 tolerance breach in the adopted runs is attributed
   to this parameter unless evidence points elsewhere.**
2. `finemo call-hits -d cuda` dropped (deprecated flag; GPU now pinned by
   `CUDA_VISIBLE_DEVICES`). Numerically inert.

`finemo -b` is *unchanged* versus region_v3 (2000 both) — it differs only from
the vertical slice's 1000, and the vertical slice was not recomputed.

## 8. Soft tolerances used for the consistency reports

Pre-registered before the adopted runs (buildout plan, Global Constraints), and
implemented in `src/motifline/equivalence.py`:

| Check | Band |
|---|---|
| per-motif hit count ratio new/old | 0.8 – 1.25 |
| best-match CWM cosine, per archived pattern | ≥ 0.85 |
| pos / neg pattern count difference | ≤ 2 |
| attr per-row Pearson (level 1 only, hard gate) | ≥ 0.999 |

Out-of-band is a **flag with an attribution**, not a failure.

## 9. Measured effect of the `-w 500 → 400` change (adopted runs, 2026-08-14)

Numbers from `motifline/results/<run>/equivalence/{modisco,hits}_report.json`.
Attributions are byte-identical to the archive in all four runs (they are the
archive, symlinked), so **`modisco -w` is the only thing that changed** between
the archived downstream and these.

| run | pos new/old | neg new/old | mean CWM cos | min cos | #<0.85 | hits new/old | ratio | exact J | tol J | hit flags |
|---|---|---|---|---|---|---|---|---|---|---|
| k562_cbp_adopted | 20/22 | 10/11 | 0.9594 | 0.7018 | 4 | 756333/748010 | 1.011 | 0.738 | 0.822 | 14 |
| k562_ag_adopted | 28/28 | 0/0 | 0.9780 | 0.8438 | 1 | 144717/151126 | 0.958 | 0.226 | 0.586 | 13 |
| gm12878_cbp_adopted | 19/23 | 0/0 | 0.9595 | 0.5949 | 2 | 230992/236176 | 0.978 | 0.798 | 0.918 | 5 |
| gm12878_ag_adopted | 33/34 | 0/0 | 0.9524 | 0.6141 | 5 | 153263/140278 | 1.093 | 0.662 | 0.731 | 18 |

Reading: the **total** hit yield is stable (0.96–1.09× across all four) and the
**high-rank motifs** are recovered with mean CWM cosine 0.95–0.98. What the
narrower discovery window costs is the **tail**: the patterns that fall below
the cosine floor and the ones whose counts move are consistently the
highest-numbered (lowest-seqlet-count) ones — `pattern_17..21` in K562-CBP,
`pattern_26..33` in GM12878-AG. A 400bp window around the peak centre admits
fewer seqlets per region than a 500bp one, so rare motifs lose the support that
kept them separable, and their seqlets are absorbed into neighbouring patterns.

Level-2 pattern counts stay inside the ±2 tolerance for three of the four runs;
`gm12878_cbp_adopted` (−4 pos) is the exception.

The K562-AG `exact_jaccard` of 0.226 against a `tolerant_jaccard` of 0.586 is a
**coordinate-offset**, not a motif-identity, effect: its lexicon matches the
archive best of all four (28/28 patterns, mean cosine 0.978), but `call-hits`
re-derives each motif's start/end by trimming the CWM at `-t 0.3`, so a slightly
different CWM shifts every hit of that motif by a base or two.
