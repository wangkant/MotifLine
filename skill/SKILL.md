---
name: motifline
description: How to drive the motifline pipeline (attribution -> MoDISco -> FiNeMo). Use when running or resuming a motifline rule, adding a new run to config/default.yaml, adopting an archived interpret_*_par_hyp directory, or reading a result out of results/.
---

# motifline

A Snakemake pipeline from model attributions to MoDISco motifs to FiNeMo hits.
Each stage directory holds only the files that stage produces.

Repo: `<KANT_ROOT>/motifline`.
This file explains **when to use which rule** and **how to read what came out**.
It contains no algorithm and no parameter values — parameters live in
`config/default.yaml`, and why each one has the value it has is
`docs/PARAMS_AUDIT.md`. What each stage writes is `docs/ARTIFACTS.md`. What has
actually been run and what came out is `docs/RUNS.md`.

## The two environments, and why you cannot merge them

| layer | interpreter | holds |
|---|---|---|
| orchestration | `envs/motifline` (py3.12) | snakemake, `motifline.*` |
| compute | `envs/chrombpnet_local` (py3.10) | torch, chrombpnet, tangermeme, modisco-lite, **finemo 0.30** |
| comparison only | `envs/mmv_verify` (py3.12) | **finemo 0.41**, used by `call_hits_v041` and nothing else |

Stage bodies under `src/motifline/steps/` run in the compute env and must never
`import motifline`; they take every site path as an argument. The Snakefile and
the equivalence reports run in the orchestration env.

## Standard invocation

```bash
cd <KANT_ROOT>/motifline
envs/motifline/bin/snakemake --rerun-triggers mtime -j <N> --resources gpu=<G> -- <target>
```

**`--rerun-triggers mtime` is not optional cosmetics.** Without it snakemake
also reruns on its *code* trigger: a rule's `run:` block here only assembles an
argv, so editing it (adding a shard argument, changing a flag's spelling)
changes snakemake's code hash while changing nothing about the artifact. On this
tree a bare `snakemake -n` therefore proposes to recompute finished attributions
and hit calls; with `--rerun-triggers mtime` it says `Nothing to be done`. Do
**not** reach for `snakemake --cleanup-metadata` instead: it deletes the trigger
information per output file and has to be repeated after every edit, whereas the
flag is one documented decision.

## Which rule to use

| you want | rule | target |
|---|---|---|
| attribute a slice or a small run in one process | `attribute` | `results/<run>/attr/attr.npy` |
| attribute a full-scale run, one shard per GPU | `attribute_sharded` | `results/<run>/attr_shards/shard_<i>/attr_<i>.npy` |
| glue the shards back into one matrix | `merge_shards` | `results/<run>/attr/attr.npy` |
| use an archived `interpret_*_par_hyp` attribution instead of computing one | `import_attr` | `results/<run>/attr/attr.npy` |
| discover motifs (MoDISco) | `discover` | `results/<run>/modisco/modisco.h5` |
| prepare regions for FiNeMo | `extract_regions` | `results/<run>/finemo_regions/regions.npz` |
| call hits against **the run's own** lexicon | `call_hits` | `results/<run>/hits/hits.tsv` |
| call hits against an **external** lexicon (gw / mmv routes) | `call_hits_lexicon` | `results/<run>/hits_lexicon_<route>/hits.tsv` |
| call the same regions+lexicon with finemo 0.41 (comparison only) | `call_hits_v041` | `results/<run>/hits_v041/hits.tsv` |
| new attributions vs archived attributions (level 1, **hard gate**) | `equivalence_attr` | `results/<run>/equivalence/attr_report.json` |
| new lexicon vs archived lexicon (level 2, report only) | `equivalence_modisco` | `results/<run>/equivalence/modisco_report.json` |
| new hits vs archived hits (level 3, report only) | `equivalence_hits` | `results/<run>/equivalence/hits_report.json` |
| 0.41's hits vs 0.30's, same inputs | `equivalence_hits_v041` | `results/<run>/equivalence/hits_v041_report.json` |
| external-lexicon hits vs the archive's, same route | `equivalence_hits_lexicon` | `results/<run>/equivalence/hits_lexicon_<route>_report.json` |

Rules are selected by what the run's config entry contains, not by a flag you
pass:

* `adopt_from` present → `import_attr` owns `attr/` (the run is *adopted*).
* `n_shards` present → `attribute_sharded` + `merge_shards` own `attr/`.
* neither → `attribute` owns `attr/`.
* `old_attr`/`old_regions` → an `equivalence_attr` target exists.
* `old_modisco`/`old_hits` → level-2/3 targets exist.
* `lexicon_hits` + `lexicon_model` → the external-lexicon route targets exist.
* `cross_version_hits: true` → the finemo-0.41 target exists.
* `cancelled: true` → the run drops out of `rule all`; every explicit target
  still resolves, so it stays resumable. Say where the decision is written with
  `cancelled_note:`.

`attribute`, `attribute_sharded`, `call_hits`, `call_hits_lexicon` and
`call_hits_v041` declare `resources: gpu=1`; give snakemake `--resources gpu=<G>`
or they will all try to start at once.

**Level 1 is the only gate.** Levels 2 and 3 are consistency reports with soft
tolerance bands: out-of-band is a `flags` entry to be attributed to a parameter
change, never an automatic verdict, and never a licence to suppress the number.

## Adding a run

Copy the shape, keep every path absolute, and delete the keys whose stages you
do not want to exist.

```yaml
runs:
  <cell>_<model>_<scope>:
    model: cbp                 # cbp | ag  -- dispatches the attribution step
    # --- computed attribution (omit the whole block for an adopted run) ---
    wo_bias:  /abs/path/chrombpnet_wo_bias.pt      # cbp only
    weights:  /abs/path/model_fold_0.safetensors   # ag only
    ag_src:   /abs/path/alphagenome-pytorch-deeplift/src   # ag only
    peaks:    /abs/path/peaks.bed
    in_window: 2114            # cbp: model input width == stored width
    # ag also needs: track: <int>, crop: 2114, n_shuffles: 20, chunk: 100,
    #                in_window: 16384
    head: null                 # int for a slice run, null for everything
    n_shards: 4                # present => sharded attribution, one GPU each
    rows: 33917                # expected row count; a drift fails the stage
    # --- adopted attribution (mutually exclusive with the block above) ---
    adopt_from:  /abs/path/interpret_cbp_par_hyp
    adopt_peaks: peaks.bed     # file inside adopt_from; 4-col AG beds are expanded
    # --- optional comparison targets ---
    old_attr:    /abs/path/attr.npy        # level 1
    old_regions: /abs/path/peaks.bed       # level 1 (alignment key)
    old_modisco: /abs/path/modisco.h5      # level 2
    old_hits:    /abs/path/finemo_hits/hits.tsv   # level 3
    # --- optional extra hit-calling stages ---
    cross_version_hits: true   # also call hits with finemo 0.41
    lexicon_model: cbp         # key into lexicons.routes (cbp | ag2048)
    lexicon_hits:              # archived per-route hits to report against
      gw:  /abs/path/hits_gw/.../hits.tsv
      mmv: /abs/path/hits_mmv/.../hits.tsv
    # --- if the run is stopped for good ---
    cancelled: true
    cancelled_note: results/<run>/CANCELLED.md
```

The canonical parameter block (`modisco:`, `finemo:`, `lexicons:`) is
**shared by every run** and holds each tool's official
default unless `docs/PARAMS_AUDIT.md` records a technical reason. Do not add a
per-run override to make a comparison come out nicer; that turns a report into a
result. If a run genuinely needs a different value, change it in the shared
block and add the audit row.

## Reading a stage

A stage directory holds only the files that stage produces — see
`docs/ARTIFACTS.md` for the file list, shapes and dtypes of each one. There is
no record file to consult: a rule either finishes and leaves its outputs, or it
exits non-zero and snakemake stops. If a stage crashed part way through, its
directory can still hold partial output (AG's `_chunks_<i>/` checkpoints in
particular), so a rerun of the target, not the presence of files, is what tells
you the stage is finished.

When quoting a number out of `results/`, name the file it came from.

## GPU discipline

* Every GPU stage takes a `gpu_lease` from `.gpu_locks/`: `nvidia-smi` alone is
  not enough, because two stages starting together both see the same idle card
  before either allocates. The lease is claimed atomically by UUID and released
  in a `finally`, and a lock whose holder PID is dead is reclaimed.
* Cards must be **fully free** (no compute apps, <100 MiB used, 0% util) and
  **one working card is always left free** (`leave_free=1`). If no card
  qualifies the stage waits (`gpu_wait_s`, default 3600 s) rather than sharing.
* The card is pinned with `CUDA_VISIBLE_DEVICES=<uuid>`, never an index —
  indices shift.
* `env_sees_cuda(python)` asks *that interpreter* whether its torch can reach a
  GPU, and a stage that cannot use a card does not lease one. A torch built
  against a newer CUDA than the node's driver imports fine, warns, and runs on
  the CPU; leasing anyway squats a card for the whole run without issuing a
  kernel. This is not hypothetical here — it is finemo 0.41 in `envs/mmv_verify`.
* `--resources gpu=<G>` bounds how many GPU stages snakemake starts; the lease
  is what makes the choice safe. Use both.

## Installing this skill in another repo

```bash
mkdir -p <repo>/.claude/skills
ln -s <KANT_ROOT>/motifline/skill <repo>/.claude/skills/motifline
```

The symlink points at the directory holding this `SKILL.md`, so the skill
tracks the pipeline it documents instead of drifting from a copy.
