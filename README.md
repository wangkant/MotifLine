# motifline

Snakemake pipeline from attribution to MoDISco to FiNeMo. Each stage directory
holds only its result files; the orchestration layer (Python 3.12) stays
separate from the compute layer (`chrombpnet_local`, Python 3.10), which is
where every stage body actually runs.

## Architecture

```
config/default.yaml ──────── runs, canonical parameters, lexicon routes
        │
        ▼
   Snakefile (envs/motifline, py3.12)          orchestration layer
        │   each rule assembles an argv and runs it as a subprocess
        │   gpu.gpu_lease: one fully-free card by UUID, one always left free
        │
        ├── attribute            ─┐
        ├── attribute_sharded ──► merge_shards ─┐   src/motifline/steps/*.py
        └── import_attr  (adopt archived attr)  │   (envs/chrombpnet_local, py3.10;
                                                │    never import motifline)
                                    attr/  ─────┤
                                                │
                    ┌───────────────────────────┴───────────────┐
                    ▼                                           ▼
              discover                                  extract_regions
            modisco/modisco.h5                       finemo_regions/regions.npz
                    │                                           │
                    └──────────────┬────────────────────────────┘
                                   ▼
                        call_hits ──► hits/hits.tsv
                        call_hits_lexicon ──► hits_lexicon_<route>/   (external lexicon)
                        call_hits_v041    ──► hits_v041/              (comparison only)
                                   │
                                   ▼
                     equivalence_attr    (level 1 — HARD GATE, r >= 0.999)
                     equivalence_modisco (level 2 — report, soft bands)
                     equivalence_hits    (level 3 — report, soft bands)
```

## Quick start

```bash
cd <KANT_ROOT>/motifline

# what would run (should say "Nothing to be done" on a finished tree)
<KANT_ROOT>/envs/motifline/bin/snakemake -n --rerun-triggers mtime

# a specific target, two GPU stages at a time
<KANT_ROOT>/envs/motifline/bin/snakemake --rerun-triggers mtime -j 2 --resources gpu=2 -- \
  results/<run>/hits/hits.tsv

# tests
<KANT_ROOT>/envs/motifline/bin/python -m pytest -q
```

**Always pass `--rerun-triggers mtime`.** A rule's `run:` block here only
assembles an argv, so editing it changes snakemake's *code* hash without
changing anything about the artifact; without the flag a bare `snakemake`
proposes to recompute finished attributions and hit calls. `snakemake
--cleanup-metadata` would suppress the same trigger by deleting information per
output file and would have to be repeated after every edit; the flag is one
documented decision instead.

## Where things are written down

| file | what it answers |
|---|---|
| `skill/SKILL.md` | which rule to use, how to add a run, GPU discipline |
| `docs/ARTIFACTS.md` | what each stage writes: files, shapes, dtypes |
| `docs/PARAMS_AUDIT.md` | every parameter vs its official source, and why each deviation exists (§13 = the canonical values on one page) |
| `docs/RUNS.md` | what has actually been executed and what came out, every number named with its file |
| `docs/FINAL_REPORT.md` | the project's end state: deliverables, all runs and their verdicts, known residuals |
| `results/<run>/CANCELLED.md` | for a stopped run: why, its checkpoint state, how to resume |

Design spec: `docs/superpowers/specs/2026-08-14-motifline-design.md`.
