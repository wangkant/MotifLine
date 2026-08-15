---
name: motifline
description: How to drive the motifline pipeline (attribution -> MoDISco -> FiNeMo) — which rule produces what, how to add a run to config/default.yaml, and the operating rules (hypothetical-only, finemo 0.30, GPU etiquette).
---

# motifline

Repo: `<KANT_ROOT>/motifline`. A stage directory holds only its
result files; parameters live in `config/default.yaml`.

## Standard invocation

```bash
cd <KANT_ROOT>/motifline
envs/motifline/bin/snakemake --rerun-triggers mtime -j <N> --resources gpu=<G> -- <target>
```

`--rerun-triggers mtime` is not optional: the `run:` blocks only assemble argv,
so snakemake's code trigger would otherwise propose recomputing finished stages
after any cosmetic Snakefile edit.

## Environments

| layer | interpreter | holds |
|---|---|---|
| orchestration | `envs/motifline` (py3.12) | snakemake, `motifline.*` |
| compute | `envs/chrombpnet_local` (py3.10) | torch, chrombpnet, tangermeme, modisco-lite, **finemo 0.30** |
| comparison only | `envs/mmv_verify` (py3.12) | finemo 0.41, used by `call_hits_v041` and nothing else (its torch runs CPU on this node — slower, same hits) |

Step scripts under `src/motifline/steps/` run in the compute env and take every
site path as an argument. AlphaGenome is not importable from the compute env —
`--ag_src` must point at `models/alphagenome-pytorch-deeplift/src`.

## Which rule produces what

| you want | rule | target |
|---|---|---|
| attribute a slice/small run in one process | `attribute` | `results/<run>/attr/attr.npy` |
| attribute full-scale, one shard per GPU | `attribute_sharded` | `results/<run>/attr_shards/shard_<i>/attr_<i>.npy` |
| glue shards into one matrix | `merge_shards` | `results/<run>/attr/attr.npy` |
| use an archived `interpret_*_par_hyp` attribution | `import_attr` | `results/<run>/attr/attr.npy` |
| discover motifs (MoDISco) | `discover` | `results/<run>/modisco/modisco.h5` |
| prepare regions for FiNeMo | `extract_regions` | `results/<run>/finemo_regions/regions.npz` |
| call hits against the run's own lexicon | `call_hits` | `results/<run>/hits/hits.tsv` |
| call hits against an external lexicon (gw/mmv) | `call_hits_lexicon` | `results/<run>/hits_lexicon_<route>/hits.tsv` |
| same regions+lexicon with finemo 0.41 (comparison) | `call_hits_v041` | `results/<run>/hits_v041/hits.tsv` |

Rules are selected by what the run's config entry contains:
`adopt_from` → adopted; `n_shards` → sharded+merge; neither → single-process;
`lexicon_hits`+`lexicon_model` → external-lexicon targets;
`cross_version_hits: true` → the 0.41 target; `cancelled: true` → out of
`rule all` but resumable by explicit target.

## Adding a run

```yaml
runs:
  <cell>_<model>_<scope>:
    model: cbp                 # cbp | ag
    # computed attribution (omit for an adopted run):
    wo_bias:  /abs/chrombpnet_wo_bias.pt      # cbp
    weights:  /abs/model_fold_0.safetensors   # ag
    ag_src:   /abs/alphagenome-pytorch-deeplift/src   # ag
    peaks:    /abs/peaks.bed
    in_window: 2114            # cbp; ag: 16384 plus track/crop/n_shuffles/chunk
    head: null                 # int for a slice run
    n_shards: 4                # present => sharded
    rows: 33917                # expected row count (merge/adopt check)
    # adopted attribution:
    adopt_from:  /abs/interpret_cbp_par_hyp
    adopt_peaks: peaks.bed
    # extras:
    cross_version_hits: true
    lexicon_model: cbp         # key into lexicons.routes
```

The shared parameter blocks (`modisco:`, `finemo:`, `lexicons:`) hold each
tool's official default; the one deviation is `lexicons.batch: 1000` (289-motif
union lexicon OOMs at 2000 on a 49GiB card). Don't add per-run overrides.

## Operating rules

- **Hypothetical only**: attribution always runs `--hypothetical`; MoDISco and
  FiNeMo consume hypothetical scores. Actual-score attributions are aux and
  must not be mixed in.
- **finemo 0.30 pinned** for production hits; 0.41 exists only for the
  comparison stage. Note the version next to any hits you keep.
- **GPU etiquette is the caller's** (nothing enforces it): `nvidia-smi` first,
  fully-free cards only (no compute apps, <100 MiB, 0% util), pin by UUID via
  `CUDA_VISIBLE_DEVICES` (indices shift on this node), always leave one working
  card free, and bound concurrency with `--resources gpu=<G>`. Two jobs started
  in the same second will pick the same idle card — stagger them or export
  distinct UUIDs yourself.
- **Archives read-only**: `new_experiment/region_v3|v4` is never written.
- peaks fed to finemo must be 10/11-column narrowPeak (summit = column 10);
  the attribution steps and `adopt.py` normalize this — don't hand-roll beds.

## Installing this skill in another repo

```bash
mkdir -p <repo>/.claude/skills
ln -s <KANT_ROOT>/motifline/skill <repo>/.claude/skills/motifline
```
