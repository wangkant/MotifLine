# motifline

A Snakemake pipeline from model attributions to MoDISco motifs to FiNeMo hits,
for ChromBPNet and AlphaGenome, on the same `tangermeme.deep_lift_shap` code
path. No bespoke algorithms: attribution is tangermeme 1.0.4, discovery is
modisco-lite 2.4.0, hit calling is finemo 0.30 (pinned), and cross-model motif
merging finishes with motiflex (`mlex`, since 2026-08-17; MotifCompendium 1.0.18
stays as the explicit-target comparison arm). Parameters are the tools' official
defaults; the deviations are `lexicons.batch: 1000` (a measured OOM limit on the
289-motif union lexicon) and `compendium.cluster_within_posneg` (below).

**Attribution geometry.** Each model attributes its own native window and stores
all of it: **ChromBPNet 2114bp, AlphaGenome 2048bp** (AG's input floor, 16 x 128bp
trunk bins). Neither is cropped to meet the other, so MoDISco sees windows 66bp
apart — motif discovery is robust to that, per-position profiles are not. The
canonical pair is `config/default.yaml: geometry`; a run may override it, and the
adopted runs do, because their archived arrays are AG 16384bp cropped to 2114.

## Layout

```
Snakefile            the DAG: attr -> discover -> extract_regions -> call_hits,
                     and, off to the side, discover x2 -> merge_motiflex
                     (-> merge_motiflex_cut -> cross-call -> occupancy
                      when the merge declares cut_height)
config/default.yaml  runs + shared canonical parameters + site paths
                     (not tracked: cp config/example.yaml config/default.yaml
                      and fill in the site paths)
src/motifline/       adopt / merge_shards / lexicons / geometry / compendium / runs
src/motifline/steps/ attr_cbp.py, attr_ag.py (+ ag_support.py), merge_compendium.py
                     -- compute-env scripts
docs/ARTIFACTS.md    what each stage writes
```

## Merging two models' motifs

The merge line finishes with [motiflex](https://github.com/wangkant/motiflex)
(2026-08-17). `rule merge_motiflex` takes the MoDISco h5 of the same regions under
both models and runs `mlex ingest -> align -> [annotate] -> merge`, which builds an
average-linkage tree over `1 - ppm_similarity` (per-pair shuffle null) and merges
**nothing** until the merge declares a `cut_height`; `rule merge_motiflex_cut` then
applies the cut and compiles a lexicon whose group representatives are medoids —
observed patterns, never cross-model averages — read back with the real finemo
loader (`--verify-roundtrip auto`). Merge definitions stay in
`config.compendium.merges`; the mlex-specific fields sit on the merge and the tool
block is `config.motiflex`:

```yaml
compendium:
  merges:
    islandA:
      cut_height: 0.05      # omit -> tree only, nothing merged
      db_yaml: /path/db.yaml # optional precomputed TomTom labels for mlex annotate
      leaves:
        cbp:    {run: <a cbp run>}   # this pipeline builds its modisco.h5 + arrays
        ag2048: {h5: ..., attr: ..., ohe: ..., bed: ...,   # external leaf: declared,
                 meta: {model: alphagenome, readout: atac_track60,   # never derived
                        union_id: AG2048, training_celltype: MULTI_TRACK_SHARED,
                        input_width: 2048, context_window: 2048}}
motiflex:
  mlex: /path/to/envs/mmv_verify/bin/mlex
  celltype: K562
  peak_universe_id: <declared universe id>
```

```bash
envs/motifline/bin/snakemake --rerun-triggers mtime -j 1 -- \
    results/merges/islandA/motiflex/lexicons/core.h5      # or .../motiflex/merge/tree.txt
```

`merge/tree.txt` + `merge_nodes.csv` carry the statistics behind every candidate
merge (the cut is chosen by reading them, not defaulted);
`merge_cut/merge_decisions.json` records each group's members, medoid
representative and evidence. `mlex ingest` refuses semantics it would have to
guess, so an `h5:` leaf must declare its arrays and metadata (`attr`/`ohe`/`bed`/
`meta`); a `run:` leaf gets all of that from the run's own config and outputs.

**The cross-call (2026-08-20).** A compiled merge no longer stops at the lexicon:
every leaf's own regions are scanned with the ONE compiled lexicon
(`rule merge_cross_regions` -> `rule merge_cross_call`, finemo `call-hits` with
`lexicons.batch`), so a difference between the hit tables is a model difference,
not a lexicon difference. `rule merge_occupancy` reduces the hit tables to
`occupancy/occupancy.tsv` (per motif x leaf: n_hits, n_peaks with >=1 hit,
frac_peaks), `agreement.tsv` (per motif x leaf pair: both / only-a / only-b /
neither on the binarised peak x motif matrices, jaccard), and one
`peak_by_motif_<leaf>.tsv.gz` count matrix per leaf. The leaves of a merge are
the same peak rows, so alignment is by row index — the step refuses to run if
the leaves' beds differ anywhere. A motif one model never calls keeps its zero
row: those zeros are the comparison's point.

`rule merge_compendium` (MotifCompendium 1.0.18, the tool that built the ENCODE
CWM motif sets) remains the comparison arm, reachable only by explicit target
`results/merges/<merge>/clusters.tsv`. Its `clusters.tsv` recovers the
`shared`/`<label>_only` bookkeeping MC does not do, and positive/negative patterns
are kept apart (`cluster_within_posneg`) because MC's similarity is sign-blind —
motiflex records metacluster mixing per node instead (`cross_meta` in the tree).

## Running

```bash
envs/motifline/bin/snakemake --rerun-triggers mtime -j <N> --resources gpu=<G> -- <target>
```

`--rerun-triggers mtime` is the standard invocation: the `run:` blocks only
assemble argv, so snakemake's code trigger would otherwise propose recomputing
finished stages after any cosmetic edit.

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
| merge two models' motifs: tree | `merge_motiflex` | `results/merges/<merge>/motiflex/merge/tree.txt` |
| apply the declared cut + compile lexicon | `merge_motiflex_cut` | `results/merges/<merge>/motiflex/lexicons/core.h5` |
| scan a leaf's regions with the compiled lexicon | `merge_cross_call` | `results/merges/<merge>/motiflex/cross_call/<leaf>/hits.tsv` |
| occupancy + cross-leaf agreement tables | `merge_occupancy` | `results/merges/<merge>/motiflex/occupancy/agreement.tsv` |
| MotifCompendium comparison arm (explicit target only) | `merge_compendium` | `results/merges/<merge>/clusters.tsv` |

Which rules a run gets is decided by what its config entry contains: `adopt_from`
→ adopted; `n_shards` → sharded+merge; neither → single-process;
`lexicon_hits`+`lexicon_model` → external-lexicon targets; `cross_version_hits`
→ the 0.41 target; `cancelled: true` → out of `rule all` but still resumable by
explicit target. Attribution geometry comes from `config.geometry` unless the run
overrides it — omit `in_window`/`crop` for new runs.


GPU selection is the caller's: `export CUDA_VISIBLE_DEVICES=<uuid>` (fully-free
cards only, pin by UUID -- indices shift on this node -- and leave one working
card free), and bound concurrency with `--resources gpu=<G>`.

## Environments

| env | role |
|---|---|
| `envs/motifline` (py3.12) | snakemake + this package |
| `envs/chrombpnet_local` (py3.10) | all compute: torch, chrombpnet, tangermeme, modisco-lite, finemo 0.30 |
| `envs/mmv_verify` (py3.12) | `mlex` (motiflex merge finish) + finemo 0.41 for the `call_hits_v041` comparison (its torch runs CPU on this node) |
| `envs/motifcompendium` (py3.11) | MotifCompendium 1.0.18, `merge_compendium` comparison arm only (editable install of `models/MotifCompendium`; no cupy on this node, so similarity runs on CPU) |

Step scripts never import this package; every site path arrives as an argument
(AlphaGenome in particular: pass `--ag_src`, the package is not importable from
the compute env).

## House rules

- Attributions are HYPOTHETICAL (`--hypothetical`), always.
- hits are produced by finemo 0.30; note the version next to any hits you keep.
- `new_experiment/region_v3|v4` archives are read-only.
- peaks fed to finemo must be 10/11-column narrowPeak (summit in column 10);
  the attribution steps normalize this themselves.
