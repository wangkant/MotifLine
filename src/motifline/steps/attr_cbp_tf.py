#!/usr/bin/env python
"""ChromBPNet DeepSHAP hypothetical attributions (counts head), official TensorFlow code.

Runs kundajelab/chrombpnet's own interpret entry point
(`chrombpnet.evaluation.interpret.interpret.main`, what `chrombpnet contribs_bw` calls) on
the run's peaks with the counts head only, then reads the `.counts_scores.h5` it writes
and stores attr.npy / ohe.npy / regions.bed after import_chrombpnet.py's checks
(hypothetical, projected = raw * shap, every row equal to the genome).

One change to the official code: it draws each of its dinucleotide-shuffled references
from a fresh, unseeded RandomState. Here shuffle k of every sequence draws from
RandomState(random_state + k), the seeding tangermeme uses on the other attribution
paths, so a run is reproducible and does not depend on sharding. The number of shuffles
is n_shuffles (the official code uses 20).

As in chrombpnet, a peak whose window runs off the chromosome is dropped and one with
non-ACGT bases is kept (all-zero columns). Scores are float16 in chrombpnet's file;
attr.npy is float32. chrombpnet's own files stay next to the arrays. Sharded runs cut the
peak list into contiguous slices first and write attr_<i>.npy / ohe_<i>.npy / regions_<i>.bed.

Runs in the chrombpnet env and must not import motifline.
"""
import argparse
import json
import os
from argparse import Namespace

import numpy as np

from import_chrombpnet import check, mismatches, read_regions

PREFIX = "chrombpnet"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_h5", required=True, help="chrombpnet_nobias.h5 (Keras)")
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hypothetical", action="store_true", help="required")
    ap.add_argument("--n_shuffles", type=int, default=20)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--in_window", type=int, default=None,
                    help="must equal the model's input length")
    ap.add_argument("--head", type=int, default=None, help="first N peaks only")
    ap.add_argument("--n_shards", type=int, default=None)
    ap.add_argument("--shard_i", type=int, default=None)
    a = ap.parse_args()
    if not a.hypothetical:
        ap.error("--hypothetical is required, as on the other attribution steps")
    if (a.shard_i is None) != (a.n_shards is None):
        ap.error("--shard_i and --n_shards must be given together")
    if a.n_shards is not None:
        if a.head is not None:
            ap.error("--head and --shard_i/--n_shards are mutually exclusive")
        if not 0 <= a.shard_i < a.n_shards:
            ap.error(f"--shard_i must be in [0, {a.n_shards})")
    return a


def peak_lines(path):
    """The peaks as 10-column lines, chrombpnet's NARROWPEAK_SCHEMA. An 11-column bed
    whose last column is all True is cut back to 10; anything else is refused."""
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    widths = {ln.count("\t") + 1 for ln in lines}
    if widths == {11}:
        bad = {ln.rsplit("\t", 1)[1] for ln in lines} - {"True"}
        if bad:
            raise ValueError(f"{path}: 11th column must be is_peak=True, saw {sorted(bad)}")
        return [ln.rsplit("\t", 1)[0] for ln in lines]
    if widths != {10}:
        raise ValueError(f"{path}: expected 10 or 11 columns, saw {sorted(widths)}")
    return lines


def repeated_summits(lines):
    centers = [(c[0], int(c[1]) + int(c[9])) for c in (ln.split("\t") for ln in lines)]
    return len(centers) - len(set(centers))


def seeded_shuffles(n_shuffles, random_state, shuffle=None):
    """chrombpnet's shap_utils.shuffle_several_times, with shuffle k of every sequence
    drawn from RandomState(random_state + k). `shuffle` is deeplift's dinuc_shuffle, the
    function the official code calls. `.calls` counts the sequences it was asked for."""
    if shuffle is None:
        from deeplift.dinuc_shuffle import dinuc_shuffle as shuffle

    def shuffle_several_times(s):
        shuffle_several_times.calls += 1
        refs = [np.array([shuffle(s[0], rng=np.random.RandomState(random_state + k))
                          for k in range(n_shuffles)])]
        if len(s) == 2:
            refs.append(np.array([s[1] for _ in range(n_shuffles)]))
        return refs
    shuffle_several_times.calls = 0
    return shuffle_several_times


def model_input_length(model_h5):
    """The input length in the Keras model's saved config, or None if it is not there."""
    import h5py
    with h5py.File(model_h5, "r") as f:
        cfg = f.attrs.get("model_config")
    if cfg is None:
        return None
    cfg = json.loads(cfg if isinstance(cfg, str) else cfg.decode())
    for layer in cfg.get("config", {}).get("layers", []):
        if layer.get("class_name") == "InputLayer":
            shape = layer.get("config", {}).get("batch_input_shape") or [None, None]
            return int(shape[1]) if shape[1] else None
    return None


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    width = model_input_length(a.model_h5)
    if a.in_window is not None and width is not None and width != a.in_window:
        raise SystemExit(f"in_window {a.in_window}, but {a.model_h5} takes {width} bp")

    lines = peak_lines(a.peaks)
    dups = repeated_summits(lines)
    if dups:
        raise SystemExit(f"{a.peaks}: {dups} peaks share a summit with an earlier peak")
    suffix = ""
    if a.n_shards is not None:
        bounds = np.linspace(0, len(lines), a.n_shards + 1).astype(int)
        lo, hi = int(bounds[a.shard_i]), int(bounds[a.shard_i + 1])
        lines = lines[lo:hi]
        suffix = f"_{a.shard_i}"
        print(f"[shard {a.shard_i}/{a.n_shards}] peaks [{lo}:{hi}]", flush=True)
    if a.head:
        lines = lines[: a.head]
    bed = os.path.join(a.out_dir, f"input_peaks{suffix}.bed")
    with open(bed, "w") as fh:
        fh.writelines(ln + "\n" for ln in lines)
    prefix = os.path.join(a.out_dir, PREFIX + suffix)

    # importing interpret turns off eager execution, which shap's DeepExplainer needs
    import chrombpnet.evaluation.interpret.interpret as interpret
    import chrombpnet.evaluation.interpret.shap_utils as shap_utils
    import tensorflow as tf
    print(f"[attr_cbp_tf] GPUs: {[d.name for d in tf.config.list_physical_devices('GPU')]}",
          flush=True)
    print(f"[attr_cbp_tf] references: {a.n_shuffles} dinucleotide shuffles, shuffle k "
          f"seeded with RandomState({a.random_state} + k)", flush=True)
    shuffles = seeded_shuffles(a.n_shuffles, a.random_state)
    shap_utils.shuffle_several_times = shuffles
    interpret.main(Namespace(genome=a.fasta, regions=bed, model_h5=a.model_h5,
                             output_prefix=prefix, debug_chr=None,
                             profile_or_counts=["counts"]))

    import pyfaidx
    import tables
    with tables.open_file(f"{prefix}.counts_scores.h5") as f:
        raw = f.get_node("/raw/seq").read()
        shap = f.get_node("/shap/seq").read()
        projected = f.get_node("/projected_shap/seq").read()
    rows = read_regions(f"{prefix}.interpreted_regions.bed")
    if shuffles.calls != raw.shape[0]:
        raise SystemExit(f"the seeded references were drawn for {shuffles.calls} of "
                         f"{raw.shape[0]} sequences: chrombpnet did not call "
                         f"shap_utils.shuffle_several_times once per sequence")
    check(raw, shap, projected, rows, len(rows), a.in_window)
    fasta = pyfaidx.Fasta(a.fasta)
    bad = mismatches(raw, rows, lambda c, s, e: fasta[c][s:e].seq, raw.shape[-1])
    if bad:
        raise SystemExit(f"{len(bad)} rows do not match {a.fasta}; first is row {bad[0]}: "
                         f"{rows[bad[0]][3]}")

    np.save(os.path.join(a.out_dir, f"attr{suffix}.npy"), shap.astype(np.float32))
    np.save(os.path.join(a.out_dir, f"ohe{suffix}.npy"), raw.astype(np.int8))
    with open(os.path.join(a.out_dir, f"regions{suffix}.bed"), "w") as fh:
        fh.writelines(r[3] + "\n" for r in rows)
    print(json.dumps({"peaks": len(lines), "rows": int(raw.shape[0]),
                      "dropped_off_chromosome": len(lines) - int(raw.shape[0]),
                      "rows_with_non_acgt": int((raw.sum(1) == 0).any(axis=1).sum()),
                      "width": int(raw.shape[-1]), "checked_against_genome": True}),
          flush=True)
    print(f"DONE attr={shap.shape}", flush=True)


if __name__ == "__main__":
    main()
