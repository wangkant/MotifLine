#!/usr/bin/env python
"""ChromBPNet DeepLIFT/SHAP hypothetical attributions (counts head).

Uses chrombpnet-pytorch's own _deep_lift_shap. A peak whose window contains a non-ACGT
base is dropped. Sharded runs split the peak list into contiguous slices before that
filter and write attr_<i>.npy / ohe_<i>.npy / regions_<i>.bed.

tangermeme's convergence check runs at its default threshold and is recorded in
convergence.json rather than switched off.

Runs in the compute env and must not import motifline.
"""
import argparse
import json
import os
import sys

from convergence import THRESHOLD, Deltas, summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrombpnet_src", required=True, help="chrombpnet-pytorch checkout")
    ap.add_argument("--wo_bias", required=True)
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--chrom_sizes", required=True)
    ap.add_argument("--task", default="counts", choices=["counts", "profile"])
    ap.add_argument("--hypothetical", action="store_true", help="required")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--in_window", type=int, default=2114)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_shuffles", type=int, default=20)
    ap.add_argument("--head", type=int, default=None, help="first N peaks only")
    ap.add_argument("--n_shards", type=int, default=None)
    ap.add_argument("--shard_i", type=int, default=None)
    a = ap.parse_args()
    if not a.hypothetical:
        ap.error("--hypothetical is required; without it the step would write projected "
                 "scores under the same file names")
    if (a.shard_i is None) != (a.n_shards is None):
        ap.error("--shard_i and --n_shards must be given together")
    if a.n_shards is not None:
        if a.head is not None:
            ap.error("--head and --shard_i/--n_shards are mutually exclusive")
        if not 0 <= a.shard_i < a.n_shards:
            ap.error(f"--shard_i must be in [0, {a.n_shards})")
    return a


def shard_bounds(n_rows, n_shards):
    import numpy as np
    return np.linspace(0, n_rows, n_shards + 1).astype(int)


def normalize_peaks_bed(peaks_path, out_dir):
    """load_region_df wants 10 columns and appends is_peak itself. An 11-column bed
    whose last column is all True is cut back to 10; anything else is refused."""
    with open(peaks_path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    widths = {ln.count("\t") + 1 for ln in lines}
    if widths == {10}:
        return peaks_path
    if widths != {11}:
        raise ValueError(f"{peaks_path}: expected 10 or 11 columns, saw {sorted(widths)}")
    bad = {ln.rsplit("\t", 1)[1] for ln in lines} - {"True"}
    if bad:
        raise ValueError(f"{peaks_path}: 11th column must be is_peak=True, saw {sorted(bad)}")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "input_peaks_10col.bed")
    with open(dst, "w") as fh:
        fh.write("".join(ln.rsplit("\t", 1)[0] + "\n" for ln in lines))
    return dst


def main():
    a = parse_args()
    sys.path.insert(0, a.chrombpnet_src)
    import numpy as np
    import pyfaidx
    import torch
    from chrombpnet.bpnet import BPNet
    from chrombpnet.data_utils import get_seq, load_region_df
    from chrombpnet.interpret import _deep_lift_shap
    from chrombpnet.model_wrappers import CountWrapper, ProfileWrapper

    os.makedirs(a.out_dir, exist_ok=True)
    model = BPNet(n_filters=512, n_layers=8, n_outputs=1, n_control_tracks=0)
    sd = torch.load(a.wo_bias, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = CountWrapper(model) if a.task == "counts" else ProfileWrapper(model)

    # in_window is passed so the edge filter matches the window get_seq fetches.
    regions_df = load_region_df(normalize_peaks_bed(a.peaks, a.out_dir),
                                chrom_sizes=a.chrom_sizes, in_window=a.in_window)
    center = (regions_df["chr"].astype(str) + ":"
              + (regions_df["start"] + regions_df["summit"]).astype(str))
    if center.duplicated().any():
        raise SystemExit(f"{a.peaks}: {int(center.duplicated().sum())} peaks share a summit "
                         f"with an earlier peak")
    n_in = len(regions_df)
    suffix = ""
    if a.n_shards is not None:
        bounds = shard_bounds(len(regions_df), a.n_shards)
        lo, hi = int(bounds[a.shard_i]), int(bounds[a.shard_i + 1])
        regions_df = regions_df.iloc[lo:hi].reset_index(drop=True)
        suffix = f"_{a.shard_i}"
        print(f"[shard {a.shard_i}/{a.n_shards}] rows [{lo}:{hi}]", flush=True)
    if a.head:
        regions_df = regions_df.iloc[: a.head].reset_index(drop=True)
    seq = get_seq(regions_df, pyfaidx.Fasta(a.fasta), a.in_window)
    if isinstance(seq, np.ndarray):
        seq = torch.tensor(seq.astype(np.float32))
    if seq.shape[-1] == 4:
        seq = seq.permute(0, 2, 1)
    # get_seq encodes non-ACGT bases as all-zero columns
    keep = ~((seq == 0).all(dim=1).any(dim=1))
    seq = seq[keep]
    regions_df = regions_df.iloc[keep.numpy()].reset_index(drop=True)
    print(f"[attr_cbp] {len(regions_df)} peaks ({n_in} in bounds); "
          f"{int((~keep).sum())} dropped for non-ACGT bases", flush=True)

    with Deltas() as d:
        attr = _deep_lift_shap(model.to("cuda"), seq, batch_size=a.batch_size, verbose=True,
                               args=None, warning_threshold=THRESHOLD, device="cuda",
                               n_shuffles=a.n_shuffles,
                               hypothetical=a.hypothetical, random_state=a.random_state)
    conv = summary(len(seq) * a.n_shuffles, d.over, d.max)
    with open(os.path.join(a.out_dir, f"convergence{suffix}.json"), "w") as fh:
        json.dump(conv, fh, indent=2)
    print(f"[attr_cbp] DeepLIFT convergence: {conv['pairs_over']} of {conv['pairs']} "
          f"(sequence, reference) pairs over {THRESHOLD}; max delta {conv['max_delta']:.3g}",
          flush=True)
    attr = attr.cpu().numpy() if isinstance(attr, torch.Tensor) else np.asarray(attr)
    np.save(os.path.join(a.out_dir, f"attr{suffix}.npy"), attr.astype(np.float32))
    np.save(os.path.join(a.out_dir, f"ohe{suffix}.npy"), seq.cpu().numpy().astype(np.int8))
    regions_df.to_csv(os.path.join(a.out_dir, f"regions{suffix}.bed"),
                      sep="\t", header=False, index=False)
    print(f"DONE attr={attr.shape}", flush=True)


if __name__ == "__main__":
    main()
