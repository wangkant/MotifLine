#!/usr/bin/env python
"""ChromBPNet DeepSHAP attribution step (compute layer, runs in chrombpnet_local py3.10).

Line-for-line the math of region_v3/k562/shap_shard.py, single-process:
same model construction, get_seq, masks and _deep_lift_shap call, so its
output is directly comparable to the archived attr.npy. stdlib+numpy+torch+
chrombpnet only -- this file must never import motiflex or motifline.

Full-scale runs shard the peak list (--shard_i/--n_shards, same linspace bounds
as the archive) and write ``attr_<i>.npy``/``ohe_<i>.npy``/``regions_<i>.bed``;
``motifline.merge_shards`` concatenates them back in shard order.
"""
import argparse
import os
import sys


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrombpnet_src", required=True, help="path to chrombpnet-pytorch checkout")
    ap.add_argument("--wo_bias", required=True)
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--chrom_sizes", required=True)
    ap.add_argument("--task", default="counts", choices=["counts", "profile"])
    ap.add_argument("--hypothetical", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--in_window", type=int, default=2114)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--head", type=int, default=None, help="take only the first N peaks (slice runs)")
    ap.add_argument("--n_shards", type=int, default=None,
                    help="split the peak list into N contiguous shards (full-scale runs)")
    ap.add_argument("--shard_i", type=int, default=None, help="index of the shard to compute, 0-based")
    a = ap.parse_args()
    if (a.shard_i is None) != (a.n_shards is None):
        ap.error("--shard_i and --n_shards must be given together")
    if a.n_shards is not None:
        if a.head is not None:
            ap.error("--head and --shard_i/--n_shards are mutually exclusive: a head-slice of a "
                     "shard is neither the slice nor the shard, and the merged output would be "
                     "silently short")
        if not 0 <= a.shard_i < a.n_shards:
            ap.error(f"--shard_i must be in [0, {a.n_shards})")
    return a


def shard_bounds(n_rows, n_shards):
    """Contiguous shard boundaries, verbatim from region_v3/k562/shap_shard.py:52-56.

    ``np.linspace(0, N, n_shards+1).astype(int)`` is what produced the archived
    shards, so keeping it (rather than a ceil-divide that looks equivalent) keeps
    shard i covering exactly the rows it covered then.
    """
    import numpy as np
    return np.linspace(0, n_rows, n_shards + 1).astype(int)


def normalize_peaks_bed(peaks_path, out_dir):
    """Hand ``load_region_df`` the 10-column narrowPeak form it expects.

    The archived region_v3 ``peaks.bed`` is chrombpnet's *output* regions bed:
    10 narrowPeak columns plus the ``is_peak`` column that ``load_region_df``
    itself appends (``data_utils.py:238``), so feeding it back verbatim gives
    12 columns against 11 names and dies. Dropping that trailing column and
    letting ``load_region_df`` re-add ``is_peak=True`` is identity, which is
    only true while every value is True -- anything else is refused rather
    than silently coerced. Files with <=10 columns are passed through
    untouched. Row order and row count are never changed here.
    """
    with open(peaks_path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    widths = {ln.count("\t") + 1 for ln in lines}
    if widths and max(widths) <= 10:
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

    regions_df = load_region_df(normalize_peaks_bed(a.peaks, a.out_dir),
                                chrom_sizes=a.chrom_sizes)
    suffix = ""
    if a.n_shards is not None:
        # Shard BEFORE the N-base masks, exactly as shap_shard.py did: the masks drop
        # different rows in different shards, so slicing after them would make the shard
        # bounds depend on sequence content and the concatenation no longer be the whole.
        bounds = shard_bounds(len(regions_df), a.n_shards)
        lo, hi = int(bounds[a.shard_i]), int(bounds[a.shard_i + 1])
        regions_df = regions_df.iloc[lo:hi].reset_index(drop=True)
        suffix = f"_{a.shard_i}"
        print(f"[shard {a.shard_i}/{a.n_shards}] rows [{lo}:{hi}] = {len(regions_df)} peaks", flush=True)
    if a.head:
        regions_df = regions_df.iloc[: a.head].reset_index(drop=True)
    seq = get_seq(regions_df, pyfaidx.Fasta(a.fasta), a.in_window)
    if isinstance(seq, np.ndarray):
        seq = torch.tensor(seq.astype(np.float32))
    if seq.shape[-1] == 4:
        seq = seq.permute(0, 2, 1)
    keep = ~((seq == 0.25).any(dim=(1, 2)))
    seq = seq[keep]
    regions_df = regions_df.iloc[keep.numpy()].reset_index(drop=True)
    keep = ~((seq == 0).all(dim=1).any(dim=1))
    seq = seq[keep]
    regions_df = regions_df.iloc[keep.numpy()].reset_index(drop=True)

    attr = _deep_lift_shap(model.to("cuda"), seq, batch_size=a.batch_size, verbose=True,
                           args=None, warning_threshold=1e8, device="cuda",
                           hypothetical=a.hypothetical, random_state=a.random_state)
    attr = attr.cpu().numpy() if isinstance(attr, torch.Tensor) else np.asarray(attr)
    np.save(os.path.join(a.out_dir, f"attr{suffix}.npy"), attr.astype(np.float32))
    np.save(os.path.join(a.out_dir, f"ohe{suffix}.npy"), seq.cpu().numpy().astype(np.int8))
    regions_df.to_csv(os.path.join(a.out_dir, f"regions{suffix}.bed"),
                      sep="\t", header=False, index=False)
    print(f"DONE attr={attr.shape}", flush=True)


if __name__ == "__main__":
    main()
