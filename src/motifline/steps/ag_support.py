#!/usr/bin/env python
"""AlphaGenome DeepLIFT helpers for attr_ag.py.

AlphaGenomeWrapper and _ag_nonlinear_ops are adapted from Ruchir Rastogi's
examples/notebooks/deeplift.ipynb (rastogiruchir/alphagenome-pytorch, branch
deeplift-support, Apache-2.0; licence in licenses/Apache-2.0.txt). Changes: the wrapper
is reformatted, the op table is a function of two flags, and the notebook's
uncached_nonlinear_ops is dropped because tangermeme 1.0.4 does not accept it.
The sequence and peak helpers are not from the notebook.

The op table must cover every nonlinear module: tangermeme gives a module without a
rule plain gradients and does not warn. At 64e9b75 everything uncovered is linear.
"""
import pandas as pd
import torch
import torch.nn as nn

from alphagenome_pytorch import layers
from alphagenome_pytorch.attribution import deeplift
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor
from tangermeme.deep_lift_shap import _nonlinear


def read_chrom_sizes(path):
    """chrom -> length from a two-column chrom.sizes file; other lines are skipped."""
    with open(path) as f:
        rows = [line.strip().split("\t") for line in f]
    return {r[0]: int(r[1]) for r in rows if len(r) == 2}


class AlphaGenomeWrapper(nn.Module):
    """log1p(sum over the central `target_length` bp of one track)."""
    def __init__(self, model, modality, track_idx, target_length, resolution=1, organism_idx=0):
        super().__init__()
        self.model = model; self.modality = modality; self.track_idx = track_idx
        self.target_length = target_length; self.resolution = resolution
        self.organism_idx = organism_idx; self.log1p = layers.Log1p()

    def forward(self, x):                          # (N, 4, L)
        x = x.transpose(1, 2)
        org = torch.full((x.shape[0],), self.organism_idx, dtype=torch.long, device=x.device)
        out = self.model(x, org, heads=(self.modality,), resolutions=(self.resolution,), channels_last=True)
        sig = out[self.modality][self.resolution][:, :, self.track_idx]
        cs = sig.shape[1] // 2 - self.target_length // 2
        sig = sig[:, cs:cs + self.target_length].sum(dim=1)
        return self.log1p(sig).unsqueeze(1)


def _ag_nonlinear_ops(use_log_ratio_softmax=True, use_log_ratio_layernorm=False):
    return {
        layers.QuickGELU: _nonlinear,
        layers.TanhSoftCap: _nonlinear,
        layers.SoftClip: _nonlinear,
        layers.Log1p: _nonlinear,
        nn.Softmax: (deeplift.softmax_log_ratio_product_rule if use_log_ratio_softmax
                     else deeplift.softmax_symmetric_product_rule),
        layers.LayerNorm: (deeplift.layernorm_log_ratio_product_rule if use_log_ratio_layernorm
                           else deeplift.layernorm_symmetric_product_rule),
        layers.BilinearOp: deeplift.bilinear_symmetric_product_rule,
    }


def window_seq(fasta, chrom, center, L):
    """L bp centred on `center`, the same centring as chrombpnet's get_seq.

    `chrom` is looked up in the FASTA exactly as the peaks name it.
    """
    return fasta[str(chrom)][center - L // 2:center + L // 2].seq


def onehot(seq):
    """(4, L); non-ACGT bases are all-zero columns."""
    return sequence_to_onehot_tensor(seq).T


def load_peaks(peaks_bed, chrom_sizes_path, in_window):
    """chr/start/end/summit of every peak whose window fits on its chromosome."""
    raw = pd.read_csv(peaks_bed, sep="\t", header=None)
    df = pd.DataFrame({"chr": raw[0].astype(str), "start": raw[1].astype(int),
                       "end": raw[2].astype(int), "summit": raw[9].astype(int)})
    cs = read_chrom_sizes(chrom_sizes_path)
    unknown = sorted(set(df["chr"]) - set(cs))
    if unknown:
        raise SystemExit(f"{peaks_bed}: chromosomes not in {chrom_sizes_path}: {unknown[:5]}")
    flank = in_window // 2
    center = df["start"] + df["summit"]
    size = df["chr"].map(lambda c: int(cs[c]))
    keep = (center - flank > 0) & (center + flank <= size)
    return df[keep].reset_index(drop=True)
