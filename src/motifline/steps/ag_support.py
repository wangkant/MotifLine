#!/usr/bin/env python
"""AlphaGenome DeepLIFT support symbols.

``AlphaGenomeWrapper`` and ``_ag_nonlinear_ops`` derive from, and modify, Ruchir
Rastogi's ``examples/notebooks/deeplift.ipynb`` (rastogiruchir/alphagenome-pytorch,
branch deeplift-support), Apache-2.0; licence text at licenses/Apache-2.0.txt. Changed:
the wrapper is reformatted, the op table became a function of two booleans, and the
notebook's ``uncached_nonlinear_ops`` is dropped because stock tangermeme 1.0.4 does not
take it. ``ag_seqs_centered`` and ``load_peaks`` are not his.

The table has to be complete: tangermeme registers no hook for a module type it has
no rule for and says nothing, so a missing nonlinearity is silently given ordinary
gradients. Measured over AlphaGenome's 639 modules at 64e9b75, everything left
uncovered is linear or structural.

They live here rather than being imported from the validation script, because that tree
is read-only and must never be on motifline's import path: the pipeline has to keep
working when it is gone, and an import would silently re-bind the pipeline's math to a
file nobody is versioning.

This module is part of the COMPUTE layer (chrombpnet_local, py3.10): it must never import
motifline. It also does no sys.path manipulation of its own -- the caller
(attr_ag.py) puts the alphagenome-pytorch-deeplift and chrombpnet-pytorch checkouts on
sys.path before importing this module, so the site paths stay in config, not in code.
"""
import numpy as np  # noqa: F401  (kept for parity with the source module's namespace)
import pandas as pd
import torch
import torch.nn as nn

from alphagenome_pytorch import layers
from alphagenome_pytorch.attribution import deeplift
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor
from tangermeme.deep_lift_shap import _nonlinear

from chrombpnet.data_utils import read_chrom_sizes


# --------------------------- AlphaGenome wrapper ---------------------------
class AlphaGenomeWrapper(nn.Module):
    """Scalar = log1p(sum of central `target_length` bp of one ATAC track at `resolution`)."""
    def __init__(self, model, modality, track_idx, target_length, resolution=1, organism_idx=0):
        super().__init__()
        self.model = model; self.modality = modality; self.track_idx = track_idx
        self.target_length = target_length; self.resolution = resolution
        self.organism_idx = organism_idx; self.log1p = layers.Log1p()

    def forward(self, x):                          # x: (N, 4, L)
        x = x.transpose(1, 2)                      # (N, L, 4)
        org = torch.full((x.shape[0],), self.organism_idx, dtype=torch.long, device=x.device)
        out = self.model(x, org, heads=(self.modality,), resolutions=(self.resolution,), channels_last=True)
        sig = out[self.modality][self.resolution][:, :, self.track_idx]   # (N, L)
        cs = sig.shape[1] // 2 - self.target_length // 2
        sig = sig[:, cs:cs + self.target_length].sum(dim=1)              # (N,)
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


# --------------------------- sequence loading ---------------------------
def ag_seqs_centered(df, fasta, L):
    """One-hot (N,4,L) centered on (start + summit), matching CBP's get_seq centering."""
    out = []
    for _, r in df.iterrows():
        c = int(r["start"]) + int(r["summit"])
        seq = fasta[f"chr{str(r['chr']).replace('chr','')}"][c - L // 2:c + L // 2].seq
        out.append(sequence_to_onehot_tensor(seq).T)        # (4, L)
    return torch.stack(out)


def load_peaks(peaks_bed, chrom_sizes_path, in_window):
    """Build chr/start/end/summit df from region_v3 peaks.bed (cols: chr,start,end,name,score,
    strand,signal,p,q,summit,is_peak); keep only rows whose (start+summit)+/-in_window//2 is in-bounds."""
    raw = pd.read_csv(peaks_bed, sep="\t", header=None)
    df = pd.DataFrame({"chr": raw[0].astype(str), "start": raw[1].astype(int),
                       "end": raw[2].astype(int), "summit": raw[9].astype(int)})
    cs = read_chrom_sizes(chrom_sizes_path)
    flank = in_window // 2
    center = df["start"] + df["summit"]
    size = df["chr"].map(lambda c: int(cs.get(c, 0)))
    keep = (center - flank > 0) & (center + flank <= size)
    return df[keep].reset_index(drop=True)
