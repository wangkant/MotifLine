"""AlphaGenome target and DeepLIFT rules for attr_ag.py.

Adapted from Ruchir Rastogi's examples/notebooks/deeplift.ipynb
(rastogiruchir/alphagenome-pytorch, branch deeplift-support, Apache-2.0; licence in
licenses/Apache-2.0.txt). Changes: the wrapper is reformatted, the op table is fixed to
the log-ratio softmax and symmetric LayerNorm rules, and uncached_nonlinear_ops is
dropped because tangermeme 1.0.4 does not accept it.

tangermeme gives a module without a rule plain gradients; at 64e9b75 every module not
listed here is linear.
"""
import torch
import torch.nn as nn

from alphagenome_pytorch import layers
from alphagenome_pytorch.attribution import deeplift
from tangermeme.deep_lift_shap import _nonlinear


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


def nonlinear_ops():
    return {
        layers.QuickGELU: _nonlinear,
        layers.TanhSoftCap: _nonlinear,
        layers.SoftClip: _nonlinear,
        layers.Log1p: _nonlinear,
        nn.Softmax: deeplift.softmax_log_ratio_product_rule,
        layers.LayerNorm: deeplift.layernorm_symmetric_product_rule,
        layers.BilinearOp: deeplift.bilinear_symmetric_product_rule,
    }
