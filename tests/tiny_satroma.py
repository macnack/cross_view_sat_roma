"""A toy-width Sat-RoMa decoder built from the package's own classes, for CPU tests without weights or data.

Real: `Decoder` (the forward, ToWarp, refiner wiring and exposed intermediates under test), `ConvRefiner` (local
correlation, displacement embedding) and `TransformerDecoder` (with no blocks: its linear `to_out` is the classifier).
Stand-in: the GP, replaced by the query tokens' own normalised position, so that `to_out` can be set to put each
token's logit peak at the cell a known translation sends it to (`plant_translation`).
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import torch
import torch.nn as nn

import bevloc.match.satroma  # noqa: F401  (puts sat_roma on the path)
from sat_roma.matcher import ConvRefiner, Decoder, RegressionMatcher
from sat_roma.transformer import TransformerDecoder

K = 56          # reference cells per side (896 px / 16)
C_ENC = 16      # toy encoder width
PROJ = 8        # toy proj_dim (= gp_dim = feat_dim)


class PosGP(nn.Module):
    """GP stand-in: channels 0/1 = the query token's normalised (x, y), the rest zero."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dummy = nn.Parameter(torch.zeros(()))           # so the stand-in has a parameter to (not) train

    def forward(self, x, y, **kw):
        b, _, h, w = x.shape
        ys = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=x.device)
        xs = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=x.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        out = torch.zeros(b, self.dim, h, w, device=x.device)
        out[:, 0], out[:, 1] = xx, yy
        return out + self.dummy


def tiny_decoder(seed=0, radius=2, emb=4):
    torch.manual_seed(seed)
    hidden = 2 * PROJ
    coarse = TransformerDecoder(nn.Sequential(), hidden, K * K + 1, is_classifier=True, amp=False, pos_enc=False)
    in_dim = 2 * PROJ + emb + (2 * radius + 1) ** 2
    refiner = nn.ModuleDict({"16": ConvRefiner(in_dim, in_dim, 3, kernel_size=5, dw=True, hidden_blocks=1,
                                               displacement_emb="linear", displacement_emb_dim=emb,
                                               local_corr_radius=radius, corr_in_other=True, amp=False)})
    # make the refiner's output non-trivial but small (random init of the last 1x1 conv is fine; scale it down)
    with torch.no_grad():
        refiner["16"].out_conv.weight.mul_(0.1)
    proj = nn.ModuleDict({"16": nn.Sequential(nn.Conv2d(C_ENC, PROJ, 1), nn.BatchNorm2d(PROJ))})
    gps = nn.ModuleDict({"16": PosGP(PROJ)})
    dec = Decoder(coarse, gps, proj, refiner, detach=True, scales=["16"])
    return dec.eval()


def plant_translation(dec, t_px, query_px=224, ref_px=896, beta=4000.0):
    """Set the classifier so token (i, j) of a query_px picture peaks at the cell of (token centre + t_px) in the
    ref_px reference (identity scale: 1 query px = 1 reference px). Works for any token grid via the GP's
    normalised positions only when the grid is the 14 x 14 picture grid."""
    lin = dec.embedding_decoder.to_out
    G = torch.linspace(-1 + 1 / K, 1 - 1 / K, K)
    gy, gx = torch.meshgrid(G, G, indexing="ij")
    Gc = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)                  # (K*K, 2) class centres (x, y)
    a = query_px / ref_px                                                   # ref_n = a * pos + b
    b = torch.tensor([a - 1 + 2 * t_px[0] / ref_px, a - 1 + 2 * t_px[1] / ref_px])
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()
        lin.weight[:-1, 0] = 2 * beta * a * Gc[:, 0]
        lin.weight[:-1, 1] = 2 * beta * a * Gc[:, 1]
        lin.bias[:-1] = 2 * beta * (Gc @ b) - beta * (Gc ** 2).sum(-1)
        lin.bias[-1] = 2.0                                                  # certainty logit
    return dec


class StubEncoder(nn.Module):
    patch_size = 16


def tiny_matcher(dec):
    """Just enough of FeatureQueryMatcher for scripts/eval_vigor.score: .model (a real RegressionMatcher, for
    exposed_intermediates), .wrapper (im_a_size / im_b_size / device / model), .reference_features."""
    model = RegressionMatcher(StubEncoder(), dec)
    wrapper = NS(im_a_size=224, im_b_size=896, device="cpu", model=model)
    torch.manual_seed(1)
    ref_proj = nn.Conv2d(3, C_ENC, 1)

    def reference_features(ref):
        with torch.no_grad():
            return {16: ref_proj(nn.functional.adaptive_avg_pool2d(ref, K))}
    return NS(model=model, wrapper=wrapper, reference_features=reference_features)


class StubPictureQuery:
    """A picture query (no `placement`): tokens from the picture by a fixed projection, validity from bev_valid."""

    def __init__(self):
        torch.manual_seed(2)
        self.proj = nn.Conv2d(3, C_ENC, 1)

    def __call__(self, batch, matcher):
        with torch.no_grad():
            f = self.proj(nn.functional.avg_pool2d(batch["bev"], 16))
        frac = nn.functional.avg_pool2d(batch["bev_valid"].float()[:, None], 16)[:, 0]
        return f, frac


def cfg_stub(solver="se2", cell_m=0.125):
    return NS(grid=NS(n=224, cell_m=cell_m),
              matcher=NS(reproj_cells=3.0, seed=0, solver=solver, min_valid_frac=0.05, use_means=False))
