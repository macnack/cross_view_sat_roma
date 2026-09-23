"""Query builders: anything that turns a sample into the (B, 1024, 14, 14) tokens the decoder expects.

  lift   — spherical Lift-Splat (learned depth per ERP token)            [05_lift_splat]
  ipm    — RGB flat-ground IPM picture through the frozen sat493m encoder [kick-off H2 lower bound]
  hybrid — dense IPM ground features + learned depth above the horizon   [task 03, Task 3]
  erp    — the panorama's own tokens; placement after matching (Loc²)     [task 03, Task 6]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevloc.model.lift_splat import SphericalLiftSplat


def _lift_from_cfg(cfg, **kw):
    L, g = cfg.lift, cfg.grid
    args = dict(dim=L.dim, depth_bins=L.depth_bins, d_min=L.d_min, d_max=L.d_max,
                n=g.n, cell=g.cell_m, max_elev_deg=L.max_elev_deg)
    args.update(kw)
    return SphericalLiftSplat(**args)


class LiftQuery(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lift = _lift_from_cfg(cfg)

    def forward(self, batch, matcher):
        erp = batch["erp"]                                  # (B, T, 3, H, W)
        B, T = erp.shape[:2]
        with torch.no_grad():
            feats = [matcher.model.encoder(erp[:, t])[16] for t in range(T)]
        hw = erp.shape[-2:]
        if T == 1:
            return self.lift(feats[0], batch["R_w2c"][:, 0], erp_hw=hw)
        return self.lift.forward_multiframe(torch.stack(feats, 1), batch["R_w2c"], batch["se2"], erp_hw=hw)


class IpmQuery(nn.Module):
    """No parameters of its own: the picture goes through the frozen encoder; the decoder is what trains."""

    def __init__(self, cfg):
        super().__init__()
        self.patch = 16

    def forward(self, batch, matcher):
        f_q = matcher.image_query_features(batch["bev"])
        frac = F.avg_pool2d(batch["bev_valid"].float()[:, None], self.patch)[:, 0]
        return f_q, frac


def build_query(cfg, mode):
    if mode == "lift":
        return LiftQuery(cfg)
    if mode == "ipm":
        return IpmQuery(cfg)
    if mode == "hybrid":
        from bevloc.model.hybrid_query import HybridQuery
        return HybridQuery(cfg)
    if mode == "erp":
        from bevloc.model.erp_query import ErpQuery
        return ErpQuery(cfg)
    raise ValueError(f"unknown query mode {mode!r}")


def load_query_state(query, state):
    """Accept both the new {"query": ..., "mode": ...} and the old {"lift": ...} checkpoint layouts."""
    if "query" in state:
        query.load_state_dict(state["query"])
    elif "lift" in state and hasattr(query, "lift"):
        query.lift.load_state_dict(state["lift"])
    elif "lift" in state and hasattr(query, "load_lift_weights"):
        query.load_lift_weights(state["lift"])


def lift_state_dict(state):
    """The raw SphericalLiftSplat weights from either checkpoint layout (for viz scripts)."""
    if "lift" in state:
        return state["lift"]
    if "query" in state:
        return {k[len("lift."):]: v for k, v in state["query"].items() if k.startswith("lift.")}
    raise KeyError("checkpoint has neither 'lift' nor 'query'")
