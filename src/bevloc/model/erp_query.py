"""ERP-token query: the panorama's own tokens are the query; placement happens after matching.

Following Loc² (Xia, Xu, Alahi, ICLR 2026): match image-plane tokens to aerial tokens, then lift only
the matched points. Here the lift is depth-free: every token looking below the horizon is placed at
its flat-ground intersection from the camera height (exact for road and pavement, wrong only for
above-ground content, which the matcher's certainty head is free to reject). Tokens above the
horizon are invalid for the coarse loss and excluded from consensus.

Placement is expressed in **virtual BEV pixel coordinates** on the 224 px / 0.25 m grid every other
query uses (u = n/2 − y/cell − 0.5, v = n/2 − x/cell − 0.5, x forward, y left), so the ground-truth
homography, the pose metrics and the matcher's pixel conversion apply unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def erp_placement(h, w, erp_hw, R_w2c, height_m, n, cell_m, max_range_m=40.0, patch=16):
    """Ground intersection of every token-centre ray as virtual BEV pixels.

    Returns xy (B, h, w, 2) float (u, v) and valid (B, h, w) bool (below the horizon, inside range)."""
    B = R_w2c.shape[0]
    H, W = int(erp_hw[0]), int(erp_hw[1])
    device, dtype = R_w2c.device, torch.float32
    dir_cam, _elev = _rays(h, w, H, W, device, dtype, patch)
    R_c2w = R_w2c.float().transpose(-1, -2)                       # (B, 3, 3)
    dir_w = torch.einsum("bij,nj->bni", R_c2w, dir_cam)           # (B, N, 3) world ENU
    fwd = R_c2w[:, :, 2].clone()
    fwd[:, 2] = 0.0
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left = torch.stack([-fwd[:, 1], fwd[:, 0], torch.zeros_like(fwd[:, 0])], -1)
    dz = dir_w[..., 2]
    down = dz < -1e-4
    t = torch.where(down, -float(height_m) / dz.clamp(max=-1e-4), torch.zeros_like(dz))   # ray length to the ground
    p = dir_w * t[..., None]
    x = (p * fwd[:, None, :]).sum(-1)
    y = (p * left[:, None, :]).sum(-1)
    rng = torch.hypot(x, y)
    valid = down & (rng <= float(max_range_m)) & (rng >= 0.0)
    u = n / 2 - y / cell_m - 0.5
    v = n / 2 - x / cell_m - 0.5
    xy = torch.stack([u, v], -1).reshape(B, h, w, 2)
    return xy, valid.reshape(B, h, w)


def _rays(h, w, H, W, device, dtype, patch):
    """Unit directions (camera x right, y down, z forward) and elevation (+up) of token centres."""
    jj, ii = torch.meshgrid(torch.arange(w, device=device, dtype=dtype),
                            torch.arange(h, device=device, dtype=dtype), indexing="xy")
    u = (jj + 0.5) * patch
    v = (ii + 0.5) * patch
    lon = (u / W - 0.5) * (2 * math.pi)
    lat = (0.5 - v / H) * math.pi
    cl = torch.cos(lat)
    dir_cam = torch.stack([torch.sin(lon) * cl, -torch.sin(lat), torch.cos(lon) * cl], -1)
    return dir_cam.reshape(-1, 3), lat.reshape(-1)


class ErpQuery(nn.Module):
    """No parameters: f_q is the frozen encoder's token map of the panorama; the decoder trains."""

    def __init__(self, cfg):
        super().__init__()
        g = cfg.grid
        self.n, self.cell = int(g.n), float(g.cell_m)
        self.height = float(cfg.ipm.height_m)
        E = getattr(cfg, "erp_query", None)
        self.max_range = float(getattr(E, "max_range_m", 40.0)) if E else 40.0
        self.patch = 16

    def placement(self, batch):
        erp = batch["erp"]                                       # (B, T, 3, H, W); T must be 1
        H, W = erp.shape[-2:]
        return erp_placement(H // self.patch, W // self.patch, (H, W), batch["R_w2c"][:, 0],
                             self.height, self.n, self.cell, self.max_range, self.patch)

    def forward(self, batch, matcher):
        erp = batch["erp"]
        if erp.shape[1] != 1:
            raise ValueError("ErpQuery is single-frame; got T=%d" % erp.shape[1])
        with torch.no_grad():
            f_q = matcher.model.encoder(erp[:, 0])[16]
        _, valid = self.placement(batch)
        return f_q, valid.float()
