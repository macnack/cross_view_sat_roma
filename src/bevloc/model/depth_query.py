"""ERP-token query with metric-depth placement (task 04, Step 2): Loc² geometry, Sat-RoMa decoder as the matcher.

The query is the panorama's own frozen ``sat493m`` token map (as in the ``erp`` mode, `bevloc.model.erp_query`),
optionally through a light trained projection head (Loc² §3.1: a few convs + one self-attention block). Every token
is placed in the ego frame from a precomputed monocular depth map along its ray (Loc² §3.2): the token-centre ray
``dir`` (unit, camera frame) scaled by the depth ``d`` (metres **along the ray**, UniK3D's ``points.norm()``, see
scripts/loc2_depth_vigor.py) gives a 3-D point; its horizontal components are the BEV position. Height is dropped,
exactly as Loc² drops z. Tokens with depth >= ``max_depth_m`` (Loc²: 35 m), non-finite or non-positive are invalid.

Written from the equations of Loc² (Xia, Xu, Alahi, 2509.09792v3 §3.2): with θ the polar angle from the zenith and φ
the azimuth, x = d·sinθ·cosφ, y = −d·sinθ·sinφ. In Loc²'s VIGOR code φ = 0 is the panorama's first column (south),
which makes its frame x = south / y = east (= the satellite image's row / column axes). Ours is x forward / y left,
forward = the camera's horizontal z axis (north for VIGOR), so for a north-aligned panorama with azimuth α clockwise
from north (α = 0 at the centre column, increasing to the right) and elevation e:

    x = d·cos(e)·cos(α)      (north)
    y = −d·cos(e)·sin(α)     (left = west; a token 90° to the right, i.e. east, has y = −d)

Rather than hard-coding this, the rays go through ``R_w2c`` like `erp_placement`, so the formula holds for any
camera attitude; tests/test_depth_query.py checks both the formulas and the consistency with VigorPairs' H / en.

Placement is returned in **virtual BEV pixels** on the cfg.grid (u = n/2 − y/cell − 0.5, v = n/2 − x/cell − 0.5),
the convention every other query mode, `coarse_targets` and `SatRoMa.consensus_from_gm` use. Points may lie outside
the 224 px virtual picture (35 m > 28 m at 0.25 m/px); nothing downstream clips them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevloc.model.erp_query import _rays, rays_at


def sample_token_depth(depth, h, w):
    """Depth at every token centre: (B, 1, H, W) or (B, H, W) metres -> (B, h, w).

    Nearest pixel to the token-centre ray (pixel row floor((i + 0.5)·H/h)), for any depth-map resolution."""
    if depth.ndim == 4:
        depth = depth[:, 0]
    H, W = depth.shape[-2:]
    rows = ((torch.arange(h, device=depth.device, dtype=torch.float64) + 0.5) * H / h).floor().long().clamp(0, H - 1)
    cols = ((torch.arange(w, device=depth.device, dtype=torch.float64) + 0.5) * W / w).floor().long().clamp(0, W - 1)
    return depth[:, rows][:, :, cols]


def ego_axes(R_w2c):
    """Forward (camera z projected on the ground plane) and left unit vectors in the world frame, (B, 3) each."""
    R_c2w = R_w2c.float().transpose(-1, -2)
    fwd = R_c2w[:, :, 2].clone()
    fwd[:, 2] = 0.0
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left = torch.stack([-fwd[:, 1], fwd[:, 0], torch.zeros_like(fwd[:, 0])], -1)
    return R_c2w, fwd, left


def depth_placement_metric(depth, h, w, R_w2c, max_depth_m=35.0, patch=16):
    """Ego-metric (x forward, y left) of every token from the depth along its ray.

    depth: (B, 1, H, W) metres along the ray. Returns xy_m (B, h, w, 2) and valid (B, h, w) bool."""
    B = R_w2c.shape[0]
    dir_cam, _ = _rays(h, w, h * patch, w * patch, R_w2c.device, torch.float32, patch)   # (N, 3)
    d = sample_token_depth(depth.float(), h, w).reshape(B, -1)
    xy, valid = _metric_along_rays(dir_cam, d, R_w2c, max_depth_m)
    return xy.reshape(B, h, w, 2), valid.reshape(B, h, w)


def sample_pixel_depth(depth, uv, erp_hw):
    """Depth at continuous ERP points uv (N, 2) of an erp_hw = (H, W) panorama: (B, N), nearest depth pixel
    (floor(u * W_d / W), the rule of `sample_token_depth`, which it equals at token centres)."""
    if depth.ndim == 4:
        depth = depth[:, 0]
    Hd, Wd = depth.shape[-2:]
    H, W = int(erp_hw[0]), int(erp_hw[1])
    uv = uv.to(torch.float64)
    cols = (uv[:, 0] * Wd / W).floor().long().clamp(0, Wd - 1)
    rows = (uv[:, 1] * Hd / H).floor().long().clamp(0, Hd - 1)
    return depth[:, rows, cols]


def depth_placement_metric_at(depth, uv, erp_hw, R_w2c, max_depth_m=35.0):
    """`depth_placement_metric` for arbitrary ERP points: uv (N, 2) continuous coords (pixel k spans [k, k + 1);
    a token centre is ((j + 0.5) * 16, (i + 0.5) * 16)); the depth is read at that pixel and the point goes along
    that pixel's own ray. Returns xy_m (B, N, 2), valid (B, N)."""
    dir_cam, _ = rays_at(uv[:, 0].float(), uv[:, 1].float(), int(erp_hw[0]), int(erp_hw[1]))
    d = sample_pixel_depth(depth.float(), uv, erp_hw)
    return _metric_along_rays(dir_cam, d, R_w2c, max_depth_m)


def _metric_along_rays(dir_cam, d, R_w2c, max_depth_m):
    """dir_cam (N, 3) camera rays, d (B, N) depth along them -> ego xy (B, N, 2) metres, valid (B, N)."""
    R_c2w, fwd, left = ego_axes(R_w2c)
    dir_w = torch.einsum("bij,nj->bni", R_c2w, dir_cam)                                 # (B, N, 3)
    valid = torch.isfinite(d) & (d > 0) & (d < float(max_depth_m))
    d = torch.where(valid, d, torch.zeros_like(d))
    p = dir_w * d[..., None]
    x = (p * fwd[:, None, :]).sum(-1)
    y = (p * left[:, None, :]).sum(-1)
    return torch.stack([x, y], -1), valid


def metric_to_bev_px(xy_m, n, cell_m):
    """(…, 2) ego metres (x forward, y left) -> virtual BEV pixels (u, v)."""
    u = n / 2 - xy_m[..., 1] / cell_m - 0.5
    v = n / 2 - xy_m[..., 0] / cell_m - 0.5
    return torch.stack([u, v], -1)


def depth_placement(depth, h, w, R_w2c, n, cell_m, max_depth_m=35.0, patch=16):
    """Token placement as virtual BEV pixels (the `ErpQuery.placement` contract): xy (B, h, w, 2), valid (B, h, w)."""
    xy_m, valid = depth_placement_metric(depth, h, w, R_w2c, max_depth_m, patch)
    return metric_to_bev_px(xy_m, n, cell_m), valid


class ProjectionHead(nn.Module):
    """Loc² §3.1-style light head on the frozen tokens: 1×1 down-projection, two 3×3 convs (circular padding along
    the azimuth, the panorama wraps), one self-attention block, 1×1 up-projection back to the encoder width.

    Residual with a zero-initialised output projection: at initialisation the decoder sees the frozen tokens
    unchanged, so the released / warm-started decoder is not disturbed by a random head."""

    def __init__(self, in_dim=1024, dim=256, heads=4, n_convs=2, ff_mult=2):
        super().__init__()
        if int(dim) % 8:
            raise ValueError(f"erp_depth.head_dim must be divisible by 8 (GroupNorm with 8 groups), got {dim}")
        if int(dim) % int(heads):
            raise ValueError(f"erp_depth.head_dim {dim} must be divisible by head_heads {heads}")
        self.down = nn.Conv2d(in_dim, dim, 1)
        self.convs = nn.ModuleList([nn.Conv2d(dim, dim, 3, padding=0) for _ in range(int(n_convs))])
        self.norms = nn.ModuleList([nn.GroupNorm(8, dim) for _ in range(int(n_convs))])
        self.attn = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=int(ff_mult * dim), dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.up = nn.Conv2d(dim, in_dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, f):
        x = self.down(f)
        for conv, norm in zip(self.convs, self.norms):
            p = F.pad(x, (1, 1, 0, 0), mode="circular")          # azimuth wraps
            p = F.pad(p, (0, 0, 1, 1), mode="replicate")         # poles do not
            x = x + F.gelu(norm(conv(p)))
        B, C, h, w = x.shape
        x = self.attn(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, h, w)
        return f + self.up(x)


class ErpDepthQuery(nn.Module):
    """ERP tokens (frozen encoder) [+ projection head]; placement from ``batch["depth"]`` (B, 1, H, W) metres."""

    def __init__(self, cfg):
        super().__init__()
        g = cfg.grid
        self.n, self.cell = int(g.n), float(g.cell_m)
        E = getattr(cfg, "erp_depth", None)
        self.max_depth = float(getattr(E, "max_depth_m", 35.0)) if E else 35.0
        self.patch = 16
        self.head = None
        if E is not None and bool(getattr(E, "head", False)):
            self.head = ProjectionHead(in_dim=1024, dim=int(getattr(E, "head_dim", 256)),
                                       heads=int(getattr(E, "head_heads", 4)),
                                       n_convs=int(getattr(E, "head_convs", 2)),
                                       ff_mult=float(getattr(E, "head_ff_mult", 2)))

    def placement(self, batch):
        erp = batch["erp"]                                       # (B, T, 3, H, W); T must be 1
        H, W = erp.shape[-2:]
        if "depth" not in batch:
            raise KeyError("erp_depth query needs batch['depth'] (VigorPairs with depth, see scripts/loc2_depth_vigor.py)")
        return depth_placement(batch["depth"], H // self.patch, W // self.patch, batch["R_w2c"][:, 0],
                               self.n, self.cell, self.max_depth, self.patch)

    def placement_at(self, batch, uv):
        """Placement of arbitrary ERP points uv (N, 2) (continuous coords): depth read at each point's own pixel,
        point along its own ray, the convention of `placement`. xy (B, N, 2) virtual BEV px, valid (B, N)."""
        H, W = batch["erp"].shape[-2:]
        xy_m, valid = depth_placement_metric_at(batch["depth"], uv, (H, W), batch["R_w2c"][:, 0], self.max_depth)
        return metric_to_bev_px(xy_m, self.n, self.cell), valid

    def forward(self, batch, matcher):
        erp = batch["erp"]
        if erp.shape[1] != 1:
            raise ValueError("ErpDepthQuery is single-frame; got T=%d" % erp.shape[1])
        with torch.no_grad():
            f_q = matcher.model.encoder(erp[:, 0])[16]
        if self.head is not None:
            f_q = self.head(f_q.float())                         # the encoder may emit fp16 under autocast
        _, valid = self.placement(batch)
        return f_q, valid.float()
