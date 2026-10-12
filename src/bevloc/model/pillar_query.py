"""Pillar feature lift (task 07): FG²'s BEV geometry with the Sat-RoMa decoder as the matcher.

Written from FG² (Xia & Alahi, CVPR 2025, 2503.18725 §3.2, Eq. 1–2) and docs/tasks/07_pillar_lift.md; nothing is
taken from third_party/FG2 (GPL-3).

Every BEV cell (x forward, y left, metres, the `bevloc.bev.ipm_sphere.cell_centres` convention) carries a pillar
of M points at heights ``z_above_ground_m`` above the ground plane z = −cam_height_m (slice 0 = the flat-ground
IPM point). Each 3-D point is expressed in the camera frame through ``R_w2c`` and projected into the panorama with
`project_erp`, the exact inverse of `bevloc.model.erp_query.rays_at` (band-aware); the frozen scale-16 token map is
sampled there bilinearly (sub-patch, decision 2026-09-19 (b)). A learned softmax over the M slices pools the
pillar into one feature per cell (FG² Eq. 2: "feature selection along height"); the pooled grid goes through a light
BEV head into the decoder at ``f_q_pyramid[16]``.

Placement is deterministic — the cell centres in virtual BEV pixels (`bevloc.model.depth_query.metric_to_bev_px`) —
so the placed-token paths (coarse targets, VCE, consensus, two-pass) apply unchanged and no depth is needed at
inference. Depth is a *training* signal only: `height_targets` turns a UniK3D range map into the slice each cell
should select (the kick-off's "LiDAR teaches geometry, camera-only afterwards", H3).

Frames: the camera is x right, y down, z forward; ``R_w2c`` maps world ENU to camera. The grid is defined in the
ego axes of `bevloc.model.depth_query.ego_axes` (forward = the camera's z flattened onto the ground, left, up),
which on VIGOR (R_NORTH) is north / west / up — FG²'s aerial-frame ground grid under known orientation.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from bevloc.model.depth_query import ego_axes


def grid_xy(n, cell_m, device=None, dtype=torch.float32):
    """Ego metres (x forward, y left) of every cell centre of an n x n grid: (n, n, 2); row 0 forward, column 0 left
    (the `ipm_sphere.cell_centres` / `metric_to_bev_px` convention)."""
    c = (torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2.0) * float(cell_m)
    x = -c[:, None].expand(n, n)
    y = -c[None, :].expand(n, n)
    return torch.stack([x, y], -1)


def pillar_points(xy_m, z_above_ground_m, cam_height_m, R_w2c, ego=True):
    """Camera-frame 3-D points of every pillar: xy_m (..., 2) ego metres, M heights above the ground plane
    (z = −cam_height_m in the camera-centred frame) -> p_cam (B, N, M, 3), N = the number of cells.

    ego=True: the grid is in the ego axes (`ego_axes(R_w2c)`: forward, left, up); ego=False: in world ENU with
    x = north, y = west (equal to the ego axes for a north-facing camera, the VIGOR case)."""
    R = R_w2c.float()
    B = R.shape[0]
    xy = xy_m.reshape(-1, 2).to(R.device, R.dtype)                              # (N, 2)
    z = torch.as_tensor(list(z_above_ground_m), device=R.device, dtype=R.dtype) - float(cam_height_m)   # (M,)
    if ego:
        _, fwd, left = ego_axes(R)                                                # (B, 3) each
    else:
        fwd = torch.tensor([[0.0, 1.0, 0.0]], device=R.device, dtype=R.dtype).expand(B, 3)
        left = torch.tensor([[-1.0, 0.0, 0.0]], device=R.device, dtype=R.dtype).expand(B, 3)
    up = torch.tensor([0.0, 0.0, 1.0], device=R.device, dtype=R.dtype)
    p_w = (xy[None, :, None, 0, None] * fwd[:, None, None, :] + xy[None, :, None, 1, None] * left[:, None, None, :]
           + z[None, None, :, None] * up)                                        # (B, N, M, 3)
    return torch.einsum("bij,bnmj->bnmi", R, p_w)


def project_erp(p_cam, H, W, band=None):
    """Camera-frame points (..., 3) -> continuous ERP coordinates (u, v) on an H x W panorama and the range.

    The exact inverse of `erp_query.rays_at`: lon = atan2(x, z) (0 at the centre column, + to the right),
    lat = atan2(−y, hypot(x, z)) (+ up); u = (lon / 2π + 0.5) W; v = (lat − top) / (bot − top) H with
    (top, bot) = (+π/2, −π/2) for the full sphere or the panorama's elevation band (lat_top, lat_bottom)."""
    x, y, z = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
    rng = p_cam.norm(dim=-1)
    lon = torch.atan2(x, z)
    lat = torch.atan2(-y, torch.hypot(x, z))
    top, bot = (math.pi / 2, -math.pi / 2) if band is None else (float(band[0]), float(band[1]))
    u = (lon / (2 * math.pi) + 0.5) * W
    v = (lat - top) / (bot - top) * H
    return u, v, rng


def slice_validity(p_cam, H, W, band=None, max_depression_deg=90.0):
    """Which pillar points carry image: inside the panorama's elevation band and not below the ego cut
    (elevation >= −max_depression_deg; 90 = no cut). p_cam (B, N, M, 3) -> bool (B, N, M)."""
    x, y, z = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
    lat = torch.atan2(-y, torch.hypot(x, z))
    ok = torch.isfinite(lat) & (lat >= -math.radians(float(max_depression_deg)))
    if band is not None:
        ok = ok & (lat <= float(band[0])) & (lat >= float(band[1]))
    return ok


def sample_pillars(f, uv, H, W):
    """Bilinear sample of the token map f (B, C, h, w) of an H x W panorama at ERP points uv (B, N, M, 2)
    -> (B, N, M, C). The azimuth wraps: one token column is circularly padded on each side, so a point just inside
    either edge interpolates with the column across the seam (`padding_mode="border"` alone would not)."""
    B, C, h, w = f.shape
    fp = F.pad(f, (1, 1, 0, 0), mode="circular")
    patch_w = W / w
    gu = (uv[..., 0] + patch_w) / (W + 2 * patch_w) * 2 - 1
    gv = uv[..., 1] / H * 2 - 1
    grid = torch.stack([gu, gv], -1).to(f.dtype)                                 # (B, N, M, 2)
    out = F.grid_sample(fp, grid, mode="bilinear", padding_mode="border", align_corners=False)   # (B, C, N, M)
    return out.permute(0, 2, 3, 1)


def sample_depth_at(depth, uv, H, W):
    """Nearest depth pixel under ERP points uv (B, N, M, 2) of an H x W panorama: depth (B, 1, Hd, Wd) or
    (B, Hd, Wd) -> (B, N, M). The rule of `depth_query.sample_pixel_depth` (floor(u · Wd / W)); the azimuth wraps,
    so u = W (a point exactly behind the camera, lon = π) reads column 0."""
    if depth.ndim == 4:
        depth = depth[:, 0]
    B, Hd, Wd = depth.shape
    cols = (uv[..., 0].double() * Wd / W).floor().long() % Wd
    rows = (uv[..., 1].double() * Hd / H).floor().long().clamp(0, Hd - 1)
    b = torch.arange(B, device=depth.device)[:, None, None].expand_as(rows)
    return depth[b, rows, cols]


def height_targets(rng, uv, slice_ok, depth, H, W, tol_m=1.0, tol_frac=0.10):
    """The slice each cell should select, from a range map: the slice whose camera range agrees best with the
    depth (metres along the ray, UniK3D's `points.norm()`) at its projected pixel; a cell has a target only when
    that agreement is within max(tol_m, tol_frac · range). rng (B, N, M), uv (B, N, M, 2), slice_ok (B, N, M),
    depth (B, 1, Hd, Wd) -> target (B, N) long, has_target (B, N) bool."""
    d = sample_depth_at(depth.float(), uv, H, W)
    err = (rng - d).abs()
    bad = ~slice_ok | ~torch.isfinite(d) | (d <= 0) | ~torch.isfinite(rng)
    err = torch.where(bad, torch.full_like(err, float("inf")), err)
    best, target = err.min(dim=-1)
    r = rng.gather(-1, target[..., None])[..., 0]
    tol = torch.clamp(float(tol_frac) * r, min=float(tol_m))
    has = torch.isfinite(best) & (best <= tol)
    return target, has


# ---- the query module ----------------------------------------------------------------------------------------------

import torch.nn as nn                                                             # noqa: E402

from bevloc.model.depth_query import ProjectionHead, metric_to_bev_px            # noqa: E402
from bevloc.model.erp_query import band_of                                        # noqa: E402

# the cfg.pillar keys that decide the module's parameter shapes: a checkpoint restores these into an evaluator's
# config (query.apply_query_cfg); the grid keys (cell_m, range_m, cam_height_m, ...) always come from the config
PILLAR_ARCH_KEYS = ("head", "head_dim", "head_convs", "head_heads", "head_ff_mult", "z_above_ground_m", "z_embed_dim")


def sinusoidal_2d(xy_m, dim, scale_m=70.0):
    """Fixed sinusoidal embedding of metric positions xy (n, n, 2) -> (dim, n, n); periods from scale_m / 2 down."""
    half = dim // 2
    k = torch.arange(half // 2, device=xy_m.device, dtype=xy_m.dtype)
    freq = (2 * math.pi / float(scale_m)) * (2.0 ** k)                            # (half/2,)
    out = []
    for a in (xy_m[..., 0], xy_m[..., 1]):
        ang = a[..., None] * freq                                                  # (n, n, half/2)
        out += [torch.sin(ang), torch.cos(ang)]
    pe = torch.cat(out, -1)                                                        # (n, n, 4 * (half/2))
    if pe.shape[-1] < dim:
        pe = torch.cat([pe, torch.zeros(*pe.shape[:-1], dim - pe.shape[-1], device=pe.device, dtype=pe.dtype)], -1)
    return pe.permute(2, 0, 1)


class PillarQuery(nn.Module):
    """Pillar lift: frozen panorama tokens sampled along M heights per BEV cell, a learned softmax over the heights,
    a light BEV head -> (B, 1024, n, n) for the decoder; placement = the cell centres (no depth at inference).

    `height_loss` (training, Row B) supervises the height softmax with the slice a depth map agrees with."""

    def __init__(self, cfg):
        super().__init__()
        g = cfg.grid
        self.n_bev, self.cell_bev = int(g.n), float(g.cell_m)
        P = cfg.pillar
        self.cell, self.range = float(P.cell_m), float(P.range_m)
        self.nq = int(round(2 * self.range / self.cell))
        self.z = [float(v) for v in P.z_above_ground_m]
        self.cam_h = float(P.cam_height_m)
        self.max_dep = float(getattr(P, "max_depression_deg", 90.0))
        self.blind = float(getattr(P, "blind_radius_m", 0.0))
        self.tol_m, self.tol_frac = float(getattr(P, "height_tol_m", 1.0)), float(getattr(P, "height_tol_frac", 0.10))
        self.patch, C = 16, 1024
        zd = int(getattr(P, "z_embed_dim", 64))
        self.z_embed = nn.Parameter(0.02 * torch.randn(len(self.z), zd))
        self.scorer = nn.Sequential(nn.Linear(C + zd, 256), nn.GELU(), nn.Linear(256, 1))
        self.head = None
        if bool(getattr(P, "head", True)):
            self.head = ProjectionHead(in_dim=C, dim=int(getattr(P, "head_dim", 256)), heads=int(getattr(P, "head_heads", 4)),
                                       n_convs=int(getattr(P, "head_convs", 2)), ff_mult=float(getattr(P, "head_ff_mult", 2)),
                                       circular=False)
        self._last = None

    # ---- geometry ---------------------------------------------------------------------------------------------

    def geometry(self, batch):
        """Projected pillars of the batch: uv (B, N, M, 2) ERP px, rng (B, N, M) camera range, ok (B, N, M) slices
        carrying image, cell_valid (B, N), xy_m (N, 2) cell centres (ego metres); N = nq * nq."""
        erp = batch["erp"]
        H, W = erp.shape[-2:]
        R = batch["R_w2c"][:, 0]
        xy = grid_xy(self.nq, self.cell, device=R.device).reshape(-1, 2)
        p_cam = pillar_points(xy, self.z, self.cam_h, R)
        band = band_of(batch)
        u, v, rng = project_erp(p_cam, H, W, band)
        ok = slice_validity(p_cam, H, W, band, self.max_dep)
        cell_valid = ok.any(-1) & (xy.norm(dim=-1) >= self.blind)[None]
        return dict(uv=torch.stack([u, v], -1), rng=rng, ok=ok, cell_valid=cell_valid, xy_m=xy, hw=(H, W))

    def placement(self, batch):
        """Cell centres in virtual BEV pixels, xy (B, n, n, 2), and the valid cells (B, n, n) bool."""
        geo = self.geometry(batch)
        B = geo["cell_valid"].shape[0]
        xy = metric_to_bev_px(geo["xy_m"], self.n_bev, self.cell_bev).reshape(1, self.nq, self.nq, 2).expand(B, -1, -1, -1)
        return xy.contiguous(), geo["cell_valid"].reshape(B, self.nq, self.nq)

    # ---- the lift ---------------------------------------------------------------------------------------------

    def forward(self, batch, matcher):
        erp = batch["erp"]
        if erp.shape[1] != 1:
            raise ValueError("PillarQuery is single-frame; got T=%d" % erp.shape[1])
        with torch.no_grad():
            f = matcher.model.encoder(erp[:, 0])[16]
        geo = self.geometry(batch)
        H, W = geo["hw"]
        s = sample_pillars(f.float(), geo["uv"], H, W)                              # (B, N, M, C)
        B, N, M, C = s.shape
        ze = self.z_embed[None, None].expand(B, N, M, -1)
        logits = self.scorer(torch.cat([s, ze], -1))[..., 0]                         # (B, N, M)
        ok = geo["ok"]
        keep = ok | ~ok.any(-1, keepdim=True)                                        # a cell with no image keeps all (no NaN)
        logits = logits.masked_fill(~keep, float("-inf"))
        wts = torch.softmax(logits, -1)
        pooled = (wts[..., None] * s).sum(-2)                                        # (B, N, C)   FG² Eq. 2
        f_q = pooled.reshape(B, self.nq, self.nq, C).permute(0, 3, 1, 2)
        valid = geo["cell_valid"].reshape(B, self.nq, self.nq)
        if self.head is not None:
            pos = sinusoidal_2d(geo["xy_m"].reshape(self.nq, self.nq, 2), self.head.down.out_channels, 2 * self.range)[None]
            f_q = self.head(f_q, valid, zero=False, pos=pos)
        self._last = dict(logits=logits, geo=geo)
        return f_q, valid.float()

    def height_loss(self, batch):
        """Cross-entropy of the height softmax of the LAST forward against the depth-derived slice (Row B):
        (loss, number of supervised cells). batch["depth"] (B, 1, H, W) metres along the ray is required."""
        if "depth" not in batch:
            raise KeyError("pillar height loss needs batch['depth'] (train with depth: VigorPairs(depth=True))")
        if self._last is None:
            raise RuntimeError("height_loss needs a forward pass first")
        logits, geo = self._last["logits"], self._last["geo"]
        H, W = geo["hw"]
        target, has = height_targets(geo["rng"], geo["uv"], geo["ok"], batch["depth"], H, W, self.tol_m, self.tol_frac)
        has = has & geo["cell_valid"]
        n = int(has.sum())
        if n == 0:
            return logits.sum() * 0.0, 0
        return F.cross_entropy(logits[has], target[has]), n
