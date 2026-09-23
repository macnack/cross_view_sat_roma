"""Dense hybrid query: IPM places ground features exactly, a depth head places the rest.

Ground cells (within ``ground_max_range_m``) bilinearly sample the projected ERP tokens at the
sub-patch position of their ground point (decision 2026-09-19: lift by grid_sample at sub-patch
resolution, never on the patch grid). Tokens looking above the ground band, elevation in
[min_elev_deg, max_elev_deg], are splatted along learned depth bins exactly as in
SphericalLiftSplat (walls, facades). Both land in one accumulator; the BEV head is shared.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevloc.bev.ipm_sphere import cell_centres, ground_pixel_coords
from bevloc.model.lift_splat import SphericalLiftSplat


class HybridQuery(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        L, g = cfg.lift, cfg.grid
        self.n, self.cell, self.height = int(g.n), float(g.cell_m), float(cfg.ipm.height_m)
        self.blind = float(cfg.ipm.blind_radius_m)
        self.max_range = float(getattr(L, "ground_max_range_m", g.n * g.cell_m / 2))
        # the ground band ends where a ray at max_range meets the ground
        band_lo = -math.degrees(math.atan2(self.height, self.max_range))
        min_elev = max(band_lo, float(getattr(L, "min_elev_deg", band_lo)))
        self.lift = SphericalLiftSplat(dim=L.dim, depth_bins=L.depth_bins, d_min=L.d_min, d_max=L.d_max,
                                       n=g.n, cell=g.cell_m, max_elev_deg=L.max_elev_deg,
                                       min_elev_deg=min_elev)
        x, y = cell_centres(self.n, self.cell)
        self.register_buffer("cell_x", torch.from_numpy(x).float(), persistent=False)
        self.register_buffer("cell_y", torch.from_numpy(y).float(), persistent=False)

    def load_lift_weights(self, lift_state):
        self.lift.load_state_dict(lift_state)

    def dense_ground(self, f_proj, R_w2c, erp_hw, se2=None):
        """Bilinear sample of projected tokens at every ground cell.

        f_proj (B, dim, h, w) tokens; R_w2c (B, 3, 3); se2 (B, 3) = (yaw, tx, ty) of the source
        origin in the query ego frame (None = the query itself). Returns (B, dim, n, n) features
        (zero outside the ground mask) and the (B, n, n) bool mask."""
        B = f_proj.shape[0]
        H, W = int(erp_hw[0]), int(erp_hw[1])
        x = self.cell_x[None].expand(B, -1, -1)
        y = self.cell_y[None].expand(B, -1, -1)
        if se2 is not None:                       # query-ego cell -> source-ego cell (inverse SE(2))
            yaw, tx, ty = se2[:, 0], se2[:, 1], se2[:, 2]
            xs = x - tx[:, None, None]
            ys = y - ty[:, None, None]
            c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
            x, y = c * xs + s * ys, -s * xs + c * ys
        grids, masks = [], []
        for b in range(B):
            xb, yb = x[b].detach().cpu().numpy(), y[b].detach().cpu().numpy()
            mu, mv = ground_pixel_coords(xb, yb, R_w2c[b].detach().cpu().numpy(), self.height, (H, W))
            u = torch.from_numpy(mu).to(f_proj.device) / W * 2 - 1
            v = torch.from_numpy(mv).to(f_proj.device) / H * 2 - 1
            grids.append(torch.stack([u, v], -1))
            rng = torch.hypot(x[b], y[b])
            masks.append((rng >= self.blind) & (rng <= self.max_range))
        grid = torch.stack(grids, 0).to(f_proj.dtype)
        g = F.grid_sample(f_proj, grid, mode="bilinear", padding_mode="border", align_corners=False)
        mask = torch.stack(masks, 0)
        return g * mask[:, None].to(g.dtype), mask

    def _accumulate(self, f_erp, R_w2c, erp_hw, se2=None):
        """Numerator / denominator of the shared accumulator for one source frame."""
        lift = self.lift
        B, _, h, w = f_erp.shape
        f_proj = lift.feat_proj(f_erp)
        g, gmask = self.dense_ground(f_proj, R_w2c, erp_hw, se2)
        alpha = lift.depth_head(f_erp).softmax(1).permute(0, 2, 3, 1).reshape(B, h * w, -1)
        feat = f_proj.reshape(B, -1, h * w)
        dir_cam, elev = lift.token_rays(h, w, int(erp_hw[0]), int(erp_hw[1]), f_erp.device, f_erp.dtype)
        x, y = lift.ego_xy(dir_cam, R_w2c, lift.depths.to(f_erp.dtype))
        if se2 is not None:
            x, y = lift.apply_se2(x, y, se2)
        elev_ok = ((elev <= lift.max_elev) & (elev >= lift.min_elev))[None].expand(B, -1)
        bev, valid = lift.splat(feat, alpha, x, y, elev_ok)
        mass = valid.to(g.dtype)[:, None]
        num = g + bev * mass
        den = gmask.to(g.dtype)[:, None] + mass
        return num, den

    def forward(self, batch, matcher):
        erp = batch["erp"]                                  # (B, T, 3, H, W)
        B, T = erp.shape[:2]
        hw = erp.shape[-2:]
        with torch.no_grad():
            feats = [matcher.model.encoder(erp[:, t])[16] for t in range(T)]
        num = den = None
        for t in range(T):
            se2 = None if T == 1 else batch["se2"][:, t]
            n_t, d_t = self._accumulate(feats[t], batch["R_w2c"][:, t], hw, se2)
            num = n_t if num is None else num + n_t
            den = d_t if den is None else den + d_t
        bev = num / den.clamp_min(1e-6)
        f_q = self.lift.bev_head(bev)
        frac = F.avg_pool2d((den[:, 0] > 1e-6).float()[:, None], self.lift.patch)[:, 0]
        return f_q, frac
