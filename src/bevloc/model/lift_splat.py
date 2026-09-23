"""Camera-only spherical Lift-Splat: sat493m ERP tokens -> soft-depth BEV -> decoder tokens.

The frozen aerial encoder sees the panorama (not a smeared RGB BEV). A small depth head
lifts each token along its ray; features soft-splat into the metric BEV; a thin CNN
emits the (B, 1024, 14, 14) query the Sat-RoMa decoder expects. No LiDAR.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SphericalLiftSplat(nn.Module):
    def __init__(self, in_dim=1024, dim=128, depth_bins=16, d_min=1.0, d_max=40.0,
                 n=224, cell=0.25, out_dim=1024, patch=16, max_elev_deg=-5.0):
        super().__init__()
        self.n, self.cell, self.patch = int(n), float(cell), int(patch)
        self.out_dim = int(out_dim)
        self.max_elev = math.radians(float(max_elev_deg))
        self.register_buffer(
            "depths", torch.linspace(float(d_min), float(d_max), int(depth_bins)), persistent=False)
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_dim, dim, 1, bias=False),
            nn.GroupNorm(math.gcd(32, dim), dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, depth_bins, 1),
        )
        self.feat_proj = nn.Conv2d(in_dim, dim, 1, bias=False)
        # 224 -> 112 -> 56 -> 28 -> 14
        ch = [dim, 128, 256, 512, out_dim]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1, bias=False),
                       nn.GroupNorm(math.gcd(32, b), b), nn.SiLU(inplace=True)]
        self.bev_head = nn.Sequential(*layers)

    def token_rays(self, h, w, H, W, device, dtype):
        """Unit directions in the camera frame for every ERP token centre.

        Camera: x right, y down, z forward. ERP: centre = forward, azimuth to the right,
        top = zenith. Returns (h*w, 3) and elevation (h*w,) with +up.
        """
        jj, ii = torch.meshgrid(torch.arange(w, device=device, dtype=dtype),
                                torch.arange(h, device=device, dtype=dtype), indexing="xy")
        u = (jj + 0.5) * self.patch
        v = (ii + 0.5) * self.patch
        lon = (u / W - 0.5) * (2 * math.pi)
        lat = (0.5 - v / H) * math.pi
        cl = torch.cos(lat)
        dir_cam = torch.stack([torch.sin(lon) * cl, -torch.sin(lat), torch.cos(lon) * cl], -1)
        return dir_cam.reshape(-1, 3), lat.reshape(-1)

    def ego_xy(self, dir_cam, R_w2c, depths):
        """Ray directions -> ego (forward, left) metres at each depth.

        R_w2c: (B, 3, 3) world ENU -> camera (``p_c = R @ p_w``). Returns x,y (B, N, D).
        """
        R_c2w = R_w2c.transpose(-1, -2)
        dir_w = torch.einsum("bij,nj->bni", R_c2w, dir_cam)   # p_w = R_c2w @ p_c
        fwd = R_c2w[:, :, 2].clone()                          # camera +z in world
        fwd[:, 2] = 0.0
        fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        left = torch.stack([-fwd[:, 1], fwd[:, 0], torch.zeros_like(fwd[:, 0])], -1)
        x = (dir_w * fwd[:, None, :]).sum(-1)[..., None] * depths
        y = (dir_w * left[:, None, :]).sum(-1)[..., None] * depths
        return x, y

    def splat(self, feat, alpha, x, y, elev_ok):
        """Soft-splat (B, C, N) features with (B, N, D) weights into a (B, C, n, n) BEV."""
        B, C, N = feat.shape
        D, n, cell = alpha.shape[-1], self.n, self.cell
        w = alpha * elev_ok[:, :, None].to(alpha.dtype)
        xf = x.reshape(B, -1)
        yf = y.reshape(B, -1)
        wf = w.reshape(B, -1)
        ff = feat[:, :, :, None].expand(B, C, N, D).reshape(B, C, -1)
        r = torch.floor(n / 2 - xf / cell).long()
        c = torch.floor(n / 2 - yf / cell).long()
        ok = (r >= 0) & (r < n) & (c >= 0) & (c < n) & (wf > 1e-8)
        bev = feat.new_zeros(B, C, n * n)
        mass = feat.new_zeros(B, n * n)
        for b in range(B):
            m = ok[b]
            if not m.any():
                continue
            idx = (r[b, m] * n + c[b, m])
            wb = wf[b, m]
            bev[b].index_add_(1, idx, ff[b, :, m] * wb)
            mass[b].index_add_(0, idx, wb)
        bev = bev / mass.clamp_min(1e-6).unsqueeze(1)
        return bev.view(B, C, n, n), (mass > 1e-6).view(B, n, n)

    @staticmethod
    def apply_se2(x, y, se2):
        """Map source-ego (x,y) into query-ego. se2: (B, 3) = (yaw, tx, ty)."""
        yaw, tx, ty = se2[:, 0], se2[:, 1], se2[:, 2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        # (B, N, D)
        xq = c[:, None, None] * x - s[:, None, None] * y + tx[:, None, None]
        yq = s[:, None, None] * x + c[:, None, None] * y + ty[:, None, None]
        return xq, yq

    def forward(self, f_erp, R_w2c, erp_hw, se2=None):
        """f_erp: (B, 1024, h, w); R_w2c: (B, 3, 3); erp_hw: (H, W) of the encoded ERP.

        Optional se2 (B, 3)=(yaw, tx, ty) remaps source-ego splat into the query ego frame.
        Returns f_q (B, 1024, 14, 14) and patch_frac (B, 14, 14) in [0, 1].
        """
        B, _, h, w = f_erp.shape
        H, W = int(erp_hw[0]), int(erp_hw[1])
        alpha = self.depth_head(f_erp).softmax(1).permute(0, 2, 3, 1).reshape(B, h * w, -1)
        feat = self.feat_proj(f_erp).reshape(B, -1, h * w)
        dir_cam, elev = self.token_rays(h, w, H, W, f_erp.device, f_erp.dtype)
        x, y = self.ego_xy(dir_cam, R_w2c, self.depths.to(dtype=f_erp.dtype))
        if se2 is not None:
            x, y = self.apply_se2(x, y, se2)
        elev_ok = elev <= self.max_elev
        bev, valid = self.splat(feat, alpha, x, y, elev_ok[None].expand(B, -1))
        f_q = self.bev_head(bev)
        frac = F.avg_pool2d(valid.float()[:, None], self.patch)[:, 0]
        return f_q, frac

    def forward_multiframe(self, f_erp, R_w2c, se2, erp_hw):
        """Fuse T source frames into one query BEV.

        f_erp: (B, T, C, h, w); R_w2c: (B, T, 3, 3); se2: (B, T, 3) with (yaw, tx, ty)
        of each source origin in the query ego frame (identity for the query itself).
        Soft-splat mass is accumulated across frames before the BEV head.
        """
        B, T, C, h, w = f_erp.shape
        H, W = int(erp_hw[0]), int(erp_hw[1])
        device, dtype = f_erp.device, f_erp.dtype
        dir_cam, elev = self.token_rays(h, w, H, W, device, dtype)
        elev_ok = elev <= self.max_elev
        depths = self.depths.to(dtype=dtype)
        n, cell = self.n, self.cell
        bev = f_erp.new_zeros(B, self.feat_proj.out_channels, n * n)
        mass = f_erp.new_zeros(B, n * n)
        for t in range(T):
            ft = f_erp[:, t]
            alpha = self.depth_head(ft).softmax(1).permute(0, 2, 3, 1).reshape(B, h * w, -1)
            feat = self.feat_proj(ft).reshape(B, -1, h * w)
            x, y = self.ego_xy(dir_cam, R_w2c[:, t], depths)
            x, y = self.apply_se2(x, y, se2[:, t])
            Cdim = feat.shape[1]
            wf = (alpha * elev_ok[None, :, None].to(dtype)).reshape(B, -1)
            xf, yf = x.reshape(B, -1), y.reshape(B, -1)
            ff = feat[:, :, :, None].expand(B, Cdim, h * w, alpha.shape[-1]).reshape(B, Cdim, -1)
            r = torch.floor(n / 2 - xf / cell).long()
            c = torch.floor(n / 2 - yf / cell).long()
            ok = (r >= 0) & (r < n) & (c >= 0) & (c < n) & (wf > 1e-8)
            for b in range(B):
                m = ok[b]
                if not m.any():
                    continue
                idx = r[b, m] * n + c[b, m]
                bev[b].index_add_(1, idx, ff[b, :, m] * wf[b, m])
                mass[b].index_add_(0, idx, wf[b, m])
        bev = (bev / mass.clamp_min(1e-6).unsqueeze(1)).view(B, -1, n, n)
        valid = (mass > 1e-6).view(B, n, n)
        f_q = self.bev_head(bev)
        frac = F.avg_pool2d(valid.float()[:, None], self.patch)[:, 0]
        return f_q, frac
