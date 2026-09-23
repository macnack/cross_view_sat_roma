"""Camera + LiDAR fusion BEV encoder whose output replaces Sat-RoMa's query features.

  ERP image --frozen DINOv3 ConvNeXt + FPN--> stride-4 feature map
  LiDAR points --project into the ERP--> sample an image feature per point ("point painting"),
      append LiDAR channels, softmax-splat into the metric 224 x 224 BEV grid
  BEV CNN (+ self-attention at 14 x 14) --> (B, 1024, 14, 14) = Sat-RoMa's `f_q_pyramid[16]`

Layout follows BEV-Patch-PF (third_party/bev-patch-pf: frozen DINOv3 ConvNeXt, lift-splat with
a learned vertical-pooling weight, residual BEV encoder with attention); re-implemented here
because their mapper is pinhole + dense depth, and ours is spherical + sparse LiDAR.

LiDAR places every feature metrically, so unlike IPM this BEV does not depend on roll/pitch
beyond cos(theta) (docs/decisions.md, 2026-09-19).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
LIDAR_CHANNELS = 5          # height above road, range, reflectivity, "pixel is visible to the camera", is-a-LiDAR-return
CELL_CHANNELS = 2           # per-cell max height, log point count


def gn(c):
    return nn.GroupNorm(math.gcd(32, c), c)


class GroundBackbone(nn.Module):
    """Frozen DINOv3 ConvNeXt (LVD weights: ground-level imagery) + a small trainable FPN."""

    def __init__(self, name="convnext_tiny.dinov3_lvd1689m", dim=128):
        super().__init__()
        import timm
        self.net = timm.create_model(name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3)).eval()
        for p in self.net.parameters():
            p.requires_grad = False
        ch = self.net.feature_info.channels()
        self.lateral = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in ch)
        self.smooth = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False), gn(dim), nn.SiLU(inplace=True))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, erp):
        """erp: (B, 3, H, W) in [0, 1] -> (B, dim, H/4, W/4)."""
        with torch.no_grad():
            feats = self.net((erp - self.mean) / self.std)
        x = self.lateral[-1](feats[-1])
        for lat, f in zip(reversed(self.lateral[:-1]), reversed(feats[:-1])):
            x = lat(f) + F.interpolate(x, size=f.shape[-2:], mode="bilinear", align_corners=False)
        return self.smooth(x)


class PointPaintSplat(nn.Module):
    """Sample an image feature for every LiDAR return and pool the returns of each BEV cell."""

    def __init__(self, dim, n=224, cell=0.25, lidar_height=1.57, R_cl=None, t_cl=None, ground_fill=True):
        super().__init__()
        self.n, self.cell, self.lidar_height = n, cell, float(lidar_height)
        self.ground_fill = ground_fill
        self.register_buffer("R_cl", torch.eye(3) if R_cl is None else torch.as_tensor(R_cl, dtype=torch.float32))
        self.register_buffer("t_cl", torch.zeros(3) if t_cl is None else torch.as_tensor(t_cl, dtype=torch.float32))
        c = dim + LIDAR_CHANNELS
        self.weight = nn.Sequential(nn.Linear(c, c // 2), nn.SiLU(inplace=True), nn.Linear(c // 2, 1))
        self.out_channels = c + CELL_CHANNELS

    def erp_grid(self, xyz):
        """LiDAR-frame points -> grid_sample coordinates in the ERP (bevloc.bev.spherical convention)."""
        p = xyz @ self.R_cl.T + self.t_cl
        rng = p.norm(dim=1).clamp_min(1e-6)
        az = torch.atan2(-p[:, 1], p[:, 0])
        el = torch.asin((p[:, 2] / rng).clamp(-1, 1))
        return torch.stack([az / math.pi, -2.0 * el / math.pi], 1), rng     # x: u/W*2-1, y: v/H*2-1

    def cell_index(self, xyz_bev):
        """xyz_bev: (N, 3) in the TARGET BEV's own level frame (ego-centred). Returns the flat
        (row*n+col) index and an in-bounds mask."""
        n = self.n
        r = torch.floor(n / 2 - xyz_bev[:, 0] / self.cell).long()
        c = torch.floor(n / 2 - xyz_bev[:, 1] / self.cell).long()
        ok = (r >= 0) & (r < n) & (c >= 0) & (c < n)
        return r * n + c, ok

    def point_vectors(self, feat_b, erp_valid, xyz_cam, refl, is_lidar=1.0):
        """Per-point feature vector, sampled from ONE frame's own image feature map.
        xyz_cam: (N, 3) in THAT frame's own camera-relative coordinates (i.e. what its ERP
        actually shows) -- this must NOT be the pose-transformed BEV-frame xyz used for
        placement, or the image sample would come from the wrong direction. feat_b: (C, h, w).
        Returns x (N, dim+5) = [img*seen, height/3, range/30, reflectivity/255, seen, is_lidar]."""
        g, rng = self.erp_grid(xyz_cam)
        g = g.view(1, -1, 1, 2)
        seen = F.grid_sample(erp_valid, g, mode="nearest", align_corners=False).view(-1, 1)
        img = F.grid_sample(feat_b[None], g, mode="bilinear", align_corners=False).view(feat_b.shape[0], -1).T
        height = xyz_cam[:, 2:3] + self.lidar_height
        return torch.cat([img * seen, height / 3.0, rng[:, None] / 30.0, refl[:, None] / 255.0, seen,
                          torch.full_like(seen, float(is_lidar))], 1).float()

    def ground_points(self, xyz_cam, erp_valid, az_bins=720, min_obj=0.3, max_obj=3.0, margin=0.5, max_range=None):
        """Virtual ground points for the road the camera sees but the LiDAR does not return from
        (wet asphalt: ~no returns on the carriageway, so a LiDAR-only BEV has no ground at all).
        One point per BEV cell centre, on the plane z = -lidar_height, kept only if (a) it lies in
        front of the first above-ground LiDAR return in its azimuth bin (the LiDAR's own
        occlusion boundary = the contact line, minus `margin`), and (b) its pixel is not ego
        vehicle / seam. Flat-ground IPM in feature space; roll = pitch = 0 as everywhere else."""
        n, cell, dev = self.n, self.cell, xyz_cam.device
        c = (n / 2 - torch.arange(n, device=dev, dtype=torch.float32) - 0.5) * cell
        gx, gy = torch.meshgrid(c, c, indexing="ij")
        g = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((n * n,), -self.lidar_height, device=dev)], 1)
        rho = g[:, :2].norm(dim=1)

        h = xyz_cam[:, 2] + self.lidar_height
        obj = xyz_cam[(h > min_obj) & (h < max_obj)]
        first = torch.full((az_bins,), float("inf"), device=dev)
        if len(obj):
            b = ((torch.atan2(obj[:, 1], obj[:, 0]) / (2 * math.pi) + 0.5) * az_bins).long().clamp(0, az_bins - 1)
            first = first.scatter_reduce(0, b, obj[:, :2].norm(dim=1), "amin", include_self=True)
            first = torch.minimum(first, torch.minimum(first.roll(1), first.roll(-1)))     # +-1 bin
        gb = ((torch.atan2(g[:, 1], g[:, 0]) / (2 * math.pi) + 0.5) * az_bins).long().clamp(0, az_bins - 1)
        ok = (rho < first[gb] - margin) & (rho > 1.0) & (rho < (max_range or n * cell / 2))
        uv, _ = self.erp_grid(g)
        seen = F.grid_sample(erp_valid, uv.view(1, -1, 1, 2), mode="nearest", align_corners=False).view(-1)
        return g[ok & (seen > 0)]

    def pool_and_scatter(self, x, idx, height3):
        """x: (N, dim+4) per-point vectors (any number of source frames, already concatenated);
        idx: (N,) target BEV cell; height3: (N, 1) height/3 for the max-height channel.
        One softmax-weighted pool per cell, over however many points (one frame or several)
        landed there. Returns flat (n*n, out_channels) and (n*n, 1) mask."""
        n = self.n
        w = self.weight(x)
        w = torch.exp(w - w.max())
        wsum = torch.zeros(n * n, 1, device=x.device).scatter_add_(0, idx[:, None], w)
        pooled = torch.zeros(n * n, x.shape[1], device=x.device)
        pooled.scatter_add_(0, idx[:, None].expand(-1, x.shape[1]), x * w / wsum[idx].clamp_min(1e-6))
        top = torch.zeros(n * n, 1, device=x.device).scatter_reduce_(
            0, idx[:, None], height3.clamp(0, 3).float(), "amax", include_self=True)
        cnt = torch.zeros(n * n, 1, device=x.device).scatter_add_(0, idx[:, None], torch.ones_like(w))
        return torch.cat([pooled, top, torch.log1p(cnt) / 5.0], 1), (cnt > 0).float()

    def _frame(self, feat_b, erp_valid, xyz_cam, refl, xyz_bev=None, to_bev=None):
        """x, flat cell index and height/3 for one source frame: its LiDAR returns plus (optionally)
        its camera-visible ground. xyz_bev: the returns already expressed in the target BEV frame
        (None = same frame); to_bev: the same relative pose as a function, needed only so that the
        frame's virtual ground points can follow it (None = ground fill for the query frame only)."""
        bev_xyz = xyz_cam if xyz_bev is None else xyz_bev
        cam, place, rf = [xyz_cam], [bev_xyz], [refl]
        lid = [torch.ones(len(xyz_cam), device=xyz_cam.device)]
        if self.ground_fill and (xyz_bev is None or to_bev is not None):
            gp = self.ground_points(xyz_cam, erp_valid)
            cam.append(gp); place.append(gp if to_bev is None else to_bev(gp))
            rf.append(torch.zeros(len(gp), device=gp.device)); lid.append(torch.zeros(len(gp), device=gp.device))
        cam, place, rf, lid = torch.cat(cam), torch.cat(place), torch.cat(rf), torch.cat(lid)
        idx, ok = self.cell_index(place)
        x = self.point_vectors(feat_b, erp_valid, cam[ok], rf[ok])
        x[:, -1] = lid[ok]
        return x, idx[ok], (place[ok][:, 2:3] + self.lidar_height) / 3.0

    def forward(self, feat, erp_valid, points, reflectivity):
        """feat (B, C, h, w); erp_valid (1, 1, H, W) float; points / reflectivity: lists of (N, 3) / (N,).
        Single frame per batch item: camera-relative xyz IS the BEV-level xyz (no relative pose).
        Returns bev (B, C + 7, n, n) and mask (B, 1, n, n)."""
        n, out, masks = self.n, [], []
        for b, (xyz, refl) in enumerate(zip(points, reflectivity)):
            pooled, m = self.pool_and_scatter(*self._frame(feat[b], erp_valid, xyz, refl))
            out.append(pooled.T.reshape(-1, n, n))
            masks.append(m.reshape(1, n, n))
        return torch.stack(out), torch.stack(masks)

    def forward_multiframe(self, frames):
        """A single BEV query built from SEVERAL source frames (a short driven sequence), each
        contributing its own LiDAR returns and its own image features -- unlike `forward`, which
        assumes one frame IS the BEV origin, this lets frame k's points be placed by a relative
        pose while still sampling frame k's own camera.

        frames: list of dicts, one per source frame (query frame included, T_rel=identity for it):
          feat     (C, h, w)  that frame's OWN backbone output
          erp_valid (1, 1, H, W)
          xyz_cam  (N, 3)     that frame's LiDAR returns, in ITS OWN camera-relative frame
                              (used for image sampling -- must not be pose-transformed)
          xyz_bev  (N, 3)     the SAME points transformed into the QUERY's level frame by the
                              relative pose (used only for BEV cell placement)
          refl     (N,)
          to_bev   optional callable xyz -> xyz, the same relative pose; lets this frame's
                              camera-visible GROUND fill follow it (without it, only the query
                              frame contributes ground)
        Returns bev (1, C + 6, n, n), mask (1, 1, n, n) -- same shape as one `forward` item.
        """
        xs, idxs, heights = [], [], []
        for fr in frames:
            same = fr["xyz_bev"] is fr["xyz_cam"] or torch.equal(fr["xyz_bev"], fr["xyz_cam"])
            x, idx, h3 = self._frame(fr["feat"], fr["erp_valid"], fr["xyz_cam"], fr["refl"],
                                     None if same else fr["xyz_bev"], fr.get("to_bev"))
            xs.append(x); idxs.append(idx); heights.append(h3)
        x, idx, height3 = torch.cat(xs), torch.cat(idxs), torch.cat(heights)
        pooled, m = self.pool_and_scatter(x, idx, height3)
        return pooled.T.reshape(1, -1, self.n, self.n), m.reshape(1, 1, self.n, self.n)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), gn(cout), nn.SiLU(inplace=True),
                               nn.Conv2d(cout, cout, 3, 1, 1, bias=False), gn(cout))
        self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), gn(cout))

    def forward(self, x):
        return F.silu(self.a(x) + self.skip(x))


class BevEncoder(nn.Module):
    """224 x 224 BEV -> 14 x 14 tokens in the matcher's feature space (stride 16, like its ViT)."""

    def __init__(self, cin, dims=(128, 192, 256, 384), out_dim=1024, n_attn=2, tokens=14):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin + 1, dims[0], 3, padding=1, bias=False), gn(dims[0]), nn.SiLU(inplace=True))
        self.stages = nn.Sequential(*[ResBlock(a, b, 2) for a, b in zip((dims[0],) + dims[:-1], dims)])
        self.pos = nn.Parameter(torch.zeros(1, tokens * tokens, dims[-1]))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.attn = nn.ModuleList(nn.TransformerEncoderLayer(dims[-1], 8, dims[-1] * 2, dropout=0.0, activation="gelu",
                                                             batch_first=True, norm_first=True) for _ in range(n_attn))
        self.head = nn.Linear(dims[-1], out_dim)
        self.norm = nn.LayerNorm(out_dim)       # the decoder was trained on LayerNorm'ed ViT tokens

    def forward(self, bev, mask):
        x = self.stages(self.stem(torch.cat([bev, mask], 1)))
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2) + self.pos
        for blk in self.attn:
            t = blk(t)
        return self.norm(self.head(t)).transpose(1, 2).reshape(b, -1, h, w)


class FusionBEV(nn.Module):
    def __init__(self, n=224, cell=0.25, lidar_height=1.57, R_cl=None, t_cl=None,
                 backbone="convnext_tiny.dinov3_lvd1689m", dim=128, out_dim=1024, patch=16, ground_fill=True):
        super().__init__()
        self.patch = patch
        self.backbone = GroundBackbone(backbone, dim)
        self.lift = PointPaintSplat(dim, n, cell, lidar_height, R_cl, t_cl, ground_fill)
        self.encoder = BevEncoder(self.lift.out_channels, out_dim=out_dim, tokens=n // patch)

    @classmethod
    def from_config(cls, cfg, calib):
        m = cfg.fusion
        return cls(cfg.grid.n, cfg.grid.cell_m, calib.lidar_height, calib.R_cl, calib.t_cl,
                   backbone=m.backbone, dim=m.dim, ground_fill=getattr(m, "ground_fill", True))

    def forward(self, erp, erp_valid, points, reflectivity):
        """Returns query features (B, 1024, 14, 14), per-patch valid fraction (B, 14, 14), BEV mask."""
        bev, mask = self.lift(self.backbone(erp), erp_valid, points, reflectivity)
        return self.encoder(bev, mask), F.avg_pool2d(mask, self.patch)[:, 0], mask

    def forward_sequence(self, frames, erp_valid):
        """One query built from several source frames (a short driven sequence) instead of one.
        frames: list of dicts {erp (3, H, W), xyz_cam (N, 3), xyz_bev (N, 3), refl (N,)} --
        see PointPaintSplat.forward_multiframe for what xyz_cam/xyz_bev mean. Runs the (frozen)
        backbone once per frame, then one combined splat and the shared encoder. Batch size 1
        only: a sequence is inherently one query, not a batch dimension."""
        for fr in frames:
            fr["feat"] = self.backbone(fr["erp"][None])[0]
            fr["erp_valid"] = erp_valid
        bev, mask = self.lift.forward_multiframe(frames)
        return self.encoder(bev, mask), F.avg_pool2d(mask, self.patch)[:, 0], mask
