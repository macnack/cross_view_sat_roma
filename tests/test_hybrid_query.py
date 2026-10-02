"""Hybrid query: exact ground placement by IPM, learned placement above the horizon."""
from __future__ import annotations

import numpy as np
import torch

from bevloc import config as C
from bevloc.bev.ipm_sphere import cell_centres, ground_pixel_coords
from bevloc.model.hybrid_query import HybridQuery
from bevloc.model.lift_splat import SphericalLiftSplat

R_NORTH = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]])


def _cfg():
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.lift.dim = 8
    return cfg


def test_min_elev_band_excludes_ground_rays():
    # A ray looking 20 deg down is excluded when min_elev_deg = -3.5; a horizontal one is kept.
    # n=32 -> 2x2 output, so GroupNorm sees more than one value per group
    m = SphericalLiftSplat(in_dim=4, dim=4, depth_bins=2, n=32, cell=1.0, out_dim=8, patch=16,
                           max_elev_deg=30.0, min_elev_deg=-3.5)
    h, w = 4, 8
    f = torch.zeros(1, 4, h, w)
    f[0, 0, 3, 4] = 1.0        # bottom token row looks ~ -56 deg: ground, must NOT be splatted
    f[0, 1, 1, 4] = 1.0        # token row 1 looks ~ +11 deg: kept
    _, elev = m.token_rays(h, w, h * 16, w * 16, "cpu", torch.float32)
    elev = elev.reshape(h, w)
    assert elev[3, 4] < np.radians(-3.5) < elev[1, 4] < np.radians(30.0)
    f_q, frac = m(f, R_NORTH, erp_hw=(h * 16, w * 16))
    assert f_q.shape == (1, 8, 2, 2) and 0.0 < float(frac.mean()) < 1.0


def test_dense_ground_samples_the_token_under_the_ground_point():
    cfg = _cfg()
    q = HybridQuery(cfg)
    h, w, H, W = 28, 56, 448, 896
    f = torch.zeros(1, 8, h, w)
    n, cell = cfg.grid.n, cfg.grid.cell_m
    r, c = n // 2 - int(round(10.0 / cell)), n // 2
    x, y = cell_centres(n, cell)
    mu, mv = ground_pixel_coords(x[r:r + 1, c:c + 1], y[r:r + 1, c:c + 1], R_NORTH[0].numpy(),
                                 cfg.ipm.height_m, (H, W))
    f[0, 3, int(mv[0, 0] // 16), int(mu[0, 0] // 16)] = 1.0     # one-hot token under that cell's ground point
    g, mask = q.dense_ground(f, R_NORTH, (H, W))
    assert g.shape == (1, 8, n, n) and mask.shape == (1, n, n)
    assert mask[0, r, c]
    assert g[0, 3, r, c] >= 0.25                    # bilinear weight of the one-hot token (>= 1/4 anywhere inside it)
    assert int(g[0, :, r, c].argmax()) == 3
    assert g[0, 3].sum() > 0 and g[0, :3].abs().sum() == 0
    assert not mask[0, n // 2, n // 2]              # blind disc under the camera


def test_forward_shapes_and_valid_fraction():
    cfg = _cfg()
    q = HybridQuery(cfg)

    class M:  # fake matcher: encoder returns zeros at stride 16 in the lift's input dim
        class model:
            @staticmethod
            def encoder(x):
                return {16: torch.zeros(x.shape[0], 1024, x.shape[-2] // 16, x.shape[-1] // 16)}

    batch = dict(erp=torch.rand(1, 1, 3, 448, 896), R_w2c=R_NORTH[None], se2=torch.zeros(1, 1, 3))
    f_q, frac = q(batch, M)
    assert f_q.shape == (1, 1024, 14, 14)
    assert 0.5 < float(frac.mean()) <= 1.0            # most of the 56 m grid is ground within range


def test_load_lift_weights_from_old_checkpoint():
    cfg = _cfg()
    q = HybridQuery(cfg)
    ref = SphericalLiftSplat(dim=8, depth_bins=cfg.lift.depth_bins, d_min=cfg.lift.d_min, d_max=cfg.lift.d_max,
                             n=cfg.grid.n, cell=cfg.grid.cell_m, max_elev_deg=cfg.lift.max_elev_deg)
    q.load_lift_weights(ref.state_dict())
    assert torch.equal(q.lift.feat_proj.weight, ref.feat_proj.weight)
