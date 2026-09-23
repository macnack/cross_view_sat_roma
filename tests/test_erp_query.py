"""ERP-token query: matching on the panorama's own tokens, placement after matching (plan Task 6)."""
from __future__ import annotations

import numpy as np
import torch

from bevloc import config as C
from bevloc.model.coarse import coarse_targets
from bevloc.model.erp_query import ErpQuery, erp_placement

# R_w2c: cam x = east, cam y = -up, cam z = north
R_NORTH = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]])


def test_placement_of_the_token_looking_10m_ahead():
    n, cell, h_cam = 224, 0.25, 1.7
    H, W = 448, 896
    hh, ww = H // 16, W // 16
    xy, valid = erp_placement(hh, ww, (H, W), R_NORTH, h_cam, n, cell, max_range_m=40.0)
    assert xy.shape == (1, hh, ww, 2) and valid.shape == (1, hh, ww)
    # ERP row whose token centre looks down by atan(h/10): forward, so column = centre column
    lat = -np.arctan2(h_cam, 10.0)
    v_px = (0.5 - lat / np.pi) * H
    row, col = int(v_px // 16), ww // 2
    assert valid[0, row, col]
    u, v = xy[0, row, col].tolist()
    # expected from the token centre's own ray: azimuth lon_c (right positive), elevation lat_c (< 0)
    lon_c = ((col + 0.5) * 16 / W - 0.5) * 2 * np.pi
    lat_c = (0.5 - ((row + 0.5) * 16) / H) * np.pi
    r_m = h_cam / np.tan(-lat_c)                             # ground range along that ray
    x_m, y_m = r_m * np.cos(lon_c), -r_m * np.sin(lon_c)     # x forward, y left (right of centre -> y < 0)
    assert abs(u - (n / 2 - y_m / cell - 0.5)) < 1e-3
    assert abs(v - (n / 2 - x_m / cell - 0.5)) < 1e-3
    assert 8.0 < r_m < 12.0
    assert not valid[0, 0, col]                              # sky
    assert not valid[0, hh - 1, col] or xy[0, hh - 1, col, 1] > n / 2 - 2 / cell   # ~1 m: valid, right under the camera


def test_coarse_targets_accept_placed_query_points():
    B, hh, ww = 1, 3, 4
    H = torch.tensor([[[1.0, 0.0, 300.0], [0.0, 1.0, 400.0], [0.0, 0.0, 1.0]]])   # identity scale, shift
    xy = torch.zeros(B, hh, ww, 2)
    xy[0, 1, 2] = torch.tensor([100.0, 50.0])                 # query px -> ref px (400, 450) -> cell (col 25, row 28)
    valid = torch.zeros(B, hh, ww, dtype=torch.bool)
    valid[0, 1, 2] = True
    idx, use = coarse_targets(H, valid, query_xy=xy)
    assert bool(use[0, 1, 2]) and int(idx[0, 1, 2]) == 28 * 56 + 25
    assert int(use.sum()) == 1


def test_erp_query_forward_returns_tokens_and_validity():
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    q = ErpQuery(cfg)

    class M:
        class model:
            @staticmethod
            def encoder(x):
                return {16: torch.ones(x.shape[0], 1024, x.shape[-2] // 16, x.shape[-1] // 16)}

    batch = dict(erp=torch.rand(1, 1, 3, 448, 896), R_w2c=R_NORTH[None], se2=torch.zeros(1, 1, 3))
    f_q, frac = q(batch, M)
    assert f_q.shape == (1, 1024, 28, 56) and frac.shape == (1, 28, 56)
    assert 0.2 < float(frac.mean()) < 0.6                    # roughly the lower half, minus the far ring
    xy, valid = q.placement(batch)
    assert torch.equal(valid, frac > 0.5)


def test_match_placed_recovers_an_injected_similarity():
    """Synthetic gm_cls: every valid token votes for the reference cell its placed point maps to under
    a known similarity (rotation 10 deg, scale 1, shift). The consensus must return that transform."""
    from bevloc.match import satroma as S
    n, cells = 224, 56
    hh, ww = 6, 8
    rng = np.random.default_rng(0)
    xy = torch.from_numpy(rng.uniform(20, 200, (hh, ww, 2)).astype(np.float32))       # placed query px
    valid = torch.ones(hh, ww, dtype=torch.bool)
    valid[0, :] = False                                                                # a "sky" row
    th = np.radians(10.0)
    H_true = np.array([[np.cos(th), -np.sin(th), 300.0], [np.sin(th), np.cos(th), 250.0], [0, 0, 1.0]])
    p = np.c_[xy.numpy().reshape(-1, 2), np.ones(hh * ww)] @ H_true.T                  # ref px
    col = np.floor((p[:, 0] + 0.5) / 16).astype(int)
    row = np.floor((p[:, 1] + 0.5) / 16).astype(int)
    gm = torch.full((cells * cells, hh, ww), -8.0)
    for k in range(hh * ww):
        r, c = divmod(k, ww)
        if 0 <= col[k] < cells and 0 <= row[k] < cells:
            gm[row[k] * cells + col[k], r, c] = 8.0

    class FakeMatcher:                       # no checkpoint needed: consensus_from_gm only reads these fields
        def __init__(self):
            self.m = type("m", (), {"im_a_size": n, "im_b_size": 896})()
            self.use_means, self.reproj, self.seed, self.solver = False, 3.0, 0, "srt"

    res = S.SatRoMa.consensus_from_gm(FakeMatcher(), gm, xy, valid)
    assert res.H is not None
    err = np.abs(res.H - H_true)
    assert err[:2, :2].max() < 0.02 and err[:2, 2].max() < 8.0      # within half a reference cell
    assert res.n_patches == int(valid.sum())
