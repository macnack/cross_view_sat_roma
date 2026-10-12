"""pillar query (task 07): FG²-style pillar lift geometry, sampling, depth-taught height targets, the module, wiring."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from bevloc.data.vigor import R_NORTH
from bevloc.model.erp_query import rays_at
from bevloc.model.pillar_query import (
    grid_xy, height_targets, pillar_points, project_erp, sample_pillars, slice_validity,
)

R_N = torch.from_numpy(R_NORTH.copy())[None]          # (1, 3, 3) world ENU -> camera (x east, y down, z north)
H, W = 448, 896
h, w = H // 16, W // 16


# ---- (a) ERP projection: the exact inverse of rays_at ------------------------------------------------------------

def test_project_erp_inverts_rays_at_on_the_full_sphere():
    g = torch.Generator().manual_seed(0)
    d = torch.randn(500, 3, generator=g)
    d = d / d.norm(dim=-1, keepdim=True)
    u, v, rng = project_erp(d * 7.0, H, W)
    assert torch.allclose(rng, torch.full((500,), 7.0), atol=1e-5)
    assert bool((u >= 0).all()) and bool((u <= W).all()) and bool((v >= 0).all()) and bool((v <= H).all())
    back, _ = rays_at(u, v, H, W)
    assert torch.allclose(back, d, atol=1e-5)


def test_project_erp_inverts_rays_at_inside_an_elevation_band():
    band = (0.30, -0.45)                                   # lat_top, lat_bottom (radians), the erp_band convention
    Hb = 192
    g = torch.Generator().manual_seed(1)
    u0 = torch.rand(300, generator=g) * W
    v0 = torch.rand(300, generator=g) * Hb
    d, _ = rays_at(u0, v0, Hb, W, band)
    u, v, _ = project_erp(d, Hb, W, band)
    assert torch.allclose(u, u0, atol=1e-3) and torch.allclose(v, v0, atol=1e-3)


def test_ground_slice_of_the_cell_10m_ahead_hits_the_centre_column_below_the_horizon():
    hc = 2.5
    xy = torch.tensor([[10.0, 0.0], [0.0, -10.0]])        # 10 m ahead (north); 10 m to the right (east, y = -10)
    p_cam = pillar_points(xy, [0.0], hc, R_N)              # (B, N, M, 3), ground slice only
    assert p_cam.shape == (1, 2, 1, 3)
    u, v, rng = project_erp(p_cam, H, W)
    assert abs(float(u[0, 0, 0]) - W / 2) < 1e-4                         # forward = centre column
    assert abs(float(v[0, 0, 0]) - (0.5 + math.atan(hc / 10.0) / math.pi) * H) < 1e-3   # below the horizon
    assert abs(float(rng[0, 0, 0]) - math.hypot(10.0, hc)) < 1e-4
    assert abs(float(u[0, 1, 0]) - 3 * W / 4) < 1e-4                     # east = 3/4 of the width


def test_slices_above_ground_rise_in_the_image_and_a_yawed_camera_shifts_the_column():
    xy = torch.tensor([[10.0, 0.0]])
    z = [0.0, 5.0, 20.0]
    u, v, _ = project_erp(pillar_points(xy, z, 2.5, R_N), H, W)
    assert float(v[0, 0, 0]) > float(v[0, 0, 1]) > float(v[0, 0, 2])     # higher slice = smaller row
    assert float(v[0, 0, 1]) < H / 2 < float(v[0, 0, 0])                  # 5 m above ground is above the horizon
    yaw = torch.tensor([[math.cos(math.pi / 2), -math.sin(math.pi / 2), 0.0],
                        [math.sin(math.pi / 2), math.cos(math.pi / 2), 0.0], [0.0, 0.0, 1.0]])
    R_yawed = R_N[0] @ yaw                                 # the camera turned 90 deg: world north is now at its left
    u2, _, _ = project_erp(pillar_points(xy, [0.0], 2.5, R_yawed[None], ego=False), H, W)
    assert abs(float(u2[0, 0, 0]) - W / 4) < 1e-3          # a WORLD-frame point 10 m north lands 1/4 width (left)


def test_grid_xy_is_the_cell_centres_convention_row0_forward_col0_left():
    xy = grid_xy(35, 2.0)                                  # (n, n, 2) metres, x forward / y left
    assert xy.shape == (35, 35, 2)
    assert torch.allclose(xy[0, 17], torch.tensor([34.0, 0.0])) and torch.allclose(xy[17, 0], torch.tensor([0.0, 34.0]))
    assert torch.allclose(xy[12, 17], torch.tensor([10.0, 0.0]))


# ---- (b) sampling the token map at the projected points ------------------------------------------------------

def _token_of(u, v):
    return int(v // 16), int(u // 16)


def test_sample_pillars_reads_the_token_under_the_ground_point():
    xy = torch.tensor([[10.0, 0.0]])
    p_cam = pillar_points(xy, [0.0, 3.0], 2.5, R_N)
    u, v, _ = project_erp(p_cam, H, W)
    i, j = _token_of(float(u[0, 0, 0]), float(v[0, 0, 0]))
    f = torch.zeros(1, 4, h, w)
    f[0, 0, i, j] = 1.0
    s = sample_pillars(f, torch.stack([u, v], -1), H, W)   # (B, N, M, C)
    assert s.shape == (1, 1, 2, 4)
    assert float(s[0, 0, 0, 0]) >= 0.25 and float(s[0, 0, 1, 0]) < 0.25   # the 3 m slice looks at another token


def test_sample_pillars_wraps_the_azimuth_seam():
    f = torch.zeros(1, 1, h, w)
    f[0, 0, 5, 0] = 1.0                                    # column 0 = azimuth -180 deg
    uv = torch.tensor([[[[0.2, 5 * 16 + 8.0]], [[W - 0.2, 5 * 16 + 8.0]]]])   # (1, 2, 1, 2): just inside both edges
    s = sample_pillars(f, uv, H, W)
    assert float(s[0, 0, 0, 0]) > 0.4 and float(s[0, 1, 0, 0]) > 0.4   # the far edge sees column 0 through the wrap


def test_slice_validity_masks_the_ego_cut_and_the_band():
    xy = torch.tensor([[2.0, 0.0], [30.0, 0.0]])
    p_cam = pillar_points(xy, [0.0, 10.0], 2.5, R_N)
    ok = slice_validity(p_cam, H, W, band=None, max_depression_deg=20.0)
    assert ok.shape == (1, 2, 2)
    assert not ok[0, 0, 0] and ok[0, 0, 1] and ok[0, 1, 0] and ok[0, 1, 1]   # ground 2 m ahead is 51 deg down
    assert bool(slice_validity(p_cam, H, W, None, 90.0).all())
    okb = slice_validity(p_cam, 192, W, band=(0.30, -0.05), max_depression_deg=90.0)
    assert not okb[0, 1, 0] and okb[0, 1, 1]                               # ground 30 m ahead (-4.8 deg) is below -2.9 deg


# ---- (c) depth-taught height targets ------------------------------------------------------------------------

def test_height_targets_pick_the_slice_whose_range_matches_the_depth():
    xy = grid_xy(9, 4.0).reshape(-1, 2)
    z = [0.0, 1.0, 3.0, 8.0, 15.0]
    p_cam = pillar_points(xy, z, 2.5, R_N)
    u, v, rng = project_erp(p_cam, H, W)
    uv = torch.stack([u, v], -1)
    ok = slice_validity(p_cam, H, W, None, 90.0)
    k = 2                                                  # render slice 2's range into the depth map at its pixels
    depth = torch.full((1, 1, H, W), 60.0)
    for n in range(uv.shape[1]):
        c, r = int(uv[0, n, k, 0]) % W, int(uv[0, n, k, 1])   # u = W is the seam (the cell straight behind)
        depth[0, 0, r, c] = rng[0, n, k]
    tgt, has = height_targets(rng, uv, ok, depth, H, W, tol_m=1.0, tol_frac=0.1)
    assert tgt.shape == (1, 81) and has.shape == (1, 81)
    assert bool(has.all()) and bool((tgt == k).all())
    far = torch.full((1, 1, H, W), 60.0)                   # 60 m everywhere: no slice agrees anywhere
    _, has2 = height_targets(rng, uv, ok, far, H, W, tol_m=1.0, tol_frac=0.1)
    assert not has2.any()
