"""Unit tests for spherical Lift-Splat geometry (no GPU / checkpoint needed)."""
from __future__ import annotations

import torch

from bevloc.model.lift_splat import SphericalLiftSplat


def test_token_rays_centre_is_forward():
    m = SphericalLiftSplat(depth_bins=4, n=8, cell=1.0, dim=8, out_dim=16, patch=16)
    # 4x8 tokens on a 64x128 ERP: token (1, 3) and (1, 4) / (2, 3) and (2, 4) straddle the centre.
    dir_cam, elev = m.token_rays(4, 8, H=64, W=128, device="cpu", dtype=torch.float32)
    mid = dir_cam.reshape(4, 8, 3)[1:3, 3:5].mean(0).mean(0)
    assert mid[2] > 0.8
    assert abs(float(mid[0])) < 0.2
    assert abs(float(elev.reshape(4, 8)[1:3, 3:5].mean())) < 0.25


def test_ego_xy_forward_depth_lands_ahead():
    m = SphericalLiftSplat(depth_bins=4, d_min=4, d_max=16, n=8, cell=1.0, dim=8, out_dim=16)
    # R_w2c @ world = cam. cam_x=east, cam_y=-up, cam_z=north.
    R = torch.tensor([[[1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0]]])
    dir_cam = torch.tensor([[0.0, 0.0, 1.0]])
    x, y = m.ego_xy(dir_cam, R, m.depths)
    assert torch.allclose(x[0, 0], m.depths, atol=1e-4)
    assert torch.allclose(y[0, 0], torch.zeros_like(m.depths), atol=1e-4)


def test_forward_shapes():
    m = SphericalLiftSplat(in_dim=32, dim=8, depth_bins=4, n=32, cell=1.0, out_dim=16, patch=16,
                           max_elev_deg=90.0)
    B, h, w = 2, 4, 8
    f = torch.randn(B, 32, h, w)
    R = torch.eye(3).expand(B, 3, 3).contiguous()
    f_q, frac = m(f, R, erp_hw=(h * 16, w * 16))
    assert f_q.shape == (B, 16, 2, 2)
    assert frac.shape == (B, 2, 2)
    assert frac.min() >= 0 and frac.max() <= 1


def test_forward_multiframe_identity_matches_single():
    """T=1 with identity SE(2) must match single-frame forward."""
    torch.manual_seed(0)
    m = SphericalLiftSplat(in_dim=32, dim=8, depth_bins=4, n=32, cell=1.0, out_dim=16, patch=16,
                           max_elev_deg=90.0)
    B, T, h, w = 1, 1, 4, 8
    f = torch.randn(B, 32, h, w)
    R = torch.eye(3).expand(B, 3, 3).contiguous()
    f_q1, frac1 = m(f, R, erp_hw=(h * 16, w * 16))
    f_q2, frac2 = m.forward_multiframe(
        f[:, None], R[:, None], torch.zeros(B, T, 3), erp_hw=(h * 16, w * 16))
    assert torch.allclose(f_q1, f_q2, atol=1e-5)
    assert torch.allclose(frac1, frac2, atol=1e-5)


def test_apply_se2_translates_forward():
    x = torch.zeros(1, 2, 1)
    y = torch.zeros(1, 2, 1)
    se2 = torch.tensor([[0.0, 5.0, 0.0]])  # 5 m forward
    xq, yq = SphericalLiftSplat.apply_se2(x, y, se2)
    assert torch.allclose(xq, torch.full_like(x, 5.0))
    assert torch.allclose(yq, torch.zeros_like(y))


def test_src_in_query_se2_ahead():
    from bevloc.data.mapillary import src_in_query_se2
    # up=0 → forward = north. Source 10 m north of query.
    yaw, tx, ty = src_in_query_se2((100.0, 200.0), 0.0, (100.0, 210.0), 0.0)
    assert abs(yaw) < 1e-9 and abs(tx - 10.0) < 1e-6 and abs(ty) < 1e-6
