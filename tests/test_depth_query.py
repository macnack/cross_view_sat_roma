"""erp_depth query (task 04, Step 2): ray geometry, depth PNGs, consistency with VigorPairs' H / en, and the query factory."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from bevloc import config as C
from bevloc.data.vigor import CITY_RES, R_NORTH, VigorPairs, collate_vigor, depth_png_path, read_depth_png
from bevloc.model.coarse import coarse_targets
from bevloc.model.depth_query import (
    ErpDepthQuery, ProjectionHead, depth_placement, depth_placement_metric, sample_token_depth,
)
from bevloc.model.query import build_query, load_query_state

R_N = torch.from_numpy(R_NORTH.copy())[None]          # (1, 3, 3) world ENU -> camera (x east, y down, z north)
FAR = 60.0                                            # beyond max_depth: invalid


def _token_depth(h, w, cells, far=FAR, patch=16):
    """(1, 1, h*patch, w*patch) depth map: `far` everywhere, cells {(i, j): d} set on their whole token block."""
    d = torch.full((1, 1, h * patch, w * patch), far)
    for (i, j), v in cells.items():
        d[..., i * patch:(i + 1) * patch, j * patch:(j + 1) * patch] = v
    return d


def _cfg(mode="erp_depth", head=False):
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.lift.query_mode = mode
    cfg.erp_depth.head = head
    return cfg


# ---- (a) ray geometry ------------------------------------------------------------------------------------------

def test_forward_token_on_the_horizon_is_placed_d_metres_ahead():
    # odd grid: token (14, 28) of 29 x 57 looks exactly at azimuth 0 (centre column = north), elevation 0
    h, w, d = 29, 57, 12.5
    xy, valid = depth_placement_metric(_token_depth(h, w, {(14, 28): d}), h, w, R_N, max_depth_m=35.0)
    assert valid.sum() == 1 and bool(valid[0, 14, 28])
    assert torch.allclose(xy[0, 14, 28], torch.tensor([d, 0.0]), atol=1e-4)      # x forward = +d, y = 0


def test_token_90deg_right_is_east_and_lands_at_minus_y():
    # 58 columns: token 43 has azimuth +90 deg (east of north), token 14 has -90 deg (west)
    h, w, d = 29, 58, 9.0
    xy, valid = depth_placement_metric(_token_depth(h, w, {(14, 43): d, (14, 14): d}), h, w, R_N)
    assert torch.allclose(xy[0, 14, 43], torch.tensor([0.0, -d]), atol=1e-4)     # east = right = y < 0 (y is left)
    assert torch.allclose(xy[0, 14, 14], torch.tensor([0.0, d]), atol=1e-4)      # west = left


def test_placement_matches_the_loc2_equation_in_our_frame():
    """x = d sinθ cosφ, y = -d sinθ sinφ with θ from the zenith and φ the azimuth clockwise from forward."""
    h, w = 28, 56
    rng = np.random.default_rng(0)
    dep = torch.from_numpy(rng.uniform(2.0, 30.0, (h, w)).astype(np.float32))
    depth = dep.repeat_interleave(16, 0).repeat_interleave(16, 1)[None, None]
    xy, valid = depth_placement_metric(depth, h, w, R_N, max_depth_m=35.0)
    assert bool(valid.all())
    i, j = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    theta = (i + 0.5) / h * np.pi                            # polar angle of the token centre
    phi = ((j + 0.5) / w - 0.5) * 2 * np.pi                  # azimuth, 0 at the centre column, + to the right
    d = dep.numpy()
    assert np.allclose(xy[0, ..., 0].numpy(), d * np.sin(theta) * np.cos(phi), atol=1e-3)
    assert np.allclose(xy[0, ..., 1].numpy(), -d * np.sin(theta) * np.sin(phi), atol=1e-3)


def test_validity_mask_and_bev_pixel_convention():
    h, w = 28, 56
    depth = _token_depth(h, w, {(13, 28): 34.9, (13, 29): 35.0, (13, 30): float("nan"), (13, 31): 0.0,
                                (14, 28): 4.0})
    xy, valid = depth_placement(depth, h, w, R_N, n=224, cell_m=0.25, max_depth_m=35.0)
    assert bool(valid[0, 13, 28]) and not valid[0, 13, 29] and not valid[0, 13, 30] and not valid[0, 13, 31]
    assert int(valid.sum()) == 2
    xm, _ = depth_placement_metric(depth, h, w, R_N)
    x, y = xm[0, 14, 28].tolist()
    u, v = xy[0, 14, 28].tolist()
    assert abs(u - (112 - y / 0.25 - 0.5)) < 1e-4 and abs(v - (112 - x / 0.25 - 0.5)) < 1e-4
    assert v < 111.5                                          # ahead = up in the BEV picture (row 0 = north)


def test_token_depth_is_read_at_the_token_centre_for_any_resolution():
    h, w = 28, 56
    full = torch.zeros(1, 1, 1024, 2048)
    rows = ((torch.arange(h) + 0.5) * 1024 / h).floor().long()
    cols = ((torch.arange(w) + 0.5) * 2048 / w).floor().long()
    full[0, 0, rows[5], cols[7]] = 17.0                      # a single pixel at token (5, 7)'s centre
    t = sample_token_depth(full, h, w)
    assert t.shape == (1, h, w) and float(t[0, 5, 7]) == 17.0 and int((t > 0).sum()) == 1


# ---- (a) consistency with VigorPairs' H and en: placed point + H = the landmark's reference pixel -----------------

def _vigor_layout(tmp_path, dy, dx, depth_png=None, city="Chicago"):
    (tmp_path / city / "panorama").mkdir(parents=True)
    (tmp_path / city / "satellite").mkdir(parents=True)
    sat = np.full((640, 640, 3), 90, np.uint8)
    cv2.imwrite(str(tmp_path / city / "satellite" / "s1.png"), sat)
    cv2.imwrite(str(tmp_path / city / "panorama" / "p1.jpg"), np.full((448, 896, 3), 128, np.uint8))
    cv2.imwrite(str(tmp_path / city / "panorama" / "p2.jpg"), np.full((448, 896, 3), 128, np.uint8))
    lab = tmp_path / "splits" / "VIGOR" / city
    lab.mkdir(parents=True)
    (lab / "satellite_list.txt").write_text("s1.png\n")
    lines = "".join(f"{p} s1.png {dy} {dx} s1.png 0 0 s1.png 0 0 s1.png 0 0\n" for p in ("p1.jpg", "p2.jpg"))
    (lab / "same_area_balanced_test.txt").write_text(lines)
    if depth_png is not None:                                # only p1 gets a depth file
        path = depth_png_path(tmp_path, city, "p1.jpg")
        path.parent.mkdir(parents=True)
        Image.fromarray(depth_png).save(path)                # exactly as scripts/loc2_depth_vigor.py writes it
    return tmp_path


def _png_mm(depth_m):
    return np.clip(depth_m * 1000.0, 0.0, 65000.0).astype(np.uint16)


def test_placed_landmark_maps_through_H_to_its_reference_pixel(tmp_path):
    """A landmark seen by one token at a known ray depth; the label puts the camera 24 px east and 40 px south of the
    tile centre (dx > 0 = west, dy > 0 = south: verified on lat/lon). The landmark's canvas pixel is computed from the
    tile geometry alone (tile px -> world metres -> tile px -> the resize into the canvas), independently of H."""
    city, dy, dx = "Chicago", 40.0, -24.0
    i, j, d_ray = 13, 35, 20.0                               # token (13, 35) of 28 x 56: azimuth 48 deg, 3 deg up
    h, w = 28, 56
    dep = np.full((448, 896), FAR, np.float32)
    dep[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = d_ray
    root = _vigor_layout(tmp_path, dy, dx, _png_mm(dep))
    cfg = _cfg()
    ds = VigorPairs(root, cfg, cities=[city], split="samearea", train=False)
    assert ds.depth and ds.keep_with_depth() == 1 and len(ds) == 1
    s = ds[0]
    assert s["depth"].shape == (1, 448, 896) and s["erp"].shape[-2:] == (448, 896)
    batch = collate_vigor([s])
    assert batch["depth"].shape == (1, 1, 448, 896)
    xy, valid = ErpDepthQuery(cfg).placement(batch)
    assert int(valid.sum()) == 1 and bool(valid[0, i, j])

    # --- independent expectation ---
    res = CITY_RES[city]                                     # m per tile px (640 px tile)
    lon = ((j + 0.5) / w - 0.5) * 2 * math.pi                # clockwise from north
    lat = (0.5 - (i + 0.5) / h) * math.pi
    horiz = d_ray * math.cos(lat)
    cam_e, cam_n = -dx * res, -dy * res                      # camera offset from the tile centre, metres
    assert abs(float(s["en"][0]) - cam_e) < 1e-6 and abs(float(s["en"][1]) - cam_n) < 1e-6
    lm_e, lm_n = cam_e + horiz * math.sin(lon), cam_n + horiz * math.cos(lon)
    col_t, row_t = 319.5 + lm_e / res, 319.5 - lm_n / res    # tile px (north up, east right)
    S = int(cfg.grid.n * cfg.reference.scale)
    k = res / cfg.grid.cell_m                                # tile px -> canvas px
    new = int(round(640 * k))
    x0 = (S - new) // 2
    ratio = new / 640.0                                      # cv2.resize maps pixel centres: (p + 0.5) * ratio - 0.5
    exp = np.array([x0 + (col_t + 0.5) * ratio - 0.5, x0 + (row_t + 0.5) * ratio - 0.5])

    u, v = xy[0, i, j].tolist()
    p = batch["H"][0].double().numpy() @ np.array([u, v, 1.0])
    got = p[:2] / p[2]
    assert np.abs(got - exp).max() < 1.0, (got, exp)        # < 1 canvas px = 0.25 m
    # and the coarse target of that token is the reference cell holding the landmark
    idx, use = coarse_targets(batch["H"], valid, query_xy=xy, ref_size=S, cells=56)
    assert bool(use[0, i, j]) and int(idx[0, i, j]) == int(exp[1] + 0.5) // 16 * 56 + int(exp[0] + 0.5) // 16


# ---- (b) depth PNG decoding ------------------------------------------------------------------------------------

def test_depth_png_round_trip_uint16_millimetres(tmp_path):
    rng = np.random.default_rng(1)
    depth = rng.uniform(0.3, 80.0, (64, 128)).astype(np.float32)
    path = tmp_path / "d.png"
    Image.fromarray(_png_mm(depth)).save(path)
    back = read_depth_png(path)
    assert back.dtype == np.float32 and back.shape == depth.shape
    near = depth < 65.0
    assert np.abs(back[near] - depth[near]).max() <= 1e-3 + 1e-6              # millimetre quantisation
    assert np.all(back[~near] == 65.0)                                        # clipped far values
    t, valid = depth_placement_metric(torch.from_numpy(back)[None, None], 4, 8, R_N, max_depth_m=35.0)
    assert t.shape == (1, 4, 8, 2)
    with pytest.raises(RuntimeError):
        read_depth_png(tmp_path / "missing.png")


def test_samples_without_depth_are_dropped_and_mixed_batches_refused(tmp_path):
    root = _vigor_layout(tmp_path, 0.0, 0.0, _png_mm(np.full((448, 896), 10.0, np.float32)))
    ds = VigorPairs(root, _cfg(), cities=["Chicago"], split="samearea", train=False)
    a, b = ds[0], ds[1]
    assert "depth" in a and "depth" not in b
    with pytest.raises(KeyError):
        collate_vigor([a, b])
    ipm = VigorPairs(root, _cfg("ipm"), cities=["Chicago"], split="samearea", train=False)
    assert not ipm.depth and "depth" not in ipm[0]            # other modes unchanged
    assert ipm.erp_w == 896 and ipm.erp_h == 448


# ---- (e) factory, head, checkpoint round trip ------------------------------------------------------------------

class _FakeMatcher:
    class model:  # noqa: N801
        @staticmethod
        def encoder(x):
            return {16: torch.nn.functional.avg_pool2d(x.repeat(1, 342, 1, 1)[:, :1024], 16)}


def test_build_query_erp_depth_forward_and_state_round_trip():
    cfg = _cfg(head=True)
    q = build_query(cfg, "erp_depth")
    assert isinstance(q, ErpDepthQuery) and isinstance(q.head, ProjectionHead)
    n_par = sum(p.numel() for p in q.parameters())
    assert 1e6 <= n_par <= 3e6, n_par
    batch = dict(erp=torch.rand(1, 1, 3, 448, 896), R_w2c=R_N[None], depth=_token_depth(28, 56, {(14, 10): 8.0}))
    f_q, frac = q(batch, _FakeMatcher)
    assert f_q.shape == (1, 1024, 28, 56) and frac.shape == (1, 28, 56) and float(frac.sum()) == 1.0
    # zero-initialised output projection: at init the head is the identity on the frozen tokens
    base = _FakeMatcher.model.encoder(batch["erp"][:, 0])[16]
    assert torch.allclose(f_q, base, atol=1e-6)
    # train a step so the state differs from init, then round-trip through the checkpoint layout
    with torch.no_grad():
        for p in q.parameters():
            p.add_(0.01 * torch.randn_like(p))
    state = {"query": q.state_dict(), "mode": "erp_depth"}
    q2 = build_query(cfg, "erp_depth")
    loaded, own = load_query_state(q2, state)
    assert loaded == own == len(q.state_dict())
    for k, v in q.state_dict().items():
        assert torch.equal(v, q2.state_dict()[k])
    assert not torch.allclose(q2(batch, _FakeMatcher)[0], base)
    # head off (default): no parameters, and an erp_depth checkpoint without a head loads trivially
    q0 = build_query(_cfg(head=False), "erp_depth")
    assert q0.head is None and sum(p.numel() for p in q0.parameters()) == 0
    assert load_query_state(q0, {"query": q0.state_dict(), "mode": "erp_depth"}) == (0, 0)


def test_placement_requires_depth():
    q = build_query(_cfg(), "erp_depth")
    with pytest.raises(KeyError):
        q.placement(dict(erp=torch.rand(1, 1, 3, 448, 896), R_w2c=R_N[None]))


def test_checkpoint_head_setting_is_restored_by_the_evaluators():
    from types import SimpleNamespace
    from bevloc.model.query import apply_query_cfg
    cfg_train = _cfg(head=True)
    cfg_train.erp_depth.head_dim = 64
    q = build_query(cfg_train, "erp_depth")
    state = {"query": q.state_dict(), "mode": "erp_depth", "erp_depth": vars(cfg_train.erp_depth)}
    cfg_eval = apply_query_cfg(_cfg(head=False), state)      # the evaluator's config has the head off
    assert isinstance(cfg_eval.erp_depth, SimpleNamespace) and cfg_eval.erp_depth.head is True
    q2 = build_query(cfg_eval, "erp_depth")
    assert load_query_state(q2, state) == (len(q.state_dict()),) * 2
    cfg_ipm = apply_query_cfg(_cfg("ipm"), {"query": {}, "mode": "ipm"})   # other checkpoints: untouched
    assert cfg_ipm.erp_depth.head is False
