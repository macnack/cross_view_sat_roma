"""VIGOR reference window (coarse-to-fine second pass): H is the exact BEV px -> window px map, off-tile is black,
the jitter distribution, and the window pose -> tile frame back-mapping. Synthetic tiles, CPU, no data."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from bevloc import config as C
from bevloc.data.vigor import (
    CITY_RES, VigorPairs, canvas_to_en, collate_vigor, en_to_canvas, pose_en, window_affine,
)

DY, DX = 40.0, -24.0              # label: 40 tile px south, 24 px EAST of the tile centre (dx > 0 = west)
SIGMA = 3.0                       # tile px, Gaussian marker at the camera


def _make(tmp_path, city="Chicago", dy=DY, dx=DX, size=640):
    (tmp_path / city / "panorama").mkdir(parents=True)
    (tmp_path / city / "satellite").mkdir(parents=True)
    xc, yc = (size - 1) / 2.0 - dx, (size - 1) / 2.0 + dy            # col_sign -1, row_sign +1 (verified defaults)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    g = np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * SIGMA ** 2))
    sat = np.full((size, size, 3), 60.0)
    sat[..., 2] += 195.0 * g                                          # BGR: marker in red
    cv2.imwrite(str(tmp_path / city / "satellite" / "s1.png"), np.round(sat).astype(np.uint8))
    cv2.imwrite(str(tmp_path / city / "panorama" / "p1,1.0,.jpg"), np.full((256, 512, 3), 128, np.uint8))
    lab = tmp_path / "splits" / "VIGOR" / city
    lab.mkdir(parents=True)
    (lab / "satellite_list.txt").write_text("s1.png\n")
    line = f"p1,1.0,.jpg s1.png {dy} {dx} s1.png 0 0 s1.png 0 0 s1.png 0 0\n"
    (lab / "pano_label_balanced.txt").write_text(line)
    (lab / "same_area_balanced_test.txt").write_text(line)
    (lab / "same_area_balanced_train.txt").write_text(line)
    return tmp_path


def _cfg(cell_m=0.0625, window=56.0, jitter=6.0, mode="ipm"):
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.lift.query_mode = mode
    cfg.grid.cell_m = cell_m
    cfg.vigor.ref_window_m = window
    cfg.vigor.ref_jitter_m = jitter
    return cfg


def _marker_centroid(ref, near, r=14):
    """Sub-pixel centroid (u, v) of the red-minus-background marker in a (3, S, S) canvas near `near`."""
    red = ref[0].numpy().astype(np.float64) * 255.0 - 60.0
    u0, v0 = int(round(near[0])), int(round(near[1]))
    win = np.clip(red[v0 - r:v0 + r + 1, u0 - r:u0 + r + 1], 0, None)
    win[win < 5.0] = 0.0                                              # quantisation noise of the flat background
    vv, uu = np.mgrid[v0 - r:v0 + r + 1, u0 - r:u0 + r + 1]
    return np.array([(win * uu).sum() / win.sum(), (win * vv).sum() / win.sum()])


def _camera_px(s, n=224):
    o = (n - 1) / 2.0
    p = s["H"].numpy().astype(np.float64) @ np.array([o, o, 1.0])
    return p[:2] / p[2]


@pytest.mark.parametrize("cell_m", [0.0625, 0.25])                  # upsampling (1.78x) and downsampling (0.45x)
@pytest.mark.parametrize("offset", [(0.0, 0.0), (3.1, -4.2), (-5.9, 0.7), (25.0, 0.0), (-20.0, 22.0)])
def test_window_H_lands_on_the_camera_marker(tmp_path, cell_m, offset):
    """For several window centres (true position + offset, including windows that run off the tile edge) the camera
    pixel H(BEV centre) is where the marker painted at the label's tile pixel is, to a small fraction of a pixel."""
    root = _make(tmp_path)
    cfg = _cfg(cell_m=cell_m, window=None, jitter=0.0)
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea")
    res = CITY_RES["Chicago"]
    en = np.array([-DX * res, -DY * res])                             # east, north metres of the camera
    centre = en + np.array(offset)
    s = ds.item(0, ref_centre_en=torch.tensor(centre))
    S = s["ref"].shape[-1]
    assert np.allclose(s["en"].numpy(), en) and np.allclose(s["ref_centre_en"].numpy(), centre)
    cam = _camera_px(s)
    assert np.allclose(cam, en_to_canvas(en, centre, cell_m, S), atol=1e-4)
    # the camera is the offset away from the canvas centre: -east -> left, -north -> down
    c = (S - 1) / 2.0
    assert np.allclose(cam, [c - offset[0] / cell_m, c + offset[1] / cell_m], atol=1e-3)
    if 16 < cam[0] < S - 16 and 16 < cam[1] < S - 16:
        tol = 0.05 if cell_m < 0.1 else 0.1                           # canvas px
        assert np.abs(_marker_centroid(s["ref"], cam) - cam).max() < tol, (_marker_centroid(s["ref"], cam), cam)
    # the back-mapping of the true H is the label position
    assert np.allclose(pose_en(s["H"].numpy(), s["ref_centre_en"].numpy(), 224, cell_m, S), en, atol=1e-5)


def test_window_off_the_tile_edge_is_black_and_the_rest_is_tile(tmp_path):
    root = _make(tmp_path)
    cfg = _cfg(window=None, jitter=0.0)
    ds = VigorPairs(root, cfg, cities=["Chicago"])
    res = CITY_RES["Chicago"]
    centre = np.array([30.0, -25.0])                                  # SE corner: the window leaves the tile E and S
    s = ds.item(0, ref_centre_en=centre)
    ref = s["ref"].numpy()
    S = ref.shape[-1]
    M = window_affine(640, 640, res, 0.0625, S, centre)
    x = M[0, 0] * np.arange(S) + M[0, 2]                              # tile column of every canvas column
    y = M[1, 1] * np.arange(S) + M[1, 2]
    off_c, in_c = x > 639.0 + 1.0, x < 639.0 - 1.0
    off_r, in_r = y > 639.0 + 1.0, y < 639.0 - 1.0
    assert off_c.any() and off_r.any() and in_c.any() and in_r.any()
    assert ref[:, :, off_c].max() == 0 and ref[:, off_r, :].max() == 0
    assert ref[:, in_r][:, :, in_c].min() > 0.2                       # 60/255 background everywhere on the tile


def test_ref_window_m_blacks_out_beyond_the_window(tmp_path):
    root = _make(tmp_path)
    cfg = _cfg(window=40.0, jitter=0.0)                              # 640 px of the 896 px canvas at 0.0625
    s = VigorPairs(root, cfg, cities=["Chicago"])[0]
    ref = s["ref"].numpy()
    d = np.abs(np.arange(896) - 447.5)
    assert ref[:, :, d > 320].max() == 0 and ref[:, d > 320, :].max() == 0
    assert ref[:, d < 319][:, :, d < 319].min() > 0.2


def test_training_jitter_is_uniform_on_the_disc_and_validation_is_fixed(tmp_path):
    root = _make(tmp_path)
    cfg = _cfg(jitter=6.0)
    tr = VigorPairs(root, cfg, cities=["Chicago"], split="samearea", train=True)
    va = VigorPairs(root, cfg, cities=["Chicago"], split="samearea", train=False)
    assert tr.jitter_seed is None and va.jitter_seed == 0
    en = tr[0]["en"].numpy()
    torch.manual_seed(0)
    offs = np.array([tr.window_centre(0, en) - en for _ in range(4000)])
    r = np.linalg.norm(offs, axis=1)
    assert r.max() <= 6.0 + 1e-9
    assert abs(np.mean(r <= 6.0 / np.sqrt(2)) - 0.5) < 0.03          # uniform on the disc: half the mass inside r/sqrt2
    assert np.all(np.abs(offs.mean(0)) < 0.2)
    torch.manual_seed(0)
    again = np.array([tr.window_centre(0, en) - en for _ in range(5)])
    assert np.allclose(again, offs[:5])                               # seeded runs repeat
    a, b = va[0], va[0]
    assert np.allclose(a["ref_centre_en"].numpy(), b["ref_centre_en"].numpy())
    assert np.linalg.norm(a["ref_centre_en"].numpy() - en) <= 6.0
    # H follows the jittered centre exactly
    assert np.allclose(pose_en(a["H"].numpy(), a["ref_centre_en"].numpy(), 224, 0.0625, 896), en, atol=1e-5)
    b = collate_vigor([va[0], va[0]])
    assert b["ref_centre_en"].shape == (2, 2)


def test_whole_tile_sample_is_unchanged_and_centred_at_zero(tmp_path):
    root = _make(tmp_path)
    cfg = _cfg(cell_m=0.125, window=None, jitter=0.0)
    s = VigorPairs(root, cfg, cities=["Chicago"])[0]
    assert np.all(s["ref_centre_en"].numpy() == 0)
    res = CITY_RES["Chicago"]
    assert np.allclose(pose_en(s["H"].numpy(), np.zeros(2), 224, 0.125, 896), [-DX * res, -DY * res], atol=1e-4)


@pytest.mark.parametrize("yaw", [0.0, 0.3])
def test_fine_pose_maps_back_to_the_tile_frame(yaw):
    """An injected fine pose (camera at en_inj, any in-plane rotation) on a window centred anywhere comes back as
    en_inj exactly; canvas_to_en and en_to_canvas are inverse."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        centre = rng.uniform(-30, 30, 2)
        en_inj = centre + rng.uniform(-20, 20, 2)
        cell, S, n = 0.0625, 896, 224
        cam = en_to_canvas(en_inj, centre, cell, S)
        o = (n - 1) / 2.0
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        t = cam - R @ np.array([o, o])
        H = np.eye(3)
        H[:2, :2], H[:2, 2] = R, t
        assert np.allclose(pose_en(H, centre, n, cell, S), en_inj, atol=1e-9)
        assert np.allclose(canvas_to_en(en_to_canvas(en_inj, centre, cell, S), centre, cell, S), en_inj)
