"""VIGOR reader on a synthetic layout (no dataset needed)."""
from __future__ import annotations

import cv2
import numpy as np

from bevloc import config as C
from bevloc.data.vigor import CITY_RES, VigorPairs, collate_vigor, find_label_root, read_labels, split_cities


def _make(tmp_path, city="Chicago", dy=40.0, dx=-24.0):
    (tmp_path / city / "panorama").mkdir(parents=True)
    (tmp_path / city / "satellite").mkdir(parents=True)
    sat = np.zeros((640, 640, 3), np.uint8)
    sat[:, :, 1] = 90
    sat[300:340, 300:340] = (255, 0, 0)                     # a marker at the tile centre (BGR blue)
    cv2.imwrite(str(tmp_path / city / "satellite" / "s1.png"), sat)
    pano = np.full((256, 512, 3), 128, np.uint8)
    cv2.imwrite(str(tmp_path / city / "panorama" / "p1,1.0,.jpg"), pano)
    lab = tmp_path / "splits" / "VIGOR" / city
    lab.mkdir(parents=True)
    (lab / "satellite_list.txt").write_text("s1.png\n")
    line = f"p1,1.0,.jpg s1.png {dy} {dx} s1.png 0 0 s1.png 0 0 s1.png 0 0\n"
    (lab / "pano_label_balanced.txt").write_text(line)
    (lab / "same_area_balanced_test.txt").write_text(line)
    return tmp_path


def _cfg(mode="ipm"):
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.lift.query_mode = mode
    return cfg


def test_label_root_and_cross_area_cities(tmp_path):
    root = _make(tmp_path)
    assert find_label_root(root).name == "VIGOR"
    assert split_cities("crossarea", train=False) == ["SanFrancisco", "Chicago"]
    assert split_cities("crossarea", train=True) == ["NewYork", "Seattle"]
    rows = read_labels(root, ["Chicago"], "crossarea", train=False)
    assert rows == [dict(city="Chicago", pano="p1,1.0,.jpg", sat="s1.png", dy=40.0, dx=-24.0)]


def test_sample_geometry_matches_the_label(tmp_path):
    root = _make(tmp_path, dy=40.0, dx=-24.0)
    cfg = _cfg("ipm")
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea", train=False, height_m=2.0, row_sign=1.0)
    s = ds[0]
    n, S = int(cfg.grid.n), int(cfg.grid.n * cfg.reference.scale)
    assert s["ref"].shape == (3, S, S) and s["bev"].shape == (3, n, n) and s["bev_valid"].shape == (n, n)
    scale = CITY_RES["Chicago"] * 640.0 / 640.0 / cfg.grid.cell_m         # tile px -> canvas px
    c = (S - 1) / 2.0
    # camera on the canvas: tile centre + (dx, dy) * scale
    cam = np.array([c - 24.0 * scale, c + 40.0 * scale, 1.0])
    o = (n - 1) / 2.0
    p = s["H"].numpy() @ np.array([o, o, 1.0])
    assert np.allclose(p[:2], cam[:2], atol=1e-4)
    # the tile marker (tile centre) sits at the canvas centre, in blue channel of the RGB canvas
    ref = s["ref"].numpy()
    assert ref[2, int(c), int(c)] > 0.9 and ref[0, int(c), int(c)] < 0.1
    # the tile is smaller than the canvas: its border is black (no data)
    assert ref[:, 0, 0].sum() == 0
    assert s["erp"].shape[0] == 1 and s["R_w2c"].shape == (1, 3, 3)
    assert abs(float(s["en"][0]) + 24.0 * CITY_RES["Chicago"]) < 1e-6      # east offset in metres


def test_row_sign_flips_the_vertical_offset(tmp_path):
    root = _make(tmp_path, dy=40.0, dx=0.0)
    cfg = _cfg("ipm")
    a = VigorPairs(root, cfg, cities=["Chicago"], row_sign=1.0)[0]["H"].numpy()
    b = VigorPairs(root, cfg, cities=["Chicago"], row_sign=-1.0)[0]["H"].numpy()
    o = (cfg.grid.n - 1) / 2.0
    S = cfg.grid.n * cfg.reference.scale
    c = (S - 1) / 2.0
    assert a[1, 2] + o > c and b[1, 2] + o < c


def test_collate_stacks_the_ipm_keys(tmp_path):
    root = _make(tmp_path)
    ds = VigorPairs(root, _cfg("ipm"), cities=["Chicago"])
    b = collate_vigor([ds[0], ds[0]])
    assert b["bev"].shape[0] == 2 and b["H"].shape == (2, 3, 3) and b["erp"].shape[:2] == (2, 1)
    assert ds.centre_guess_m(0) == float(np.hypot(24.0, 40.0) * CITY_RES["Chicago"])
