"""Panoramas cropped to an elevation band (KITScenes ring cameras without black pixels): rays, placement, reader, crop."""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import torch
from PIL import Image

from bevloc import config as C
from bevloc.data import kitscenes as K
from bevloc.data.vigor import VigorPairs, collate_vigor, depth_png_path
from bevloc.model.depth_query import ErpDepthQuery, depth_placement_metric
from bevloc.model.erp_query import band_of, rays_at
from vigor_labels import corrected


def test_full_sphere_is_the_default_band():
    u, v = torch.rand(50) * 896, torch.rand(50) * 448
    a, la = rays_at(u, v, 448, 896)
    b, lb = rays_at(u, v, 448, 896, (math.pi / 2, -math.pi / 2))
    assert torch.equal(a, b) and torch.equal(la, lb)


def test_band_rays_equal_the_rays_of_the_same_rows_of_the_full_sphere():
    """A crop of rows r0:r1 of a full ERP (Hf rows): cropped pixel (u, v) sees the ray of full-ERP pixel (u, v + r0)."""
    Hf, W, r0, r1 = 1024, 2048, 330, 694
    top, bot = math.radians(90.0 - r0 * 180.0 / Hf), math.radians(90.0 - r1 * 180.0 / Hf)
    u, v = torch.rand(200) * W, torch.rand(200) * (r1 - r0)
    cropped, lat_c = rays_at(u, v, r1 - r0, W, (top, bot))
    full, lat_f = rays_at(u, v + r0, Hf, W)
    assert torch.allclose(cropped, full, atol=1e-5) and torch.allclose(lat_c, lat_f, atol=1e-6)
    edge, lat = rays_at(torch.tensor([0.0]), torch.tensor([0.0]), r1 - r0, W, (top, bot))
    assert math.isclose(float(lat[0]), top, abs_tol=1e-6)                     # the first row's top edge is lat_top


def test_depth_placement_of_a_band_matches_the_full_sphere_placement():
    """The same physical token (full-ERP token row i) placed from a band crop and from the full sphere."""
    Hf, Wf = 448, 896                                                        # 28 x 56 tokens
    rows = (14 - 5, 14 + 5)                                                  # token rows 9..14: elevations +-5 token rows
    top = math.radians(90.0 - rows[0] * 16 * 180.0 / Hf)
    bot = math.radians(90.0 - rows[1] * 16 * 180.0 / Hf)
    depth_full = torch.full((1, 1, Hf, Wf), 12.0)
    R = torch.eye(3)[None]
    R = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])[None]         # R_NORTH: camera x east, y down, z north
    xy_f, ok_f = depth_placement_metric(depth_full, 28, 56, R)
    depth_band = depth_full[:, :, rows[0] * 16:rows[1] * 16]
    xy_b, ok_b = depth_placement_metric(depth_band, rows[1] - rows[0], 56, R, band=(top, bot))
    assert torch.allclose(xy_b, xy_f[:, rows[0]:rows[1]], atol=1e-4) and ok_b.all()


def test_band_of_batch():
    assert band_of({}) is None
    assert band_of({"erp_band": torch.tensor([[0.5, -0.5], [0.5, -0.5]])}) == (0.5, -0.5)
    try:
        band_of({"erp_band": torch.tensor([[0.5, -0.5], [0.4, -0.4]])})
    except ValueError:
        return
    raise AssertionError("different bands in one batch must raise")


def test_symmetric_band_has_no_black_row_and_is_centred_on_the_horizon():
    h, w = 1024, 64
    valid = np.zeros((h, w), bool)
    valid[330:694] = True
    valid[340:350, 10] = False                                               # a seam gap at one azimuth narrows the band
    r0, r1, top, bot = K.symmetric_band(valid)
    assert r0 == h - r1 and r1 - r0 >= 2                                     # symmetric about row h / 2
    assert valid[r0:r1].all()
    assert math.isclose(top, -bot, abs_tol=1e-9) and math.isclose(top, 90.0 - r0 * 180.0 / h)
    assert not (valid[r0 - 1].all() and valid[r1].all())                     # the band is the largest such one


def _layout(tmp_path, with_band, rows=366, city="Chicago"):
    (tmp_path / city / "panorama").mkdir(parents=True)
    (tmp_path / city / "satellite").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / city / "satellite" / "s1.png"), np.full((640, 640, 3), 90, np.uint8))
    h = rows if with_band else 1024
    cv2.imwrite(str(tmp_path / city / "panorama" / "p1.jpg"), np.full((h, 2048, 3), 128, np.uint8))
    lab = tmp_path / "splits" / "VIGOR" / city
    lab.mkdir(parents=True)
    (lab / "satellite_list.txt").write_text("s1.png\n")
    (lab / "same_area_balanced_test.txt").write_text("p1.jpg s1.png 10 -5 s1.png 0 0 s1.png 0 0 s1.png 0 0\n")
    corrected(lab)
    path = depth_png_path(tmp_path, city, "p1.jpg")
    path.parent.mkdir(parents=True)
    Image.fromarray(np.full((h, 2048), 9000, np.uint16)).save(path)
    if with_band:
        top = 90.0 - (512 - rows // 2) * 180.0 / 1024
        (tmp_path / city / "erp_band.json").write_text(json.dumps(dict(top_deg=top, bottom_deg=-top, rows=rows, full_width=2048)))
    return tmp_path


def _cfg():
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.lift.query_mode = "erp_depth"
    return cfg


def test_reader_resizes_a_band_panorama_to_whole_tokens_and_exposes_the_band(tmp_path):
    root = _layout(tmp_path, True, rows=366)
    ds = VigorPairs(root, _cfg(), cities=["Chicago"], split="samearea", train=False)
    s = ds[0]
    assert s["erp"].shape[-2:] == (160, 896)                                 # 366 rows * 896 / 2048 = 160.1 -> 10 token rows
    assert s["depth"].shape[-2:] == (160, 896)
    top = float(s["erp_band"][0])
    assert math.isclose(top, math.radians(90.0 - (512 - 183) * 180.0 / 1024), rel_tol=1e-5)
    b = collate_vigor([s, s])
    assert b["erp_band"].shape == (2, 2) and band_of(b) is not None
    q = ErpDepthQuery(_cfg())
    xy, valid = q.placement(b)
    assert xy.shape == (2, 10, 56, 2) and valid.shape == (2, 10, 56)


def test_a_full_sphere_dataset_is_unchanged(tmp_path):
    root = _layout(tmp_path, False)
    ds = VigorPairs(root, _cfg(), cities=["Chicago"], split="samearea", train=False)
    s = ds[0]
    assert "erp_band" not in s and s["erp"].shape[-2:] == (448, 896)


def test_band_depth_is_computed_on_the_full_sphere_canvas_and_cropped_back(monkeypatch):
    """infer_band_distance puts the band at its rows of a full-sphere canvas, runs the full-sphere model, returns the rows."""
    from bevloc.baselines import loc2

    Hf, W, r0, rows = 1024, 64, 355, 314
    top, bot = math.radians(90.0 - r0 * 180.0 / Hf), math.radians(90.0 - (r0 + rows) * 180.0 / Hf)
    seen = {}

    def fake(model, canvas, vfov_half=1.5708):
        seen["shape"], seen["vfov"] = canvas.shape, vfov_half
        seen["band_rows"] = canvas[r0:r0 + rows]
        seen["outside_black"] = not canvas[:r0].any() and not canvas[r0 + rows:].any()
        return np.repeat(np.arange(canvas.shape[0], dtype=np.float32)[:, None], canvas.shape[1], 1)   # value = the row index

    monkeypatch.setattr(loc2, "infer_distance", fake)
    img = np.full((rows, W, 3), 77, np.uint8)
    d = loc2.infer_band_distance(None, img, top, bot)
    assert seen["shape"] == (Hf, W, 3) and seen["vfov"] == 1.5708           # the full-sphere camera
    assert (seen["band_rows"] == 77).all() and seen["outside_black"]
    assert d.shape == (rows, W) and (d[:, 0] == np.arange(r0, r0 + rows)).all()


def test_on_the_fly_crop_equals_a_dataset_cropped_on_disk(tmp_path, monkeypatch):
    """BEVLOC_CROP_BAND_DEG on a full-sphere dataset gives the sample of the same rows stored as a band dataset."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:1024, 0:2048]                                           # smooth, so JPEG round trips agree
    pano = np.stack([128 + 100 * np.sin(yy / 60.0), 128 + 100 * np.sin(xx / 90.0), 128 + 100 * np.sin((yy + xx) / 120.0)], -1).astype(np.uint8)
    depth = rng.integers(2000, 20000, (1024, 2048)).astype(np.uint16)
    r0 = int(round((90.0 - 27.6) / 180.0 * 1024))                              # 355: rows 355..669 (314 rows)
    top = 90.0 - r0 * 180.0 / 1024

    def make(root, p, d, band):
        _layout(root, False)                                                   # tiles, labels, folders
        cv2.imwrite(str(root / "Chicago" / "panorama" / "p1.jpg"), p, [cv2.IMWRITE_JPEG_QUALITY, 100])
        Image.fromarray(d).save(depth_png_path(root, "Chicago", "p1.jpg"))
        if band:
            (root / "Chicago" / "erp_band.json").write_text(json.dumps(dict(top_deg=top, bottom_deg=-top, rows=314, full_width=2048)))
        return root

    full = make(tmp_path / "full", pano, depth, False)
    disk = make(tmp_path / "disk", pano[r0:1024 - r0], depth[r0:1024 - r0], True)
    monkeypatch.setenv("BEVLOC_CROP_BAND_DEG", "27.6")
    a = VigorPairs(full, _cfg(), cities=["Chicago"], split="samearea", train=False)[0]
    monkeypatch.delenv("BEVLOC_CROP_BAND_DEG")
    b = VigorPairs(disk, _cfg(), cities=["Chicago"], split="samearea", train=False)[0]
    assert a["erp"].shape == b["erp"].shape and a["erp"].shape[-2:] == (128, 896)
    assert torch.equal(a["depth"], b["depth"])
    assert torch.allclose(a["erp"], b["erp"], atol=0.03)
    assert torch.allclose(a["erp_band"], b["erp_band"], atol=1e-4)
