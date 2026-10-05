"""KITScenes depth masking: black (no-image) pixels and the ego-car rows get depth 0, the rest is untouched."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("kit_mask_depth", Path(__file__).resolve().parents[1] / "scripts" / "kit_mask_depth.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_black_rows_and_ego_rows_are_invalid_and_the_rest_is_kept():
    h, w = 64, 128
    d = np.full((h, w), 5000, np.uint16)
    pano = np.full((h, w, 3), 120, np.uint8)
    pano[:20] = 0                                                   # no image in the top rows
    out, valid = M.mask_depth(d, pano, 20.0)
    el = (0.5 - (np.arange(h) + 0.5) / h) * 180.0
    keep = (np.arange(h) >= 20) & (el >= -20.0)
    assert (out[:20] == 0).all()
    assert (out[el < -20.0] == 0).all()
    assert (out[keep] == 5000).all() and keep.sum() > 10
    assert (valid == (out > 0)).all()
    assert (d == 5000).all()                                        # the input is not modified


def test_jpeg_speckle_in_the_black_area_does_not_make_pixels_valid():
    h, w = 64, 128
    d = np.full((h, w), 3000, np.uint16)
    pano = np.zeros((h, w, 3), np.uint8)
    pano[10, 10] = 9                                                # one isolated noisy pixel in a black region
    out, _ = M.mask_depth(d, pano, 90.0)
    assert (out == 0).all()
