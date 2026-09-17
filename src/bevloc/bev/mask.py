"""ERP validity mask: ego vehicle (estimated from data) + dual-fisheye stitching seams."""
from __future__ import annotations

import cv2
import numpy as np


def temporal_stats(erps, highpass_sigma=6.0):
    """Temporal std and |mean| of the high-passed grey ERP over MOVING frames.
    Background decorrelates (high std, ~0 mean); anything bolted to the car does not."""
    s = s2 = None
    n = 0
    for erp in erps:
        g = cv2.cvtColor(erp, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hp = g - cv2.GaussianBlur(g, (0, 0), highpass_sigma)   # insensitive to exposure changes
        s = hp if s is None else s + hp
        s2 = hp ** 2 if s2 is None else s2 + hp ** 2
        n += 1
    mean = s / n
    return np.sqrt(np.maximum(s2 / n - mean ** 2, 0)), np.abs(mean), n


def ego_mask(std, mean_hp, std_thresh=5.0, meanhp_thresh=6.0, min_blob_px=80, margin_px=6,
             horizon_frac=0.53):
    """Per-column envelope: the car occludes everything below its highest point in a column."""
    H, W = std.shape
    cand = ((cv2.GaussianBlur(std, (0, 0), 2) < std_thresh) |
            (cv2.GaussianBlur(mean_hp, (0, 0), 1.5) > meanhp_thresh)).astype(np.uint8)
    cand[:int(horizon_frac * H)] = 0
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
    keep = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_blob_px])
    top = np.where(keep.any(0), keep.argmax(0), H).astype(np.float32)
    top = cv2.erode(top[None], np.ones((1, 15)))[0] - margin_px
    return np.arange(H)[:, None] >= top[None], top


def seam_mask(shape, halfwidth_px=4):
    H, W = shape
    m = np.zeros((H, W), bool)
    for c in (W // 4, 3 * W // 4):                 # lens borders at azimuth -90 / +90 deg
        m[:, c - halfwidth_px:c + halfwidth_px + 1] = True
    return m


def blind_extents(top_row, shape, camera_height):
    """Per sector: elevation of the car's upper boundary and the range where ground becomes visible."""
    H, W = shape
    el = (0.5 - (top_row + 0.5) / H) * 180
    az = ((np.arange(W) + 0.5) / W - 0.5) * 360
    blind = camera_height / np.tan(np.radians(np.maximum(-el, 1)))
    out = {}
    for k, c in (("front", 0), ("right", 90), ("back", 180), ("left", -90)):
        m = np.abs((az - c + 180) % 360 - 180) < 20
        out[k] = dict(ego_top_elevation_deg=float(np.median(el[m])), blind_radius_m=float(np.median(blind[m])),
                      blind_radius_min_max_m=[float(blind[m].min()), float(blind[m].max())])
    return out
