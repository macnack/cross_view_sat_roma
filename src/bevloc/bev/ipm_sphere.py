"""Flat-ground IPM for a spherical (equirectangular) camera with a known height.

Camera axes: x right, y down, z forward. ``R_w2c`` maps world ENU to camera (Mapillary's
``computed_rotation``). Ego BEV: x forward, y left; row 0 forward, column 0 left
(bevloc.bev.grid convention). Camera-only: no depth, no network.
"""
from __future__ import annotations

import cv2
import numpy as np


def ego_to_world_dirs(R_w2c):
    forward = np.asarray(R_w2c, float).T @ np.array([0.0, 0.0, 1.0])
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0], 0.0])
    return forward, left


def ground_pixel_coords(x, y, R_w2c, height_m, erp_hw):
    """ERP pixel (mu, mv) of the ground point at ego (x forward, y left) metres. Same shape as x."""
    H, W = int(erp_hw[0]), int(erp_hw[1])
    R = np.asarray(R_w2c, float)
    forward, left = ego_to_world_dirs(R)
    x, y = np.asarray(x, float), np.asarray(y, float)
    p_w = np.stack([x * forward[0] + y * left[0], x * forward[1] + y * left[1],
                    np.full_like(x, -float(height_m))], -1).reshape(-1, 3)
    p_c = p_w @ R.T
    lon = np.arctan2(p_c[:, 0], p_c[:, 2])
    lat = np.arctan2(-p_c[:, 1], np.hypot(p_c[:, 0], p_c[:, 2]))
    mu = ((lon / (2 * np.pi) + 0.5) * W).astype(np.float32).reshape(x.shape)
    mv = ((0.5 - lat / np.pi) * H).astype(np.float32).reshape(x.shape)
    return mu, mv


def cell_centres(n, cell_m):
    """Ego (x forward, y left) metres of every BEV cell centre; row 0 forward, column 0 left."""
    c = (np.arange(n) - (n - 1) / 2.0) * cell_m
    x = -c[:, None] * np.ones((1, n))
    y = -c[None, :] * np.ones((n, 1))
    return x, y


def bev_affine(se2, n, cell_m):
    """2x3 affine taking SOURCE BEV pixels to QUERY BEV pixels for a source origin at
    se2 = (yaw, tx, ty) in the query ego frame (rotate then translate; bevloc.data.mapillary convention).

    Pixel (u, v) <-> ego (x, y): u = n/2 - y/cell - 0.5, v = n/2 - x/cell - 0.5 (x forward, y left)."""
    yaw, tx, ty = float(se2[0]), float(se2[1]), float(se2[2])
    c, s = np.cos(yaw), np.sin(yaw)
    o = (n - 1) / 2.0
    # source px -> source ego: x = (o - v) cell, y = (o - u) cell ; query ego: xq = c x - s y + tx, yq = s x + c y + ty
    # query px: u' = o - yq/cell, v' = o - xq/cell
    # u' = o - [s (o - v) + c (o - u) + ty/cell] = c u + s v + (o - s o - c o - ty/cell)
    # v' = o - [c (o - v) - s (o - u) - tx/cell]... careful with signs:
    # xq/cell = c (o - v) - s (o - u) + tx/cell  ->  v' = o - xq/cell = -s u + c v + (o - c o + s o - tx/cell)
    A = np.array([[c, s, o - s * o - c * o - ty / cell_m],
                  [-s, c, o - c * o + s * o - tx / cell_m]], np.float64)
    return A


def mosaic_ipm(pictures, valids, se2s, n, cell_m):
    """Composite several IPM pictures into the frame of the first one (the query).

    pictures[t] (n, n, 3) uint8 and valids[t] (n, n) bool are each in their own ego frame; se2s[t]
    = (yaw, tx, ty) of source t's origin in the query frame (the query itself is (0, 0, 0)).
    Each query cell takes the pixel of the valid source whose camera is nearest to it
    (IPM is most accurate close to the camera). Returns (img, valid)."""
    n = int(n)
    ys, xs = np.mgrid[0:n, 0:n]
    o = (n - 1) / 2.0
    ego_x = (o - ys) * cell_m
    ego_y = (o - xs) * cell_m
    best = np.full((n, n), np.inf)
    out = np.zeros((n, n, 3), np.uint8)
    out_valid = np.zeros((n, n), bool)
    for pic, val, se2 in zip(pictures, valids, se2s):
        A = bev_affine(se2, n, cell_m)
        w_img = cv2.warpAffine(pic, A.astype(np.float32), (n, n), flags=cv2.INTER_LINEAR, borderValue=0)
        w_val = cv2.warpAffine(val.astype(np.uint8), A.astype(np.float32), (n, n),
                               flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
        rng = np.hypot(ego_x - float(se2[1]), ego_y - float(se2[2]))     # distance of each cell to this camera
        take = w_val & (rng < best)
        out[take] = w_img[take]
        best[take] = rng[take]
        out_valid |= take
    return out, out_valid


def ipm_erp(erp_rgb, R_w2c, height_m, n, cell_m, blind_radius_m=1.2, erp_valid=None):
    """(n, n, 3) uint8 ground picture and (n, n) bool validity.

    A cell is valid outside the blind disc and, when ``erp_valid`` (H, W) bool is given, only if the
    ERP pixel it samples is valid (ego-vehicle body, dynamic objects, ... are masked there)."""
    x, y = cell_centres(n, cell_m)
    mu, mv = ground_pixel_coords(x, y, R_w2c, height_m, erp_rgb.shape[:2])
    img = cv2.remap(erp_rgb, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    valid = np.hypot(x, y) >= float(blind_radius_m)
    if erp_valid is not None:
        src_ok = cv2.remap(erp_valid.astype(np.uint8), mu, mv, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        valid &= src_ok.astype(bool)
    img[~valid] = 0
    return img, valid


def contact_feet(sem, erp_rgb, class_ids, R_w2c, height_m, n, cell_m, max_range_m=40.0,
                 facade_rows=12, min_run=3):
    """Camera-only contact line (kick-off §3.1 with a semantic map instead of LiDAR).

    For every ERP column, the lowest pixel of the given classes (building/wall/fence, or vegetation)
    is where that object meets the ground as seen from the camera; its flat-ground intersection is
    the object's foot on the map (the footprint edge an orthophoto shows). Returns a list of
    (x, y, colour) with ego metres and the mean facade colour sampled ``facade_rows`` above the foot.
    Columns whose lowest such pixel is above the horizon or beyond ``max_range_m`` are skipped;
    ``min_run`` consecutive class pixels are required so single mislabelled pixels do not paint."""
    H, W = sem.shape[:2]
    cls = np.isin(sem, list(class_ids))
    rows = np.arange(H)[:, None]
    lowest = np.where(cls, rows, -1).max(0)                       # (W,) lowest class row per column, -1 = none
    feet = []
    for u in range(W):
        v = int(lowest[u])
        if v < 0 or v - min_run + 1 < 0 or not cls[v - min_run + 1:v + 1, u].all():
            continue
        lat = (0.5 - (v + 0.5) / H) * np.pi
        if lat >= 0:
            continue
        lon = ((u + 0.5) / W - 0.5) * 2 * np.pi
        # ray in camera axes (x right, y down, z forward) -> world -> ground intersection
        d_cam = np.array([np.sin(lon) * np.cos(lat), -np.sin(lat), np.cos(lon) * np.cos(lat)])
        d_w = np.asarray(R_w2c, float).T @ d_cam
        if d_w[2] >= -1e-6:
            continue
        t = -float(height_m) / d_w[2]
        p = d_w * t
        fwd, left = ego_to_world_dirs(R_w2c)
        x, y = float(p @ fwd), float(p @ left)
        if np.hypot(x, y) > max_range_m:
            continue
        top = max(0, v - facade_rows)
        colour = erp_rgb[top:v + 1, u].reshape(-1, 3).mean(0)
        feet.append((x, y, colour))
    return feet


def paint_feet(img, valid, feet, n, cell_m, thickness=2):
    """Draw the contact feet into an IPM picture in place (row 0 forward, col 0 left) and mark them valid."""
    o = (n - 1) / 2.0
    for x, y, colour in feet:
        r = int(round(o - x / cell_m))
        c = int(round(o - y / cell_m))
        if 0 <= r < n and 0 <= c < n:
            cv2.circle(img, (c, r), max(0, thickness - 1), tuple(int(v) for v in colour), -1)
            valid[max(0, r - thickness + 1):r + thickness, max(0, c - thickness + 1):c + thickness] = True
    return img, valid


def depression_mask(erp_hw, max_depression_deg):
    """(H, W) bool: True where the ERP pixel looks less than ``max_depression_deg`` below the horizon.
    On a car-mounted 360° camera everything steeper than ~20° down is the vehicle's own body."""
    H, W = int(erp_hw[0]), int(erp_hw[1])
    lat = (0.5 - (np.arange(H) + 0.5) / H) * 180.0
    rows_ok = lat >= -float(max_depression_deg)
    return np.repeat(rows_ok[:, None], W, axis=1)
