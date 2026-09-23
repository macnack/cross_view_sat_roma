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


def ipm_erp(erp_rgb, R_w2c, height_m, n, cell_m, blind_radius_m=1.2):
    """(n, n, 3) uint8 ground picture and (n, n) bool validity (outside the blind disc)."""
    x, y = cell_centres(n, cell_m)
    mu, mv = ground_pixel_coords(x, y, R_w2c, height_m, erp_rgb.shape[:2])
    img = cv2.remap(erp_rgb, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    valid = np.hypot(x, y) >= float(blind_radius_m)
    img[~valid] = 0
    return img, valid
