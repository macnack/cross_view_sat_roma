"""Direct LiDAR -> dual-fisheye projection with Dur360BEV's lens model.

Float re-implementation of `XYZ2uv_poly` / `dualfisheye_project` from
third_party/Dur360BEV/dense/utils/project_pcd2image.py (E, Wenke; Durham), kept
numerically identical except that pixel coordinates are returned unrounded.
Target image: the *pre-rotated* dual fisheye (Dur360Frames.fisheye), 1280x640,
front lens on the right half.

Lens model: radius is linear in the off-axis angle phi with slope 2/fov_inner up
to `alpha` rad, then a second linear piece reaching r = 1 at fov_outer/2
(sigmoid-blended, k = 20). Dur360BEV defaults: 196 deg / 203 deg / alpha = 1.0.
"""
from __future__ import annotations

import numpy as np


def lidar_to_dualfisheye(xyz, width=1280, height=640, fov_outer_deg=203.0, alpha=1.0,
                         fov_inner_deg=196.0, R_cl=None, t_cl=None, k=20.0):
    p = np.asarray(xyz, np.float64)
    if R_cl is not None:
        p = p @ np.asarray(R_cl).T
    if t_cl is not None:
        p = p + np.asarray(t_cl)
    X, Y, Z = p[:, 0], p[:, 2], -p[:, 1]          # forward, up, right (their axis permutation)

    theta = np.arctan2(Y, Z)
    phi = np.arctan(np.sqrt(Y ** 2 + Z ** 2) / (X + 1e-6))   # >0 front lens, <0 back lens
    front = phi > 0

    a = 2.0 / np.radians(fov_inner_deg)
    half = np.radians(fov_outer_deg) / 2.0
    b = (1.0 - a * alpha) / (half - alpha)
    d = 1.0 - b * half

    def radius(x):
        s = 1.0 / (1.0 + np.exp(-k * (x - alpha)))
        return (1.0 - s) * (a * x) + s * (b * x + d)

    r = np.where(front, radius(np.abs(phi)), -radius(np.abs(phi)))
    x = r * np.cos(theta)
    y = np.abs(r) * np.sin(theta)
    x = np.where(X > 0, (x + 1) / 2, (x - 1) / 2)
    u = (x + 1) / 2 * width
    v = (-y + 1) / 2 * height
    return u, v, np.linalg.norm(p, axis=1)


def erp_maps(width=1280, height=640, fe_width=1280, fe_height=640, **lens):
    """cv2.remap maps building an ERP (forward at centre, azimuth to the right, zenith on
    top) from the pre-rotated dual fisheye, using the lens model above instead of
    fisheye_tools' ideal 203 deg equidistant assumption."""
    az = ((np.arange(width) + 0.5) / width - 0.5) * 2 * np.pi
    el = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    az, el = np.meshgrid(az, el)
    d = np.stack([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)], -1).reshape(-1, 3)
    u, v, _ = lidar_to_dualfisheye(d, fe_width, fe_height, **lens)
    return (u.reshape(height, width).astype(np.float32) - 0.5,
            v.reshape(height, width).astype(np.float32) - 0.5)
