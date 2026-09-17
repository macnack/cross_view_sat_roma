"""Spherical-camera geometry for Dur360BEV.

LiDAR frame: x forward, y left, z up. With no published camera<->LiDAR
extrinsics we start from SI2BEV's assumption (same origin, axis permutation
only) and expose the residual as (R_cl, t_cl), mapping LiDAR -> camera.
"""
from __future__ import annotations

import numpy as np


def lidar_to_erp(xyz, width, height, R_cl=None, t_cl=None):
    """Project LiDAR points to ERP pixels.

    ERP convention (Dur360BEV dualfisheye2equi, center_angle=0): image centre
    is vehicle-forward, azimuth increases to the right, top row is zenith.
    Returns u, v (float pixels) and range [m].
    """
    p = np.asarray(xyz, dtype=np.float64)
    if R_cl is not None:
        p = p @ np.asarray(R_cl).T
    if t_cl is not None:
        p = p + np.asarray(t_cl)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    rng = np.linalg.norm(p, axis=1)
    az = np.arctan2(-y, x)                      # + to the right of forward
    el = np.arcsin(np.clip(z / np.maximum(rng, 1e-9), -1, 1))  # + up
    u = (az / (2 * np.pi) + 0.5) * width
    v = (0.5 - el / np.pi) * height
    return u, v, rng
