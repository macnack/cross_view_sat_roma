"""Fixed-scale SE(2) consensus on Sat-RoMa correspondences (kick-off H7).

Two correspondences determine (theta, tx, ty); at inlier ratio 0.3 and p = 0.99 that is 49 trials
against 567 for a homography. Scale is one because query and reference share the GSD and the
query is metric from the camera height. Coordinates are whatever the caller passes (here the
matcher's patch / cell units, where 1 query patch = 1 reference cell at the 4:1 setting).
"""
from __future__ import annotations

import numpy as np


def rigid_fit(a, b):
    """Least-squares rotation + translation (no scale) taking points a onto b. Returns (theta, t)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ca, cb = a.mean(0), b.mean(0)
    Hm = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d if d != 0 else 1.0]) @ U.T
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    t = cb - R @ ca
    return theta, t


def _H(theta, t):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, t[0]], [s, c, t[1]], [0.0, 0.0, 1.0]])


def se2_ransac(a, b, thresh, n_iter=500, seed=0, min_inliers=4):
    """2-point RANSAC for a rigid transform a -> b. Returns (H 3x3 or None, inlier mask (N,))."""
    a, b = np.asarray(a, float).reshape(-1, 2), np.asarray(b, float).reshape(-1, 2)
    N = len(a)
    best, best_inl = None, np.zeros(N, bool)
    if N < 2:
        return None, best_inl
    rng = np.random.default_rng(seed)
    ah = np.c_[a, np.ones(N)]
    for _ in range(int(n_iter)):
        i, j = rng.choice(N, 2, replace=False)
        if np.linalg.norm(a[i] - a[j]) < 1e-9:
            continue
        theta, t = rigid_fit(a[[i, j]], b[[i, j]])
        H = _H(theta, t)
        err = np.linalg.norm((ah @ H.T)[:, :2] - b, axis=1)
        inl = err <= thresh
        if inl.sum() > best_inl.sum():
            best, best_inl = H, inl
    if best is None or best_inl.sum() < min_inliers:
        return None, best_inl
    theta, t = rigid_fit(a[best_inl], b[best_inl])
    return _H(theta, t), best_inl
