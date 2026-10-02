"""2-point SE(2) RANSAC with fixed metric scale."""
from __future__ import annotations

import numpy as np

from bevloc.match.se2 import rigid_fit, se2_ransac


def _pairs(theta_deg=12.0, t=(3.0, -2.0), n=60, outliers=24, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 14, (n, 2))
    c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    b = a @ np.array([[c, -s], [s, c]]).T + np.asarray(t) + rng.normal(0, noise, (n, 2))
    b[:outliers] = rng.uniform(0, 56, (outliers, 2))
    return a, b


def test_rigid_fit_recovers_exact_transform():
    a, b = _pairs(outliers=0, noise=0.0)
    th, t = rigid_fit(a, b)
    assert abs(np.degrees(th) - 12.0) < 1e-6 and np.allclose(t, (3.0, -2.0), atol=1e-6)


def test_se2_ransac_rejects_outliers():
    a, b = _pairs()
    H, inl = se2_ransac(a, b, thresh=0.5, n_iter=300)
    assert H is not None and inl.sum() >= 30 and not inl[:24].any()
    assert abs(np.degrees(np.arctan2(H[1, 0], H[0, 0])) - 12.0) < 0.3
    assert np.allclose([H[0, 2], H[1, 2]], (3.0, -2.0), atol=0.1)
    assert abs(np.hypot(H[0, 0], H[1, 0]) - 1.0) < 1e-9           # scale exactly one


def test_se2_ransac_needs_two_points():
    H, inl = se2_ransac(np.zeros((1, 2)), np.zeros((1, 2)), thresh=0.5)
    assert H is None and inl.shape == (1,)
