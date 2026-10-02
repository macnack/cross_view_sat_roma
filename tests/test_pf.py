"""SE(2) particle filter on a synthetic route."""
from __future__ import annotations

import numpy as np

from bevloc.track.pf import SE2ParticleFilter


def test_pf_tracks_a_straight_route_with_gaussian_likelihood():
    rng = np.random.default_rng(0)
    pf = SE2ParticleFilter(n=400, seed=0)
    truth = np.array([100.0, 200.0])
    yaw = 30.0
    pf.init(truth + rng.normal(0, 5, 2), yaw + 3.0, sigma_xy=10.0, sigma_yaw=5.0)
    errs = []
    for _ in range(40):
        truth = truth + 2.0 * np.array([np.sin(np.radians(yaw)), np.cos(np.radians(yaw))])
        pf.predict(2.0 + rng.normal(0, 0.2), rng.normal(0, 0.2), rng.normal(0, 0.5), sigma_xy=0.5, sigma_yaw=1.0)
        d = np.linalg.norm(pf.particles[:, :2] - truth, axis=1)
        pf.update(-0.5 * (d / 3.0) ** 2)
        en, _ = pf.estimate()
        errs.append(np.linalg.norm(en - truth))
    assert np.median(errs[10:]) < 1.5
    assert pf.ess() > 0.2 * 400


def test_pf_yaw_estimate_wraps_around_north():
    pf = SE2ParticleFilter(n=200, seed=1)
    pf.init((0.0, 0.0), 359.0, sigma_xy=1.0, sigma_yaw=3.0)
    _, yaw = pf.estimate()
    assert min(abs(yaw - 359.0), abs(yaw + 1.0)) < 2.0      # circular mean, not ~180
