"""Minimal SE(2) particle filter in the map frame (E, N, up-bearing degrees CW from grid north).

Motion is applied in the ego frame (x forward, y left), matching ``bevloc.data.mapillary``.
Weights are kept as log-weights; systematic resampling when the effective sample size drops
below half the particle count.
"""
from __future__ import annotations

import numpy as np


class SE2ParticleFilter:
    def __init__(self, n=512, seed=0):
        self.n = int(n)
        self.rng = np.random.default_rng(seed)
        self.particles = np.zeros((self.n, 3))      # e, n, yaw_deg
        self.logw = np.zeros(self.n)

    def init(self, en, yaw_deg, sigma_xy=10.0, sigma_yaw=5.0):
        self.particles[:, :2] = np.asarray(en, float) + self.rng.normal(0, sigma_xy, (self.n, 2))
        self.particles[:, 2] = (float(yaw_deg) + self.rng.normal(0, sigma_yaw, self.n)) % 360.0
        self.logw[:] = 0.0

    def predict(self, dx, dy, dyaw_deg, sigma_xy=1.0, sigma_yaw=1.0):
        """Ego-frame motion (dx forward, dy left, dyaw) with additive Gaussian noise per particle."""
        b = np.radians(self.particles[:, 2])
        fwd = np.stack([np.sin(b), np.cos(b)], 1)
        left = np.stack([-np.cos(b), np.sin(b)], 1)
        dxs = float(dx) + self.rng.normal(0, sigma_xy, self.n)
        dys = float(dy) + self.rng.normal(0, sigma_xy, self.n)
        self.particles[:, :2] += dxs[:, None] * fwd + dys[:, None] * left
        self.particles[:, 2] = (self.particles[:, 2] + float(dyaw_deg)
                                + self.rng.normal(0, sigma_yaw, self.n)) % 360.0

    def update(self, loglik):
        self.logw += np.asarray(loglik, float)
        self.logw -= self.logw.max()
        if self.ess() < 0.5 * self.n:
            self._resample()

    def weights(self):
        w = np.exp(self.logw)
        return w / w.sum()

    def ess(self):
        w = self.weights()
        return 1.0 / float((w ** 2).sum())

    def _resample(self):
        w = self.weights()
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.searchsorted(np.cumsum(w), pos)
        self.particles = self.particles[np.minimum(idx, self.n - 1)].copy()
        self.logw[:] = 0.0

    def estimate(self):
        """Weighted mean position and circular-mean bearing."""
        w = self.weights()
        en = (w[:, None] * self.particles[:, :2]).sum(0)
        b = np.radians(self.particles[:, 2])
        yaw = np.degrees(np.arctan2((w * np.sin(b)).sum(), (w * np.cos(b)).sum())) % 360.0
        return en, float(yaw)
