"""Mosaic of per-frame BEVs from relative poses (LiDAR odometry or OxTS)."""
from __future__ import annotations

import cv2
import numpy as np


def se2_from_pose(T_ref_k):
    """(R 2x2, t 2) of frame k in the reference frame; assumes near-level motion."""
    yaw = np.arctan2(T_ref_k[1, 0], T_ref_k[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]]), np.asarray(T_ref_k[:2, 3], float)


def warp_matrix(R, t, grid, canvas_n):
    """2x3 affine taking BEV pixel (col, row) of frame k to canvas pixel (col, row) of the reference.
    Both grids: row 0 = farthest forward, col 0 = farthest left, vehicle at the centre."""
    s, a, A = grid.cell, (grid.n / 2 - 0.5) * grid.cell, canvas_n / 2 - 0.5
    return np.array([[R[1, 1], R[1, 0], A - (R[1, 0] * a + R[1, 1] * a + t[1]) / s],
                     [R[0, 1], R[0, 0], A - (R[0, 0] * a + R[0, 1] * a + t[0]) / s]], np.float64)


def vehicle_pixel(t, grid, canvas_n):
    return np.array([canvas_n / 2 - 0.5 - t[1] / grid.cell, canvas_n / 2 - 0.5 - t[0] / grid.cell])


class Mosaic:
    """Nearest-range-wins compositing: where frames overlap, keep the observation made from
    the shortest range (smallest pixel footprint on the ground). Edges are OR-ed."""

    def __init__(self, grid, canvas_n):
        self.grid, self.n = grid, canvas_n
        self.img = np.zeros((canvas_n, canvas_n, 3), np.uint8)
        self.best = np.full((canvas_n, canvas_n), np.inf, np.float32)
        self.edge = np.zeros((canvas_n, canvas_n), bool)
        self.track = []
        x, y = grid.cell_centres()
        self._rho = np.hypot(x, y).astype(np.float32)

    def add(self, img, valid, edge, T_ref_k):
        R, t = se2_from_pose(T_ref_k)
        M, size = warp_matrix(R, t, self.grid, self.n), (self.n, self.n)
        wi = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR)
        wo = cv2.warpAffine(valid.astype(np.uint8), M, size, flags=cv2.INTER_NEAREST) > 0
        wr = cv2.warpAffine(self._rho, M, size, flags=cv2.INTER_LINEAR, borderValue=1e6)
        wr[~wo] = np.inf
        m = wr < self.best
        self.img[m], self.best[m] = wi[m], wr[m]
        if edge is not None:
            self.edge |= cv2.warpAffine(edge.astype(np.uint8), M, size, flags=cv2.INTER_NEAREST) > 0
        self.track.append(vehicle_pixel(t, self.grid, self.n))

    @property
    def valid(self):
        return np.isfinite(self.best)

    def crop_reference(self):
        """The reference frame's own grid window (the matcher query)."""
        c0 = (self.n - self.grid.n) // 2
        sl = slice(c0, c0 + self.grid.n)
        return self.img[sl, sl], self.valid[sl, sl], self.edge[sl, sl]
