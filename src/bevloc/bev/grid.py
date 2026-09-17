"""Metric, ego-centred, gravity-aligned BEV grid; IPM and LiDAR splatting.

Frames
  body  : LiDAR frame, x forward / y left / z up (FLU), origin at the LiDAR.
  level : body rotated by roll/pitch only (NOT yaw) so that z is anti-gravity;
          x stays the horizontal projection of vehicle-forward. Origin unchanged.
  BEV image: looks like a north-up map when the vehicle heads north:
          row 0 = farthest forward, col 0 = farthest left.
          cell (r, c) centre:  x = (N/2 - r - 0.5) * s,  y = (N/2 - c - 0.5) * s.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .spherical import lidar_to_erp


@dataclass(frozen=True)
class BevGrid:
    n: int = 224          # cells per side
    cell: float = 0.25    # metres

    @property
    def extent(self) -> float:
        return self.n * self.cell

    def cell_centres(self):
        """(n, n) arrays of level-frame x (forward) and y (left) per cell."""
        k = (self.n / 2 - np.arange(self.n) - 0.5) * self.cell
        return np.repeat(k[:, None], self.n, 1), np.repeat(k[None, :], self.n, 0)

    def to_cell(self, x, y):
        """Level-frame metres -> integer (row, col) and in-bounds mask."""
        r = np.floor(self.n / 2 - np.asarray(x) / self.cell).astype(int)
        c = np.floor(self.n / 2 - np.asarray(y) / self.cell).astype(int)
        ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
        return r, c, ok


def level_from_body(roll: float, pitch: float) -> np.ndarray:
    """Rotation taking FLU body vectors to the gravity-aligned level frame.

    roll, pitch in radians with the OxTS/aerospace sign (roll + = right side
    down, pitch + = nose up). In FLU that is R_y(-pitch) @ R_x(roll).
    """
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])  # R_y(-pitch)
    return Ry @ Rx


def ipm_bev(erp, grid: BevGrid, roll, pitch, sensor_height, R_cl=None, t_cl=None,
            erp_valid=None, ground_z=None):
    """Flat-ground inverse perspective mapping of an ERP image.

    sensor_height: height of the body-frame origin (LiDAR) above the ground.
    ground_z: optional (n, n) per-cell ground height in the level frame,
        overriding the flat plane z = -sensor_height.
    Returns (n, n, 3) uint8 image and (n, n) bool validity.
    """
    h, w = erp.shape[:2]
    x, y = grid.cell_centres()
    z = np.full_like(x, -float(sensor_height)) if ground_z is None else ground_z
    P_level = np.stack([x, y, z], -1).reshape(-1, 3)
    P_body = P_level @ level_from_body(roll, pitch)      # == R^T applied to rows
    u, v, _ = lidar_to_erp(P_body, w, h, R_cl, t_cl)
    mu = (u % w).astype(np.float32).reshape(grid.n, grid.n)
    mv = v.astype(np.float32).reshape(grid.n, grid.n)
    img = cv2.remap(erp, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    valid = np.isfinite(mv) & (mv >= 0) & (mv <= h - 1)
    if erp_valid is not None:
        valid &= cv2.remap(erp_valid.astype(np.uint8), mu, mv, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_WRAP) > 0
    img[~valid] = 0
    return img, valid


def lidar_bev(erp, points_xyz, grid: BevGrid, roll, pitch, R_cl=None, t_cl=None,
              erp_valid=None, select="top", z_max=None):
    """Splat LiDAR points, coloured from the ERP, into the BEV grid.

    select: which point colours a cell when several fall in it:
        "top" (highest z, what a nadir camera sees), "bottom", or "mean".
    Returns image (n,n,3) uint8, validity (n,n) bool, height map (n,n) float
    (level-frame z of the selected point; NaN where invalid).
    """
    h, w = erp.shape[:2]
    p = np.asarray(points_xyz, np.float64)
    u, v, _ = lidar_to_erp(p, w, h, R_cl, t_cl)
    ui = np.clip(u.astype(int), 0, w - 1)
    vi = np.clip(v.astype(int), 0, h - 1)
    P = p @ level_from_body(roll, pitch).T
    r, c, ok = grid.to_cell(P[:, 0], P[:, 1])
    if erp_valid is not None:
        ok &= erp_valid[vi, ui]
    if z_max is not None:
        ok &= P[:, 2] <= z_max
    r, c, z, rgb = r[ok], c[ok], P[ok, 2], erp[vi[ok], ui[ok]].astype(np.float64)
    flat = r * grid.n + c

    img = np.zeros((grid.n * grid.n, 3), np.float64)
    hgt = np.full(grid.n * grid.n, np.nan)
    if select == "mean":
        cnt = np.bincount(flat, minlength=grid.n ** 2).astype(float)
        for k in range(3):
            img[:, k] = np.bincount(flat, rgb[:, k], grid.n ** 2) / np.maximum(cnt, 1)
        hgt = np.where(cnt > 0, np.bincount(flat, z, grid.n ** 2) / np.maximum(cnt, 1), np.nan)
    else:
        order = np.argsort(z if select == "top" else -z)   # last write wins
        img[flat[order]] = rgb[order]
        hgt[flat[order]] = z[order]
    valid = np.isfinite(hgt).reshape(grid.n, grid.n)
    return (img.reshape(grid.n, grid.n, 3).round().astype(np.uint8), valid,
            hgt.reshape(grid.n, grid.n))
