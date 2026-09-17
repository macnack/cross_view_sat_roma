"""LiDAR ground segmentation and the per-azimuth ground/obstacle contact line.

Training-time / oracle only: at inference the contact line is what the geometry
head has to predict from the image.

Contact = the FOOT of the lowest non-ground return in an image column: that point's own
(x, y), dropped to the local floor. (Using the row of the lowest object return and
intersecting it with the ground plane overshoots by ground_tol / sensor_height of the range,
~1 m at 8 m, because the bottom `ground_tol` of every wall is labelled ground.)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .spherical import lidar_to_erp


def segment_ground(P_level, sensor_height, cell=0.5, extent=60.0, ground_tol=0.20,
                   max_dev=1.2, max_range=40.0, return_floor=False):
    """Label level-frame points as ground.

    A point is ground if it lies within `ground_tol` of the local floor (min-z over a `cell`
    grid, median-smoothed), and that floor is itself within `max_dev` of the nominal plane
    z = -sensor_height (rejects "floors" that are really canopy or roofs with nothing visible
    beneath). Points beyond max_range are never ground.
    Returns bool array; with return_floor also the local floor height per point (nominal
    plane where unknown).
    """
    n = int(extent / cell)
    ix = np.floor((P_level[:, 0] + extent / 2) / cell).astype(int)
    iy = np.floor((P_level[:, 1] + extent / 2) / cell).astype(int)
    inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (np.linalg.norm(P_level[:, :2], axis=1) < max_range)
    floor = np.full(n * n, np.inf)
    np.minimum.at(floor, ix[inside] * n + iy[inside], P_level[inside, 2])
    floor = floor.reshape(n, n)
    ok = np.isfinite(floor) & (np.abs(floor + sensor_height) < max_dev)
    f = cv2.medianBlur(np.where(ok, floor, -sensor_height).astype(np.float32), 5)
    ground = np.zeros(len(P_level), bool)
    floor_z = np.full(len(P_level), -float(sensor_height))
    floor_z[inside] = f[ix[inside], iy[inside]]
    ground[inside] = (P_level[inside, 2] - floor_z[inside] < ground_tol) & ok[ix[inside], iy[inside]]
    return (ground, floor_z) if return_floor else ground


@dataclass
class Contact:
    row: np.ndarray        # (W,) v_c(u): image row of the contact foot; horizon row where no obstacle
    foot_xy: np.ndarray    # (W, 2) body-frame x, y of the foot; NaN where no obstacle
    top_row: np.ndarray    # (W,) row of the highest non-ground return (horizon row where none)
    height: np.ndarray     # (W,) object height above its floor [m]; 0 where none (LiDAR FOV tops at +22.5 deg)

    @property
    def has_obstacle(self):
        return np.isfinite(self.foot_xy[:, 0])


def contact_line(points_body, is_ground, width, height, R_cl=None, t_cl=None, foot_z_body=None,
                 sensor_height=None, min_range=1.5, margin_px=1, contact_max_height=0.7) -> Contact:
    """Per ERP column: the foot of the lowest non-ground return, and the object's top.

    foot_z_body: per-point z of the local floor in the body frame (from segment_ground's floor);
    if None, the flat plane z = -sensor_height is used.
    contact_max_height: only returns this close to the floor can define a contact. Overhanging
    content (canopy, signs, barrier arms) has no ground contact and must not cut off the ground
    visible beneath it; it still counts towards the column's object height.
    """
    sel = ~is_ground & (np.linalg.norm(points_body, axis=1) > min_range)
    obj = points_body[sel]
    fz = np.full(len(obj), -float(sensor_height)) if foot_z_body is None else np.asarray(foot_z_body)[sel]
    foot = np.c_[obj[:, :2], fz]
    u, v, _ = lidar_to_erp(obj, width, height, R_cl, t_cl)
    _, vf, _ = lidar_to_erp(foot, width, height, R_cl, t_cl)
    ui = np.clip(u.astype(int), 0, width - 1)

    horizon = height / 2.0
    row = np.full(width, horizon)
    top = np.full(width, horizon)
    foot_xy = np.full((width, 2), np.nan)
    hgt = np.zeros(width)
    order = np.argsort(vf)                         # ascending: the last write per column is the lowest foot
    o = order[(vf[order] > horizon) & ((obj[order, 2] - fz[order]) < contact_max_height)]
    row[ui[o]] = vf[o]
    foot_xy[ui[o]] = foot[o, :2]
    np.minimum.at(top, ui, v)
    np.maximum.at(hgt, ui, obj[:, 2] - fz)
    row = np.where(np.isfinite(foot_xy[:, 0]), row + margin_px, horizon)
    return Contact(row, foot_xy, top, hgt)


def ground_pixel_mask(contact_row, height, erp_valid=None, col_dilate=2):
    """ERP pixels that are ground: below the contact row (taken conservatively over +-col_dilate columns)."""
    k = 2 * col_dilate + 1
    vc = np.r_[contact_row[-col_dilate:], contact_row, contact_row[:col_dilate]].astype(np.float32)
    vc = cv2.dilate(vc[None], np.ones((1, k)))[0][col_dilate:-col_dilate]
    m = np.arange(height)[:, None] > vc[None, :]
    return m if erp_valid is None else (m & erp_valid)


def contact_edges(grid, contact: Contact, R_level_body=None, min_obj_height=1.5, max_jump_m=0.75, thickness=1):
    """Draw the contact feet of tall objects into the BEV as thin edges: a wall is a line in a
    top-down view. Neighbouring columns are joined only if their feet are < max_jump_m apart,
    so separate objects are not bridged. Returns (n, n) bool."""
    xy = contact.foot_xy
    if R_level_body is not None:                  # gravity alignment (small angles: xy only)
        xy = (np.c_[xy, np.zeros(len(xy))] @ np.asarray(R_level_body).T)[:, :2]
    ok = contact.has_obstacle & (contact.height > min_obj_height)
    xy = np.nan_to_num(xy)
    col = grid.n / 2 - xy[:, 1] / grid.cell - 0.5
    row = grid.n / 2 - xy[:, 0] / grid.cell - 0.5
    ok &= (np.abs(col - grid.n / 2) < grid.n) & (np.abs(row - grid.n / 2) < grid.n)   # keep cv2 coordinates sane
    img = np.zeros((grid.n, grid.n), np.uint8)
    S, w = 8, len(xy)
    for i in range(w):
        j = (i + 1) % w
        if ok[i] and ok[j] and np.hypot(*(xy[i] - xy[j])) < max_jump_m:
            p = (int(round(col[i] * S)), int(round(row[i] * S)))
            q = (int(round(col[j] * S)), int(round(row[j] * S)))
            cv2.line(img, p, q, 1, thickness, cv2.LINE_8, shift=3)
    return img > 0


def collapse_above_contact(erp, grid, contact: Contact, sensor_height, erp_valid=None, R_cl=None, t_cl=None,
                           min_obj_height=0.4, depth_per_height=1.0, min_depth=0.5, max_depth=8.0):
    """REJECTED variant (radial fans; kept for the record, see docs/decisions.md).

    Paint the mean colour of each column between the object's top and its contact into the cells
    radially behind the foot, over a depth proportional to the object's height."""
    h, w = erp.shape[:2]
    rows = np.arange(h)[:, None]
    band = (rows >= np.floor(contact.top_row)[None]) & (rows < np.floor(contact.row)[None])
    if erp_valid is not None:
        band &= erp_valid
    cnt = band.sum(0)
    colour = (erp.astype(np.float64) * band[..., None]).sum(0) / np.maximum(cnt, 1)[:, None]
    rho_c = np.nan_to_num(np.hypot(contact.foot_xy[:, 0], contact.foot_xy[:, 1]), nan=np.inf)
    has = contact.has_obstacle & (cnt >= 2) & (contact.height > min_obj_height)
    depth = np.clip(depth_per_height * contact.height, min_depth, max_depth)

    x, y = grid.cell_centres()
    P = np.stack([x, y, np.full_like(x, -sensor_height)], -1).reshape(-1, 3)
    u, _, _ = lidar_to_erp(P, w, h, R_cl, t_cl)
    ui = np.clip(u.astype(int), 0, w - 1)
    rho = np.hypot(P[:, 0], P[:, 1])
    m = has[ui] & (rho >= rho_c[ui]) & (rho <= rho_c[ui] + depth[ui])
    img = np.zeros((grid.n * grid.n, 3), np.uint8)
    img[m] = colour[ui[m]].round().astype(np.uint8)
    return img.reshape(grid.n, grid.n, 3), m.reshape(grid.n, grid.n)
