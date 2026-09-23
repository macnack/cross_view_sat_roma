"""Flat-ground IPM for an equirectangular camera."""
from __future__ import annotations

import numpy as np

from bevloc.bev.ipm_sphere import cell_centres, ground_pixel_coords, ipm_erp

# R_w2c: cam x = east, cam y = -up, cam z = north (the convention of tests/test_lift_splat.py)
R_NORTH = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def test_ground_point_ahead_maps_to_erp_centre_column():
    h, W, H = 1.7, 1280, 640
    mu, mv = ground_pixel_coords(np.array([10.0]), np.array([0.0]), R_NORTH, h, (H, W))
    assert abs(mu[0] - W / 2) < 1e-6                       # straight ahead = centre column
    lat = -np.arctan2(h, 10.0)
    assert abs(mv[0] - (0.5 - lat / np.pi) * H) < 1e-4     # below the horizon by atan(h/10)


def test_ipm_marker_lands_in_the_forward_cell():
    h, W, H, n, cell = 1.7, 1280, 640, 64, 0.5
    r, c = n // 2 - int(10.0 / cell), n // 2            # row 0 is forward, col 0 is left
    x, y = cell_centres(n, cell)
    assert abs(x[r, c] - 9.75) < 1e-9 and abs(y[r, c] + 0.25) < 1e-9
    erp = np.zeros((H, W, 3), np.uint8)
    mu, mv = ground_pixel_coords(x[r:r + 1, c:c + 1], y[r:r + 1, c:c + 1], R_NORTH, h, (H, W))
    u, v = int(round(float(mu[0, 0]))), int(round(float(mv[0, 0])))
    erp[v - 1: v + 2, u - 1: u + 2] = 255              # 3 px marker at that cell's ground point
    img, valid = ipm_erp(erp, R_NORTH, h, n, cell)
    assert img[r, c].max() >= 200                       # bilinear sample inside the marker
    assert img[r - 4, c].max() == 0                     # 2 m further ahead is not the marker
    assert valid[r, c] and not valid[n // 2, n // 2]    # blind disc under the camera
