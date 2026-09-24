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


def test_mosaic_ipm_warps_a_source_frame_by_its_relative_pose_and_prefers_the_nearer_camera():
    from bevloc.bev.ipm_sphere import mosaic_ipm
    n, cell = 64, 0.5
    q = np.zeros((n, n, 3), np.uint8)
    s = np.zeros((n, n, 3), np.uint8)
    # source frame sits 4 m BEHIND the query (its origin at query ego x = -4), same heading
    se2_src = (0.0, -4.0, 0.0)
    # a marker 10 m ahead of the SOURCE camera -> 6 m ahead of the query camera
    rs, cs = n // 2 - int(round(10.0 / cell)) , n // 2
    s[rs - 1:rs + 2, cs - 1:cs + 2] = (0, 255, 0)
    valid = np.ones((n, n), bool)
    rq = n // 2 - int(round(6.0 / cell))
    q_valid = valid.copy()
    q_valid[rq - 2:rq + 3, :] = False                 # the query cannot see that strip (say, its blind disc)
    img, vmask = mosaic_ipm([q, s], [q_valid, valid], [(0.0, 0.0, 0.0), se2_src], n, cell)
    assert img[rq, cs, 1] >= 200 and vmask[rq, cs]   # the source fills it, warped 4 m forward
    # a cell 2 m ahead of the query is nearer to the query camera than to the source: the query's pixel wins
    q2 = q.copy(); q2[:] = (255, 0, 0)
    s2 = s.copy(); s2[:] = (0, 0, 255)
    img2, _ = mosaic_ipm([q2, s2], [valid, valid], [(0.0, 0.0, 0.0), se2_src], n, cell)
    r2 = n // 2 - int(round(2.0 / cell))
    assert tuple(img2[r2, cs]) == (255, 0, 0)
    # a cell 3 m BEHIND the query is nearer to the source camera (1 m) than to the query (3 m): source wins
    r3 = n // 2 + int(round(3.0 / cell))
    assert tuple(img2[r3, cs]) == (0, 0, 255)


def test_erp_validity_and_depression_mask_propagate_to_cells():
    from bevloc.bev.ipm_sphere import depression_mask
    h, W, H, n, cell = 1.7, 1280, 640, 64, 0.5
    erp = np.full((H, W, 3), 200, np.uint8)
    dep = depression_mask((H, W), 20.0)
    assert dep[0, 0] and not dep[H - 1, 0]                       # sky row valid, nadir row invalid
    img, valid = ipm_erp(erp, R_NORTH, h, n, cell, blind_radius_m=0.0, erp_valid=dep)
    x, y = cell_centres(n, cell)
    rng = np.hypot(x, y)
    r_cut = h / np.tan(np.radians(20.0))                          # 4.67 m: the ego-body cut on the ground
    assert valid[rng > r_cut + 1.0].all()
    assert not valid[rng < r_cut - 1.0].any()
    assert (img[~valid] == 0).all()


def test_contact_feet_paint_the_facade_colour_at_the_wall_foot():
    from bevloc.bev.ipm_sphere import contact_feet, paint_feet
    h, W, H, n, cell = 1.7, 1280, 640, 64, 0.5
    # a "building" occupying ERP rows 200..(contact row) in columns around the centre; ground 10 m ahead
    mu, mv = ground_pixel_coords(np.array([10.0]), np.array([0.0]), R_NORTH, h, (H, W))
    vc = int(round(mv[0]))                              # contact row of that wall on the ground at 10 m
    sem = np.full((H, W), 10, np.uint8)                 # sky everywhere
    sem[vc + 1:, :] = 0                                 # road below the contact
    u0 = int(round(mu[0]))
    sem[200:vc + 1, u0 - 20:u0 + 21] = 2                # building down to the contact row
    erp = np.zeros((H, W, 3), np.uint8)
    erp[200:vc + 1, u0 - 20:u0 + 21] = (200, 50, 50)    # a red facade
    feet = contact_feet(sem, erp, (2, 3, 4), R_NORTH, h, n, cell, max_range_m=40.0)
    assert len(feet) > 0
    img = np.zeros((n, n, 3), np.uint8)
    valid = np.zeros((n, n), bool)
    paint_feet(img, valid, feet, n, cell, thickness=1)
    r, c = n // 2 - int(round(10.0 / cell)), n // 2
    band = img[r - 2:r + 3, c - 1:c + 2]
    assert band[..., 0].max() >= 150 and band[..., 1].max() <= 80   # red foot mark near (10 m, 0)
    assert valid[r - 2:r + 3, c - 1:c + 2].any()
    assert not valid[n // 2 + 10, c]                                # nothing painted behind the camera


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
