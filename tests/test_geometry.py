import numpy as np

from bevloc import config as C
from bevloc.bev.fisheye import erp_maps, lidar_to_dualfisheye
from bevloc.bev.grid import BevGrid
from bevloc.bev.ground import contact_edges, contact_line, ground_pixel_mask, segment_ground
from bevloc.bev.mosaic import Mosaic, se2_from_pose, warp_matrix
from bevloc.bev.spherical import lidar_to_erp
from bevloc.data.calib import Calib
from bevloc.data.dur360 import POINT_BYTES, SCAN_COLS, SCAN_ROWS, read_scan

G = BevGrid(224, 0.25)


def test_config_and_calib():
    cfg = C.load()
    cal = Calib.from_config(cfg)
    assert (cfg.grid.n, cfg.grid.cell_m) == (224, 0.25)
    assert np.allclose(cal.R_cl, np.eye(3)) and np.allclose(cal.t_cl, [0, 0, -0.27])
    assert np.isclose(cal.camera_height, cfg.calib.lidar_height_m + 0.27)   # camera above the LiDAR


def test_fisheye_front_back_and_sides():
    u, v, _ = lidar_to_dualfisheye(np.array([[10.0, 0, 0], [-10.0, 0, 0]]))
    assert np.allclose([u[0], v[0]], [960, 320], atol=1e-3)      # forward = centre of the right (front) lens
    assert np.allclose([u[1], v[1]], [320, 320], atol=1e-3)      # backward = centre of the left lens
    ur, _, _ = lidar_to_dualfisheye(np.array([[10.0, -3.0, 0]]))
    assert ur[0] > 960                                           # a point to the right lands right of centre


def test_erp_built_from_lens_model_is_consistent_with_projection():
    """A direction must hit the same fisheye pixel via ERP lookup and via direct projection."""
    W, H = 1280, 640
    mx, my = erp_maps(W, H)
    rng = np.random.default_rng(0)
    d = rng.normal(size=(200, 3))
    d[:, 2] = -np.abs(d[:, 2]) * 0.5                             # around / below the horizon
    u, v, _ = lidar_to_erp(d, W, H)
    uf, vf, _ = lidar_to_dualfisheye(d)
    ui, vi = np.clip(u.astype(int), 0, W - 1), np.clip(v.astype(int), 0, H - 1)
    err = np.hypot(mx[vi, ui] + 0.5 - uf, my[vi, ui] + 0.5 - vf)
    assert np.median(err) < 1.0 and np.percentile(err, 90) < 2.5   # px; limited by ERP pixel quantisation


def test_read_scan_layout(tmp_path):
    raw = np.zeros((SCAN_ROWS * SCAN_COLS, POINT_BYTES), np.uint8)
    f, u = raw.view(np.float32).reshape(-1, 9), raw.view(np.uint32).reshape(-1, 9)
    f[5, :3], f[5, 3], u[5, 4], f[5, 5], u[5, 7], u[5, 8] = (3.0, 4.0, 0.0), 7.0, 123456, 42.0, 900, 5000
    p = tmp_path / "s.bin"
    raw.tofile(p)
    s = read_scan(p)
    assert s.xyz.shape == (128, 2048, 3) and s.valid.sum() == 1
    assert np.isclose(s.range_m[0, 5], 5.0) and s.t_ns[0, 5] == 123456
    assert s.intensity[0, 5] == 7.0 and s.reflectivity[0, 5] == 42.0 and s.ambient[0, 5] == 900
    assert np.allclose(s.points(), [[3, 4, 0]])


def _scene():
    """Flat ground at z = -1.57 plus a 3 m wall 8 m ahead spanning y in [-4, 4]."""
    gx, gy = np.meshgrid(np.arange(-20, 20, 0.2), np.arange(-20, 20, 0.2))
    ground = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, -1.57)]
    ground = ground[~((ground[:, 0] > 8) & (np.abs(ground[:, 1]) < 4))]          # nothing visible behind the wall
    wy, wz = np.meshgrid(np.arange(-4, 4, 0.05), np.arange(-1.57, 1.5, 0.05))
    wall = np.c_[np.full(wy.size, 8.0), wy.ravel(), wz.ravel()]
    return ground, wall


def test_ground_segmentation_and_contact_line():
    ground, wall = _scene()
    pts = np.vstack([ground, wall])
    is_g = segment_ground(pts, 1.57)
    assert is_g[:len(ground)].mean() > 0.99
    assert is_g[len(ground):][wall[:, 2] > -1.2].mean() < 0.01                    # wall above 0.35 m is never ground
    W, H = 1280, 640
    ct = contact_line(pts, is_g, W, H, sensor_height=1.57, margin_px=0)
    dep = np.degrees(np.arctan(1.57 / 8.0))                                       # wall foot, straight ahead
    assert abs(ct.row[W // 2] - (0.5 + dep / 180) * H) < 1.0
    assert np.allclose(ct.foot_xy[W // 2], [8.0, 0.0], atol=0.06)
    assert ct.height[W // 2] > 2.8                                                # 3 m wall
    assert ct.row[W // 4] == H / 2 and not ct.has_obstacle[W // 4]                # nothing to the left: horizon
    m = ground_pixel_mask(ct.row, H)
    assert m[H - 1, W // 2] and not m[H // 2 + 2, W // 2]


def test_wall_becomes_a_straight_edge_at_its_footprint():
    ground, wall = _scene()
    pts = np.vstack([ground, wall])
    is_g = segment_ground(pts, 1.57)
    W, H = 1280, 640
    edge = contact_edges(G, contact_line(pts, is_g, W, H, sensor_height=1.57))
    r, c = np.nonzero(edge)
    r_wall, _, _ = G.to_cell([8.0], [0.0])
    assert len(r) > 20 and np.abs(r - r_wall[0]).max() <= 1       # a straight line AT 8 m (not 9 m), not a fan
    assert c.min() < 112 - 12 and c.max() > 112 + 12              # spanning most of the +-4 m wall


def test_mosaic_warp_places_a_forward_moved_frame_correctly():
    T = np.eye(4)
    T[0, 3] = 5.0                                                 # frame k is 5 m ahead of the reference
    R, t = se2_from_pose(T)
    M = warp_matrix(R, t, G, 320)
    centre_k = M @ [111.5, 111.5, 1]                              # vehicle of frame k, in canvas px (col, row)
    assert np.allclose(centre_k, [159.5, 159.5 - 20])             # 5 m = 20 cells towards row 0 (forward)
    m = Mosaic(G, 320)
    img = np.full((224, 224, 3), 200, np.uint8)
    m.add(img, np.ones((224, 224), bool), None, np.eye(4))
    m.add(np.full((224, 224, 3), 50, np.uint8), np.ones((224, 224), bool), None, T)
    assert m.img[160, 160].mean() > 150                           # near the reference vehicle: reference wins
    assert m.img[160 - 40, 160].mean() < 100                      # 10 m ahead: the closer frame k wins
    assert m.crop_reference()[1].all()


def test_overhanging_canopy_does_not_cut_the_ground():
    ground, wall = _scene()
    cy, cx = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(4, 6, 0.1))
    canopy = np.c_[cx.ravel(), cy.ravel(), np.full(cx.size, 1.5)]          # slab 3 m above ground, 4-6 m ahead
    pts = np.vstack([ground, wall, canopy])
    is_g = segment_ground(pts, 1.57)
    ct = contact_line(pts, is_g, 1280, 640, sensor_height=1.57, margin_px=0)
    assert np.allclose(ct.foot_xy[640], [8.0, 0.0], atol=0.06)             # contact is still the wall, not the canopy
    assert ct.height[640] > 2.8
