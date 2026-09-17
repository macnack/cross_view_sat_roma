import numpy as np

from bevloc.bev.grid import BevGrid, ipm_bev, level_from_body, lidar_bev
from bevloc.bev.spherical import lidar_to_erp

G = BevGrid(224, 0.25)


def test_grid_extent_and_cell_of_known_points():
    assert G.extent == 56.0
    # 10 m ahead, 5 m to the LEFT -> above centre, left of centre
    r, c, ok = G.to_cell([10.0], [5.0])
    assert ok[0] and (r[0], c[0]) == (112 - 40, 112 - 20)
    # 10 m behind, 5 m to the RIGHT (first cell on the far side of the axis: 112 + 40 - 1 + 1)
    r, c, ok = G.to_cell([-10.0], [-5.0])
    assert (r[0], c[0]) == (152, 132)
    assert not G.to_cell([28.1], [0.0])[2][0]


def test_cell_centres_roundtrip():
    x, y = G.cell_centres()
    r, c, ok = G.to_cell(x.ravel(), y.ravel())
    assert ok.all()
    assert (r.reshape(224, 224) == np.arange(224)[:, None]).all()
    assert (c.reshape(224, 224) == np.arange(224)[None, :]).all()


def test_erp_convention():
    u, v, _ = lidar_to_erp(np.array([[5.0, 0, 0], [0, -5.0, 0], [0, 5.0, 0], [5.0, 0, 5.0]]), 1280, 640)
    assert np.allclose(u[:3], [640, 960, 320])       # forward centre, right at 3/4, left at 1/4
    assert np.allclose(v[:3], 320) and np.isclose(v[3], 160)  # 45 deg up -> quarter height


def test_level_from_body_signs():
    # nose up by 10 deg: body-forward points upward in the level frame
    f = level_from_body(0.0, np.radians(10)) @ [1, 0, 0]
    assert f[2] > 0 and np.isclose(f[2], np.sin(np.radians(10)))
    # roll +10 deg = right side down: body-left points upward
    l = level_from_body(np.radians(10), 0.0) @ [0, 1, 0]
    assert l[2] > 0


def _paint(erp, P_level, roll, pitch, colour):
    p_body = np.asarray(P_level) @ level_from_body(roll, pitch)
    u, v, _ = lidar_to_erp(p_body[None], erp.shape[1], erp.shape[0])
    uu, vv = int(round(u[0])), int(round(v[0]))
    erp[vv - 3:vv + 4, uu - 3:uu + 4] = colour


def test_ipm_places_ground_point_in_expected_cell():
    roll, pitch, h = np.radians(3.0), np.radians(-4.0), 1.9
    erp = np.zeros((640, 1280, 3), np.uint8)
    _paint(erp, [8.0, -6.0, -h], roll, pitch, (255, 0, 0))   # 8 m ahead, 6 m right, on the ground
    img, valid = ipm_bev(erp, G, roll, pitch, h)
    rr, cc = np.nonzero(img[..., 0] > 128)
    r0, c0, _ = G.to_cell([8.0], [-6.0])
    assert valid.all() and len(rr) > 0
    assert abs(rr.mean() - r0[0]) <= 1 and abs(cc.mean() - c0[0]) <= 1
    # ignoring roll/pitch must misplace it: the test is sensitive to gravity alignment
    img0, _ = ipm_bev(erp, G, 0.0, 0.0, h)
    rr0, cc0 = np.nonzero(img0[..., 0] > 128)
    assert np.hypot(rr0.mean() - r0[0], cc0.mean() - c0[0]) > 3


def test_lidar_bev_cell_colour_and_top_selection():
    erp = np.zeros((640, 1280, 3), np.uint8)
    lo, hi = np.array([6.10, 4.10, -1.9]), np.array([6.15, 4.15, 1.0])   # same cell, ground and roof
    _paint(erp, lo, 0, 0, (0, 255, 0))
    _paint(erp, hi, 0, 0, (0, 0, 255))
    r0, c0, _ = G.to_cell([6.10], [4.10])
    for sel, ch in (("top", 2), ("bottom", 1)):
        img, valid, hgt = lidar_bev(erp, np.stack([lo, hi]), G, 0.0, 0.0, select=sel)
        assert valid.sum() == 1 and valid[r0[0], c0[0]]
        assert img[r0[0], c0[0], ch] == 255
    assert np.isclose(hgt[r0[0], c0[0]], -1.9)
