"""Geometry of the Poznań three-way comparison (synthetic data): heading protocol, ERP roll and its camera rotation,
local <-> map frames, reference window construction, pose back to the map frame, and the end-to-end chain
depth-placed token -> GT homography -> reference pixel of the same world point."""
from pathlib import Path

import numpy as np
import pytest
import torch

from bevloc.baselines import fixtor as fx
from bevloc.baselines.common import (depth_png_for, heading_setup, pose_from_homography, resolve_panorama,
                                     roll_shift, select_entries, vehicle_yaw, wrap180)
from bevloc.data.ortho import Oriented, gt_homography
from bevloc.data.poznan_pano import (canvas_oriented, ego_rows_mask, forward_bearing, local_to_map, map_to_local,
                                     render_area, residual_rotation_deg, roll_matrix, rolled_R_w2c)
from bevloc.data.vigor import canvas_to_en, en_to_canvas, pose_en
from bevloc.model.depth_query import depth_placement_metric_at, metric_to_bev_px
from bevloc.model.erp_query import rays_at


def entry(up=81.4, crop_rot=7.0, en=(360026.0, 504344.0), off=(3.0, -12.0)):
    crop_up = up + crop_rot
    b = np.radians(crop_up)
    right, upv = np.array([np.cos(b), -np.sin(b)]), np.array([np.sin(b), np.cos(b)])
    centre = np.asarray(en) - (off[0] * right + off[1] * upv)
    return dict(frame_id="1", year=2025, en=list(en), up_bearing_deg=up, crop_up_bearing_deg=crop_up,
                crop_rot_deg=crop_rot, crop_centre_en=centre.tolist(), crop_offset_m=list(off),
                panorama="data/mapillary/Fixtor/SEQ/images/1.jpg")


def R_from(heading_true_deg, tilt_deg=1.5, roll_deg=-0.8):
    """World ENU -> camera (x right, y down, z forward) for a camera heading `heading_true_deg` (cw from north)
    with a small pitch / roll."""
    h = np.radians(heading_true_deg)
    fwd = np.array([np.sin(h), np.cos(h), 0.0])
    right = np.array([np.cos(h), -np.sin(h), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    R0 = np.stack([right, down, fwd])                     # rows: camera axes in world

    def rot(axis, a):
        a = np.radians(a)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    return rot(np.array([1.0, 0, 0]), tilt_deg) @ rot(np.array([0, 0, 1.0]), roll_deg) @ R0


def test_heading_setup_protocols():
    e = entry(up=81.4, crop_rot=7.0)
    b, assumed, rho = heading_setup(e, "prior", "crop")
    assert b == pytest.approx(88.4) and assumed == pytest.approx(88.4) and rho == pytest.approx(0.0)
    b, assumed, rho = heading_setup(e, "gt", "crop")                 # the reported rows: roll by crop_rot
    assert assumed == pytest.approx(81.4) and rho == pytest.approx(7.0)
    b, assumed, rho = heading_setup(e, "prior", "north")
    assert b == 0.0 and rho == pytest.approx(wrap180(-88.4))
    assert residual_rotation_deg(e, "prior") == pytest.approx(-7.0)
    assert residual_rotation_deg(e, "gt") == pytest.approx(0.0)
    # vehicle heading from the rolled column's heading: gt protocol, perfect estimate -> the proxy heading
    assert vehicle_yaw(88.4, 7.0) == pytest.approx(81.4)


@pytest.mark.parametrize("rho", [0.0, 7.3, -84.2, 170.0, -179.0])
def test_roll_brings_rho_to_centre(rho):
    W = 2048
    bearing = ((np.arange(W) + 0.5) / W - 0.5) * 360.0          # relative bearing of every column (centre = 0)
    shift, rho_eff = roll_shift(rho, W)
    rolled = np.roll(bearing, shift)
    assert abs(rho_eff - rho) <= 360.0 / W / 2 + 1e-9 or abs(abs(rho_eff - rho) - 360) < 1e-6
    centre = rolled[W // 2] - 360.0 / W / 2                     # the column whose left edge is the centre
    assert wrap180(centre - rho_eff) == pytest.approx(0.0, abs=1e-6)
    # the torch path of the baselines rolls the same way
    t = fx.roll_erp(torch.from_numpy(bearing)[None], rho)[0][0].numpy()
    assert np.array_equal(t, rolled)


def test_rolled_rotation_matches_rolled_pixels():
    """A pixel of the rolled panorama under R' = M(rho)^T R looks along the same world ray as the source pixel
    under R (the ERP convention of bevloc.model.erp_query.rays_at)."""
    H, W = 1024, 2048
    R = R_from(63.0)
    shift, rho = roll_shift(-84.2, W)
    Rr = rolled_R_w2c(R, rho)
    u_new = torch.tensor([10.5, 700.25, 1023.5, 2040.0])
    v = torch.tensor([300.5, 520.0, 700.75, 100.0])
    u_old = (u_new - shift) % W
    d_new, _ = rays_at(u_new, v, H, W)
    d_old, _ = rays_at(u_old, v, H, W)
    w_new = d_new.double().numpy() @ Rr                   # world = R'^T d  (row vectors)
    w_old = d_old.double().numpy() @ R
    np.testing.assert_allclose(w_new, w_old, atol=1e-6)
    np.testing.assert_allclose(roll_matrix(rho) @ roll_matrix(-rho), np.eye(3), atol=1e-12)
    # the rolled camera's forward turns by rho (small tilt: to ~0.05 deg)
    assert wrap180(forward_bearing(Rr) - forward_bearing(R) - rho) == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("beta", [0.0, 88.4, -135.0])
def test_local_frame_and_canvas(beta):
    origin = np.array([360000.0, 504000.0])
    pts = np.array([[360012.0, 503990.5], [359970.0, 504031.0]])
    loc = map_to_local(pts, origin, beta)
    np.testing.assert_allclose(local_to_map(loc, origin, beta), pts, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(loc, axis=1), np.linalg.norm(pts - origin, axis=1), atol=1e-9)
    # canvas: vigor.canvas_to_en in the local frame == the Oriented canvas' pixel -> map, for a shifted window
    size, cell, centre = 896, 0.0625, np.array([3.2, -7.9])
    o = canvas_oriented(origin, beta, centre, size, cell)
    uv = np.array([[0.0, 0.0], [447.5, 447.5], [895.0, 12.0], [100.0, 800.0]])
    world = (o.px_to_world @ np.c_[uv, np.ones(len(uv))].T).T[:, :2]
    np.testing.assert_allclose(map_to_local(world, origin, beta), canvas_to_en(uv, centre, cell, size), atol=1e-7)
    np.testing.assert_allclose(en_to_canvas(map_to_local(world, origin, beta), centre, cell, size), uv, atol=1e-6)


@pytest.mark.parametrize("heading,ref_up", [("prior", "north"), ("prior", "crop"), ("gt", "north"), ("gt", "crop")])
def test_gt_pose_round_trip_to_map(heading, ref_up):
    """GT H of the sample -> pose_en (tile/local frame, what eval_vigor scores) and pose_from_homography (map
    frame) both give the proxy position; the recovered rolled-column heading minus rho is the proxy heading; the
    rotation left in H is the protocol's residual."""
    e = entry(up=81.4, crop_rot=-6.0)
    beta, _assumed, rho = heading_setup(e, heading, ref_up)
    q_up = e["up_bearing_deg"] + rho
    n, cell, size = 224, 0.125, 896
    for centre in (np.zeros(2), np.array([4.0, -2.5])):
        ref_o = canvas_oriented(e["crop_centre_en"], beta, centre, size, cell)
        H = gt_homography(Oriented(tuple(e["en"]), q_up, n, cell), ref_o)
        en_l = pose_en(H, centre, n, cell, size)
        np.testing.assert_allclose(en_l, map_to_local(e["en"], e["crop_centre_en"], beta), atol=1e-6)
        en_m, yaw = pose_from_homography(H, n, ref_o)
        np.testing.assert_allclose(en_m, e["en"], atol=1e-6)
        assert wrap180(vehicle_yaw(yaw, rho) - e["up_bearing_deg"]) == pytest.approx(0.0, abs=1e-6)
        rot = np.degrees(np.arctan2(H[1, 0], H[0, 0]))       # image-plane rotation left in H (cw on screen)
        assert wrap180(rot - (q_up - beta)) == pytest.approx(0.0, abs=1e-6)
    assert wrap180(q_up - beta) == pytest.approx(residual_rotation_deg(e, heading))


def test_depth_placed_token_lands_on_its_world_point():
    """End to end: a ground point seen by the (tilted, rolled) panorama, placed from its depth along the ray
    (bevloc.model.depth_query), mapped with the GT homography, lands on the reference pixel of that point."""
    gamma = 0.7                                              # grid convergence: grid bearing = true bearing + gamma
    R = R_from(80.0, tilt_deg=1.8, roll_deg=-1.1)
    e = entry(up=forward_bearing(R) + gamma, crop_rot=5.0)
    beta, _a, rho_req = heading_setup(e, "prior", "north")
    Hh, W = 448, 896
    shift, rho = roll_shift(rho_req, 2048)
    Rr = rolled_R_w2c(R, rho)
    q_up = (forward_bearing(Rr) + gamma) % 360.0
    n, cell, size = 224, 0.125, 896
    ref_o = canvas_oriented(e["crop_centre_en"], beta, np.zeros(2), size, cell)
    Hgt = gt_homography(Oriented(tuple(e["en"]), q_up, n, cell), ref_o)
    for dE, dN, dz in ((10.0, 5.0, -1.65), (-7.0, -12.0, -1.6), (0.5, 20.0, 3.0)):
        bt = np.radians(np.degrees(np.arctan2(dE, dN)) - gamma)     # grid bearing -> true bearing
        r = np.hypot(dE, dN)
        w = np.array([np.sin(bt) * r, np.cos(bt) * r, dz])
        d = Rr @ (w / np.linalg.norm(w))
        lon, lat = np.arctan2(d[0], d[2]), np.arcsin(-d[1])
        uv = torch.tensor([[(lon / (2 * np.pi) + 0.5) * W, (0.5 - lat / np.pi) * Hh]], dtype=torch.float64)
        depth = torch.full((1, 1, Hh, W), float(np.linalg.norm(w)))
        xy_m, valid = depth_placement_metric_at(depth, uv, (Hh, W), torch.from_numpy(Rr).float()[None], 35.0)
        assert bool(valid[0, 0])
        bev = metric_to_bev_px(xy_m.double(), n, cell)[0, 0].numpy()
        p = Hgt @ np.array([bev[0], bev[1], 1.0])
        got = p[:2] / p[2]
        want = (np.linalg.inv(ref_o.px_to_world) @ np.array([e["en"][0] + dE, e["en"][1] + dN, 1.0]))[:2]
        np.testing.assert_allclose(got, want, atol=0.02)                # px at 0.125 m: < 3 mm


def test_render_area_places_and_orients(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from bevloc.data.mapillary import PoznanOrtho
    gsd, npx = 0.05, 1200                                    # 60 m tile at 5 cm (the 2024 orthophoto's GSD)
    E0, N0 = 360000.0, 504060.0                              # top-left corner
    img = np.zeros((3, npx, npx), np.uint8)
    # a 2 m white square whose centre sits 6 m east of the tile centre
    cx, cy = npx / 2 + 6.0 / gsd, npx / 2
    img[:, int(cy - 20):int(cy + 20), int(cx - 20):int(cx + 20)] = 255
    p = tmp_path / "t.tif"
    with rasterio.open(p, "w", driver="GTiff", width=npx, height=npx, count=3, dtype="uint8",
                       transform=from_origin(E0, N0, gsd, gsd)) as ds:
        ds.write(img)
    o = PoznanOrtho([p])
    centre = (E0 + 30.0, N0 - 30.0)
    for up, (du, dv) in ((0.0, (6.0, 0.0)), (90.0, (0.0, -6.0)), (-90.0, (0.0, 6.0))):
        rgb, valid = render_area(o, Oriented(centre, up, 128, 0.125))
        assert valid.all()
        ys, xs = np.nonzero(rgb[..., 0] > 127)
        c = 63.5
        assert xs.mean() == pytest.approx(c + du / 0.125, abs=0.6)
        assert ys.mean() == pytest.approx(c + dv / 0.125, abs=0.6)
        assert rgb[..., 0].sum() / 255.0 == pytest.approx((2.0 / 0.125) ** 2, rel=0.1)    # area preserved
    o.close()


def test_fixtor_decode_pose_axes_and_signs():
    c, up = (100.0, 200.0), 30.0
    b = np.radians(up)
    upv, right = np.array([np.sin(b), np.cos(b)]), np.array([np.cos(b), -np.sin(b)])
    # Loc²: t = (image-down, image-right); FG²: t = (image-up, image-left)
    en, _ = fx.decode_pose([2.0, 3.0], np.eye(2), c, up, "loc2", fx.LOC2_YAW_SIGN)
    np.testing.assert_allclose(en, np.asarray(c) - 2.0 * upv + 3.0 * right)
    en, _ = fx.decode_pose([2.0, 3.0], np.eye(2), c, up, "fg2", fx.FG2_YAW_SIGN)
    np.testing.assert_allclose(en, np.asarray(c) + 2.0 * upv - 3.0 * right)
    a = np.radians(5.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    assert fx.decode_pose([0, 0], R, c, up, "loc2", fx.LOC2_YAW_SIGN)[1] == pytest.approx(25.0)
    assert fx.decode_pose([0, 0], R, c, up, "fg2", fx.FG2_YAW_SIGN)[1] == pytest.approx(35.0)


def test_ego_rows_and_flat_depth():
    m = ego_rows_mask(448, 20.0)
    lat = (0.5 - (np.arange(448) + 0.5) / 448) * 180
    assert m.sum() > 0 and (lat[m] < -20).all() and (lat[~m] >= -20).all()
    assert np.array_equal(fx.ego_rows(448, 20.0), m) and not fx.ego_rows(448, 0).any()
    d = fx.loc2_flat_depth(714, 1428)
    assert d.shape == (1, 714, 1428)
    theta = np.linspace(0, np.pi, 714)
    r = 600                                               # a row below the horizon: d = h / -cos(theta)
    assert float(d[0, r, 0]) == pytest.approx(1.65 / -np.cos(theta[r]), rel=1e-5)
    assert float(d[0, 100, 0]) == 35.0


def test_manifest_helpers(tmp_path, monkeypatch):
    man = {"frames": [dict(frame_id=str(k // 2), year=2025 if k % 2 == 0 else 2024) for k in range(10)]
           + [dict(frame_id="9", year=2021)]}
    assert len(select_entries(man, [2025, 2024])) == 10
    sub = select_entries(man, [2025, 2024], 2)
    assert [(f["frame_id"], f["year"]) for f in sub] == [("0", 2025), ("0", 2024), ("1", 2025), ("1", 2024)]
    p = resolve_panorama("/elsewhere/checkout/data/mapillary/Fixtor/S/images/7.jpg")
    assert p.parts[-6:] == ("data", "mapillary", "Fixtor", "S", "images", "7.jpg")
    monkeypatch.delenv("POZNAN_DEPTH_DIR", raising=False)
    from bevloc.data.vigor import DEPTH_DIR
    assert depth_png_for(Path("/d/Fixtor/S/images/7.jpg")) == Path("/d/Fixtor/S") / DEPTH_DIR / "7.png"
    monkeypatch.setenv("POZNAN_DEPTH_DIR", str(tmp_path))
    assert depth_png_for(Path("/d/Fixtor/S/images/7.jpg")) == tmp_path / "S" / DEPTH_DIR / "7.png"


# --- FG² / Loc² adapters against each method's own conventions (review 2026-09-29) --------------------------------
def _kabsch(A, B):
    """Unweighted 2-D Procrustes in the third-party form (H = A_c^T B_c, R = V diag(1, det) U^T, t = B_mean -
    A_mean R^T): b = R a + t."""
    Am, Bm = A.mean(0), B.mean(0)
    U, _S, Vt = np.linalg.svd((A - Am).T @ (B - Bm))
    Z = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = Vt.T @ Z @ U.T
    return R, Bm - Am @ R.T


def _third_party_solvers():
    """FG²'s weighted_procrustes_2d and Loc²'s weighted_procrustes_2d_with_scale from third_party/, or None."""
    import importlib.util
    import sys
    from bevloc.baselines import fg2 as fg2_wrap
    from bevloc.baselines import loc2 as loc2_wrap
    f, l_ = fg2_wrap.FG2_ROOT / "utils" / "utils.py", loc2_wrap.LOC2_ROOT / "models" / "utils.py"
    if not (f.is_file() and l_.is_file()):
        return None
    out = []
    for name, p in (("_review_fg2_utils", f), ("_review_loc2_utils", l_)):
        spec = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(p.parent.parent))
        try:
            spec.loader.exec_module(m)
        except Exception:                                   # noqa: BLE001 — optional cross-check only
            return None
        finally:
            sys.path.pop(0)
        out.append(m)
    return out[0].weighted_procrustes_2d, out[1].weighted_procrustes_2d_with_scale


def _planted_pair(e, heading, W=1428, seed=0):
    """Synthetic correspondences for manifest entry e under a heading protocol, in both methods' metric frames:
    sat points S (image-down, image-right metres from the crop centre, crop rendered at beta) and the same points in
    the rolled camera's ground grid G (backward, right). Returns (S, G, beta, rho_eff)."""
    beta, _assumed, rho = heading_setup(e, heading, "crop")
    _shift, rho_eff = roll_shift(rho, W)
    b = np.radians(beta)
    up, right = np.array([np.sin(b), np.cos(b)]), np.array([np.cos(b), -np.sin(b)])
    d = np.asarray(e["en"]) - np.asarray(e["crop_centre_en"])
    p = np.array([-(d @ up), d @ right])                    # camera in sat (down, right)
    psi = np.radians(e["up_bearing_deg"] + rho_eff - beta)  # rolled centre column, cw of the crop's up
    M = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])   # sat -> ground (down, right)
    S = np.random.default_rng(seed).uniform(-30.0, 30.0, (300, 2))
    return S, (S - p) @ M.T, beta, rho_eff


@pytest.mark.parametrize("heading", ["prior", "gt"])
@pytest.mark.parametrize("crop_rot,off", [(7.0, (3.0, -12.0)), (-9.5, (-21.0, 18.0)), (4.0, (15.0, 20.0))])
def test_fixtor_adapters_recover_planted_pose(heading, crop_rot, off):
    """A planted pose (crop offset up to 28 m, crop rotation up to 9.5 deg) through each method's own solver
    direction -> fixtor.decode_pose -> vehicle_yaw gives back the proxy position and heading. Fails for a wrong yaw
    sign (error 2 |crop_rot| under prior) and for FG²'s p = -t decoding (error ~|p| |crop_rot|)."""
    e = entry(up=81.4, crop_rot=crop_rot, off=off)
    S, G, beta, rho_eff = _planted_pair(e, heading)
    tp = _third_party_solvers()
    for method, sign in (("fg2", fx.FG2_YAW_SIGN), ("loc2", fx.LOC2_YAW_SIGN)):
        sols = [(*(_kabsch(S, G) if method == "fg2" else _kabsch(G, S)), 1e-6)]
        if tp is not None:
            St, Gt = torch.tensor(S[None], dtype=torch.float32), torch.tensor(G[None], dtype=torch.float32)
            w = torch.ones(1, len(S))
            if method == "fg2":
                R, t, _ok = tp[0](St, Gt, w=w, use_weights=True, use_mask=True)
            else:
                R, t, _s, _ok = tp[1](Gt, St, w=w, use_weights=True, use_mask=True)
            sols.append((R[0].double().numpy(), t[0, 0].double().numpy(), 2e-3))          # float32 solvers
        for R, t, tol in sols:
            en, vyaw = fx.decode_pose(t, R, e["crop_centre_en"], beta, method, sign)
            np.testing.assert_allclose(en, e["en"], rtol=0, atol=tol)
            assert wrap180(vehicle_yaw(vyaw, rho_eff) - e["up_bearing_deg"]) == pytest.approx(0.0, abs=tol)
    # the wrong conventions fail on this pose (prior: a real rotation is left to the solver)
    if heading == "prior":
        _R, t = _kabsch(S, G)
        en_old, _ = fx.decode_pose(t, np.eye(2), e["crop_centre_en"], beta, "fg2", 1.0)   # task-02: p = -t
        assert np.linalg.norm(np.asarray(en_old) - np.asarray(e["en"])) > 0.5
        for method, sign in (("fg2", -fx.FG2_YAW_SIGN), ("loc2", -fx.LOC2_YAW_SIGN)):
            R, t = _kabsch(S, G) if method == "fg2" else _kabsch(G, S)
            _en, vyaw = fx.decode_pose(t, R, e["crop_centre_en"], beta, method, sign)
            assert abs(wrap180(vehicle_yaw(vyaw, rho_eff) - e["up_bearing_deg"])) == pytest.approx(2 * abs(crop_rot),
                                                                                                    abs=1e-6)
