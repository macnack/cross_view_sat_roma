"""KITScenes wrapper: synthetic geometry checks (no data) + checks on the real val scene when it is extracted
(data/kitscenes, `make kitscenes-fetch`)."""
import numpy as np
import pytest

from bevloc.bev.spherical import lidar_to_erp
from bevloc.data import kitscenes as K

SCENE = K.Path(__file__).resolve().parents[1] / "data/kitscenes/data/val/142f1419-b6f2-4215-4055-6eb161f63043"
needs_scene = pytest.mark.skipif(not (SCENE / "calibration/calib.json").exists(), reason="KITScenes scene not extracted")


def _ring(yaws_deg, f=211.0, size=(400, 260), tilt=0.0):
    """Synthetic ring: OpenCV cameras at the origin looking along yaw (ccw from forward), level, like the real rig."""
    cams = {}
    for i, yaw in enumerate(yaws_deg):
        a = np.radians(yaw)
        fwd = np.array([np.cos(a), np.sin(a), 0.0])
        right = np.array([np.sin(a), -np.cos(a), 0.0])
        down = np.array([0.0, 0.0, -1.0])
        T = np.eye(4)
        T[:3, :3] = np.stack([right, down, fwd], 1)
        K_ = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
        cams[f"c{i}"] = K.Camera(f"c{i}", K_, T, size)
    return cams


def test_erp_axes():
    w, h = 400, 200
    d = K.erp_rays(w, h)
    mid = d[h // 2, w // 2]
    assert mid[0] > 0.999                                    # centre = forward
    right = d[h // 2, int(0.75 * w)]
    assert right[1] < -0.999                                 # a quarter of the way right = ego -y (right)
    assert d[0, w // 2][2] > 0.99 and d[-1, w // 2][2] < -0.99   # zenith on top
    assert np.allclose(np.linalg.norm(d, axis=-1), 1)


def test_erp_rays_match_lidar_to_erp():
    """K.erp_rays and bevloc.bev.spherical.lidar_to_erp are inverse conventions (so LiDAR overlays line up)."""
    w, h = 400, 200
    d = K.erp_rays(w, h)
    pts = (d[::17, ::23] * 10.0).reshape(-1, 3)
    u, v, r = lidar_to_erp(pts, w, h)
    gu, gv = np.meshgrid(np.arange(w)[::23] + 0.5, np.arange(h)[::17] + 0.5)
    assert np.abs(u - gu.ravel()).max() < 1e-6 and np.abs(v - gv.ravel()).max() < 1e-6
    assert np.allclose(r, 10.0)


def test_project_principal_point():
    cam = _ring([30.0])["c0"]
    p = cam.centre + cam.R[:, 2] * 7.0
    uv, z = K.project(cam, p[None])
    assert np.allclose(uv[0], cam.K[:2, 2]) and np.isclose(z[0], 7.0)


def test_stitch_round_trip():
    """A known world panorama rendered into six ring cameras and stitched back reproduces it inside the coverage."""
    yaws = [0, -60, -120, 180, 120, 60]
    cams = _ring(yaws)
    W, H = 720, 360
    rng = np.random.default_rng(0)
    base = cv2_blur(rng.integers(0, 255, (H // 8, W // 8, 3)).astype(np.uint8), (W, H))
    # smooth, direction-indexed texture: world(d) = base at the ERP pixel of d
    def world(dirs):
        az = np.arctan2(-dirs[..., 1], dirs[..., 0])
        el = np.arcsin(np.clip(dirs[..., 2], -1, 1))
        u = ((az / (2 * np.pi) + 0.5) * W - 0.5).astype(np.float32)
        v = ((0.5 - el / np.pi) * H - 0.5).astype(np.float32)
        return cv2_remap(base, u, v)
    imgs = {}
    for n, c in cams.items():
        w_, h_ = c.size
        uu, vv = np.meshgrid(np.arange(w_), np.arange(h_))
        rays = np.stack([(uu - c.K[0, 2]) / c.K[0, 0], (vv - c.K[1, 2]) / c.K[1, 1], np.ones_like(uu, float)], -1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)      # world() takes unit directions (arcsin(z))
        imgs[n] = world(rays @ c.R.T)
    st = K.ErpStitcher(cams, size=(W, H), pano_centre=np.zeros(3), scale=1.0)
    erp, valid = st(imgs)
    # coverage: |elevation| < ~25 deg is covered by every azimuth (vertical half-FOV = atan(130/211) = 31.6 deg)
    el = np.abs((0.5 - (np.arange(H) + 0.5) / H) * 180)
    band = np.zeros((H, W), bool)
    band[el < 25] = True
    assert valid[band].all()
    assert not valid[el > 40].any()                           # nothing above / below the cameras
    ref = world(K.erp_rays(W, H))
    err = np.abs(erp.astype(float) - ref.astype(float)).mean(-1)
    assert err[band].mean() < 4.0, err[band].mean()           # smooth texture, bilinear resampling


def test_clean_crop_has_no_invalid_pixel():
    """Crop = rows covered at every azimuth: no black pixel left; the ego cut and the elevation cap only shrink it."""
    cams = _ring([0, -60, -120, 180, 120, 60])
    W, H = 720, 360
    st = K.ErpStitcher(cams, size=(W, H), pano_centre=np.zeros(3), scale=1.0)
    imgs = {n: np.full((c.size[1], c.size[0], 3), 200, np.uint8) for n, c in cams.items()}
    erp, valid = st(imgs)
    assert not valid.all() and valid.any()                         # the full ERP does have holes
    e, m, info = K.crop_clean(erp, valid)
    assert m.all() and (e > 0).all()                               # crop: every pixel available, none black
    assert e.shape[0] == info["row1"] - info["row0"] and e.shape[1] == W
    assert info["elevation_top_deg"] > 0 > info["elevation_bottom_deg"]
    assert info["elevation_top_deg"] <= 31.7                       # cannot exceed the vertical half-FOV
    # seam limited: narrower than the band the camera centres cover (+-31.6 deg)
    assert info["elevation_top_deg"] < 31.0
    e2, m2, info2 = K.crop_clean(erp, valid, ego_mask_deg=10.0)
    assert m2.all() and info2["elevation_bottom_deg"] >= -10.5 and info2["row1"] <= info["row1"]
    e3, _, info3 = K.crop_clean(erp, valid, max_elevation_deg=12.0)
    assert info3["elevation_top_deg"] <= 12.5 and info3["row0"] >= info["row0"]
    with pytest.raises(ValueError):
        K.clean_band(np.zeros((H, W), bool))


def test_sheet_does_not_draw_on_the_panorama():
    """Regression: the figure's labels were once drawn into the panorama itself (a slice that was not copied), so the
    cropped panorama written afterwards carried the yellow azimuth ticks."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("kitscenes_pano", K.Path(__file__).resolve().parents[1] / "scripts/kitscenes_pano.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cams = _ring([0, -60, -120, 180, 120, 60])
    st = K.ErpStitcher(cams, size=(720, 360), pano_centre=np.zeros(3), scale=1.0)
    erp, valid = st({n: np.full((c.size[1], c.size[0], 3), 90, np.uint8) for n, c in cams.items()})
    before, ov = erp.copy(), erp.copy()
    mod.sheet(erp, ov, valid, cams, "t")
    assert np.array_equal(erp, before) and np.array_equal(ov, before)


def cv2_blur(small, size):
    import cv2
    return cv2.resize(small, size, interpolation=cv2.INTER_CUBIC)


def cv2_remap(img, u, v):
    import cv2
    return cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def test_quat_to_R():
    R = K.quat_to_R([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])     # +90 deg about z
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)


# --- real scene ---------------------------------------------------------------------------------------------------

@needs_scene
def test_real_calibration():
    cams = K.load_calibration(SCENE)
    assert list(cams) == list(K.RING_CAMERAS)
    yaws = np.sort([c.yaw_deg for c in cams.values()])
    gaps = np.diff(np.r_[yaws, yaws[0] + 360.0])
    assert np.abs(gaps - 60.0).max() < 3.0, gaps               # six cameras ~60 deg apart, full 360 deg ring
    assert abs(cams["camera_ring_front"].yaw_deg) < 2.0         # measured: +1.0 deg (rear: -179.0)
    for c in cams.values():
        assert c.size == (3504, 2272) and 1800 < c.K[0, 0] < 1900
        assert abs(np.linalg.det(c.R) - 1) < 1e-3 and np.allclose(c.R.T @ c.R, np.eye(3), atol=2e-3)
        assert abs(c.R[2, 2]) < 0.02 and c.R[2, 1] < -0.99     # level: optical axis horizontal, image-down = -z
        assert abs(c.centre[2] + 0.18) < 0.02                  # all ~18 cm below the lidar_top origin
    assert np.allclose(K.lidar_extrinsic(SCENE), np.eye(4))     # reference frame == lidar_top


@needs_scene
def test_real_lidar_erp_matches_camera_projection():
    """A far LiDAR point seen by the front camera lands on the ERP pixel whose lookup map points at the same camera
    pixel (to within the parallax of the 0.2 m camera offset)."""
    sc = K.KitScene(SCENE)
    cam = sc.cams["camera_ring_front"]
    st = K.ErpStitcher(sc.cams, size=(2048, 1024), scale=0.25)
    pts, _ = sc.lidar(0)
    uv, z = K.project(cam, pts)
    r = np.linalg.norm(pts - st.pano_centre, axis=1)
    sel = (z > 0) & (r > 30) & (uv[:, 0] > 300) & (uv[:, 0] < 3200) & (uv[:, 1] > 300) & (uv[:, 1] < 1900)
    assert sel.sum() > 500
    u, v, _ = lidar_to_erp(pts[sel], 2048, 1024, t_cl=-st.pano_centre)
    mu, mv, w = st.maps["camera_ring_front"]
    iu, iv = np.clip(u.astype(int), 0, 2047), np.clip(v.astype(int), 0, 1023)
    s = st.scale
    back = np.stack([(mu[iv, iu] + 0.5) / s - 0.5, (mv[iv, iu] + 0.5) / s - 0.5], 1)
    err = np.linalg.norm(back - uv[sel], axis=1)
    assert np.median(err) < 8.0, np.median(err)               # full-res px; 1 ERP px = ~7 camera px


@needs_scene
def test_real_stitch_covers_ring():
    sc = K.KitScene(SCENE)
    st = K.ErpStitcher(sc.cams, size=(1024, 512), scale=0.25)
    erp, valid = st(sc.images(0))
    assert erp.shape == (512, 1024, 3) and valid[256 - 40:256 + 40].all()   # +-28 deg band fully covered, all azimuths
    assert erp[valid].mean() > 20                                            # not black


@needs_scene
def test_real_pose_heading_follows_travel():
    sc = K.KitScene(SCENE)
    res = sc.heading_vs_travel()
    assert len(res) > 20
    assert abs(np.median(res)) < 5.0, np.median(res)          # quaternion = ego -> map, x-forward
