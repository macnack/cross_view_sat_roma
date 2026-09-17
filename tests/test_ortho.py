import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from bevloc.data.ortho import OrthoMap, Oriented, gt_homography, lonlat_to_bng, sample_reference

E0, N0, GSD_MAP, NPX = 427000.0, 542500.0, 0.125, 4800   # 600 m tile near Durham, centre (427300, 542200)
MARK = (427310.0, 542190.0)


@pytest.fixture(scope="module")
def tif(tmp_path_factory):
    rng = np.random.default_rng(0)
    img = cv2.resize(rng.integers(70, 190, (100, 100, 3), dtype=np.uint8), (NPX, NPX),
                     interpolation=cv2.INTER_CUBIC)
    col, row = (MARK[0] - E0) / GSD_MAP, (N0 - MARK[1]) / GSD_MAP
    cv2.circle(img, (int(col), int(row)), 12, (255, 0, 0), -1)
    p = tmp_path_factory.mktemp("ortho") / "synthetic_bng.tif"
    with rasterio.open(p, "w", driver="GTiff", height=NPX, width=NPX, count=3, dtype="uint8",
                       crs="EPSG:27700", transform=from_origin(E0, N0, GSD_MAP, GSD_MAP)) as d:
        d.write(np.moveaxis(img, -1, 0))
    return p


def test_durham_lonlat_to_bng():
    from pyproj import Transformer
    e, n = lonlat_to_bng(-1.5849, 54.7753)     # central Durham
    assert 420e3 < e < 435e3 and 535e3 < n < 550e3
    lon, lat = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True).transform(e, n)
    assert abs(lon + 1.5849) < 1e-7 and abs(lat - 54.7753) < 1e-7


def test_px_to_world_axes():
    o = Oriented((100.0, 200.0), 0.0, 224, 0.25)
    A = o.px_to_world
    assert np.allclose(A @ [111.5, 111.5, 1], [100, 200, 1])
    assert np.allclose(A @ [112.5, 111.5, 1], [100.25, 200, 1])      # +u = east when north-up
    assert np.allclose(A @ [111.5, 110.5, 1], [100, 200.25, 1])      # -v = north
    east_up = Oriented((100.0, 200.0), 90.0, 224, 0.25).px_to_world  # image-up = east
    assert np.allclose(east_up @ [111.5, 110.5, 1], [100.25, 200, 1])


def test_marker_lands_where_predicted(tif):
    m = OrthoMap(tif)
    ref = Oriented((MARK[0] + 30.0, MARK[1] - 45.0), 37.0, 896, 0.25)
    img, valid = m.render(ref)
    assert valid.all()
    red = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] < 60)
    v, u = np.nonzero(red)
    pred = np.linalg.inv(ref.px_to_world) @ [MARK[0], MARK[1], 1]
    assert abs(u.mean() - pred[0]) < 1.0 and abs(v.mean() - pred[1]) < 1.0   # < 0.25 m


def test_gt_homography_is_consistent_with_rendering(tif):
    m = OrthoMap(tif)
    rng = np.random.default_rng(1)
    q = Oriented((427300.0, 542200.0), 250.0, 224, 0.25)      # vehicle heading 250 deg
    for _ in range(5):
        r = sample_reference(q, rng)
        assert abs(((r.up_bearing_deg - q.up_bearing_deg) + 180) % 360 - 180) <= 55
        H = gt_homography(q, r)
        corners = np.array([[0, 0, 1], [223, 0, 1], [223, 223, 1], [0, 223, 1]], float) @ H.T
        assert corners[:, :2].min() >= 0 and corners[:, :2].max() <= 895   # footprint inside
        centre = H @ [111.5, 111.5, 1]
        assert np.abs(centre[:2] - 447.5).max() <= 0.30 * 896 + 1e-6
        qi, _ = m.render(q)
        ri, _ = m.render(r)
        back = cv2.warpPerspective(ri, np.linalg.inv(H), (224, 224), flags=cv2.INTER_LINEAR)
        diff = np.abs(back[8:-8, 8:-8].astype(int) - qi[8:-8, 8:-8].astype(int)).mean()
        assert diff < 3.0, diff


def test_rejects_wrong_crs(tmp_path):
    p = tmp_path / "wgs.tif"
    with rasterio.open(p, "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(0, 0, 1, 1)) as d:
        d.write(np.zeros((1, 4, 4), np.uint8))
    with pytest.raises(ValueError):
        OrthoMap(p)
