"""Mapillary pairs helpers (no network, no GPU)."""
from __future__ import annotations

import pytest


def test_poznan_tiles_uses_env_root(tmp_path, monkeypatch):
    from bevloc.data.mapillary import poznan_tiles, sat_data_root
    monkeypatch.setenv("SAT_DATA_DIR", str(tmp_path))
    assert sat_data_root() == tmp_path
    for i in range(9):
        d = tmp_path / f"geoportal_poznan_15km2_e{i}_n{i}_gmix"
        d.mkdir()
        (d / "year_2025.tif").write_bytes(b"")
    assert len(poznan_tiles(2025)) == 9
    with pytest.raises(FileNotFoundError):
        poznan_tiles(2024)


import json

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin


def _synthetic_route(tmp_path):
    """One 500x500 m EPSG:2180 tile at 0.25 m (a 224 m crop fits inside) and one 64x32 ERP frame near its centre."""
    tile = tmp_path / "geoportal_poznan_15km2_e0_n0_gmix"
    tile.mkdir()
    img = np.random.default_rng(0).integers(1, 255, (3, 2000, 2000), np.uint8)
    with rasterio.open(tile / "year_2025.tif", "w", driver="GTiff", width=2000, height=2000, count=3,
                       dtype="uint8", crs="EPSG:2180", transform=from_origin(0.0, 500.0, 0.25, 0.25)) as ds:
        ds.write(img)
    seq = tmp_path / "seq"
    (seq / "images").mkdir(parents=True)
    cv2.imwrite(str(seq / "images" / "1.jpg"), np.full((32, 64, 3), 128, np.uint8))
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True).transform(250.0, 250.0)
    (seq / "images.json").write_text(json.dumps([{
        "id": "1", "captured_at": 0, "computed_compass_angle": 0.0,
        "computed_geometry": {"coordinates": [lon, lat]}, "computed_rotation": [0.0, 0.0, 0.0]}]))
    return tile / "year_2025.tif", seq


def test_sample_for_uses_the_given_reference(tmp_path):
    from bevloc import config as C
    from bevloc.data.mapillary import MapillaryPairs, PoznanOrtho, load_frames
    from bevloc.data.ortho import Oriented, gt_homography
    tif, seq = _synthetic_route(tmp_path)
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    ortho = PoznanOrtho([tif])
    frames = load_frames([seq], ortho, margin_m=0.0)
    ds = MapillaryPairs(frames, {2025: ortho}, cfg, train=False, erp_size=(64, 32), years=[2025])
    ref_o = Oriented((240.0, 255.0), 7.0, 224 * 4, 0.25)     # deliberately off-centre and rotated
    s = ds.sample_for("1", ref_o, 2025)
    q = Oriented(frames[0]["_en"], ds._up_of(frames[0]), 224, 0.25)
    assert np.allclose(s["H"].numpy(), gt_homography(q, ref_o), atol=1e-4)
    assert s["erp"].shape == (1, 3, 32, 64) and s["ref"].shape == (3, 896, 896)
    ortho.close()
