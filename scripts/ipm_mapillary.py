"""One-frame smoke test: Mapillary spherical panorama -> flat-ground BEV on a Poznań orthophoto.

The panorama is equirectangular. ``computed_rotation`` is the world-to-camera
angle-axis rotation in east-north-up, camera axes x right / y down / z forward.
Its forward bearing matches ``computed_compass_angle``. Camera height above the
road is not in the file; it is a single assumed metres value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Geod, Transformer
from rasterio.windows import Window

from bevloc.bev.grid import BevGrid

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "data/mapillary/Fixtor/gXabFhpwk2dcl0i4518mDQ"
ORTHO = {
    "e357585_n505324": Path.home() / "Github/sat_data/geoportal_poznan_15km2_e357585_n505324_gmix/year_2025.tif",
    "e359710_n505264": Path.home() / "Github/sat_data/geoportal_poznan_15km2_e359710_n505264_gmix/year_2025.tif",
}


def rodrigues(r):
    r = np.asarray(r, np.float64)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def ipm(erp, grid: BevGrid, R_w2c, height):
    """(n, n, 3) uint8 ground view. Row 0 is camera-forward, col 0 is left."""
    x, y = grid.cell_centres()
    forward = R_w2c.T @ np.array([0.0, 0.0, 1.0])
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0], 0.0])
    east = x * forward[0] + y * left[0]
    north = x * forward[1] + y * left[1]
    up = np.full_like(x, -float(height))
    p_cam = np.stack([east, north, up], -1).reshape(-1, 3) @ R_w2c.T
    xc, yc, zc = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    lon = np.arctan2(xc, zc)
    lat = np.arctan2(-yc, np.hypot(xc, zc))
    h, w = erp.shape[:2]
    mu = ((lon / (2 * np.pi) + 0.5) * w).astype(np.float32).reshape(grid.n, grid.n)
    mv = ((0.5 - lat / np.pi) * h).astype(np.float32).reshape(grid.n, grid.n)
    img = cv2.remap(erp, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    # The capture rig sits on the nadir. Drop the disc under the camera.
    img[np.hypot(x, y) < 1.2] = 0
    return img


def grid_bearing(lon, lat, true_bearing):
    """Bearing clockwise from grid north, for a true-north compass bearing."""
    geod = Geod(ellps="WGS84")
    lon2, lat2, _ = geod.fwd(lon, lat, 0.0, 200.0)
    t = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    e0, n0 = t.transform(lon, lat)
    e1, n1 = t.transform(lon2, lat2)
    gamma = np.degrees(np.arctan2(e1 - e0, n1 - n0))
    return (true_bearing + gamma) % 360, (e0, n0)


def ortho_crop(path, centre_en, up_bearing, n, gsd):
    b = np.radians(up_bearing)
    right = np.array([np.cos(b), -np.sin(b)])
    up = np.array([np.sin(b), np.cos(b)])
    c = (n - 1) / 2.0
    uu, vv = np.meshgrid(np.arange(n), np.arange(n))
    E = centre_en[0] + gsd * ((uu - c) * right[0] - (vv - c) * up[0])
    N = centre_en[1] + gsd * ((uu - c) * right[1] - (vv - c) * up[1])
    with rasterio.open(path) as ds:
        inv = ~ds.transform
        col, row = inv * (E, N)
        col, row = col - 0.5, row - 0.5
        c0, r0 = int(np.floor(col.min())) - 2, int(np.floor(row.min())) - 2
        c1, r1 = int(np.ceil(col.max())) + 3, int(np.ceil(row.max())) + 3
        src = ds.read(indexes=[1, 2, 3], window=Window(c0, r0, c1 - c0, r1 - r0),
                      boundless=True, fill_value=0)
        src = np.moveaxis(src, 0, -1)
        k = gsd / abs(ds.transform.a)
        if k > 1.5:
            s = max(1, int(round(k)))
            src = cv2.blur(src, (s, s))
        img = cv2.remap(np.ascontiguousarray(src), (col - c0).astype(np.float32),
                        (row - r0).astype(np.float32), cv2.INTER_LINEAR)
        inside = (col >= 0) & (col <= ds.width - 1) & (row >= 0) & (row <= ds.height - 1)
    img[~inside] = 0
    return img


def tile_for(en):
    for name, path in ORTHO.items():
        with rasterio.open(path) as ds:
            if (ds.bounds.left <= en[0] <= ds.bounds.right
                    and ds.bounds.bottom <= en[1] <= ds.bounds.top):
                return name, path
    raise SystemExit(f"GPS point {en} is outside the two 2025 tiles")


def panel(img, title):
    bar = np.zeros((36, img.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--index", type=int, default=300)
    a.add_argument("--height", type=float, default=1.7)
    a.add_argument("--gsd", type=float, default=0.1)
    a.add_argument("--extent", type=float, default=56.0)
    a.add_argument("--out", type=Path, default=ROOT / "experiments/04_mapillary_ipm")
    args = a.parse_args()

    frames = json.loads((SEQ / "images.json").read_text())
    fr = frames[args.index]
    erp = cv2.cvtColor(cv2.imread(str(SEQ / "images" / f"{fr['id']}.jpg")), cv2.COLOR_BGR2RGB)
    R = rodrigues(fr["computed_rotation"])
    lon, lat = fr["computed_geometry"]["coordinates"]
    up, en = grid_bearing(lon, lat, fr["computed_compass_angle"])
    n = int(round(args.extent / args.gsd))
    grid = BevGrid(n=n, cell=args.gsd)
    bev = ipm(erp, grid, R, args.height)
    tile, path = tile_for(en)
    ref = ortho_crop(path, en, up, n, args.gsd)
    blend = cv2.addWeighted(bev, 0.5, ref, 0.5, 0)

    args.out.mkdir(parents=True, exist_ok=True)
    sheet = np.hstack([panel(ref, "2025 orthophoto"), panel(bev, f"IPM  h={args.height:.1f} m"),
                       panel(blend, "overlay")])
    cv2.imwrite(str(args.out / "smoke.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    meta = {
        "id": fr["id"], "index": args.index, "height_m": args.height, "gsd_m": args.gsd,
        "extent_m": args.extent, "computed_compass_deg": fr["computed_compass_angle"],
        "compass_angle_deg": fr["compass_angle"], "grid_up_deg": up, "tile": tile,
        "east": en[0], "north": en[1],
    }
    (args.out / "smoke.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("wrote", args.out / "smoke.jpg", sheet.shape)


if __name__ == "__main__":
    main()
