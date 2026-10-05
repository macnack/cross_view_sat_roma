"""KITScenes scene -> a dataset in VIGOR's layout, so the VIGOR evaluators (PanoRoMa, FG², Loc²) run on it unchanged.

  make kitscenes-vigor SCENE_DIR=data/kitscenes/data/val/<uuid> OUT=data/kit_vigor/<name> [STRIDE=1 PRIOR_M=17.8 SEED=0]

For every `stride`-th frame of the scene:
  * panorama: the six ring cameras stitched to a 2048 x 1024 ERP (bevloc.data.kitscenes.ErpStitcher; ~+-28 deg of elevation
    carry image, the rest is black) rolled so that the centre column is GRID NORTH and azimuth grows clockwise: VIGOR's
    convention, the known-orientation protocol. The roll is bearing / 360 * 2048 columns, the bearing of the ego's
    forward axis (clockwise from grid north) from the pose's quaternion;
  * tile: the free 20 cm orthophoto of the scene's state (Hessen DOP20 or LGL-BW DOP20 WMS, EPSG:25832, the frame of
    the poses: UTM 32N metres minus maps/origin.json projected to 25832; verified against the GNSS fixes) of 640 px at
    the GSD of VIGOR's Chicago tiles (0.111262 m/px, 71.2 m), north-up, centred at the true position plus a random offset
    inside the disc of radius `prior_m` (VIGOR's positive tile has the camera in its central quarter, 17.8 m);
  * label: VIGOR's (dy, dx) in tile px, dy > 0 = panorama SOUTH of the tile centre, dx > 0 = panorama WEST of it.
With --crop the panorama is cut to the largest band centred on the horizon that has image at every azimuth
(`bevloc.data.kitscenes.symmetric_band`): no black pixel is left, the JPEGs are 2048 x rows, and
`<out>/Chicago/erp_band.json` records the elevations (top_deg / bottom_deg), rows and full_width. The VIGOR reader
(`bevloc.data.vigor.read_band`) then resizes panorama and depth to the band at the model's pixel scale, and the PanoRoMa
query (`erp_band` in the batch) computes every token's ray from the band's elevations; the depth script gives UniK3D's
spherical camera the band's vertical field of view. FG² and Loc² assume a full sphere: run them on the uncropped dataset.
The layout is `<out>/Chicago/{panorama,satellite}` and `<out>/splits__corrected/Chicago/*` (the corrected-label names the
reader requires; the labels here are exact by construction). The folder is named Chicago only because FG²'s loader
recognises four VIGOR city names and takes the ground resolution from the name: ours is Chicago's, so metres are right.
The test list is `same_area_balanced_test`; train files are empty. KITScenes frames are CC BY-NC: `<out>` is never committed.

Sanity checks run first (--check, default on): the LiDAR depth edges of two frames must coincide with the colour edges of
the ROLLED panorama (bevloc.data.kitscenes.depth_edge_alignment, peak shift 0 px): this tests the roll direction and the
heading, with the same code that verified the unrolled stitch.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

from bevloc.data import kitscenes as K
from bevloc.data.vigor import CITY_RES

WMS = {
    "hessen": dict(url="https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-images.ows", layer="he_dop20_rgb",
                   credit="Hessische Verwaltung für Bodenmanagement und Geoinformation, DOP20 (Datenlizenz Deutschland Zero 2.0)"),
    "lgl": dict(url="https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_DOP_20_C", layer="IMAGES_DOP_20_RGB",
                credit="LGL-BW (year) Datenlizenz Deutschland – Namensnennung – Version 2.0, www.lgl-bw.de"),
}
TILE_PX = 640
RES = CITY_RES["Chicago"]            # m/px of the tile; the city name is only a label (module doc)
NATIVE_M = 0.2                        # DOP20 ground resolution


def default_wms(lat, lon):
    """Frankfurt (Hessen) vs Baden-Württemberg (Karlsruhe, Sindelfingen) by the scene origin."""
    return "hessen" if 49.5 <= lat <= 50.6 and 8.0 <= lon <= 9.3 else "lgl"


def bearing_cw_from_north(R):
    """Bearing (deg, clockwise from grid north) of the ego's forward axis, from its ego -> map rotation (poses.txt: x east,
    y north)."""
    return float((90.0 - np.degrees(np.arctan2(R[1, 0], R[0, 0]))) % 360.0)


def label_from_offset(e, n, res=RES):
    """VIGOR label (dy, dx) in tile px of a panorama at (e, n) metres east / north of the tile centre."""
    return -n / res, -e / res


def fetch_tile(wms, centre_en, cache_dir, retries=3):
    """640 px north-up tile of 640 * RES m centred at centre_en (EPSG:25832): the WMS at its native 0.2 m (size / 0.2 px),
    then bicubic to 640 px. Cached by the request."""
    half = TILE_PX * RES / 2.0
    bbox = (centre_en[0] - half, centre_en[1] - half, centre_en[0] + half, centre_en[1] + half)
    npx = int(round(2 * half / NATIVE_M))
    q = dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS=WMS[wms]["layer"], STYLES="", CRS="EPSG:25832",
             BBOX=",".join(f"{v:.3f}" for v in bbox), WIDTH=npx, HEIGHT=npx, FORMAT="image/png")
    url = WMS[wms]["url"] + "?" + urllib.parse.urlencode(q, safe=",:")
    cache = Path(cache_dir) / (f"{wms}_{bbox[0]:.1f}_{bbox[1]:.1f}.png")
    if not cache.exists():
        for k in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    data = r.read()
                if not data.startswith(b"\x89PNG"):
                    raise RuntimeError(f"not a PNG: {data[:200]!r}")
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
                break
            except Exception as ex:
                if k == retries - 1:
                    raise
                time.sleep(2.0 * (k + 1))
        time.sleep(0.3)                                             # one request at a time, politely
    img = cv2.imread(str(cache), cv2.IMREAD_COLOR)
    return cv2.resize(img, (TILE_PX, TILE_PX), interpolation=cv2.INTER_CUBIC)


def check_roll(sc, st, frame, o_en, pano_centre, W):
    """Peak shift (px) of the LiDAR-depth-edge / colour-edge alignment on the ROLLED panorama of `frame`."""
    R, t = sc.ego_pose(frame)
    erp, _ = st(sc.images(frame))
    b = bearing_cw_from_north(R)
    erp = np.roll(erp, int(round(b / 360.0 * W)), axis=1)
    pts, _refl, ring = sc.lidar(frame, with_ring=True)
    world = pts @ R.T                                                 # ego frame -> map axes (east, north, up), centred on the ego
    centre = R @ np.asarray(pano_centre)
    rel = world - centre
    ne = np.stack([rel[:, 1], -rel[:, 0], rel[:, 2]], 1)              # ERP frame of the rolled pano: x = north, y = west, z = up
    return K.depth_edge_alignment(erp, ne, ring, np.zeros(3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--prior-m", type=float, default=17.8, help="radius of the disc the tile centre is drawn from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wms", choices=sorted(WMS), default=None, help="default: by the scene origin")
    ap.add_argument("--erp", type=int, nargs=2, default=(2048, 1024))
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--crop", action="store_true", help="cut the panoramas to the image band (no black pixel); see the module doc")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sc = K.KitScene(a.scene_dir)
    W, H = a.erp
    st = K.ErpStitcher(sc.cams, size=(W, H))
    org = json.loads((Path(a.scene_dir) / "maps/origin.json").read_text())
    to_utm = Transformer.from_crs(4326, 25832, always_xy=True)
    to_ll = Transformer.from_crs(25832, 4326, always_xy=True)
    o_en = np.array(to_utm.transform(org["longitude"], org["latitude"]))
    wms = a.wms or default_wms(org["latitude"], org["longitude"])
    print(f"scene {sc.path.name}: {len(sc)} frames, origin {org}, UTM32 {o_en.round(1).tolist()}, WMS {wms}", flush=True)

    if not a.no_check:
        # the alignment curve of one frame can be flat (few depth edges on an open bridge): sum the peak-normalised curves
        # of 8 frames spread over the scene and take the peak of the sum
        tot, n_edge = {}, 0
        for f in sc.frame_ids[:: max(1, len(sc) // 8)][:8]:
            r = check_roll(sc, st, f, o_en, st.pano_centre, W)
            top = max(r["curve"].values())
            for sh, v in r["curve"].items():
                tot[sh] = tot.get(sh, 0.0) + v / top
            n_edge += r["n_edge_points"]
        peak = max(tot, key=tot.get)
        print(f"roll check (8 frames, {n_edge} edge points): LiDAR/colour edge alignment peak at {peak} px; "
              f"curve {{{', '.join(f'{k}: {v:.2f}' for k, v in sorted(tot.items()) if k % 12 == 0)}}}", flush=True)
        if abs(peak) > 8:
            raise SystemExit(f"the rolled panorama is misaligned with the LiDAR (peak {peak} px): heading or roll direction wrong")

    band = None
    if a.crop:
        _, v0 = st(sc.images(sc.frame_ids[0]))                     # the coverage is the cameras' geometry: the same for every frame
        r0, r1, top, bot = K.symmetric_band(v0)
        band = dict(top_deg=top, bottom_deg=bot, rows=int(r1 - r0), full_width=int(W), row0=int(r0), row1=int(r1))
        print(f"crop to rows {r0}..{r1} of {H}: elevation {top:+.1f} .. {bot:+.1f} deg ({r1 - r0} rows)", flush=True)

    out = Path(a.out)
    pdir, sdir = out / "Chicago/panorama", out / "Chicago/satellite"
    lab = out / "splits__corrected/Chicago"
    for d in (pdir, sdir, lab):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    lines, sats, meta = [], [], []
    frames = sc.frame_ids[::a.stride]
    if a.limit:
        frames = frames[:a.limit]
    for k, f in enumerate(frames):
        R, t = sc.ego_pose(f)
        p = t + R @ st.pano_centre                                   # the panorama centre in the pose frame
        e_n = o_en + p[:2]                                           # EPSG:25832 east / north of the camera
        lon, lat = to_ll.transform(*e_n)
        b = bearing_cw_from_north(R)
        erp, valid = st(sc.images(f))
        erp = np.roll(erp, int(round(b / 360.0 * W)), axis=1)
        if band is not None:
            erp = erp[band["row0"]:band["row1"]]
        r, ang = a.prior_m * np.sqrt(rng.random()), 2 * np.pi * rng.random()
        off = np.array([r * np.cos(ang), r * np.sin(ang)])           # camera minus tile centre (east, north)
        c_en = e_n - off
        tile = fetch_tile(wms, c_en, out / "wms_cache")
        clon, clat = to_ll.transform(*c_en)
        pano = f"kit{sc.path.name[:8]}_{f:04d},{lat:.6f},{lon:.6f},.jpg"
        sat = f"satellite_{clat:.12f}_{clon:.12f}.png"
        cv2.imwrite(str(pdir / pano), cv2.cvtColor(erp, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(sdir / sat), tile)
        dy, dx = label_from_offset(off[0], off[1])
        lines.append(f"{pano} {sat} {dy:.3f} {dx:.3f} {sat} 0 0 {sat} 0 0 {sat} 0 0")
        sats.append(sat)
        meta.append(dict(frame=int(f), pano=pano, sat=sat, camera_en=e_n.tolist(), tile_centre_en=c_en.tolist(),
                         offset_en=off.tolist(), bearing_deg=b, lat=lat, lon=lon))
        if (k + 1) % 10 == 0 or k + 1 == len(frames):
            print(f"  {k + 1}/{len(frames)} frames", flush=True)
    body = "\n".join(lines) + "\n"
    for name in ("same_area_balanced_test", "pano_label_balanced"):
        for suffix in (".txt", "__corrected.txt"):
            (lab / f"{name}{suffix}").write_text(body)
    for suffix in (".txt", "__corrected.txt"):
        (lab / f"same_area_balanced_train{suffix}").write_text("")
    (lab / "satellite_list.txt").write_text("\n".join(sorted(set(sats))) + "\n")
    if band is not None:
        (out / "Chicago/erp_band.json").write_text(json.dumps(band, indent=1))
    (out / "kit_meta.json").write_text(json.dumps(dict(scene=sc.path.name, wms=wms, credit=WMS[wms]["credit"], res_m=RES,
                                                       prior_m=a.prior_m, seed=a.seed, band=band, frames=meta), indent=1))
    print(f"wrote {len(lines)} samples to {out}", flush=True)


if __name__ == "__main__":
    main()
