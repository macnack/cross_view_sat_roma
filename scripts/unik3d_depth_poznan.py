"""UniK3D metric depth for every panorama of a Fixtor × Poznań manifest, in Loc²'s layout transposed to a Mapillary
sequence (<seq>/<DEPTH_DIR>/<id>.png, DEPTH_DIR = unik3d_depth_v2 (bevloc.data.vigor; v1 had row stripes), uint16
millimetres along the ray, clipped at 65 m), plus the camera-height sanity check. Stripe-guard rejects are listed at
the end and in <out>/rejects_<DEPTH_DIR>_<manifest>.json (the job then exits non-zero; the rest is written).

  make poznan-depth MANIFEST=experiments/06_fg2_bevsplat/manifest.json          # GPU (Eagle: ~1 s / panorama)
  make poznan-depth MANIFEST=... DEPTH_ARGS="--check-only"                      # only the height check, from the PNGs

Model and post-processing as scripts/loc2_depth_vigor.py (= third_party/Loc2/preprocess/infer_depth_vigor.py):
lpiccinelli/unik3d-vitl, resolution level 9, the full-sphere Spherical camera of the panorama size, depth =
|points|. The Fixtor panorama (7680 x 3840) is first resized (INTER_AREA) to cfg.poznan.depth_size = 2048 x 1024, the
VIGOR panorama size UniK3D saw in Loc²'s preprocessing; the PNG is stored at that size. Existing PNGs are skipped.

Height check: every pixel whose camera-frame ray looks between cfg.poznan.height_band_deg below the horizon (under
the 20 deg car-body limit) and within cfg.poznan.height_azimuth_deg of straight ahead or behind (the road in front of
and behind the car; restricted to Cityscapes road pixels when <seq>/semantic/<id>.png exists) gives an implied
camera height -z of its 3-D point, with the attitude from Mapillary's computed_rotation (levels the ~1 deg tilt).
Per panorama: the median; reported: the distribution over panoramas against the 1.65 m flat-ground proxy of the
reported Loc² row (and ipm.height_m = 1.7 m of the IPM row). Output: experiments/12_poznan_three_way/depth_check/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C
from bevloc.baselines import loc2 as loc2_wrap
from bevloc.baselines.common import depth_png_for, load_manifest, resolve_panorama, select_entries
from bevloc.data.mapillary import rodrigues
from bevloc.data.vigor import DEPTH_DIR, STRIPE_MAX_M, read_depth_png


def frame_meta(pano: Path, cache: dict) -> dict:
    """The Mapillary images.json record of a panorama (<seq>/images/<id>.jpg), cached per sequence."""
    seq = pano.parent.parent
    if seq not in cache:
        cache[seq] = {str(f["id"]): f for f in json.loads((seq / "images.json").read_text())}
    return cache[seq][pano.stem]


def implied_heights(depth, R_w2c, band_deg, az_deg, keep=None):
    """-z (metres) of the 3-D points of the selected pixels: depth (H, W) along the ray, R_w2c world(ENU)->camera
    (x right, y down, z forward), band (lo, hi) camera-frame elevation in degrees, |azimuth| <= az_deg or
    >= 180 - az_deg. keep (H, W) bool: optional extra pixel mask (road). Returns a 1-D array."""
    H, W = depth.shape
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    lon = (u - 0.5) * 2 * np.pi
    lat = (0.5 - v) * np.pi
    lo, hi = np.radians(band_deg[0]), np.radians(band_deg[1])
    rows = (lat >= lo) & (lat <= hi)
    az = np.abs(np.degrees(lon))
    cols = (az <= az_deg) | (az >= 180.0 - az_deg)
    m = rows[:, None] & cols[None, :] & np.isfinite(depth) & (depth > 0.05)
    if keep is not None:
        m &= keep
    ii, jj = np.nonzero(m)
    la, lo_ = lat[ii], lon[jj]
    d_cam = np.stack([np.sin(lo_) * np.cos(la), -np.sin(la), np.cos(lo_) * np.cos(la)], -1)
    z = (d_cam @ np.asarray(R_w2c, float))[:, 2]          # world z of each unit ray: (R^T d)_z = d . R[:, 2]
    return -(z * depth[ii, jj])


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--manifest", default=None, help="default cfg.poznan.manifest")
    ap.add_argument("--years", default=None, help="comma list (default: every year of the manifest; depth is per frame)")
    ap.add_argument("--limit", type=int, default=0, help="first N unique frames (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="skip inference, only the height check")
    ap.add_argument("--out", default="experiments/12_poznan_three_way/depth_check")
    ap.add_argument("--stripe-max", type=float, default=STRIPE_MAX_M,
                    help="row-stripe guard (m); maps above it are not written but listed (<out>/rejects_*.json)")
    a = ap.parse_args()
    cfg = C.load(a.config)
    P = cfg.poznan
    manifest = a.manifest or P.manifest
    years = [int(y) for y in a.years.split(",")] if a.years else None
    entries = select_entries(load_manifest(manifest), years, a.limit)
    panos = list(dict.fromkeys(resolve_panorama(e["panorama"]) for e in entries))
    W, H = (int(v) for v in P.depth_size)
    todo = [p for p in panos if a.overwrite or not depth_png_for(p).is_file()]
    n_rejects = 0
    print(f"{len(panos)} panoramas in {manifest}, {len(todo)} without depth", flush=True)
    if todo and not a.check_only:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = loc2_wrap.load_unik3d(dev, P.depth_model, int(P.depth_resolution_level))
        writer = loc2_wrap.DepthWriter(a.stripe_max)
        t0 = time.time()
        for k, pano in enumerate(todo, 1):
            img = cv2.cvtColor(cv2.imread(str(pano), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            depth = loc2_wrap.infer_distance(model, img)          # fresh camera per panorama (v2, see DEPTH_DIR)
            writer.write(depth, depth_png_for(pano), str(pano))
            if k % 20 == 0 or k == len(todo) or k == 1:
                print(f"  {k}/{len(todo)}  {(time.time() - t0) / k:.2f} s/img  {pano.stem} median {np.median(depth):.1f} m",
                      flush=True)
        n_rejects = writer.finish(Path(a.out) / f"rejects_{DEPTH_DIR}_{Path(manifest).stem}.json")

    # --- camera-height sanity check ---
    cache, per = {}, []
    for pano in panos:
        dp = depth_png_for(pano)
        if not dp.is_file():
            continue
        depth = read_depth_png(dp)
        R = rodrigues(frame_meta(pano, cache)["computed_rotation"])
        sem_p = pano.parent.parent / "semantic" / (pano.stem + ".png")
        keep = None
        if sem_p.is_file():
            sem = cv2.imread(str(sem_p), cv2.IMREAD_UNCHANGED)
            keep = cv2.resize(sem, depth.shape[::-1], interpolation=cv2.INTER_NEAREST) == 0     # Cityscapes road
        h = implied_heights(depth, R, tuple(P.height_band_deg), float(P.height_azimuth_deg), keep)
        per.append(dict(frame_id=pano.stem, n_px=int(h.size), road_only=keep is not None,
                        height_median_m=float(np.median(h)) if h.size else None,
                        height_p25_m=float(np.percentile(h, 25)) if h.size else None,
                        height_p75_m=float(np.percentile(h, 75)) if h.size else None,
                        depth_median_m=float(np.median(depth))))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hs = np.array([r["height_median_m"] for r in per if r["height_median_m"] is not None])
    summary = dict(manifest=manifest, n_panoramas=len(panos), n_with_depth=len(per),
                   n_road_only=int(sum(r["road_only"] for r in per)),
                   band_deg=list(P.height_band_deg), azimuth_deg=float(P.height_azimuth_deg),
                   height_median_m=float(np.median(hs)) if hs.size else None,
                   height_p10_m=float(np.percentile(hs, 10)) if hs.size else None,
                   height_p90_m=float(np.percentile(hs, 90)) if hs.size else None,
                   loc2_flat_proxy_m=1.65, ipm_height_m=float(cfg.ipm.height_m), depth_size=[W, H],
                   model=P.depth_model, resolution_level=int(P.depth_resolution_level))
    (out / "height_check.json").write_text(json.dumps(dict(summary=summary, frames=per), indent=2))
    if hs.size:
        print(f"camera height implied by UniK3D (road {P.height_band_deg} deg below the horizon, +-{P.height_azimuth_deg} "
              f"deg fore/aft): median {summary['height_median_m']:.2f} m (p10 {summary['height_p10_m']:.2f}, p90 "
              f"{summary['height_p90_m']:.2f}) over {hs.size} panoramas; flat proxy 1.65 m, ipm.height_m "
              f"{cfg.ipm.height_m} m", flush=True)
    print(f"wrote {out / 'height_check.json'}", flush=True)
    if n_rejects:
        raise SystemExit(f"{n_rejects} panoramas rejected by the stripe guard (listed above); every other map is written")


if __name__ == "__main__":
    main()
