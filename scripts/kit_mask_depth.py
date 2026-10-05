"""Mask the depth maps of a KITScenes VIGOR-layout dataset where the panorama has no image or shows the ego car.

  python scripts/kit_mask_depth.py --root <kit_vigor/name> [--src unik3d_depth_v2] [--dst unik3d_depth_v2_kitmask] [--ego-mask-deg 20]

The stitched ring-camera panorama carries image only inside about +-28 deg of elevation (and less at the seams between
cameras); the rest is black. UniK3D still returns a depth for every pixel, including those black ones, and PanoRoMa
(like Loc²) places every token that has a depth below its range limit on the ground. Black tokens are then placed at
arbitrary ranges: outliers that a panorama from a 360 deg camera never produces. This writes a copy of the depth maps in
`--dst` with depth 0 (= invalid, dropped by the placement) at
  * every pixel where the panorama is black (all channels <= 4 of 255, opened with a 5 x 5 kernel against JPEG speckle);
  * every row steeper than `--ego-mask-deg` below the horizon (the ego car's bonnet and roof sensors; the value of
    poznan.ego_mask_deg in configs/default.yaml, the Fixtor car body there).
Use it with BEVLOC_DEPTH_DIR=<dst> (bevloc.data.vigor.DEPTH_DIR). The original maps are never touched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def mask_depth(depth_mm, pano_bgr, ego_mask_deg):
    """depth_mm uint16 (H, W), pano_bgr uint8 (H, W, 3) at the same size -> uint16 depth with invalid pixels set to 0."""
    h = depth_mm.shape[0]
    valid = (pano_bgr.max(axis=2) > 4).astype(np.uint8)
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    el = (0.5 - (np.arange(h) + 0.5) / h) * 180.0                    # row-centre elevation, + up
    valid &= (el >= -float(ego_mask_deg))[:, None]
    out = depth_mm.copy()
    out[~valid] = 0
    return out, valid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--city", default="Chicago")
    ap.add_argument("--src", default="unik3d_depth_v2")
    ap.add_argument("--dst", default="unik3d_depth_v2_kitmask")
    ap.add_argument("--ego-mask-deg", type=float, default=20.0)
    a = ap.parse_args()
    root = Path(a.root) / a.city
    out_dir = root / a.dst
    out_dir.mkdir(parents=True, exist_ok=True)
    n, frac = 0, []
    for dp in sorted((root / a.src).glob("*.png")):
        pano = cv2.imread(str(root / "panorama" / (dp.stem + ".jpg")), cv2.IMREAD_COLOR)
        if pano is None:
            raise FileNotFoundError(f"no panorama for {dp.name}")
        d = np.array(Image.open(dp))
        if d.shape[:2] != pano.shape[:2]:
            pano = cv2.resize(pano, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked, valid = mask_depth(d, pano, a.ego_mask_deg)
        Image.fromarray(masked).save(out_dir / dp.name)
        frac.append(valid.mean())
        n += 1
    print(f"{n} depth maps masked -> {out_dir}; valid share of pixels: mean {np.mean(frac):.3f} "
          f"(min {np.min(frac):.3f}, max {np.max(frac):.3f})")


if __name__ == "__main__":
    main()
