"""Overlay LiDAR points on the ERP image to check camera<->LiDAR alignment.

Usage:
  python scripts/check_lidar_erp.py --root data/dur360bev --frames 0000000000 ... \
      --out experiments/00_calib/lidar_in_erp [--rpy_deg r p y] [--t x y z]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bevloc.bev.spherical import lidar_to_erp  # noqa: E402
from bevloc.data.dur360 import Dur360Frames  # noqa: E402


def overlay(erp, pts, R_cl, t_cl, max_range=40.0, stride=1):
    h, w = erp.shape[:2]
    u, v, rng = lidar_to_erp(pts[::stride, :3], w, h, R_cl, t_cl)
    keep = (rng < max_range) & (rng > 1.0)
    u, v, rng = u[keep], v[keep], rng[keep]
    col = cv2.applyColorMap(
        np.clip(rng / max_range * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    out = cv2.cvtColor(erp, cv2.COLOR_RGB2BGR).copy()
    ui = np.clip(u.astype(int), 0, w - 1)
    vi = np.clip(v.astype(int), 0, h - 1)
    out[vi, ui] = col
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dur360bev")
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/00_calib/lidar_in_erp")
    ap.add_argument("--rpy_deg", nargs=3, type=float, default=[0, 0, 0])
    ap.add_argument("--t", nargs=3, type=float, default=[0, 0, 0])
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R_cl = Rotation.from_euler("xyz", a.rpy_deg, degrees=True).as_matrix()
    ds = Dur360Frames(a.root)
    for name in a.frames:
        erp, pts = ds.erp(name), ds.points(name)
        img = overlay(erp, pts, R_cl, np.array(a.t), stride=a.stride)
        # top: sparse overlay on image, bottom: raw image, for edge comparison
        both = np.vstack([img, cv2.cvtColor(erp, cv2.COLOR_RGB2BGR)])
        cv2.imwrite(str(out / f"{name}.jpg"), both, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(name, "points:", len(pts))


if __name__ == "__main__":
    main()
