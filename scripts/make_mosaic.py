"""Mosaic of per-frame BEVs at given travelled distances from a reference frame.

  python scripts/make_mosaic.py --ref 200 --dists 0 1 5 10

Poses: npz from scripts/lidar_odometry.py (names, poses 4x4). OxTS poses will replace it.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C, viz
from bevloc.bev.grid import BevGrid
from bevloc.bev.mosaic import Mosaic
from bevloc.bev.variants import build_variants
from bevloc.run import context


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--poses", default="experiments/00_calib/kiss_icp_poses_130_299.npz")
    ap.add_argument("--ref", default="200")
    ap.add_argument("--dists", nargs="+", type=float, default=[0, 1, 5, 10])
    ap.add_argument("--variant", default="ipm_cl")
    ap.add_argument("--canvas", type=int, default=320, help="mosaic size in cells")
    ap.add_argument("--out", default="experiments/01_kickoff/mosaic")
    a = ap.parse_args()
    cfg = C.load(a.config)
    ds, calib, erp_valid = context(cfg)
    out = Path(a.out)
    C.snapshot(cfg, out, dict(poses=a.poses))

    Z = np.load(a.poses)
    names, poses = [str(n) for n in Z["names"]], Z["poses"]
    ref = f"{int(a.ref):010d}"
    i0 = names.index(ref)
    travelled = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[i0:, :2, 3], axis=0), axis=1))]
    picks = [i0 + int(np.argmin(np.abs(travelled - d))) for d in a.dists]
    Tinv, grid = np.linalg.inv(poses[i0]), BevGrid(cfg.grid.n, cfg.grid.cell_m)

    cache = {}
    def bev(i):
        if i not in cache:
            b = build_variants(ds.erp(names[i]), ds.points(names[i]), erp_valid, calib, cfg,
                               variants=("ipm_edge",) if a.variant == "ipm_edge" else (a.variant,))
            cache[i] = (*b.get(a.variant), b.edge)
        return cache[i]

    def mosaic(idx):
        m = Mosaic(grid, a.canvas)
        for i in idx:
            m.add(*bev(i), Tinv @ poses[i])
        return m

    pad = (a.canvas - grid.n) // 2
    P = lambda x: cv2.copyMakeBorder(x, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    singles = []
    for i in picks:
        img, ok, edge = bev(i)
        singles.append(viz.bev_tile(P(img), P(ok.astype(np.uint8)) > 0, f"single {names[i]} (+{travelled[i - i0]:.1f} m)",
                                    cfg.grid.cell_m, edge=P(edge.astype(np.uint8)) > 0))
    steps = []
    for k in range(1, len(picks) + 1):
        m = mosaic(picks[:k])
        steps.append(viz.bev_tile(m.img, m.valid, "mosaic +" + "/".join(f"{d:g}" for d in a.dists[:k]) + " m",
                                  cfg.grid.cell_m, edge=m.edge, track=m.track))
    dense_idx = list(range(i0, picks[-1] + 1))
    md = mosaic(dense_idx)
    dense = viz.bev_tile(md.img, md.valid, f"mosaic of all {len(dense_idx)} frames over {travelled[picks[-1] - i0]:.1f} m",
                         cfg.grid.cell_m, edge=md.edge, track=md.track)

    q = [cv2.IMWRITE_JPEG_QUALITY, 90]
    cv2.imwrite(str(out / f"{ref}_singles.jpg"), viz.grid_of(singles, 2), q)
    cv2.imwrite(str(out / f"{ref}_mosaics.jpg"), viz.grid_of(steps, 2), q)
    cv2.imwrite(str(out / f"{ref}_dense.jpg"), dense, q)
    ins = lambda m: m.crop_reference()[1].mean()
    print(f"valid inside the reference {grid.extent:g} m grid: single {ins(mosaic(picks[:1])):.0%}, "
          f"{len(picks)}-frame mosaic {ins(mosaic(picks)):.0%}, dense {ins(md):.0%}")


if __name__ == "__main__":
    main()
