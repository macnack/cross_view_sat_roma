"""LiDAR return statistics, BEV occupancy and road-plane height over many frames.

Motivates: wet-asphalt dropout (oracle variants), sparse BEV coverage, LiDAR height estimate.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bevloc import config as C
from bevloc.bev.grid import BevGrid
from bevloc.run import context


def plane_ransac(g, rng, thr=0.03, iters=300):
    best = None
    for _ in range(iters):
        s = g[rng.choice(len(g), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < 0.9:
            continue
        inl = np.abs((g - s[0]) @ n) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    A = np.c_[g[best][:, :2], np.ones(best.sum())]
    c, *_ = np.linalg.lstsq(A, g[best][:, 2], rcond=None)
    return c, int(best.sum()), float(np.std(A @ c - g[best][:, 2]))


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", default="experiments/00_calib/lidar_stats.json")
    a = ap.parse_args()
    cfg = C.load(a.config)
    ds, _, _ = context(cfg, need_mask=False)
    grid, rng, rows = BevGrid(cfg.grid.n, cfg.grid.cell_m), np.random.default_rng(0), []
    for n in ds.names(("scan",), a.start, a.stop, a.stride):
        s = ds.scan(n)
        ok, xyz = s.valid, s.xyz
        az = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
        col_az = np.array([np.median(az[:, j][ok[:, j]]) if ok[:, j].any() else np.nan for j in range(az.shape[1])])
        sector = (np.abs(col_az) < 30) | (np.abs(col_az) > 150)          # fore / aft: where the road is
        low = slice(88, 128)                                             # downward beams
        p = xyz[low][:, sector][ok[low][:, sector]]
        p = p[(np.linalg.norm(p[:, :2], axis=1) < 15) & (p[:, 2] < -1.0)]
        pts = s.points(0.5)
        r, c, inb = grid.to_cell(pts[:, 0], pts[:, 1])
        occ = np.zeros((grid.n, grid.n), bool)
        occ[r[inb], c[inb]] = True
        d = dict(frame=n, return_rate=float(ok.mean()), road_sector_rate=float(ok[low][:, sector].mean()),
                 bev_cells_hit=float(occ.mean()))
        if len(p) > 500:
            co, n_inl, rms = plane_ransac(p, rng)
            d.update(height=float(-co[2]), slope_x_deg=float(np.degrees(np.arctan(co[0]))),
                     slope_y_deg=float(np.degrees(np.arctan(co[1]))), rms=rms, n_inl=n_inl)
        rows.append(d)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)

    R = lambda k: np.array([x.get(k, np.nan) for x in rows])
    print("frames", len(rows))
    for k in ("return_rate", "road_sector_rate", "bev_cells_hit"):
        print(f"  {k:17s} median {np.nanmedian(R(k)):.2f}  p10 {np.nanpercentile(R(k), 10):.2f}  p90 {np.nanpercentile(R(k), 90):.2f}")
    # road plane: accept only near-level fits (steep ones are grass banks)
    good = (R("n_inl") > 2000) & (R("rms") < 0.03) & (np.abs(R("slope_x_deg")) < 2.5) & (np.abs(R("slope_y_deg")) < 2.5)
    if good.sum() >= 3:
        h = R("height")[good]
        print(f"  LiDAR height above road: median {np.median(h):.3f} m, std {h.std():.3f}, n = {good.sum()} frames"
              f" (configured: {cfg.calib.lidar_height_m})")
    else:
        print(f"  only {good.sum()} frames with a usable near-level road plane")


if __name__ == "__main__":
    main()
