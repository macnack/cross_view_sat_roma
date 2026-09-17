"""Camera<-LiDAR translation from the range-dependence of edge offsets.

For p_cam = p_lidar + t (small t), a point at range r, elevation el, azimuth az
(to the right of forward), horizontal range rho, moves in the image by
    d_el = tz * cos(el) / r            (+ a constant: pitch-like bias, beam quantisation)
    d_az = (y * tx - x * ty) / rho^2   (+ a constant: yaw)
Offsets are measured per LiDAR depth-edge point as the sub-pixel displacement to
the strongest image gradient within +-search px, perpendicular to the edge:
  top edges of objects   -> vertical search   -> tz
  left/right side edges  -> horizontal search -> tx, ty, yaw  (both sides used, so the
                                                 inside-the-silhouette bias cancels)
A constant offset cannot mimic a 1/r law, which is what makes this well posed.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bevloc.bev.spherical import lidar_to_erp  # noqa: E402
from bevloc.data.dur360 import Dur360Frames  # noqa: E402

W, H = 2560, 1280
PX = 2 * np.pi / W          # rad per pixel (both axes)


def edge_points(root, name, rmin=2.0, rmax=30.0, inject=(0, 0, 0), min_z=-1.0):
    raw = np.fromfile(Path(root) / "ouster_points/data" / f"{name}.bin", np.float32)
    xyz = raw.reshape(128, 2048, 9)[..., :3].astype(np.float64)
    r = np.linalg.norm(xyz, axis=-1)
    # min_z: >= ~0.5 m above ground (LiDAR ~1.57 m up); consecutive ground rings otherwise pass as 'edges'
    ok = (r > rmin) & (r < rmax) & (xyz[..., 2] > min_z)
    jump = lambda rn: (rn > 0.5) & (rn - r > np.maximum(1.0, 0.15 * r))
    top = ok & jump(np.roll(r, 1, 0)); top[0] = False            # row above (higher beam) is farther
    side = ok & (jump(np.roll(r, 1, 1)) | jump(np.roll(r, -1, 1)))
    return xyz[top] + np.asarray(inject, float), xyz[side] + np.asarray(inject, float)


def offsets(grad, u, v, axis, search=9):
    """Sub-pixel offset (px) of the strongest gradient along `axis` within +-search."""
    k = np.arange(-search, search + 1)
    uu = u[:, None] + (k[None] if axis == "u" else np.zeros_like(k)[None])
    vv = v[:, None] + (k[None] if axis == "v" else np.zeros_like(k)[None])
    mu, mv = (uu % W).astype(np.float32), (vv + 0 * uu).astype(np.float32)
    prof = np.concatenate([cv2.remap(grad, mu[j:j + 30000], mv[j:j + 30000], cv2.INTER_LINEAR)
                           for j in range(0, len(u), 30000)])      # cv2.remap: dst rows < 32767
    i = prof.argmax(1)
    inside = (i > 0) & (i < len(k) - 1)
    ii = np.clip(i, 1, len(k) - 2)
    n = np.arange(len(u))
    a, b, c = prof[n, ii - 1], prof[n, ii], prof[n, ii + 1]
    sub = 0.5 * (a - c) / (a - 2 * b + c - 1e-9)
    strength = b / (np.median(prof, 1) + 1e-6)
    return k[ii] + sub, inside & (strength > 3.0)


def robust_lsq(A, y, iters=20, c=1.5):
    w = np.ones(len(y))
    for _ in range(iters):
        x, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        res = y - A @ x
        s = 1.4826 * np.median(np.abs(res)) + 1e-12
        w = np.sqrt(np.minimum(1.0, c * s / (np.abs(res) + 1e-12)))   # Huber
    return x


def bootstrap(A, y, groups, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    est = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids))
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        est.append(robust_lsq(A[idx], y[idx], iters=8))
    return np.std(est, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dur360bev")
    ap.add_argument("--frames", nargs="+", default=None)
    ap.add_argument("--out", default="experiments/00_calib/cam_lidar_parallax.json")
    ap.add_argument("--inject", nargs=3, type=float, default=[0, 0, 0], help="shift LiDAR points by this (m): estimator should return minus it")
    a = ap.parse_args()
    frames = a.frames or ["0000000000", "0000000110", "0000000120", "0000001600", "0000001610", "0000001620",
                          "0000001630", "0000001640", "0000001650", "0000001660", "0000001670", "0000001680",
                          "0000001690", "0000001700"]            # static / near-static only (no sweep smear)
    ds = Dur360Frames(a.root, erp_size=(W, H), lens="dur360")
    ego_row = int(0.61 * H)

    Av, yv, gv, Au, yu, gu = [], [], [], [], [], []
    for fi, name in enumerate(frames):
        g = cv2.GaussianBlur(cv2.cvtColor(ds.erp(name), cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.2)
        gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
        gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0))
        top, side = edge_points(a.root, name, inject=a.inject)

        u, v, r = lidar_to_erp(top, W, H)
        d, ok = offsets(gy, u, v, "v"); ok &= v < ego_row - 10
        el = np.arcsin(top[:, 2] / r)
        Av.append(np.c_[np.cos(el) / r, np.ones(len(r))][ok]); yv.append((-d * PX)[ok]); gv.append(np.full(ok.sum(), fi))

        u, v, r = lidar_to_erp(side, W, H)
        d, ok = offsets(gx, u, v, "u"); ok &= v < ego_row - 10
        rho2 = side[:, 0] ** 2 + side[:, 1] ** 2
        Au.append(np.c_[side[:, 1] / rho2, -side[:, 0] / rho2, np.ones(len(r))][ok]); yu.append((d * PX)[ok]); gu.append(np.full(ok.sum(), fi))

    Av, yv, gv, Au, yu, gu = map(np.concatenate, (Av, yv, gv, Au, yu, gu))
    xv, sv = robust_lsq(Av, yv), bootstrap(Av, yv, gv)
    xu, su = robust_lsq(Au, yu), bootstrap(Au, yu, gu)
    res = dict(frames=frames, n_top=int(len(yv)), n_side=int(len(yu)),
               tz_m=[float(xv[0]), float(sv[0])], el_bias_deg=[float(np.degrees(xv[1])), float(np.degrees(sv[1]))],
               tx_m=[float(xu[0]), float(su[0])], ty_m=[float(xu[1]), float(su[1])],
               yaw_bias_deg=[float(np.degrees(xu[2])), float(np.degrees(su[2]))])
    print(f"top-edge points {len(yv)}, side-edge points {len(yu)}, frames {len(frames)} (bootstrap over frames)")
    for k in ("tz_m", "tx_m", "ty_m", "el_bias_deg", "yaw_bias_deg"):
        print(f"  {k:13s} {res[k][0]:+.3f} ± {res[k][1]:.3f}")
    # binned check of the 1/r law for tz: mean elevation offset per range bin
    rr = 1.0 / Av[:, 0]
    print("  elevation offset by range (deg):", "  ".join(
        f"{lo}-{hi}m:{np.degrees(np.median(yv[(rr >= lo) & (rr < hi)])):+.2f}" for lo, hi in ((2, 4), (4, 6), (6, 9), (9, 14), (14, 30))))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
