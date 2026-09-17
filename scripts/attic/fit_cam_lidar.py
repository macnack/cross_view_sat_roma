"""Fit the camera<-LiDAR residual (rotation + translation) by edge alignment.

p_cam = R(rpy) @ p_lidar + t, starting from SI2BEV's identity assumption.
Score: mean normalised image-gradient magnitude at projected LiDAR
depth-discontinuity points (foreground side of horizontal range jumps).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bevloc.bev.spherical import lidar_to_erp  # noqa: E402
from bevloc.data.dur360 import Dur360Frames  # noqa: E402

W, H = 2048, 1024
EGO_ROW = int(0.62 * H)     # below this the ERP is car body
BOUNDS = [(-4, 4)] * 3 + [(-0.6, 0.6)] * 3   # deg, m: sensors share one roof rack


def lidar_edges(root, name, jump=0.5, rel=0.08, rmin=2.0, rmax=40.0):
    raw = np.fromfile(Path(root) / "ouster_points/data" / f"{name}.bin", np.float32).reshape(128, 2048, 9)
    xyz = raw[..., :3].astype(np.float64)
    r = np.linalg.norm(xyz, axis=-1)
    ok = (r > rmin) & (r < rmax)
    pts = []
    for sh in (1, -1):
        rn, okn = np.roll(r, sh, 1), np.roll(r > 0.5, sh, 1)
        fg = ok & okn & (rn - r > np.maximum(jump, rel * r))   # neighbour is farther: we are foreground
        pts.append(xyz[fg])
    return np.concatenate(pts)


def edge_image(erp):
    g = cv2.cvtColor(erp, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)          # vertical structures -> horizontal gradient
    # horizontal range jumps are vertical structures: score |gx| only, otherwise the
    # optimiser drifts points onto the horizon/skyline (a strong gy edge)
    m = cv2.GaussianBlur(np.abs(gx), (0, 0), 2.0)
    m[EGO_ROW:] = 0
    return m / (np.percentile(m[:EGO_ROW], 99) + 1e-6)


def score(x, data):
    R = Rotation.from_euler("xyz", x[:3], degrees=True).as_matrix()
    s = []
    for E, P in data:
        u, v, _ = lidar_to_erp(P, W, H, R, x[3:6])
        val = cv2.remap(E, (u % W).astype(np.float32)[None], v.astype(np.float32)[None],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)[0]
        s.append(val[v < EGO_ROW].mean())
    return float(np.mean(s))


def load(root, names):
    ds = Dur360Frames(root, erp_size=(W, H))
    return [(edge_image(ds.erp(n)), lidar_edges(root, n)) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dur360bev")
    ap.add_argument("--fit", nargs="+", default=["0000000000", "0000000110", "0000000120", "0000001600", "0000001700"])
    ap.add_argument("--val", nargs="+", default=["0000000500", "0000001000", "0000002000", "0000002300", "0000002600"])
    ap.add_argument("--out", default="experiments/00_calib/cam_lidar_fit.json")
    a = ap.parse_args()

    fit, val = load(a.root, a.fit), load(a.root, a.val)
    print("edge points per fit frame:", [len(p) for _, p in fit])
    x0 = np.zeros(6)
    print(f"identity: fit {score(x0, fit):.4f}  val {score(x0, val):.4f}")

    # 1-D scans around identity, for a readable picture of each parameter's sensitivity
    names = ["roll_deg", "pitch_deg", "yaw_deg", "tx_m", "ty_m", "tz_m"]
    grids = [np.linspace(-3, 3, 25)] * 3 + [np.linspace(-0.6, 0.6, 25)] * 3
    scans = {}
    for i, (n, g) in enumerate(zip(names, grids)):
        sc = []
        for gv in g:
            x = x0.copy(); x[i] = gv
            sc.append(score(x, fit))
        scans[n] = dict(grid=g.tolist(), score=sc)
        print(f"  scan {n:10s} best at {g[int(np.argmax(sc))]:+.2f}  (score {max(sc):.4f})")

    best = None
    rng = np.random.default_rng(0)
    starts = [x0] + [x0 + rng.normal(0, [1, 1, 1, .2, .2, .2]) for _ in range(7)]
    sols = []
    for s in starts:
        s = np.clip(s, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
        r = minimize(lambda x: -score(x, fit), s, method="Powell", bounds=BOUNDS,
                     options=dict(xtol=1e-3, ftol=1e-6, maxiter=4000))
        sols.append(np.r_[r.x, -r.fun])
        if best is None or -r.fun > best[1]:
            best = (r.x, -r.fun)
    sols = np.array(sols)
    top = sols[sols[:, 6] >= best[1] - 0.002]      # restarts that reached (nearly) the same optimum
    x = best[0]
    print("restarts reaching the optimum:", len(top), "/", len(sols))
    for i, n in enumerate(names):
        print(f"  {n:10s} {x[i]:+.3f}   spread over good restarts ±{top[:, i].std():.3f}")
    print(f"fitted:   fit {score(x, fit):.4f}  val {score(x, val):.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(params=dict(zip(names, x.tolist())), spread=dict(zip(names, top[:, :6].std(0).tolist())),
                   score_identity=dict(fit=score(x0, fit), val=score(x0, val)),
                   score_fitted=dict(fit=score(x, fit), val=score(x, val)),
                   fit_frames=a.fit, val_frames=a.val, scans=scans), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
