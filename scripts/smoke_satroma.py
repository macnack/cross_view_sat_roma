"""End-to-end smoke test of the Sat-RoMa wrapper on a synthetic pair with known ground truth.

The query is warped out of the package's bundled 896 px reference (rotation 25 deg, off-centre),
so checkpoint loading, GMM extraction, RANSAC and our metrics all run. Needs a GPU or ~15 s CPU.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C
from bevloc.eval.metrics import pose_errors
from bevloc.match import satroma


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--out", default="experiments/00_smoke")
    a = ap.parse_args()
    cfg = C.load(a.config)
    out = Path(a.out)
    C.snapshot(cfg, out)
    ref_path = Path(satroma.PACKAGE_DIR) / "examples/reference.png"
    ref = cv2.cvtColor(cv2.imread(str(ref_path)), cv2.COLOR_BGR2RGB)
    th, q0, cx, cy = np.radians(25), 111.5, 560.0, 330.0
    c, s = np.cos(th), np.sin(th)
    H_gt = np.array([[c, -s, cx - (c * q0 - s * q0)], [s, c, cy - (s * q0 + c * q0)], [0, 0, 1]])
    query = cv2.warpPerspective(ref, np.linalg.inv(H_gt), (224, 224), flags=cv2.INTER_LINEAR)

    results = {}
    for use_means in (False, True):
        m = satroma.SatRoMa.from_config(cfg, use_means=use_means)
        r = m.match(query, ref)
        err = pose_errors(r.H, H_gt, 224, 1.0) if r.H is not None else None
        results[f"use_means={use_means}"] = dict(modes=r.n_modes, patches=r.n_patches, multimodal=r.n_multimodal,
                                                 inlier_ratio=r.inlier_ratio, error_px=err)
        print(f"use_means={use_means}: modes {r.n_modes}, inliers {r.inlier_ratio:.2f}, error px {err}")
    json.dump(results, open(out / "metrics.json", "w"), indent=1)

    o = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
    gt = (np.c_[[[0, 0], [223, 0], [223, 223], [0, 223]], np.ones(4)] @ H_gt.T)[:, :2]
    cv2.polylines(o, [gt.astype(np.int32)], True, (0, 255, 0), 2)
    if r.corners is not None:
        cv2.polylines(o, [r.corners.astype(np.int32)], True, (0, 0, 255), 2)
    q = cv2.resize(cv2.cvtColor(query, cv2.COLOR_RGB2BGR), (896, 896), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out / "synthetic_pair.jpg"), np.hstack([q, o]))
    ok = all(v["error_px"] and v["error_px"]["position_m"] < 8 for v in results.values())
    print("SMOKE TEST", "PASSED" if ok else "FAILED", "(centre error < 8 px = half a coarse cell)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
