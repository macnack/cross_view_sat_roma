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
    ap.add_argument("--solver", default="", help="srt | se2 (default cfg.matcher.solver)")
    a = ap.parse_args()
    cfg = C.load(a.config)
    if a.solver:
        cfg.matcher.solver = a.solver
    out = Path(a.out)
    C.snapshot(cfg, out)
    ref_path = Path(satroma.PACKAGE_DIR) / "examples/reference.png"
    ref = cv2.cvtColor(cv2.imread(str(ref_path)), cv2.COLOR_BGR2RGB)
    th, q0, cx, cy = np.radians(25), 111.5, 560.0, 330.0
    c, s = np.cos(th), np.sin(th)
    H_gt = np.array([[c, -s, cx - (c * q0 - s * q0)], [s, c, cy - (s * q0 + c * q0)], [0, 0, 1]])
    query = cv2.warpPerspective(ref, np.linalg.inv(H_gt), (224, 224), flags=cv2.INTER_LINEAR)

    results, est = {}, {}
    for use_means in (False, True):
        m = satroma.SatRoMa.from_config(cfg, use_means=use_means)
        r = m.match(query, ref)
        err = pose_errors(r.H, H_gt, 224, 1.0) if r.H is not None else None
        results[f"use_means={use_means}"] = dict(modes=r.n_modes, patches=r.n_patches, multimodal=r.n_multimodal,
                                                 inlier_ratio=r.inlier_ratio, error_px=err)
        est[use_means] = (r, err)
        print(f"solver={cfg.matcher.solver} use_means={use_means}: modes {r.n_modes}, "
              f"inliers {r.inlier_ratio:.2f}, error px {err}")
    json.dump(results, open(out / "metrics.json", "w"), indent=1)

    write_overlay(out / "synthetic_pair.jpg", query, ref, H_gt, est, cfg.matcher.checkpoint)
    ok = all(v["error_px"] and v["error_px"]["position_m"] < 8 for v in results.values())
    print("SMOKE TEST", "PASSED" if ok else "FAILED", "(centre error < 8 px = half a coarse cell)")
    raise SystemExit(0 if ok else 1)


CORNERS = np.c_[[[0, 0], [223, 0], [223, 223], [0, 223]], np.ones(4)]
ESTIMATES = ((False, (60, 60, 255)), (True, (255, 190, 40)))     # BGR


def write_overlay(path, query, ref, H_gt, est, checkpoint, zoom=5, pad=182):
    """query | reference with GT and both estimates | zoom on one corner, under a text header.

    Text goes in the header, not on the imagery: on this reference every colour is
    illegible somewhere in the frame.
    """
    def txt(im, s, xy, c, sc=0.62):
        cv2.putText(im, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, c, 2, cv2.LINE_AA)

    o = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
    gt = (CORNERS @ H_gt.T)[:, :2]
    cv2.polylines(o, [np.round(gt).astype(np.int32)], True, (0, 255, 0), 3)
    for use_means, col in ESTIMATES:
        c = est[use_means][0].corners
        if c is not None:
            cv2.polylines(o, [np.round(c).astype(np.int32)], True, col, 2)

    n = o.shape[0]
    half = n // (2 * zoom)
    cx, cy = np.clip(np.round(gt[0]).astype(int), half, n - half)      # zoom on the GT top-left corner
    z = cv2.resize(o[cy - half:cy + half, cx - half:cx + half], (n, n), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(o, (cx - half, cy - half), (cx + half, cy + half), (255, 255, 255), 2)

    q = cv2.resize(cv2.cvtColor(query, cv2.COLOR_RGB2BGR), (n, n), interpolation=cv2.INTER_NEAREST)
    body = np.hstack([q, o, z])
    head = np.zeros((pad, body.shape[1], 3), np.uint8)
    txt(head, f"Sat-RoMa smoke test, checkpoint {checkpoint}  -  aerial-to-aerial, NOT the BEV-to-ortho task",
        (12, 34), (255, 255, 255), 0.8)
    txt(head, f"query 224 px: a crop of the reference itself, rotated 25 deg   |   reference 896 px   |   "
              f"{zoom}x zoom on the white box", (12, 68), (190, 190, 190), 0.55)
    txt(head, "ground truth", (12, 104), (0, 255, 0))
    for (use_means, col), dy in zip(ESTIMATES, (134, 164)):
        e = est[use_means][1]
        txt(head, f"use_means={use_means}: corner error {e['corner_m']:.2f} px, yaw error "
                  f"{e['yaw_deg']:+.3f} deg", (12, dy), col)
    cv2.imwrite(str(path), np.vstack([head, body]), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main()
