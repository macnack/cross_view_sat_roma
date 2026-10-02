"""Failure diagnostics for one eval_*.json: does a confidence gate exist, are misses year-consistent, how heavy is the tail.

  make pose-diag EVAL_JSON=experiments/05_lift_splat/eval/eval_ipm_manifest_test.json YEAR=2025
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from bevloc import config as C


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--row", default="peak")
    a = ap.parse_args()
    d = json.loads(open(a.eval_json).read())
    rows = [r for r in d["frames"] if int(r["year"]) == a.year]
    k = f"pose_{a.row}_m"
    e = np.array([np.inf if r[k] is None else r[k] for r in rows])
    inl = np.array([r[f"inliers_{a.row}"] for r in rows], float)
    modes = np.array([r[f"modes_{a.row}"] for r in rows], float)
    yaw = np.array([np.nan if r[f"yaw_{a.row}_deg"] is None else r[f"yaw_{a.row}_deg"] for r in rows])
    good, miss = e <= 5, e > 10
    print(f"{a.eval_json}  year {a.year}  row {a.row}  n {len(e)}  good(<=5m) {good.sum()}  miss(>10m) {miss.sum()}")
    print(f"inlier ratio   good {inl[good].mean():.2f}   miss {inl[miss].mean():.2f}")
    print(f"n_modes        good {modes[good].mean():.0f}   miss {modes[miss].mean():.0f}")
    print(f"yaw err (med)  good {np.nanmedian(yaw[good]):.1f} deg   miss {np.nanmedian(yaw[miss]):.1f} deg")
    print(f"tail: p90 {np.percentile(e, 90):.1f}  p95 {np.percentile(e, 95):.1f}  max {np.max(e[np.isfinite(e)]):.1f} m")
    for t in (0.6, 0.7, 0.8, 0.9):
        sel = inl >= t
        if sel.any():
            print(f"gate inliers>={t:.1f}: keeps {sel.mean():.2f} of frames, median {np.median(e[sel]):.1f} m, "
                  f">10 m {(e[sel] > 10).mean():.2f}")
    other = [y for y in sorted({int(r['year']) for r in d['frames']}) if y != a.year]
    for y in other:
        ey = {r["frame_id"]: (np.inf if r[k] is None else r[k]) for r in d["frames"] if int(r["year"]) == y}
        e2 = np.array([ey.get(r["frame_id"], np.inf) for r in rows])
        c = np.corrcoef(np.log1p(np.minimum(e, 100)), np.log1p(np.minimum(e2, 100)))[0, 1]
        print(f"year {y}: misses of {a.year} also miss in {y}: {(e2[miss] > 10).mean():.2f};  corr(log err) {c:.2f}")


if __name__ == "__main__":
    main()
