"""Remove a model's constant pose offset: fit per city on held-out TRAINING frames, subtract on the test frames.

  make vigor-offset-calib CALIB_JSON=<eval_vigor.py --calib json> TEST_JSON=<eval_vigor.py test json>

CALIB_JSON: `eval_vigor.py --calib` (the --val-frac part of the training list the model never trained on; positions
en_gt / en_coarse / en_fine per frame). TEST_JSON: the test run to correct. Fits bevloc.eval.pose_offset.fit_offsets per
city on CALIB_JSON, subtracts the offset of each test frame's city from its final position and re-scores. The test
frames play no part in the fit (frame ids checked disjoint), and the uncorrected final errors are first reproduced from
the positions (must equal `pose_fine_gated_m`, or the position bookkeeping is wrong). Writes <out>/offset_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from bevloc.eval import pose_offset as P


def stats(e, rng):
    e = np.where(np.isfinite(e), e, 1e3)
    b = [np.median(e[rng.integers(0, len(e), len(e))]) for _ in range(300)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(median=float(np.median(e)), ci=[float(lo), float(hi)], mean=float(e.mean()), r5=float(np.mean(e <= 5)),
                r10=float(np.mean(e <= 10)), over5=float(np.mean(e > 5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-json", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--tag", default="cl_new")
    ap.add_argument("--out", default="experiments/18_panoroma_corrected_labels")
    ap.add_argument("--max-err", type=float, default=5.0)
    ap.add_argument("--min-frames", type=int, default=100)
    a = ap.parse_args()

    calib = json.load(open(a.calib_json))["frames"]
    test = json.load(open(a.test_json))["frames"]
    shared = {r["id"] for r in calib} & {r["id"] for r in test}
    if shared:
        raise SystemExit(f"{len(shared)} frames are in both the calibration and the test json (e.g. {sorted(shared)[:2]})")
    off = P.fit_offsets(calib, max_err_m=a.max_err, min_frames=a.min_frames)
    for city in {r["city"] for r in test}:
        if city not in off:
            raise SystemExit(f"no calibration frames for {city}: the offset cannot be fitted")

    raw = np.array([np.inf if P.final_en(r) is None else float(np.linalg.norm(np.subtract(P.final_en(r), r["en_gt"])))
                    for r in test])
    ref = np.array([np.inf if r["pose_fine_gated_m"] is None else r["pose_fine_gated_m"] for r in test])
    fin = np.isfinite(raw) & np.isfinite(ref)
    if not np.allclose(raw[fin], ref[fin], atol=1e-3) or (np.isfinite(raw) != np.isfinite(ref)).any():
        raise SystemExit(f"final positions do not reproduce pose_fine_gated_m (max diff {np.abs(raw[fin] - ref[fin]).max():.4f} m)")
    cor = P.corrected_errors(test, off)

    rng = np.random.default_rng(0)
    res = dict(calib_json=a.calib_json, test_json=a.test_json, n_calib=len(calib), n_test=len(test), offsets=off,
               uncorrected=stats(raw, rng), corrected=stats(cor, rng),
               frac_frames_better=float(np.mean(cor[np.isfinite(raw)] < raw[np.isfinite(raw)])))
    for city in sorted(off):
        sel = np.array([r["city"] == city for r in test])
        res.setdefault("per_city", {})[city] = dict(n=int(sel.sum()), uncorrected=stats(raw[sel], rng), corrected=stats(cor[sel], rng))
    for city, o in off.items():
        print(f"{city:13s} offset E {o['b_en'][0]:+.2f} N {o['b_en'][1]:+.2f} m  (n {o['n']}, scatter {o['scatter_en'][0]:.2f}/{o['scatter_en'][1]:.2f})")
    for nm in ("uncorrected", "corrected"):
        s = res[nm]
        print(f"{nm:12s} median {s['median']:.3f} ({s['ci'][0]:.3f}-{s['ci'][1]:.3f}) mean {s['mean']:.3f} "
              f"<=5 {s['r5']:.3f} <=10 {s['r10']:.3f} >5 m {100 * s['over5']:.1f}%")
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"offset_{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
