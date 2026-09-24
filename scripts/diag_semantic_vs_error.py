"""Do the IPM query's misses coincide with panoramas that show no above-ground structure?

  make pose-diag-semantic EVAL_JSON=experiments/05_lift_splat/eval/eval_ipm_manifest_test.json YEAR=2025

Joins each evaluated frame's pose error with the class fractions of its precomputed SegFormer map
(scripts/semantic_precompute.py), restricted to the band between the horizon and the ego-body cut
(what the IPM picture and a contact-line picture would see). Prints hit/miss class fractions and the
recall inside quartiles of "structure" (building + wall + fence + pole + traffic sign fraction).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C
from bevloc.baselines.common import load_manifest
from bevloc.data.mapillary import MAP_ROOT

CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
           "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
STRUCT = [2, 3, 4, 5, 7]
GROUP = {"road": [0, 1, 9], "structure": STRUCT, "vegetation": [8], "vehicles": [13, 14, 15]}


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--manifest", default="experiments/06_fg2_bevsplat/manifest_test.json")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--band-deg", type=float, nargs=2, default=(-20.0, 15.0),
                    help="elevation band (deg, +up) counted: ego-body cut .. a bit above the horizon")
    a = ap.parse_args()
    d = json.loads(Path(a.eval_json).read_text())
    err = {r["frame_id"]: (np.inf if r["pose_peak_m"] is None else r["pose_peak_m"])
           for r in d["frames"] if int(r["year"]) == a.year}
    man = load_manifest(a.manifest)
    seq = MAP_ROOT / "Fixtor" / man["meta"]["held_out_seq"]
    rows = []
    for e in man["frames"]:
        if int(e["year"]) != a.year or e["frame_id"] not in err:
            continue
        p = seq / "semantic" / f"{e['frame_id']}.png"
        sem = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if sem is None:
            continue
        H = sem.shape[0]
        lat = (0.5 - (np.arange(H) + 0.5) / H) * 180.0
        band = (lat >= a.band_deg[0]) & (lat <= a.band_deg[1])
        s = sem[band]
        fr = {g: float(np.isin(s, ids).mean()) for g, ids in GROUP.items()}
        fr["err"] = err[e["frame_id"]]
        rows.append(fr)
    if not rows:
        raise SystemExit("no frames joined; are the semantic maps present?")
    E = np.array([r["err"] for r in rows])
    hit, miss = E <= 5, E > 10
    print(f"n {len(rows)}  hit(<=5 m) {hit.sum()}  miss(>10 m) {miss.sum()}   band {a.band_deg} deg")
    for g in GROUP:
        v = np.array([r[g] for r in rows])
        print(f"  {g:10s} fraction  hit {v[hit].mean():.3f}   miss {v[miss].mean():.3f}   corr(log err) {np.corrcoef(v, np.log1p(np.minimum(E, 100)))[0, 1]:+.2f}")
    v = np.array([r["structure"] for r in rows])
    q = np.quantile(v, [0.25, 0.5, 0.75])
    for lo, hi, name in ((-1, q[0], "Q1 least structure"), (q[0], q[1], "Q2"), (q[1], q[2], "Q3"), (q[2], 2, "Q4 most structure")):
        sel = (v > lo) & (v <= hi)
        print(f"  {name:20s} n {sel.sum():3d}  median {np.median(E[sel]):5.1f} m  R@5 {(E[sel] <= 5).mean():.2f}  R@10 {(E[sel] <= 10).mean():.2f}")


if __name__ == "__main__":
    main()
