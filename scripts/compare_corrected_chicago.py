"""PanoRoMa D (retrained on corrected labels) vs FG², Loc² and the original-label PanoRoMa D, all scored on SliceMatch's
corrected labels, full Chicago same-area test list; failure overlap with FG² and the effect of the constant offset.

  make vigor-compare-corrected

Inputs are experiments/17_corrected_labels (FG², Loc², old PanoRoMa D) and experiments/18_panoroma_corrected_labels
(the retrain). Bootstrap CI: 300 resamples of the median, seed 0. Writes <new>/compare_corrected.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


def pano(path):
    return {r["id"].split("/", 1)[1]: r for r in json.load(open(path))["frames"]}


def csv_err(path):
    return {r["pano"]: (float(r["ransac_m"]) if r["ransac_m"] not in ("", "None") else 1e3) for r in csv.DictReader(open(path))}


def final(r):
    return 1e3 if r["pose_fine_gated_m"] is None else r["pose_fine_gated_m"]


def row(e, rng):
    b = [np.median(e[rng.integers(0, len(e), len(e))]) for _ in range(300)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(median=float(np.median(e)), ci=[float(lo), float(hi)], mean=float(np.minimum(e, 1e3).mean()),
                r5=float(np.mean(e <= 5)), r10=float(np.mean(e <= 10)), over5=float(np.mean(e > 5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="experiments/17_corrected_labels")
    ap.add_argument("--new", default="experiments/18_panoroma_corrected_labels")
    ap.add_argument("--new-json", default="eval_vigor_cl_new_chicago_pano_D_samearea.json")
    a = ap.parse_args()
    new = pano(os.path.join(a.new, a.new_json))
    old = pano(os.path.join(a.old, "eval_vigor_cl_chicago_pano_D_samearea.json"))
    fg = csv_err(os.path.join(a.old, "eval_fg2_cl_chicago_fg2_samearea.csv"))
    lo = csv_err(os.path.join(a.old, "eval_loc2_cl_chicago_loc2_samearea.csv"))
    keys = [k for k in new if k in old and k in fg and k in lo]
    e = {"FG2": np.array([fg[k] for k in keys]), "PanoRoMa D retrained on corrected labels": np.array([final(new[k]) for k in keys]),
         "PanoRoMa D (original labels)": np.array([final(old[k]) for k in keys]), "Loc2": np.array([lo[k] for k in keys]),
         "PanoRoMa D retrained, first pass": np.array([1e3 if new[k]["pose_peak_m"] is None else new[k]["pose_peak_m"] for k in keys])}
    rng = np.random.default_rng(0)
    res = dict(n=len(keys), rows={nm: row(v, rng) for nm, v in e.items()})
    n, g = e["PanoRoMa D retrained on corrected labels"], e["FG2"]
    res["overlap_vs_fg2"] = dict(both_within_5=float(np.mean((n < 5) & (g < 5))), only_fg2_fails=float(np.mean((n < 5) & (g >= 5))),
                                 only_new_fails=float(np.mean((n >= 5) & (g < 5))), both_fail=float(np.mean((n >= 5) & (g >= 5))))
    d = np.array([np.subtract(new[k]["en_fine"], new[k]["en_gt"]) if new[k]["en_fine"] is not None else [np.nan, np.nan] for k in keys])
    ok = np.isfinite(d[:, 0]) & (np.linalg.norm(np.nan_to_num(d), axis=1) < 5)
    b = np.median(d[ok], 0)
    res["offset"] = dict(median_en=[float(x) for x in b], median_err_raw=float(np.nanmedian(np.linalg.norm(d, axis=1))),
                         median_err_offset_removed=float(np.nanmedian(np.linalg.norm(d - b, axis=1))))
    for nm, r in res["rows"].items():
        print(f"{nm:44s} median {r['median']:.2f} ({r['ci'][0]:.2f}-{r['ci'][1]:.2f}) mean {r['mean']:.2f} "
              f"<=5 {r['r5']:.3f} <=10 {r['r10']:.3f} >5 m {100 * r['over5']:.1f}%")
    print("n", res["n"], json.dumps(res["overlap_vs_fg2"]), json.dumps(res["offset"]))
    json.dump(res, open(os.path.join(a.new, "compare_corrected.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
