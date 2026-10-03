"""Effect of SliceMatch's corrected VIGOR labels: the same three methods scored on the original and on the corrected labels.

  make vigor-label-effect

Inputs: the full Chicago same-area test runs on the original labels (--old, experiments/16_full_test) and on
`splits__corrected` (--new, experiments/17_corrected_labels); same panoramas, same checkpoints, nothing retrained.
For each method: median / mean / R@5 / R@10 on both label sets and the per-frame change. For PanoRoMa, whose json keeps
the ground truth and the predictions in metres east/north of the tile centre, it also measures how far the label moved
per frame and whether the predictions sit closer to the original or to the corrected label (a model trained on the
original labels can learn their offset). Writes <new>/label_effect.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


def stats(e):
    e = np.asarray(e, float)
    e = np.where(np.isfinite(e), e, 1e3)  # no pose = a miss
    return dict(n=int(e.size), median=float(np.median(e)), mean=float(np.mean(np.minimum(e, 1e3))),
                r5=float(np.mean(e <= 5)), r10=float(np.mean(e <= 10)))


def fg2_csv(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["pano"]] = float(r["ransac_m"]) if r["ransac_m"] not in ("", "None") else np.nan
    return out


def pano_json(path):
    return {r["id"].split("/", 1)[1]: r for r in json.load(open(path))["frames"]}


def final_m(r):
    v = r["pose_fine_gated_m"]
    return np.nan if v is None else v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="experiments/16_full_test")
    ap.add_argument("--new", default="experiments/17_corrected_labels")
    a = ap.parse_args()

    res = {}
    for name, old_f, new_f in [("FG2", "eval_fg2_full_chicago_fg2_samearea.csv", "eval_fg2_cl_chicago_fg2_samearea.csv"),
                               ("Loc2", "eval_loc2_full_chicago_loc2_samearea.csv", "eval_loc2_cl_chicago_loc2_samearea.csv")]:
        o, n = fg2_csv(os.path.join(a.old, old_f)), fg2_csv(os.path.join(a.new, new_f))
        keys = [k for k in o if k in n]
        eo, en = np.array([o[k] for k in keys]), np.array([n[k] for k in keys])
        res[name] = dict(original=stats(eo), corrected=stats(en),
                         frac_better=float(np.mean(np.nan_to_num(en, nan=1e3) < np.nan_to_num(eo, nan=1e3))))

    o = pano_json(os.path.join(a.old, "eval_vigor_full_chicago_pano_D_samearea.json"))
    n = pano_json(os.path.join(a.new, "eval_vigor_cl_chicago_pano_D_samearea.json"))
    keys = [k for k in o if k in n]
    res["PanoRoMa D two-pass"] = dict(original=stats([final_m(o[k]) for k in keys]),
                                      corrected=stats([final_m(n[k]) for k in keys]))
    res["PanoRoMa D first pass"] = dict(original=stats([o[k]["pose_peak_m"] for k in keys]),
                                        corrected=stats([n[k]["pose_peak_m"] for k in keys]))

    # label shift and which label the predictions follow (both runs deterministic on the same tile; check it)
    g_old = np.array([o[k]["en_gt"] for k in keys])
    g_new = np.array([n[k]["en_gt"] for k in keys])
    shift = g_new - g_old
    ok = np.array([n[k]["en_fine"] is not None and o[k]["en_fine"] is not None for k in keys])
    p_new = np.array([n[k]["en_fine"] if n[k]["en_fine"] is not None else [np.nan, np.nan] for k in keys])
    p_old = np.array([o[k]["en_fine"] if o[k]["en_fine"] is not None else [np.nan, np.nan] for k in keys])
    same_pred = float(np.nanmax(np.linalg.norm(p_new[ok] - p_old[ok], axis=1)))
    d_to_old = np.linalg.norm(p_new - g_old, axis=1)[ok]
    d_to_new = np.linalg.norm(p_new - g_new, axis=1)[ok]
    near = (d_to_old < 5) | (d_to_new < 5)
    mag = np.linalg.norm(shift, axis=1)
    # mean residual (prediction - label) of near frames, in east/north metres: a learned offset shows up as a bias
    bias_old = np.nanmean((p_new - g_old)[ok][near], axis=0)
    bias_new = np.nanmean((p_new - g_new)[ok][near], axis=0)
    res["label_shift"] = dict(
        median_m=float(np.median(mag)), mean_m=float(np.mean(mag)), p90_m=float(np.percentile(mag, 90)),
        max_m=float(np.max(mag)), frac_over_1m=float(np.mean(mag > 1)),
        mean_shift_en=[float(x) for x in shift.mean(0)],
        max_prediction_change_between_runs_m=same_pred,
        panoroma_closer_to_original_frac=float(np.mean(d_to_old[near] < d_to_new[near])),
        panoroma_median_to_original_m=float(np.median(d_to_old)),
        panoroma_median_to_corrected_m=float(np.median(d_to_new)),
        panoroma_bias_vs_original_en=[float(x) for x in bias_old],
        panoroma_bias_vs_corrected_en=[float(x) for x in bias_new])

    for k, v in res.items():
        if "original" in v:
            so, sn = v["original"], v["corrected"]
            print(f"{k:24s} median {so['median']:.2f} -> {sn['median']:.2f} m  mean {so['mean']:.2f} -> {sn['mean']:.2f} m  "
                  f"R@5 {so['r5']:.3f} -> {sn['r5']:.3f}  R@10 {so['r10']:.3f} -> {sn['r10']:.3f}")
    print(json.dumps(res["label_shift"], indent=1))
    json.dump(res, open(os.path.join(a.new, "label_effect.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
