"""One scorer for every method on the Fixtor × Poznań manifest (the Poznań three-way comparison).

A row per manifest entry: the predicted map pose (CS92 east / north, heading in degrees clockwise from grid north)
against the manifest's pose proxy (`en`, `up_bearing_deg`) with `common.se2_map_error`; failures are rows with
ok = False and count as misses (inf) in every statistic. Summaries use `bevloc.eval.report.summarise_pose` (median
with percentile-bootstrap interval, R@5 / R@10 with intervals, > 30 m share) plus R@1, the mean capped at 1 km and
the heading-error median / mean over the frames with a pose.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from bevloc.baselines.common import se2_map_error
from bevloc.eval.report import summarise_pose


def open_orthos(years):
    from bevloc.data.mapillary import PoznanOrtho, poznan_tiles
    return {int(y): PoznanOrtho(poznan_tiles(int(y))) for y in years}


def centre_guess_m(entry) -> float:
    """Chance level of the protocol: predict the crop centre."""
    return float(np.hypot(*entry["crop_offset_m"]))


def base_row(entry, method: str, **extra) -> dict:
    return dict(frame_id=str(entry["frame_id"]), year=int(entry["year"]), method=method,
                gt_e=float(entry["en"][0]), gt_n=float(entry["en"][1]), gt_yaw=float(entry["up_bearing_deg"]),
                crop_rot_deg=float(entry["crop_rot_deg"]), centre_guess_m=centre_guess_m(entry), **extra)


def pose_row(entry, method: str, pred_en, pred_yaw_deg, **extra) -> dict:
    """Scored row; pred_yaw_deg None = the method gives no heading (err_deg None)."""
    row = base_row(entry, method, **extra)
    err_m, err_deg = se2_map_error(pred_en, 0.0 if pred_yaw_deg is None else pred_yaw_deg,
                                   entry["en"], entry["up_bearing_deg"])
    row.update(ok=True, pred_e=float(pred_en[0]), pred_n=float(pred_en[1]),
               pred_yaw=None if pred_yaw_deg is None else float(pred_yaw_deg), err_m=float(err_m),
               err_deg=None if pred_yaw_deg is None else float(err_deg), error=None)
    return row


def failed_row(entry, method: str, error: str, **extra) -> dict:
    row = base_row(entry, method, **extra)
    row.update(ok=False, pred_e=None, pred_n=None, pred_yaw=None, err_m=None, err_deg=None, error=str(error))
    return row


def summarise(rows, key="err_m", yaw_key="err_deg") -> dict:
    """Statistics of one row set (misses = None / non-finite count as inf)."""
    e = [r.get(key) for r in rows]
    s = summarise_pose(e)
    v = np.asarray([np.inf if x is None or not np.isfinite(x) else float(x) for x in e], float)
    s["recall@1m"] = float((v <= 1.0).mean()) if len(v) else float("nan")
    s["mean_capped_m"] = float(np.minimum(v, 1e3).mean()) if len(v) else float("nan")
    y = np.asarray([r[yaw_key] for r in rows if r.get(yaw_key) is not None], float)
    s["median_deg"] = float(np.median(y)) if len(y) else None
    s["mean_deg"] = float(np.mean(y)) if len(y) else None
    s["n_heading"] = int(len(y))
    return s


def summarise_by_year(rows, key="err_m", yaw_key="err_deg") -> dict:
    out = {"all": summarise(rows, key, yaw_key)}
    for y in sorted({int(r["year"]) for r in rows}):
        out[str(y)] = summarise([r for r in rows if int(r["year"]) == y], key, yaw_key)
    return out


def write_rows(out: Path, rows, name="frames.csv"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (out / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in r.items()})
    return out / name


def read_rows(path) -> list[dict]:
    """frames.csv written by `write_rows` (or the task-02 frames.csv of the main checkout) -> list of dicts with
    err_m / err_deg / year / ok parsed."""
    rows = []
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            d = dict(r)
            for k in ("err_m", "err_deg", "pred_e", "pred_n", "pred_yaw", "gt_yaw", "crop_rot_deg", "centre_guess_m"):
                if k in d:
                    d[k] = None if d[k] in ("", "None", None) else float(d[k])
            d["year"] = int(d["year"])
            d["ok"] = str(d.get("ok", "True")) == "True"
            rows.append(d)
    return rows


def print_summary(tag, summary):
    for name, s in summary.items():
        md = "n/a" if s["median_deg"] is None else f"{s['median_deg']:.1f}"
        print(f"{tag:28s} {name:5s} n {s['n']:4d}  median {s['median_m']:.2f} m "
              f"{tuple(round(v, 2) for v in s['median_ci'])}  mean {s['mean_capped_m']:.2f} m  R@1 {s['recall@1m']:.2f}  "
              f"R@5 {s['recall@5m']:.2f}  R@10 {s['recall@10m']:.2f}  >30m {s['frac_gt_30m']:.2f}  heading median "
              f"{md} deg", flush=True)
