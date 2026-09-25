"""Run a query checkpoint on VIGOR (known orientation): position error in metres per city, with chance.

  make vigor-eval CKPT=checkpoints/05_lift_splat_fixtor_ipm_long_best.pt SPLIT=crossarea TAG=ipm_zeroshot LIMIT=0
  make vigor-calibrate CKPT=...           # settles row_sign / camera height on 150 samples (see below)

Per sample: IPM picture of the north-aligned panorama (or the checkpoint's own query mode) -> frozen
encoder + decoder -> Sat-RoMa consensus (peak and means rows) against the positive tile. The estimated
pose is the panorama's position in the tile; the error is the distance in metres (per-city resolution).
Also reported: the centre guess (predict the tile centre), which is this protocol's chance level.
Queries with placed tokens (erp, erp_depth) go through the placed-token consensus (`SatRoMa.consensus_from_gm`, the
consensus half of `match_placed`); an erp_depth checkpoint evaluates only the panoramas of the draw that have a depth
file (`make loc2-depth` with the same SPLIT / CITIES / LIMIT), and reports how many were dropped.
Calibration mode scores row_sign in {+1, -1} x height in {1.6, 2.0, 2.5, 3.0} on a small subset and prints
the table; the best pair is what `vigor:` in configs/default.yaml should carry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from bevloc import config as C
from bevloc.data.vigor import CITY_RES, VigorPairs, split_cities
from bevloc.eval.metrics import pose_errors
from bevloc.eval.report import summarise_pose
from bevloc.match.satroma import SatRoMa, consensus_for_query
from bevloc.model.coarse import FeatureQueryMatcher
from bevloc.model.query import apply_query_cfg, build_query, load_query_state


def score(ds, query, matcher, cons, cfg, dev, n_max=0, verbose=False):
    rows = []
    n = int(cfg.grid.n)
    idx = range(len(ds)) if not n_max else range(min(n_max, len(ds)))
    for i in idx:
        try:
            s = ds[i]
        except RuntimeError as e:                                   # unreadable / missing image
            print(f"  skip {i}: {e}", flush=True)
            continue
        batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = query(batch, matcher)
            f_s = matcher.reference_features(batch["ref"])
            sf = float(((f_q.shape[-2] * 16) * (f_q.shape[-1] * 16)) ** 0.5 / 560.0)
            with matcher.model.exposed_intermediates():
                gm = matcher.model.decoder({16: f_q}, f_s, scale_factor=sf)[16]["gm_cls"][0]
        H = s["H"].numpy().astype(float)
        row = dict(id=s["id"], city=s["city"], centre_guess_m=ds.centre_guess_m(i))
        for tag, c in cons.items():
            m = consensus_for_query(c, gm, query, batch, frac, n, min_frac=0.05)
            err = pose_errors(m.H, H, n, float(cfg.grid.cell_m)) if m.H is not None else None
            row[f"pose_{tag}_m"] = None if err is None else err["position_m"]
            row[f"yaw_{tag}_deg"] = None if err is None else err["yaw_deg"]
            row[f"inliers_{tag}"] = m.inlier_ratio
        rows.append(row)
        if verbose and len(rows) % 25 == 0:
            print(f"  {len(rows)} done  peak median so far "
                  f"{np.median([np.inf if r['pose_peak_m'] is None else r['pose_peak_m'] for r in rows]):.1f} m", flush=True)
    return rows


def med(rows, key="pose_peak_m"):
    return float(np.median([np.inf if r[key] is None else r[key] for r in rows])) if rows else float("nan")


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--split", default="crossarea", choices=("crossarea", "samearea"))
    ap.add_argument("--train-split", action="store_true", help="evaluate on the training cities/labels instead")
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="random subset per run (0 = all)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--row-sign", type=float, default=None)
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--solver", default="")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default="experiments/09_vigor")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    cfg = C.load(a.config)
    if a.solver:
        cfg.matcher.solver = a.solver
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = state.get("mode", "lift")
    cfg.lift.query_mode = mode
    apply_query_cfg(cfg, state)
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    query = build_query(cfg, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    cons = {tag: SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=means, min_valid_frac=0.05)
            for tag, means in (("peak", False), ("means", True))}
    cities = a.cities or split_cities(a.split, a.train_split)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.calibrate:
        print(f"calibration on {a.limit or 150} samples, cities {cities}", flush=True)
        table = []
        for rs in (1.0, -1.0):
            for h in (1.6, 2.0, 2.5, 3.0):
                ds = VigorPairs(a.root, cfg, cities=cities, split=a.split, train=a.train_split,
                                limit=a.limit or 150, row_sign=rs, height_m=h)
                rows = score(ds, query, matcher, cons, cfg, dev)
                table.append(dict(row_sign=rs, height_m=h, median_m=med(rows), median_means_m=med(rows, "pose_means_m"),
                                  centre_guess_median_m=float(np.median([r["centre_guess_m"] for r in rows])), n=len(rows)))
                print(f"  row_sign {rs:+.0f}  height {h:.1f} m  ->  peak median {table[-1]['median_m']:.2f} m  "
                      f"means {table[-1]['median_means_m']:.2f} m  (centre guess {table[-1]['centre_guess_median_m']:.2f} m)", flush=True)
        best = min(table, key=lambda r: r["median_m"])
        (out / f"calibration_{a.tag}.json").write_text(json.dumps(dict(table=table, best=best), indent=2))
        print(f"best: row_sign {best['row_sign']:+.0f}, height {best['height_m']:.1f} m  ({best['median_m']:.2f} m)")
        return

    ds = VigorPairs(a.root, cfg, cities=cities, split=a.split, train=a.train_split, limit=a.limit,
                    stride=a.stride, row_sign=a.row_sign, height_m=a.height)
    n_no_depth = ds.keep_with_depth() if ds.depth else 0      # after the draw: a subset of the same samples
    if ds.depth:
        print(f"depth: {n_no_depth} panoramas of the draw have no depth file (skipped)", flush=True)
    print(f"{len(ds)} samples, split {a.split}, cities {cities}, row_sign {ds.row_sign:+.0f}, height {ds.height} m", flush=True)
    rows = score(ds, query, matcher, cons, cfg, dev, verbose=True)
    summary = {}
    for name in ["all"] + sorted({r["city"] for r in rows}):
        sub = rows if name == "all" else [r for r in rows if r["city"] == name]
        summary[name] = {
            "peak": summarise_pose([r["pose_peak_m"] for r in sub]),
            "means": summarise_pose([r["pose_means_m"] for r in sub]),
            "centre_guess": summarise_pose([r["centre_guess_m"] for r in sub]),
            "mean_peak_m": float(np.mean([min(np.inf if r["pose_peak_m"] is None else r["pose_peak_m"], 1e3) for r in sub])),
        }
    path = out / f"eval_vigor_{a.tag}_{a.split}.json"
    path.write_text(json.dumps(dict(meta=dict(ckpt=a.ckpt, mode=mode, split=a.split, cities=cities, n=len(rows),
                                              row_sign=ds.row_sign, height_m=ds.height, solver=cfg.matcher.solver,
                                              skipped_no_depth=n_no_depth,
                                              city_res=CITY_RES),
                                    frames=rows, summary=summary), indent=2))
    for name, s in summary.items():
        p, c = s["peak"], s["centre_guess"]
        print(f"{name:13s} n {p['n']:5d}  peak median {p['median_m']:.2f} m {tuple(round(v, 2) for v in p['median_ci'])}  "
              f"mean {s['mean_peak_m']:.2f} m  R@5 {p['recall@5m']:.2f}  R@10 {p['recall@10m']:.2f}  "
              f"| centre guess median {c['median_m']:.2f} m", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
