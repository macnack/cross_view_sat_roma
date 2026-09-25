"""Overlay sheet for a VIGOR evaluation: the worst (or a spread of) frames of an eval_vigor.py json, re-matched
with the same checkpoint so the RANSAC pose can be drawn.

  make vigor-viz CKPT=checkpoints/vigor_chicago_same_30k_best.pt EVAL_JSON=experiments/09_vigor/eval_vigor_ft_chicago_same_30k_samearea.json \\
       TAG=chicago30k_worst N=20 PICK=worst

Panels per frame: panorama | the flat-ground IPM picture the network matched | reference canvas zoomed on the
tile (green cross = label, red cross = RANSAC pose, the tile is the bright square inside the black canvas) |
the vote heat-map of the decoder's cell classifier (where the matcher thought the query lies). The caption
gives the error split into east/north so along-road vs across-road misses can be read off, and the
inlier ratio. PICK=worst takes the largest errors; PICK=spread spans the error range (best .. worst).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_pose import label  # noqa: E402

from bevloc import config as C  # noqa: E402
from bevloc.data.vigor import VigorPairs, split_cities  # noqa: E402
from bevloc.eval.metrics import pose_errors  # noqa: E402
from bevloc.match.satroma import SatRoMa, consensus_for_query  # noqa: E402
from bevloc.model.coarse import FeatureQueryMatcher  # noqa: E402
from bevloc.model.query import build_query, load_query_state  # noqa: E402
from bevloc.viz import fit, pose_overlay  # noqa: E402


def heatmap(gm, size):
    """Per-reference-cell vote mass of the classifier (sum over query patches), as a colour image."""
    v = gm.float().sum(dim=tuple(range(1, gm.dim()))) if gm.dim() > 2 else gm.float().sum(0)
    v = v.reshape(int(v.numel() ** 0.5), -1).cpu().numpy()
    v = (255 * (v - v.min()) / max(1e-9, v.max() - v.min())).astype(np.uint8)
    return cv2.applyColorMap(cv2.resize(v, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pick", default="worst", choices=("worst", "spread"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="experiments/09_vigor/viz")
    a = ap.parse_args()
    cfg = C.load(a.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = json.loads(Path(a.eval_json).read_text())
    meta, rows = ev["meta"], ev["frames"]
    rows = sorted(rows, key=lambda r: -(np.inf if r["pose_peak_m"] is None else r["pose_peak_m"]))
    if a.pick == "worst":
        picks = rows[:a.n]
    else:
        idx = np.unique(np.round(np.linspace(0, len(rows) - 1, a.n)).astype(int))
        picks = [rows[i] for i in idx]
    split = meta["split"]
    cities = meta.get("cities") or split_cities(split, False)
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = state.get("mode", "lift")
    cfg.lift.query_mode = mode
    ds = VigorPairs(a.root, cfg, cities=cities, split=split, train=False,
                    row_sign=meta.get("row_sign"), height_m=meta.get("height_m"))
    by_id = {f"{lab['city']}/{lab['pano']}": i for i, lab in enumerate(ds.labels)}
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    query = build_query(cfg, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    cons = SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=False, min_valid_frac=0.05)
    n, cell = int(cfg.grid.n), float(cfg.grid.cell_m)
    S = int(cfg.grid.n * cfg.reference.scale)
    sheets = []
    for r in picks:
        i = by_id[r["id"]]
        s = ds[i]
        batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = query(batch, matcher)
            f_s = matcher.reference_features(batch["ref"])
            sf = float(((f_q.shape[-2] * 16) * (f_q.shape[-1] * 16)) ** 0.5 / 560.0)
            with matcher.model.exposed_intermediates():
                gm = matcher.model.decoder({16: f_q}, f_s, scale_factor=sf)[16]["gm_cls"][0]
        H_gt = s["H"].numpy().astype(float)
        m = consensus_for_query(cons, gm, query, batch, frac, n, min_frac=0.05)
        err = pose_errors(m.H, H_gt, n, cell) if m.H is not None else None
        o = np.array([[(n - 1) / 2.0, (n - 1) / 2.0, 1.0]])
        g = (o @ H_gt.T)[0]
        gx, gy = g[0] / g[2], g[1] / g[2]
        if m.H is not None:
            e = (o @ np.asarray(m.H, float).T)[0]
            ex, ey = e[0] / e[2], e[1] / e[2]
            d_east, d_north = (ex - gx) * cell, -(ey - gy) * cell
        else:
            d_east = d_north = float("nan")
        erp = (s["erp"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ref = (s["ref"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        qpic = (s["bev"].permute(1, 2, 0).numpy() * 255).astype(np.uint8) if "bev" in s else erp
        ref_bgr = pose_overlay(ref, H_gt, m.H, n)
        rz = 224
        x0, y0 = int(np.clip(gx - rz, 0, S - 2 * rz)), int(np.clip(gy - rz, 0, S - 2 * rz))
        zoom = ref_bgr[y0:y0 + 2 * rz, x0:x0 + 2 * rz]
        hm = heatmap(gm, S)[y0:y0 + 2 * rz, x0:x0 + 2 * rz]
        cv2.drawMarker(hm, (int(gx - x0), int(gy - y0)), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        h = 300
        parts = [label(fit(cv2.cvtColor(erp, cv2.COLOR_RGB2BGR), h), "1 panorama (north at the centre column)"),
                 label(fit(cv2.cvtColor(qpic, cv2.COLOR_RGB2BGR), h), (f"2 query: IPM picture ({n * cell:.0f} m, {cell} m/px)" if "bev" in s else "2 query: panorama tokens")),
                 label(fit(zoom, h), f"3 tile, {2 * rz * cell:.0f} m window: green = label, red = RANSAC"),
                 label(fit(hm, h), "4 classifier votes over reference cells (green = label)")]
        sheet = np.hstack(parts)
        verdict = "no match" if err is None else (f"{err['position_m']:.1f} m  (east {d_east:+.1f}, north {d_north:+.1f})  "
                                                  f"yaw {err['yaw_deg']:.0f} deg  "
                                                  + ("GOOD" if err["position_m"] <= 5 else "MID" if err["position_m"] <= 10 else "MISS"))
        tag = f"{r['id']}  RANSAC {verdict}  inliers {m.inlier_ratio:.2f}  centre guess {r['centre_guess_m']:.1f} m  (eval json {r['pose_peak_m']:.1f} m)"
        sheets.append(label(sheet, tag, h=34))
        print(tag, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"vigor_{a.tag}_{a.pick}{a.n}.jpg"
    cv2.imwrite(str(path), np.vstack(sheets), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
