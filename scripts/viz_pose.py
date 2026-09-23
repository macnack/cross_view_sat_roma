"""Per-frame overlay sheet for any scored checkpoint on a manifest: what the matcher saw and where it put us.

  make pose-viz CKPT=checkpoints/05_lift_splat_fixtor_ipm_best.pt MANIFEST=experiments/06_fg2_bevsplat/manifest_test.json \
       TAG=ipm_test EVAL_JSON=experiments/05_lift_splat/eval/eval_ipm_manifest_test.json

Panels per frame: panorama | the query picture the network matched (IPM BEV for the ipm query; the
panorama again for the other modes) | reference crop with the Mapillary pose proxy (green) and the
RANSAC pose (red). Frames are picked to span the error range of the given evaluation json (best,
quartiles, worst) so the sheet shows successes and failures, not a cherry-picked handful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C
from bevloc.baselines.common import load_manifest
from bevloc.data.mapillary import MAP_ROOT, MapillaryPairs, PoznanOrtho, load_frames, poznan_tiles
from bevloc.data.ortho import Oriented
from bevloc.eval.metrics import pose_errors
from bevloc.match.satroma import SatRoMa
from bevloc.model.coarse import FeatureQueryMatcher
from bevloc.model.query import build_query, load_query_state
from bevloc.viz import fit, pose_overlay


def label(img, text, h=30):
    bar = np.full((h, img.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def pick_frames(entries, eval_json, year, n):
    """Spread the picks over the error quantiles of a previous evaluation, else take a stride."""
    ents = [e for e in entries if int(e["year"]) == year]
    if eval_json:
        d = json.loads(Path(eval_json).read_text())
        err = {r["frame_id"]: (np.inf if r["pose_peak_m"] is None else r["pose_peak_m"])
               for r in d["frames"] if int(r["year"]) == year}
        ents = sorted(ents, key=lambda e: err.get(e["frame_id"], np.inf))
        idx = np.unique(np.round(np.linspace(0, len(ents) - 1, n)).astype(int))
        return [(ents[i], err.get(ents[i]["frame_id"])) for i in idx]
    return [(e, None) for e in ents[:: max(1, len(ents) // n)][:n]]


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="experiments/06_fg2_bevsplat/manifest_test.json")
    ap.add_argument("--eval-json", default="", help="pick frames across this evaluation's error range")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="experiments/05_lift_splat/viz")
    a = ap.parse_args()
    cfg = C.load(a.config)
    L = cfg.lift
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = load_manifest(a.manifest)
    picks = pick_frames(man["frames"], a.eval_json, a.year, a.n)
    ortho = PoznanOrtho(poznan_tiles(a.year))
    seq_dir = MAP_ROOT / "Fixtor" / man["meta"]["held_out_seq"]
    frames = load_frames([seq_dir], ortho, margin_m=L.margin_m)
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = state.get("mode", "lift")
    L.query_mode = mode
    ds = MapillaryPairs(frames, {a.year: ortho}, cfg, train=False, seed=cfg.train.seed,
                        erp_size=tuple(L.erp_size), years=[a.year])
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    query = build_query(cfg, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    ransac = SatRoMa.from_config(cfg, use_means=False, min_valid_frac=L.min_patch_valid)
    ransac.m.model.decoder.load_state_dict(state["decoder"], strict=False)

    rows = []
    n = cfg.grid.n
    for e, prev_err in picks:
        ref_o = Oriented(tuple(e["crop_centre_en"]), float(e["crop_up_bearing_deg"]), int(e["ref_size"]), float(e["gsd_m"]))
        s = ds.sample_for(e["frame_id"], ref_o, a.year)
        batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = query(batch, matcher)
            f_s = matcher.reference_features(batch["ref"])
        H_gt = np.asarray(e["H_gt"], float)
        sf = float(((f_q.shape[-2] * 16) * (f_q.shape[-1] * 16)) ** 0.5 / 560.0)
        if hasattr(query, "placement"):
            xy, valid = query.placement(batch)
            m = ransac.match_placed(f_q, f_s[16], xy[0], valid[0], scale_factor=sf, H_gt=H_gt)
        else:
            mask = torch.nn.functional.interpolate(frac[:, None].float(), size=(n, n), mode="nearest")[0, 0]
            m = ransac.match_encoded(f_q, f_s[16], scale_factor=sf, mask=(mask >= L.min_patch_valid).cpu().numpy(), H_gt=H_gt)
        err = pose_errors(m.H, H_gt, n, cfg.grid.cell_m) if m.H is not None else None
        erp = (s["erp"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ref = (s["ref"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        h = 300
        if "bev" in s:
            qpic = (s["bev"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            qlabel = "2 query: IPM picture (h=1.7 m, 56 m, 0.25 m/px)"
        else:
            qpic = erp
            qlabel = f"2 query tokens come from the panorama ({mode})"
        ref_bgr = pose_overlay(ref, H_gt, m.H, n)
        # zoom the reference around the proxy pose so the boxes are readable: 448 px = 112 m window
        c = np.array([[n / 2 - 0.5, n / 2 - 0.5, 1.0]]) @ H_gt.T
        cx, cy = int(round(c[0, 0] / c[0, 2])), int(round(c[0, 1] / c[0, 2]))
        r = 224
        x0, y0 = int(np.clip(cx - r, 0, ref.shape[1] - 2 * r)), int(np.clip(cy - r, 0, ref.shape[0] - 2 * r))
        ref_zoom = ref_bgr[y0:y0 + 2 * r, x0:x0 + 2 * r]
        parts = [label(fit(cv2.cvtColor(erp, cv2.COLOR_RGB2BGR), h), "1 panorama"),
                 label(fit(cv2.cvtColor(qpic, cv2.COLOR_RGB2BGR), h), qlabel),
                 label(fit(ref_zoom, h), "3 orthophoto (112 m window): green = pose proxy, red = RANSAC"),
                 label(fit(ref_bgr, h), "4 full 224 m reference crop")]
        sheet = np.hstack(parts)
        verdict = "no match" if err is None else (f"{err['position_m']:.1f} m / {err['yaw_deg']:.0f} deg  "
                                                  + ("GOOD" if err["position_m"] <= 5 else "MID" if err["position_m"] <= 10 else "MISS"))
        tag = f"id {e['frame_id']}  {mode}  RANSAC {verdict}  modes {m.n_modes}  inliers {m.inlier_ratio:.2f}"
        if prev_err is not None:
            tag += f"  (eval json: {prev_err:.1f} m)"
        rows.append(label(sheet, tag, h=34))
        print(tag, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"pose_{a.tag}_y{a.year}.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {path}")
    ortho.close()


if __name__ == "__main__":
    main()
