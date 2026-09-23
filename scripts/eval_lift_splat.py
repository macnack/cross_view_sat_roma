"""RANSAC pose eval for a trained spherical Lift-Splat checkpoint.

Argmax cell distance is not a pose. This scores Sat-RoMa's multi-hypothesis
RANSAC (use_means=False = published peak; True = distinct GMM means) on a
held-out Mapillary route against a Poznań orthophoto year.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C
from bevloc.data.mapillary import VAL_SEQS, MapillaryPairs, PoznanOrtho, load_frames, poznan_tiles
from bevloc.eval.metrics import pose_errors, recall
from bevloc.match.satroma import SatRoMa
from bevloc.model.coarse import FeatureQueryMatcher, coarse_targets, ref_cell_validity
from bevloc.model.lift_splat import SphericalLiftSplat

CELL_M = 4.0


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="experiments/05_lift_splat/fixtor_multi")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--n", type=int, default=48, help="val frames to score")
    ap.add_argument("--max-offset", type=float, default=0.10)
    ap.add_argument("--max-rot", type=float, default=10.0)
    a = ap.parse_args()
    cfg = C.load(a.config)
    cfg.reference.max_offset_frac = a.max_offset
    cfg.reference.max_rot_deg = a.max_rot
    L = cfg.lift
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ortho = PoznanOrtho(poznan_tiles(a.year))
    frames = load_frames(VAL_SEQS, ortho, margin_m=L.margin_m)
    stride = max(1, len(frames) // a.n)
    frames = frames[::stride][: a.n]
    ds = MapillaryPairs(frames, {a.year: ortho}, cfg, train=False, seed=cfg.train.seed,
                        erp_size=tuple(L.erp_size), years=[a.year])

    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    lift = SphericalLiftSplat(
        dim=L.dim, depth_bins=L.depth_bins, d_min=L.d_min, d_max=L.d_max,
        n=cfg.grid.n, cell=cfg.grid.cell_m, max_elev_deg=L.max_elev_deg,
    ).to(dev)
    lift.load_state_dict(state["lift"])
    lift.eval()

    # Two RANSAC modes reuse the same SatRoMa wrapper for estimate_homography.
    ransac_peak = SatRoMa.from_config(cfg, use_means=False, min_valid_frac=L.min_patch_valid)
    ransac_means = SatRoMa.from_config(cfg, use_means=True, min_valid_frac=L.min_patch_valid)
    # Point them at our fine-tuned decoder weights.
    ransac_peak.m.model.decoder.load_state_dict(state["decoder"], strict=False)
    ransac_means.m.model.decoder.load_state_dict(state["decoder"], strict=False)

    rows = []
    for i in range(len(ds)):
        sample = ds[i]
        erp = sample["erp"].to(dev)
        ref = sample["ref"][None].to(dev)
        R = sample["R_w2c"].to(dev)
        H = sample["H"].numpy()
        with torch.no_grad():
            if erp.ndim == 3:  # (3, H, W) — should not happen after multi-frame change
                erp = erp[None]
                R = R[None]
                f_erp = matcher.model.encoder(erp)[16]
                f_q, frac = lift(f_erp, R, erp_hw=erp.shape[-2:])
            elif erp.ndim == 4 and erp.shape[0] == 1:
                # (T=1, 3, H, W)
                f_erp = matcher.model.encoder(erp)[16]
                f_q, frac = lift(f_erp, R, erp_hw=erp.shape[-2:])
            else:
                # (T, 3, H, W)
                T = erp.shape[0]
                feats = [matcher.model.encoder(erp[t:t + 1])[16] for t in range(T)]
                f_erp = torch.stack(feats, 1)  # (1, T, C, h, w)
                se2 = sample["se2"][None].to(dev)
                f_q, frac = lift.forward_multiframe(
                    f_erp, R[None], se2, erp_hw=erp.shape[-2:])
            f_s = matcher.reference_features(ref)
            out = matcher.model.decoder({16: f_q}, f_s, scale_factor=0.4)
            gm = out[16]["gm_cls"][0]
        # argmax over supervised patches
        rv = ref_cell_validity(ref)
        idx, use = coarse_targets(sample["H"][None].to(dev), (frac >= L.min_patch_valid), ref_valid=rv)
        if use.any():
            logits = gm.permute(1, 2, 0)[use[0]]
            tgt = idx[0][use[0]]
            k = int(round(gm.shape[0] ** 0.5))
            am = logits.argmax(-1)
            arg_m = float(torch.hypot((am // k - tgt // k).float(),
                                      (am % k - tgt % k).float()).median() * CELL_M)
        else:
            arg_m = None
        mask = torch.nn.functional.interpolate(
            frac[:, None].float(), size=(cfg.grid.n, cfg.grid.n), mode="nearest")[0, 0]
        mask = (mask >= L.min_patch_valid).cpu().numpy()
        row = dict(id=sample["id"], argmax_m=arg_m)
        for tag, wrap in (("peak", ransac_peak), ("means", ransac_means)):
            m = wrap.match_encoded(f_q, f_s[16], scale_factor=0.4, mask=mask, H_gt=H)
            if m.H is None:
                pos = yaw = None
            else:
                e = pose_errors(m.H, H, cfg.grid.n, cfg.grid.cell_m)
                pos, yaw = e["position_m"], e["yaw_deg"]
            row[f"pose_{tag}_m"] = pos
            row[f"yaw_{tag}_deg"] = yaw
            row[f"modes_{tag}"] = m.n_modes
            row[f"multimodal_{tag}"] = m.n_multimodal
        rows.append(row)
        print(f"  {sample['id']}  arg {None if arg_m is None else round(arg_m,1)}  "
              f"peak {None if row['pose_peak_m'] is None else round(row['pose_peak_m'],1)}  "
              f"means {None if row['pose_means_m'] is None else round(row['pose_means_m'],1)}  "
              f"modes {row['modes_peak']}/{row['modes_means']}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"pose_eval_y{a.year}.json"
    path.write_text(json.dumps({"ckpt": a.ckpt, "year": a.year, "frames": rows}, indent=2))
    for tag in ("peak", "means"):
        pos = [r[f"pose_{tag}_m"] for r in rows]
        arg = [r["argmax_m"] for r in rows if r["argmax_m"] is not None]
        med = float(np.nanmedian([np.inf if p is None else p for p in pos]))
        print(f"EVAL {tag}  n {len(rows)}  {recall(pos)}  median_pose {med:.1f} m  "
              f"median_argmax {np.median(arg):.1f} m  "
              f"matched {sum(p is not None for p in pos)}/{len(pos)}", flush=True)
    print(f"wrote {path}", flush=True)
    ortho.close()


if __name__ == "__main__":
    main()
