"""Layer sheet for the hybrid query: ERP | IPM ground mask | BEV PCA | f_q PCA | reference + pose boxes.

  make hybrid-viz CKPT=checkpoints/05_lift_splat_fixtor_hybrid_best.pt TAG=hybrid
The ground panel must show road texture placed by IPM, not the radial dots of the depth splat.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C
from bevloc.data.mapillary import VAL_SEQS, MapillaryPairs, PoznanOrtho, load_frames, poznan_tiles
from bevloc.eval.metrics import pose_errors
from bevloc.match.satroma import SatRoMa
from bevloc.model.coarse import FeatureQueryMatcher
from bevloc.model.query import build_query, load_query_state
from bevloc.viz import fit, pca_rgb, pose_overlay

OUT = C.REPO / "experiments/05_lift_splat/viz"


def label(img, text):
    bar = np.full((28, img.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--tag", default="hybrid")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    cfg = C.load(a.config)
    L = cfg.lift
    cfg.reference.max_offset_frac, cfg.reference.max_rot_deg = L.max_offset_frac, L.max_rot_deg
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ortho = PoznanOrtho(poznan_tiles(a.year))
    frames = load_frames(VAL_SEQS, ortho, margin_m=L.margin_m)
    frames = frames[:: max(1, len(frames) // a.n)][: a.n]
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = state.get("mode", "hybrid")
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
    for i in range(len(ds)):
        s = ds[i]
        batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_erp = matcher.model.encoder(batch["erp"][:, 0])[16]
            f_proj = query.lift.feat_proj(f_erp)
            g, gmask = query.dense_ground(f_proj, batch["R_w2c"][:, 0], batch["erp"].shape[-2:])
            f_q, frac = query(batch, matcher)
            f_s = matcher.reference_features(batch["ref"])
        H_gt = s["H"].numpy()
        mask = torch.nn.functional.interpolate(frac[:, None].float(), size=(cfg.grid.n, cfg.grid.n),
                                               mode="nearest")[0, 0]
        m = ransac.match_encoded(f_q, f_s[16], scale_factor=0.4,
                                 mask=(mask >= L.min_patch_valid).cpu().numpy(), H_gt=H_gt)
        err = pose_errors(m.H, H_gt, cfg.grid.n, cfg.grid.cell_m) if m.H is not None else None
        erp = (s["erp"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ref = (s["ref"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ground = cv2.cvtColor((gmask[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        h = 280
        parts = [label(fit(cv2.cvtColor(erp, cv2.COLOR_RGB2BGR), h), "1 ERP"),
                 label(fit(ground, h), "2 IPM ground cells (white)"),
                 label(fit(pca_rgb(g), h), "3 dense ground feat PCA"),
                 label(fit(pca_rgb(f_q), h), "4 f_q PCA (14x14)"),
                 label(fit(pose_overlay(ref, H_gt, m.H, cfg.grid.n), h), "5 ref: green=proxy pose, red=RANSAC")]
        sheet = np.hstack(parts)
        pose_txt = "-" if err is None else f"{err['position_m']:.1f} m / {err['yaw_deg']:.0f} deg"
        txt = f"id {s['id']}  {mode}  RANSAC {pose_txt}"
        rows.append(label(sheet, txt))
        print(txt, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"layers_{a.tag}.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {path}")
    ortho.close()


if __name__ == "__main__":
    main()
