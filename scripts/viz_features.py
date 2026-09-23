"""Visualize the actual feature maps in both branches -- what the network sees before the
Sat-RoMa decoder ever compares them -- as PCA-to-RGB false colour, in the same top-down layout
as the BEV and the reference:

  1. raw BEV grid   (224x224, LiDAR-painted DINOv3-ConvNeXt features, pre BevEncoder)
  2. query tokens    (14x14x1024, post BevEncoder -- this is what the decoder actually receives)
  3. reference tokens (56x56x1024, frozen DINOv3 ViT-L -- what the decoder compares it against)

Every panel has its OWN PCA: colours show structure within a panel and are not comparable
across panels (a joint query+reference PCA is dominated by the offset between the two token
clouds and paints the query one flat colour -- an artefact, see docs/decisions.md 2026-09-20).

  python scripts/viz_features.py --checkpoint checkpoints/02_fusion_real_nlp2021_run3_2000steps.pt \
      --ortho data/ortho/durham/2021_nlp/durham_2021_nlp_intensity_1m_uint8.tif --frames 200
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C, viz
from bevloc.bev.variants import build_variants
from bevloc.data.calib import Calib
from bevloc.data.dur360 import Dur360Frames
from bevloc.data.ortho import Oriented, OrthoMap, gt_homography, sample_reference
from bevloc.data.oxts import read_track
from bevloc.model.coarse import FeatureQueryMatcher
from bevloc.model.fusion_bev import FusionBEV
from bevloc.run import frame_names


def load(ckpt_path, cfg, calib, dev):
    ck = torch.load(ckpt_path, map_location=dev)
    bev = FusionBEV.from_config(cfg, calib).to(dev).eval()
    bev.load_state_dict(ck["bev"])
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=True).eval()
    matcher.model.decoder.load_state_dict(ck["decoder"])
    print(f"loaded {ckpt_path} (trained to step {ck.get('step', '?')})")
    return bev, matcher


def pca_rgb(*grids, clip_pct=2.0):
    """PCA(3) fit jointly on all given (h, w, C) feature grids (channel count must match across
    them), returns one (h, w, 3) uint8 image per input, on ONE shared colour scale. Robust to
    outliers via a percentile clip before rescaling to [0, 255]."""
    flat = [g.reshape(-1, g.shape[-1]) for g in grids]
    X = np.concatenate(flat, 0).astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:3].T                                   # (N, 3)
    lo, hi = np.percentile(P, clip_pct, axis=0), np.percentile(P, 100 - clip_pct, axis=0)
    P = np.clip((P - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    out, i = [], 0
    for g in grids:
        n = g.shape[0] * g.shape[1]
        out.append((P[i:i + n] * 255).astype(np.uint8).reshape(g.shape[0], g.shape[1], 3))
        i += n
    return out


def label(img, text, h=32):
    head = np.zeros((h, img.shape[1], 3), np.uint8)
    cv2.putText(head, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, img])


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ortho", required=True)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/02_fusion/features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=448, help="output panel size in px")
    a = ap.parse_args()
    cfg = C.load(a.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ds, calib = Dur360Frames.from_config(cfg), Calib.from_config(cfg)
    bev, matcher = load(a.checkpoint, cfg, calib, dev)
    erp_valid = torch.from_numpy(np.load(C.REPO / cfg.erp.valid_mask)).float()[None, None].to(dev)
    ortho_map = OrthoMap(a.ortho)
    tr = read_track(ds.root)
    rng = np.random.default_rng(a.seed)

    for n in frame_names(ds, a.frames):
        i = int(n)
        q = Oriented(tuple(tr.en[i]), float(tr.bearing(cfg.oxts.convention, grid=True)[i]), cfg.grid.n, cfg.grid.cell_m)
        r = sample_reference(q, rng, cfg.reference.scale, cfg.reference.max_offset_frac, cfg.reference.max_rot_deg)
        ref_img, ref_ok = ortho_map.render(r)
        if not ref_ok.any() or (ref_img.sum(-1) > 0).mean() < 0.5:
            print(f"{n}: skipped, reference mostly black at this pose"); continue

        erp0 = ds.erp(n)
        scan = ds.scan(n)
        keep = scan.range_m > 2.0
        erp_t = torch.from_numpy(erp0).permute(2, 0, 1).float().to(dev) / 255.0
        pts_t = torch.from_numpy(scan.xyz[keep]).float().to(dev)
        refl_t = torch.from_numpy(scan.reflectivity[keep]).float().to(dev)
        ref_t = torch.from_numpy(ref_img).permute(2, 0, 1).float().to(dev) / 255.0

        with torch.no_grad():
            raw_bev, raw_mask = bev.lift(bev.backbone(erp_t[None]), erp_valid, [pts_t], [refl_t])
            f_q = bev.encoder(raw_bev, raw_mask)
            ref_feat = matcher.reference_features(ref_t[None])[16]

        raw_np = raw_bev[0].permute(1, 2, 0).cpu().numpy()          # (224, 224, C)
        q_np = f_q[0].permute(1, 2, 0).cpu().numpy()                # (14, 14, 1024)
        r_np = ref_feat[0].permute(1, 2, 0).cpu().numpy()           # (56, 56, 1024)

        (raw_rgb,) = pca_rgb(raw_np)
        # SEPARATE PCAs: the two token clouds sit ~5 reference-spreads apart, so a joint PCA spends its
        # axes on that offset and the percentile clip paints every query token one colour (a viz artefact
        # that was misread as 'flat tokens' -- docs/decisions.md 2026-09-20).
        (q_rgb,), (r_rgb,) = pca_rgb(q_np), pca_rgb(r_np)

        S = a.size
        b = build_variants(erp0, ds.points(n), np.load(C.REPO / cfg.erp.valid_mask), calib, cfg, name=n,
                           variants=("ipm_cl",))
        bev_rgb_img = cv2.resize(viz.enhance(b.images["ipm_cl"]), (S, S), interpolation=cv2.INTER_NEAREST)
        raw_feat_img = cv2.resize(raw_rgb, (S, S), interpolation=cv2.INTER_NEAREST)
        q_feat_img = cv2.resize(q_rgb, (S, S), interpolation=cv2.INTER_NEAREST)
        ref_img_disp = cv2.resize(ref_img, (S, S), interpolation=cv2.INTER_AREA)
        r_feat_img = cv2.resize(r_rgb, (S, S), interpolation=cv2.INTER_NEAREST)

        row1 = np.hstack([label(bev_rgb_img, "1. IPM-RGB BEV (context only)"),
                          label(raw_feat_img, "2. raw BEV grid features, PCA(3) (own scale)"),
                          label(q_feat_img, "3. query tokens f_q, 14x14 (own PCA)")])
        row2 = np.hstack([label(cv2.cvtColor(ref_img_disp, cv2.COLOR_RGB2BGR), "4. reference (NLP intensity)"),
                          label(r_feat_img, "5. reference tokens, 56x56 (own PCA)"),
                          label(np.zeros((S, S, 3), np.uint8), "")])
        body = np.vstack([row1, row2])
        head = np.zeros((60, body.shape[1], 3), np.uint8)
        cv2.putText(head, f"Feature visualization, frame {n}, checkpoint {Path(a.checkpoint).name}",
                   (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(head, "each feature panel has its OWN PCA: colours show structure within a panel, they are not comparable across panels",
                   (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{n}_features.jpg"), np.vstack([head, body]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{n}: wrote {out / f'{n}_features.jpg'}")


if __name__ == "__main__":
    main()
