"""Visual sheet: Lift-Splat RANSAC on the held-out Fixtor route.

Each row: flat-ground IPM | orthophoto with Mapillary-pose footprint (green,
approximate) and RANSAC estimate (red). Position error uses Mapillary lat/lon as
the GT proxy; the green rectangle is not a precise GT box.
Scores a probe set live, then shows 3 good / 1 mid / 2 miss.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C
from bevloc.bev.grid import BevGrid
from bevloc.data.mapillary import PoznanOrtho, grid_bearing, load_frames, rodrigues, poznan_tiles
from bevloc.data.ortho import Oriented, gt_homography, sample_reference
from bevloc.eval.metrics import pose_errors
from bevloc.match.satroma import SatRoMa
from bevloc.model.coarse import FeatureQueryMatcher
from bevloc.model.lift_splat import SphericalLiftSplat

_spec = importlib.util.spec_from_file_location("ipm_mapillary", C.REPO / "scripts/ipm_mapillary.py")
_ipm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ipm)
ipm = _ipm.ipm

MAP_ROOT = C.REPO / "data/mapillary"
VAL_SEQ = MAP_ROOT / "Fixtor/IcRzj0wTLZX874qitxVsQa"
OUT = C.REPO / "experiments/05_lift_splat/viz"


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", default="checkpoints/05_lift_splat_fixtor_multi_best.pt")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--tag", default="y2025")
    ap.add_argument("--probe", type=int, default=24)
    a = ap.parse_args()
    cfg = C.load(a.config)
    cfg.reference.max_offset_frac = 0.10
    cfg.reference.max_rot_deg = 10.0
    L = cfg.lift
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    ortho_paths = poznan_tiles(a.year)
    ortho = PoznanOrtho(ortho_paths)
    frames = load_frames([VAL_SEQ], ortho, margin_m=L.margin_m)
    stride = max(1, len(frames) // a.probe)
    frames = frames[::stride][: a.probe]

    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    lift = SphericalLiftSplat(
        dim=L.dim, depth_bins=L.depth_bins, d_min=L.d_min, d_max=L.d_max,
        n=cfg.grid.n, cell=cfg.grid.cell_m, max_elev_deg=L.max_elev_deg,
    ).to(dev).eval()
    lift.load_state_dict(state["lift"])
    ransac = SatRoMa.from_config(cfg, use_means=False, min_valid_frac=L.min_patch_valid)
    ransac.m.model.decoder.load_state_dict(state["decoder"], strict=False)

    scored = []
    for fr in frames:
        pose_m, yaw, panel = render_row(cfg, L, fr, ortho, matcher, lift, ransac, dev)
        if pose_m is None:
            continue
        scored.append((pose_m, yaw, fr["id"], panel))
        print(f"  {fr['id']}  {pose_m:.1f} m  yaw {yaw:.0f}", flush=True)
    scored.sort(key=lambda t: t[0])
    picks = []
    for i, tag in ((0, "good"), (1, "good"), (2, "good"),
                   (len(scored) // 2, "mid"), (-2, "miss"), (-1, "miss")):
        pose_m, yaw, fid, panel = scored[i]
        picks.append(label_panel(panel, tag, fid, pose_m, yaw))
        print(f"pick {tag} {fid} {pose_m:.1f} m", flush=True)

    sheet = np.vstack(picks)
    path = OUT / f"pose_{a.tag}.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", path, sheet.shape, flush=True)
    ortho.close()


def label_panel(panel, tag, fid, pose_m, yaw):
    bar = np.zeros((44, panel.shape[1], 3), np.uint8)
    cv2.putText(bar, f"{tag.upper()}  id {fid}  pose {pose_m:.1f} m  yaw {yaw:.0f} deg",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    legend = np.zeros((28, panel.shape[1], 3), np.uint8)
    cv2.putText(legend, "left: IPM h=1.7 m   right: ortho  green=Mapillary pose approx  red=RANSAC",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, panel, legend])


def render_row(cfg, L, fr, ortho, matcher, lift, ransac, dev):
    path = Path(fr["_seq"]) / "images" / f"{fr['id']}.jpg"
    erp_full = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    R = rodrigues(fr["computed_rotation"]).astype(np.float32)
    lon, lat = fr["computed_geometry"]["coordinates"]
    up, en = grid_bearing(lon, lat, fr["computed_compass_angle"])

    grid = BevGrid(n=560, cell=0.1)
    bev = ipm(erp_full, grid, R, 1.7)
    bev_bgr = cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)

    erp = cv2.resize(erp_full, tuple(L.erp_size), interpolation=cv2.INTER_AREA)
    query = Oriented(en, up, cfg.grid.n, cfg.grid.cell_m)
    rng = np.random.default_rng([cfg.matcher.seed, int(fr["id"]) % (2**32)])
    ref_o = sample_reference(query, rng, cfg.reference.scale,
                             cfg.reference.max_offset_frac, cfg.reference.max_rot_deg)
    ref, _ = ortho.render(ref_o)
    H_gt = gt_homography(query, ref_o)

    erp_t = torch.from_numpy(erp).permute(2, 0, 1).float().div(255.0)[None].to(dev)
    ref_t = torch.from_numpy(np.ascontiguousarray(ref)).permute(2, 0, 1).float().div(255.0)[None].to(dev)
    R_t = torch.from_numpy(R)[None].to(dev)
    with torch.no_grad():
        f_erp = matcher.model.encoder(erp_t)[16]
        f_q, frac = lift(f_erp, R_t, erp_hw=erp_t.shape[-2:])
        f_s = matcher.reference_features(ref_t)
        mask = torch.nn.functional.interpolate(
            frac[:, None].float(), size=(cfg.grid.n, cfg.grid.n), mode="nearest")[0, 0]
        mask = (mask >= L.min_patch_valid).cpu().numpy()
        m = ransac.match_encoded(f_q, f_s[16], scale_factor=0.4, mask=mask, H_gt=H_gt)

    canvas = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR).copy()
    box(canvas, H_gt, cfg.grid.n, (0, 220, 0), 2)
    if m.H is None:
        return None, None, None
    box(canvas, m.H, cfg.grid.n, (0, 0, 255), 2)
    e = pose_errors(m.H, H_gt, cfg.grid.n, cfg.grid.cell_m)
    h = 420
    panel = np.hstack([fit_h(bev_bgr, h), np.zeros((h, 12, 3), np.uint8), fit_h(canvas, h)])
    return e["position_m"], e["yaw_deg"], panel


def box(img, H, n, color, thick):
    corners = np.array([[0, 0, 1], [n - 1, 0, 1], [n - 1, n - 1, 1], [0, n - 1, 1]], float)
    p = corners @ np.asarray(H, float).T
    pts = np.ascontiguousarray(np.round(p[:, :2] / p[:, 2:3]).astype(np.int32)).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)


def fit_h(img, h):
    s = h / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
