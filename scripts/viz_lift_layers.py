"""Layer sheet for the LSS track: ERP → depth → BEV → f_q → match → pose.

Writes experiments/05_lift_splat/viz/layers_<tag>.jpg (and per-frame tiles).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from bevloc import config as C
from bevloc.data.mapillary import PoznanOrtho, grid_bearing, load_frames, rodrigues
from bevloc.data.ortho import Oriented, gt_homography, sample_reference
from bevloc.eval.metrics import pose_errors
from bevloc.match.satroma import SatRoMa
from bevloc.model.coarse import FeatureQueryMatcher
from bevloc.model.lift_splat import SphericalLiftSplat

MAP_ROOT = C.REPO / "data/mapillary"
VAL_SEQ = MAP_ROOT / "Fixtor/IcRzj0wTLZX874qitxVsQa"
OUT = C.REPO / "experiments/05_lift_splat/viz"


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", default="checkpoints/05_lift_splat_fixtor_seq_best.pt")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--tag", default="seq")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    cfg = C.load(a.config)
    cfg.reference.max_offset_frac = 0.08
    cfg.reference.max_rot_deg = 8.0
    L = cfg.lift
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    ortho_paths = sorted(Path.home().glob(f"Github/sat_data/geoportal_poznan_15km2_*/year_{a.year}.tif"))
    ortho = PoznanOrtho(ortho_paths)
    frames = load_frames([VAL_SEQ], ortho, margin_m=L.margin_m)
    # spread across the held-out route
    idx = np.linspace(0, len(frames) - 1, a.n, dtype=int)
    frames = [frames[i] for i in idx]

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

    rows = []
    for fr in frames:
        row, pose_m = render_layers(cfg, L, fr, ortho, matcher, lift, ransac, dev)
        rows.append(row)
        print(f"  {fr['id']}  pose {pose_m}", flush=True)

    sheet = np.vstack(rows)
    path = OUT / f"layers_{a.tag}.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", path, sheet.shape, flush=True)
    ortho.close()


def render_layers(cfg, L, fr, ortho, matcher, lift, ransac, dev):
    path = Path(fr["_seq"]) / "images" / f"{fr['id']}.jpg"
    erp_full = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    R = rodrigues(fr["computed_rotation"]).astype(np.float32)
    lon, lat = fr["computed_geometry"]["coordinates"]
    up, en = grid_bearing(lon, lat, fr["computed_compass_angle"])

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
        depth_map, bev, valid, f_q, frac = lift_intermediates(lift, f_erp, R_t, erp_t.shape[-2:])
        f_s = matcher.reference_features(ref_t)
        with matcher.model.exposed_intermediates():
            out = matcher.model.decoder({16: f_q}, f_s, scale_factor=0.4)
        gm = out[16]["gm_cls"]
        cert = out[16].get("gm_certainty")
        mask = F.interpolate(frac[:, None].float(), size=(cfg.grid.n, cfg.grid.n), mode="nearest")[0, 0]
        mask_np = (mask >= L.min_patch_valid).cpu().numpy()
        m = ransac.match_encoded(f_q, f_s[16], scale_factor=0.4, mask=mask_np, H_gt=H_gt)

    pose_txt = "fail"
    pose_m = None
    if m.H is not None:
        e = pose_errors(m.H, H_gt, cfg.grid.n, cfg.grid.cell_m)
        pose_m = e["position_m"]
        pose_txt = f"{pose_m:.1f} m  yaw {e['yaw_deg']:.0f}"

    heat = soft_vote_heatmap(gm)  # (K, K)
    cert_img = certainty_img(cert)

    tiles = [
        ("1 ERP", fit(cv2.cvtColor(erp, cv2.COLOR_RGB2BGR), 280)),
        ("2 E[depth] m", fit(depth_to_bgr(depth_map), 280)),
        ("3 BEV mass", fit(mass_to_bgr(valid), 280)),
        ("4 BEV feat PCA", fit(pca_rgb(bev), 280)),
        ("5 f_q PCA", fit(pca_rgb(f_q), 280)),
        ("6 ortho", fit(cv2.cvtColor(ref, cv2.COLOR_RGB2BGR), 280)),
        ("7 certainty", fit(cert_img, 280)),
        ("8 vote heat", fit(heat_to_bgr(heat, ref.shape[:2]), 280)),
        ("9 pose", fit(pose_overlay(ref, H_gt, m.H, cfg.grid.n), 280)),
    ]

    row_imgs = []
    for title, img in tiles:
        bar = np.full((28, img.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        row_imgs.append(np.vstack([bar, img]))

    # equal height already; pad widths with gaps
    h = max(t.shape[0] for t in row_imgs)
    gap = np.full((h, 6, 3), 18, np.uint8)
    parts = []
    for i, t in enumerate(row_imgs):
        if t.shape[0] < h:
            pad = np.full((h - t.shape[0], t.shape[1], 3), 18, np.uint8)
            t = np.vstack([t, pad])
        parts.append(t)
        if i + 1 < len(row_imgs):
            parts.append(gap)
    panel = np.hstack(parts)

    header = np.full((36, panel.shape[1], 3), 20, np.uint8)
    cv2.putText(header, f"id {fr['id']}   RANSAC {pose_txt}   green=Mapillary pose approx  red=RANSAC",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, panel, np.full((10, panel.shape[1], 3), 18, np.uint8)]), pose_m


def lift_intermediates(lift, f_erp, R_w2c, erp_hw):
    """Same as SphericalLiftSplat.forward but also returns depth map + BEV."""
    B, _, h, w = f_erp.shape
    H, W = int(erp_hw[0]), int(erp_hw[1])
    alpha = lift.depth_head(f_erp).softmax(1)  # (B, D, h, w)
    depths = lift.depths.to(dtype=f_erp.dtype)
    expect = (alpha * depths[None, :, None, None]).sum(1)[0].cpu().numpy()  # (h, w)
    feat = lift.feat_proj(f_erp).reshape(B, -1, h * w)
    dir_cam, elev = lift.token_rays(h, w, H, W, f_erp.device, f_erp.dtype)
    x, y = lift.ego_xy(dir_cam, R_w2c, depths)
    elev_ok = elev <= lift.max_elev
    aflat = alpha.permute(0, 2, 3, 1).reshape(B, h * w, -1)
    bev, valid = lift.splat(feat, aflat, x, y, elev_ok[None].expand(B, -1))
    f_q = lift.bev_head(bev)
    frac = F.avg_pool2d(valid.float()[:, None], lift.patch)[:, 0]
    return expect, bev, valid[0].cpu().numpy(), f_q, frac


def soft_vote_heatmap(gm_cls):
    """Sum softmax mass over query patches → (K, K) ref cells."""
    B, K2, h, w = gm_cls.shape
    K = int(round(K2 ** 0.5))
    p = gm_cls.float().softmax(1).sum(dim=(0, 2, 3)).reshape(K, K).cpu().numpy()
    return p


def certainty_img(cert):
    if cert is None:
        return np.full((112, 112, 3), 40, np.uint8)
    c = cert[0, 0] if cert.ndim == 4 else cert[0]
    c = torch.sigmoid(c).cpu().numpy()
    u8 = (c * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)


def depth_to_bgr(depth_hw):
    d = depth_hw.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    u8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def mass_to_bgr(valid_hw):
    u8 = (np.clip(valid_hw.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_BONE)


def pca_rgb(feat_bchw):
    """First 3 PCA components of channels → RGB uint8 (H, W, 3)."""
    f = feat_bchw[0].detach().float().cpu().numpy()  # C,H,W
    C, H, W = f.shape
    X = f.reshape(C, -1).T  # HW, C
    X = X - X.mean(0, keepdims=True)
    # thin PCA via covariance on channels
    cov = (X.T @ X) / max(X.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    basis = vecs[:, -3:][:, ::-1]  # C,3
    Y = X @ basis  # HW,3
    for i in range(3):
        lo, hi = np.percentile(Y[:, i], [2, 98])
        Y[:, i] = np.clip((Y[:, i] - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = (Y.reshape(H, W, 3) * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def heat_to_bgr(heat_kk, ref_hw):
    h = heat_kk / (heat_kk.max() + 1e-8)
    u8 = (h * 255).astype(np.uint8)
    cm = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return cv2.resize(cm, (ref_hw[1], ref_hw[0]), interpolation=cv2.INTER_NEAREST)


def pose_overlay(ref_rgb, H_gt, H_est, n):
    canvas = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR).copy()
    box(canvas, H_gt, n, (0, 220, 0), 2)
    if H_est is not None:
        box(canvas, H_est, n, (0, 0, 255), 2)
    return canvas


def box(img, H, n, color, thick):
    corners = np.array([[0, 0, 1], [n - 1, 0, 1], [n - 1, n - 1, 1], [0, n - 1, 1]], float)
    p = corners @ np.asarray(H, float).T
    pts = np.ascontiguousarray(np.round(p[:, :2] / p[:, 2:3]).astype(np.int32)).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)


def fit(img, h):
    s = h / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
