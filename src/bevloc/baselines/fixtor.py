"""FG² and Loc² on the Fixtor × Poznań manifest: the adapters of the task-02 zero-shot rows, ported from the main
checkout's uncommitted `bevloc.baselines.{fg2,loc2}` Fixtor sections (2026-09-24) onto the VIGOR wrappers of this
branch (`bevloc.baselines.fg2`, `bevloc.baselines.loc2`: checkpoints, sys.path isolation, the mmcv shim). Third-party
code stays in third_party/ and is imported from there (FG² GPL-3.0, Loc² AGPL-3.0: wrapped, never copied).

Per entry (all conventions of the reported rows, unchanged):
  * panorama: Mapillary ERP (centre column = camera forward = the proxy heading, verified: Mapillary's
    computed_rotation forward equals computed_compass_angle to 1e-3 deg) resized to the method's 714 x 1428 and rolled
    so that the relative bearing rho = beta - assumed sits at the centre (`common.heading_setup`, `common.roll_shift`);
  * reference: the Poznań orthophoto rendered at the manifest's crop centre and crop_up bearing, 630 px over 71 m;
  * pose: Procrustes t (metres on the method's metric grid) -> map EN, exactly as the reported rows decoded it
    (FG²: t = (image-up, image-left); Loc²: t = (image-down, image-right); both audited on Fixtor in task 02);
  * heading: the Procrustes R gives the rotation of the ROLLED panorama's centre column relative to the crop's up;
    the vehicle heading is that minus rho (`common.vehicle_yaw`). The reported rows compared the rolled column's
    heading with the vehicle's, so their heading errors (~5.7 deg median) were |crop_rot| (the manifest noise), not
    the method's error; the positions are unaffected by this.

Loc² depth: "flat" = the 1.65 m flat-ground ray proxy of the reported row (UniK3D was not available locally), or
"unik3d" = UniK3D metric depth from scripts/unik3d_depth_poznan.py (`common.depth_png_for`), with the Fixtor car body
(camera-frame rows steeper than ego_mask_deg below the horizon) set to max_depth, which Loc² masks.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bevloc.baselines import fg2 as fg2_wrap
from bevloc.baselines import loc2 as loc2_wrap
from bevloc.baselines.common import roll_shift
from bevloc.data.ortho import Oriented

GRID_M = 71.0                   # FG² config.ini grid_size_h; Loc² uses the same 71 m aerial extent on VIGOR
LOC2_CAMERA_HEIGHT_M = 1.65     # the reported flat-ground proxy row
LOC2_DINO_DOWN = 14             # DINOv2 patch


def sat_gsd_m(size: int = 630) -> float:
    """Metres per pixel so a 630 px crop spans 71 m (both methods)."""
    return GRID_M / float(size)


def oriented(crop_centre_en, crop_up_bearing_deg, size: int = 630) -> Oriented:
    return Oriented(tuple(crop_centre_en), float(crop_up_bearing_deg), int(size), sat_gsd_m(size))


def load_erp(path, hw=(714, 1428)) -> torch.Tensor:
    """(3, H, W) float in [0, 1]: torchvision Resize (bilinear, antialias) + ToTensor, as the reported rows."""
    from torchvision import transforms
    t = transforms.Compose([transforms.Resize(list(hw)), transforms.ToTensor()])
    return t(Image.open(path).convert("RGB"))


def roll_erp(grd: torch.Tensor, rho_deg: float):
    """Roll (…, W) so the relative bearing rho_deg is at the centre. Returns (rolled, shift, rho_eff_deg)."""
    shift, rho_eff = roll_shift(rho_deg, grd.shape[-1])
    return (torch.roll(grd, shift, dims=-1) if shift else grd), shift, rho_eff


def render_sat(ortho, crop_centre_en, crop_up_bearing_deg, size: int = 630):
    """Poznań orthophoto at 71 m / 630 px (PoznanOrtho.render, as the reported rows). (3, S, S) tensor, Oriented,
    valid (S, S) bool."""
    o = oriented(crop_centre_en, crop_up_bearing_deg, size)
    rgb, valid = ortho.render(o)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0), o, valid


def decode_pose(t_m, R, crop_centre_en, crop_up_bearing_deg, method: str, yaw_sign: float):
    """(t, R) of either method -> map EN and the heading of the rolled panorama's centre column (deg cw from grid
    north). t axes: FG² (image-up, image-left), Loc² (image-down, image-right) — the task-02 audit. The heading is
    crop_up + yaw_sign * atan2(R[1,0], R[0,0]); yaw_sign is FG2_YAW_SIGN / LOC2_YAW_SIGN (module constants)."""
    t = np.asarray(t_m, float).reshape(-1)
    b = np.radians(float(crop_up_bearing_deg))
    up = np.array([np.sin(b), np.cos(b)])
    right = np.array([np.cos(b), -np.sin(b)])
    c = np.asarray(crop_centre_en, float)
    if method == "fg2":
        en = c + t[0] * up - t[1] * right
    elif method == "loc2":
        en = c - t[0] * up + t[1] * right
    else:
        raise ValueError(method)
    R = np.asarray(R, float)
    yaw_r = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return (float(en[0]), float(en[1])), float((float(crop_up_bearing_deg) + yaw_sign * yaw_r) % 360.0)


# The sign of the Procrustes rotation relative to a clockwise (bearing) rotation of the panorama's centre column.
# Settled on the Fixtor smoke (20 frame ids x 2025/2024) with heading "prior", where the rolled column looks
# crop_rot_deg anticlockwise of the crop's up (true virtual heading = crop_up - crop_rot): over the frames with a
# position error < 10 m, Loc²'s atan2(R10, R00) follows +crop_rot (Theil-Sen slope 1.16, Spearman 0.82, n 31) and
# FG²'s follows -crop_rot (slope -1.11, Spearman -0.85, n 29); the opposite signs give heading errors of 8.6 / 6.9 deg
# median instead of 1.7 / 2.7 deg. FG²'s sign is what the task-02 code assumed; Loc²'s was never exercised there (the
# panorama was aligned with the crop, so R was ~I). The two t-axis conventions differ in handedness, so do the Rs.
FG2_YAW_SIGN = 1.0
LOC2_YAW_SIGN = -1.0


# --------------------------------------------------------------------------------------------------------- FG²
def fg2_estimate(score, device, n_samples=None):
    """FG²'s weighted Procrustes on its matching scores (utils.utils of third_party/FG2): (R 2x2, t (2,)) or
    (None, None)."""
    fg2_wrap.ensure_fg2_on_path()
    prev = os.getcwd()
    try:
        os.chdir(fg2_wrap.FG2_ROOT)
        from utils.utils import create_metric_grid, weighted_procrustes_2d  # noqa: WPS433
    finally:
        os.chdir(prev)
    N = fg2_wrap.NATIVE
    B, _n_sat, n_grd = score.shape
    n = int(n_samples or N["num_samples_matches"])
    sat = create_metric_grid(N["grid_size_h"], N["sat_bev_res"], B).to(device)
    grd = create_metric_grid(N["grid_size_h"], N["grd_bev_res"], B).to(device)
    flat = score.flatten(1)
    bidx = torch.arange(B, device=device).view(B, 1).expand(B, n)
    s = torch.multinomial(flat, n)
    X = sat[bidx, torch.div(s, n_grd, rounding_mode="trunc"), :]
    Y = grd[bidx, s % n_grd, :]
    R, t, ok = weighted_procrustes_2d(X, Y, use_weights=True, use_mask=True, w=flat[bidx, s])
    if t is None or ok is False:
        return None, None
    return R[0].detach().cpu().numpy(), t[0, 0].detach().cpu().numpy()


@torch.no_grad()
def fg2_localize(model, dino, grd, sat, device, crop_centre_en, crop_up_bearing_deg):
    """One FG² pair: dict(ok, en, virtual_yaw_deg, t_m, R) or dict(ok=False, error)."""
    gf = dino(grd.unsqueeze(0).to(device))
    sf = dino(sat.unsqueeze(0).to(device))
    score, _score_orig, _hidx = model(gf, sf)
    R, t = fg2_estimate(score, device)
    if t is None:
        return {"ok": False, "error": "procrustes_failed"}
    en, vyaw = decode_pose(t, R, crop_centre_en, crop_up_bearing_deg, "fg2", FG2_YAW_SIGN)
    return {"ok": True, "en": en, "virtual_yaw_deg": vyaw, "t_m": [float(v) for v in np.ravel(t)[:2]],
            "R": np.asarray(R, float).tolist(), "yaw_r_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))}


# --------------------------------------------------------------------------------------------------------- Loc²
def loc2_flat_depth(h: int, w: int, camera_height_m: float = LOC2_CAMERA_HEIGHT_M, max_depth: float | None = None):
    """The reported row's flat-ground ray depth (Loc² spherical convention: theta from +z over [0, pi] by rows)
    (1, h, w): rays below the horizon hit the plane z = -h_cam, everything else gets max_depth (masked by Loc²)."""
    d_max = float(max_depth if max_depth is not None else loc2_wrap.NATIVE["max_depth_m"])
    theta = torch.linspace(0, np.pi, h)[:, None].expand(h, w)
    depth = torch.full((h, w), d_max, dtype=torch.float32)
    cos_t = torch.cos(theta)
    down = cos_t < -1e-4
    depth[down] = float(camera_height_m) / (-cos_t[down])
    depth.clamp_(0.2, d_max)
    depth[~down] = d_max
    return depth.unsqueeze(0)


def ego_rows(h: int, ego_mask_deg: float) -> np.ndarray:
    """(h,) bool: ERP rows whose centre looks more than ego_mask_deg below the camera's horizon (the car body of a
    roof-mounted Fixtor rig; cfg.ipm.max_depression_deg). 0 or None = no rows."""
    if not ego_mask_deg:
        return np.zeros(int(h), bool)
    lat = (0.5 - (np.arange(int(h)) + 0.5) / int(h)) * 180.0
    return lat < -float(ego_mask_deg)


def loc2_unik3d_depth(png_path, hw=(714, 1428), ego_mask_deg: float = 20.0, max_depth: float | None = None):
    """UniK3D depth PNG (uint16 mm along the ray) -> (1, H, W) metres at Loc²'s ground size, nearest resampling as
    Loc²'s dataloader, clipped at max_depth, car-body rows set to max_depth (masked)."""
    from bevloc.data.vigor import read_depth_png
    d_max = float(max_depth if max_depth is not None else loc2_wrap.NATIVE["max_depth_m"])
    d = torch.from_numpy(read_depth_png(png_path))[None, None]
    d = F.interpolate(d, size=tuple(hw), mode="nearest")[0]
    d = torch.clip(d, 0, d_max)
    d[:, torch.from_numpy(ego_rows(hw[0], ego_mask_deg))] = d_max
    return d


def loc2_spherical_grids(hw, device):
    """Loc² eval_vigor.py create_spherical_grids for one sample: theta, phi (1, 1, h/14, w/14)."""
    phi = torch.linspace(0, 2 * np.pi, int(hw[1] / LOC2_DINO_DOWN), device=device)
    theta = torch.linspace(0, np.pi, int(hw[0] / LOC2_DINO_DOWN), device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    return theta[None, None], phi[None, None]


def loc2_metric_grid(device):
    axis = torch.linspace(-GRID_M / 2, GRID_M / 2, int(loc2_wrap.NATIVE["sat_bev_res"]), device=device)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x.reshape(-1), y.reshape(-1)), -1)[None]


@torch.no_grad()
def loc2_localize(matcher, dino, grd, sat, depth, device, crop_centre_en, crop_up_bearing_deg, ransac=True):
    """One Loc² pair (the reported row's solver: RANSAC Procrustes with Loc²'s eval_vigor defaults, or plain
    weighted Procrustes): dict(ok, en, virtual_yaw_deg, t_m, R, scale) or dict(ok=False, error)."""
    def imp():
        from models.utils import e2eProbabilisticProcrustesSolver, weighted_procrustes_2d_with_scale  # noqa: WPS433
        return e2eProbabilisticProcrustesSolver, weighted_procrustes_2d_with_scale
    Solver, procrustes = loc2_wrap._import_from_loc2(imp)
    N = loc2_wrap.NATIVE
    d_max = float(N["max_depth_m"])
    gf = dino(grd.unsqueeze(0).to(device))
    sf = dino(sat.unsqueeze(0).to(device))
    d = torch.clip(depth.unsqueeze(0).to(device), 0, d_max)
    d_low = F.interpolate(d, size=gf.shape[-2:], mode="nearest")
    mask = ~(d_low == d_low.amax()).flatten(1)
    theta, phi = loc2_spherical_grids(N["ground_image_size"], device)
    gx = d_low * torch.sin(theta) * torch.cos(phi)
    gy = d_low * torch.sin(theta) * (-torch.sin(phi))
    grd_xy = torch.cat((gx.flatten(2), gy.flatten(2)), 1).permute(0, 2, 1)
    score, _ = matcher(gf, sf, mask)
    sat_grid = loc2_metric_grid(device)
    if ransac:
        R, t, scale, _, _ = Solver(100, 20, 8192, 3, 4, 2.5, 5.0, sat_grid, grd_xy).estimate_pose(
            score, return_inliers=False)
    else:
        n = int(N["num_samples_matches"])
        _, _, n_grd = score.shape
        flat = score.flatten(1)
        s = torch.multinomial(flat, n)
        bidx = torch.zeros_like(s)
        X = sat_grid[bidx, torch.div(s, n_grd, rounding_mode="trunc")]
        Y = grd_xy[bidx, s % n_grd]
        R, t, scale, _ = procrustes(Y, X, use_weights=True, use_mask=True, w=flat[bidx, s])
    if t is None:
        return {"ok": False, "error": "procrustes_failed"}
    if not (bool(torch.isfinite(t).all()) and bool(torch.isfinite(R).all())):
        return {"ok": False, "error": "singular_transform"}
    R_np = R[0].detach().cpu().numpy()
    t_np = t[0, 0].detach().cpu().numpy()
    en, vyaw = decode_pose(t_np, R_np, crop_centre_en, crop_up_bearing_deg, "loc2", LOC2_YAW_SIGN)
    return {"ok": True, "en": en, "virtual_yaw_deg": vyaw, "t_m": [float(t_np[0]), float(t_np[1])],
            "R": R_np.tolist(), "yaw_r_deg": float(np.degrees(np.arctan2(R_np[1, 0], R_np[0, 0]))),
            "scale": None if scale is None else float(scale.reshape(-1)[0].item()),
            "n_valid_tokens": int(mask.sum().item())}


def side_by_side(sat, grd, marks, label):
    """Sat | panorama strip (uint8 RGB) with (en->uv function, en, colour) markers drawn on the sat."""
    sat_rgb = (sat.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    grd_rgb = (grd.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    sh = sat_rgb.shape[0]
    grd_r = cv2.resize(grd_rgb, (int(grd_rgb.shape[1] * sh / grd_rgb.shape[0]), sh))
    for uv, colour, kind in marks:
        cv2.drawMarker(sat_rgb, tuple(int(round(v)) for v in uv), colour, kind, 18, 2)
    canvas = np.concatenate([sat_rgb, grd_r], axis=1)
    cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas
