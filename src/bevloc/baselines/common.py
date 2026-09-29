"""Shared Fixtor×Poznań protocol helpers for FG² / BevSplat / LSS comparison.

Coordinate conventions (match ``bevloc.data.mapillary``):
  - Map frame: Polish CS92 (EPSG:2180), metres east / north.
  - Ego frame: x forward, y left, gravity-aligned.
  - Heading: grid up-bearing in degrees clockwise from grid north
    (``computed_compass_angle`` after meridian convergence via ``grid_bearing``).
  - Predicted pose is always reported as map-frame ``(e, n, yaw_deg)`` where
    ``yaw_deg`` is the same up-bearing convention.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bevloc.data.mapillary import grid_bearing
from bevloc.data.ortho import Oriented, gt_homography, sample_reference


@dataclass(frozen=True)
class ManifestFrame:
    """One immutable held-out evaluation sample."""

    frame_id: str
    seq: str
    panorama: str
    lon: float
    lat: float
    en: tuple[float, float]
    up_bearing_deg: float
    compass_angle: float
    year: int
    crop_centre_en: tuple[float, float]
    crop_up_bearing_deg: float
    crop_offset_m: tuple[float, float]  # (right, up) in the crop's own axes
    crop_rot_deg: float
    query_size: int
    ref_size: int
    gsd_m: float
    H_gt: list  # 3x3 query-px → ref-px


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def se2_ego_to_map(en: tuple[float, float], up_bearing_deg: float,
                   dx: float, dy: float, dyaw_deg: float):
    """Apply an ego-frame SE(2) offset and return map-frame pose.

    ``(dx, dy)`` is the vehicle displacement in ego metres (x forward, y left).
    ``dyaw_deg`` is added to the up-bearing (CW from grid north).
    """
    b = np.radians(float(up_bearing_deg))
    fwd = np.array([np.sin(b), np.cos(b)])   # (E, N)
    left = np.array([-np.cos(b), np.sin(b)])
    en2 = np.asarray(en, float) + dx * fwd + dy * left
    yaw2 = (float(up_bearing_deg) + float(dyaw_deg)) % 360.0
    return (float(en2[0]), float(en2[1])), float(yaw2)


HEADINGS = ("prior", "gt")


def wrap180(a):
    """Angle(s) in degrees wrapped to [-180, 180)."""
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def heading_setup(entry: dict, heading: str = "prior", ref_up: str = "crop"):
    """How a panorama method sees one manifest entry's orientation (the Poznań three-way protocol).

    Returns (beta, assumed, rho) in degrees:
      beta     bearing of the reference's image-up: the entry's crop_up_bearing_deg ("crop", what FG² / Loc² /
               the IPM row get) or 0 ("north", VIGOR-like north-up reference);
      assumed  the vehicle heading the method is told: crop_up_bearing_deg under heading "prior" (the manifest's
               noisy prior, U(-10, 10) deg off the proxy heading), up_bearing_deg under "gt" (the proxy heading
               itself, which is what the reported FG² / Loc² "native" rows used: their panorama was rolled by
               crop_rot_deg, so its centre looked exactly along the crop's up, i.e. orientation was known);
      rho      the relative bearing (clockwise from the panorama's centre column, = vehicle forward) that is rolled
               to the image centre so that the believed reference-up direction faces forward: beta - assumed.
    The true residual rotation the matcher must recover is up_bearing_deg - assumed: -crop_rot_deg under
    "prior", 0 under "gt"."""
    if heading not in HEADINGS:
        raise ValueError(f"heading must be one of {HEADINGS}, got {heading!r}")
    if ref_up not in ("crop", "north"):
        raise ValueError(f"ref_up must be 'crop' or 'north', got {ref_up!r}")
    crop_up = float(entry["crop_up_bearing_deg"])
    beta = crop_up if ref_up == "crop" else 0.0
    assumed = crop_up if heading == "prior" else float(entry["up_bearing_deg"])
    return beta, assumed, float(wrap180(beta - assumed))


def roll_shift(rho_deg: float, width: int):
    """Integer column shift (np.roll / torch.roll along the width) that brings the relative bearing rho_deg
    (clockwise from the centre column) to the centre of a `width` px panorama, and the bearing it actually
    brings there (rho quantised to whole pixels): (shift, rho_eff_deg). new[u] = old[u - shift]."""
    shift = int(round(-float(rho_deg) / 360.0 * int(width)))
    return shift, float(wrap180(-shift * 360.0 / int(width)))


def vehicle_yaw(virtual_yaw_deg: float, rho_deg: float) -> float:
    """Heading of the vehicle (its panorama's centre column) from the estimated heading of the rolled panorama's
    centre column: the roll moved the column at relative bearing rho to the centre."""
    return float((float(virtual_yaw_deg) - float(rho_deg)) % 360.0)


def se2_map_error(pred_en, pred_yaw_deg, gt_en, gt_yaw_deg):
    """Position (m) and absolute yaw (°) errors in the map frame."""
    pe = float(np.linalg.norm(np.asarray(pred_en, float) - np.asarray(gt_en, float)))
    dy = (float(pred_yaw_deg) - float(gt_yaw_deg) + 180.0) % 360.0 - 180.0
    return pe, float(abs(dy))


def pose_from_homography(H, query_size: int, ref: Oriented):
    """Recover map-frame vehicle pose from a query→reference pixel homography.

    The query centre is the vehicle. Image-up (−v) is ego forward / the
    Oriented ``up_bearing``; a unit step toward image-up maps to world and
    yields that bearing via ``atan2(dE, dN)``.
    """
    H = np.asarray(H, float)
    s = query_size - 1
    c = np.array([s / 2.0, s / 2.0, 1.0])
    up = np.array([s / 2.0, s / 2.0 - 1.0, 1.0])  # −v = image-up = ego forward
    pc = H @ c
    pu = H @ up
    pc = pc[:2] / pc[2]
    pu = pu[:2] / pu[2]
    A = ref.px_to_world
    wc = A @ np.array([pc[0], pc[1], 1.0])
    wu = A @ np.array([pu[0], pu[1], 1.0])
    d = wu[:2] - wc[:2]
    yaw = float(np.degrees(np.arctan2(d[0], d[1])) % 360.0)
    return (float(wc[0]), float(wc[1])), yaw


def inject_and_recover(en, up_bearing_deg, dx=5.0, dy=-3.0, dyaw_deg=7.0,
                       query_size=224, gsd=0.25, scale=4, tol_m=1e-3, tol_deg=1e-2):
    """Unit-test helper: GT at ``en``, inject SE(2), recover via homography.

    Returns recovered ``(en, yaw)`` and the numerical residuals.
    """
    query = Oriented(tuple(en), float(up_bearing_deg), query_size, gsd)
    # Build a reference whose crop is offset so the GT pose is NOT at the crop
    # centre — then place the predicted pose via the ego offset.
    rng = np.random.default_rng(0)
    ref = sample_reference(query, rng, scale=scale, max_offset_frac=0.10, max_rot_deg=10.0)
    H_gt = gt_homography(query, ref)
    pred_en, pred_yaw = se2_ego_to_map(en, up_bearing_deg, dx, dy, dyaw_deg)
    pred_q = Oriented(pred_en, pred_yaw, query_size, gsd)
    H_pred = gt_homography(pred_q, ref)
    rec_en, rec_yaw = pose_from_homography(H_pred, query_size, ref)
    pe, ye = se2_map_error(rec_en, rec_yaw, pred_en, pred_yaw)
    # Also verify GT identity recovery
    gt_rec_en, gt_rec_yaw = pose_from_homography(H_gt, query_size, ref)
    gt_pe, gt_ye = se2_map_error(gt_rec_en, gt_rec_yaw, en, up_bearing_deg)
    return {
        "injected_en": pred_en,
        "injected_yaw": pred_yaw,
        "recovered_en": rec_en,
        "recovered_yaw": rec_yaw,
        "err_m": pe,
        "err_deg": ye,
        "gt_err_m": gt_pe,
        "gt_err_deg": gt_ye,
        "ok": pe <= tol_m and ye <= tol_deg and gt_pe <= tol_m and gt_ye <= tol_deg,
    }


def write_manifest(frames: list[ManifestFrame], path: Path, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "frames": [asdict(f) for f in frames],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def resolve_panorama(path) -> Path:
    """A manifest's panorama path on this machine: relative paths are taken from the repo root (the committed
    manifests), absolute ones as they are, and an absolute path from another checkout (the first manifest stored
    /home/.../data/mapillary/...) falls back to this repo's data/mapillary/<the same tail>."""
    from bevloc import config as C
    p = Path(path)
    if not p.is_absolute():
        return C.REPO / p
    if p.is_file():
        return p
    parts = p.parts
    for k in range(len(parts) - 1):
        if parts[k] == "data" and parts[k + 1] == "mapillary":
            return C.REPO.joinpath(*parts[k:])
    return p


def depth_png_for(panorama, root=None) -> Path:
    """Loc²'s depth layout transposed to a Mapillary sequence: <seq>/images/<id>.jpg ->
    <seq>/unik3d_depth/<id>.png (uint16 millimetres along the ray, like <City>/unik3d_depth/ for VIGOR).
    root (default $POZNAN_DEPTH_DIR): put the tree <seq name>/unik3d_depth/ under that directory instead (a
    checkout whose data/mapillary is a read-only link)."""
    import os
    p = Path(panorama)
    root = root or os.environ.get("POZNAN_DEPTH_DIR")
    base = Path(root) / p.parent.parent.name if root else p.parent.parent
    return base / "unik3d_depth" / (p.stem + ".png")


def select_entries(man: dict, years=None, n_frames: int = 0):
    """Manifest entries of `years` (None = all), restricted to the first `n_frames` unique frame ids in manifest
    order (0 = all) with every year of each kept id: the smoke subset of scripts/eval_{fg2,loc2}.py."""
    ys = None if not years else {int(y) for y in years}
    rows = [f for f in man["frames"] if ys is None or int(f["year"]) in ys]
    if not n_frames:
        return rows
    keep, order = set(), []
    for f in rows:
        if f["frame_id"] not in keep:
            if len(keep) >= n_frames:
                break
            keep.add(f["frame_id"])
            order.append(f["frame_id"])
    return [f for f in rows if f["frame_id"] in keep]


def frame_pose_proxy(fr: dict) -> tuple[tuple[float, float], float]:
    """Mapillary pose proxy: computed_geometry + compass → CS92 EN + up-bearing."""
    lon, lat = fr["computed_geometry"]["coordinates"]
    up, en = grid_bearing(lon, lat, fr["computed_compass_angle"])
    return en, up
