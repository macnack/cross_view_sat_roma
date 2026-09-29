"""Fixtor × Poznań manifest entries as VIGOR-style samples for the panorama-token matcher PanoRoMa (erp_depth query,
Sat-RoMa decoder; Poznań three-way comparison, decision 2026-09-29).

`ManifestPanoPairs` yields the dict `bevloc.data.vigor.VigorPairs` yields (id, city, erp, R_w2c, H, en, ref,
ref_centre_en, depth, ...), so scripts/eval_vigor.py's `score` / `DecoderFine` / `fine_pass_row` run on it unchanged.

Frames (all metres, bearings in degrees clockwise from CS92 grid north):
  * map: CS92 (EPSG:2180) east / north, the manifest's frame;
  * local ("tile frame" of VigorPairs): origin at the manifest's crop centre (the prior position every method gets),
    axes rotated so that local north = the reference's up bearing beta (`common.heading_setup`: 0 for ref_up
    "north", crop_up_bearing_deg for "crop"); `map_to_local` / `local_to_map`. `canvas_to_en` / `pose_en` of
    bevloc.data.vigor hold in this frame because the reference canvas is rendered with image-up = beta;
  * ego / BEV: x forward, y left, forward = the horizontal projection of the camera's z axis under R_w2c. The
    panorama is Mapillary's (centre column = camera forward; computed_rotation's forward equals
    computed_compass_angle), rolled by rho = beta - assumed (the relative bearing the method believes faces the
    reference's up is brought to the centre); R_w2c is Mapillary's computed_rotation composed with that roll
    (`rolled_R_w2c`), so metric depth placement (bevloc.model.depth_query) stays exact under the ~1 deg tilt.

Ground truth: the query footprint Oriented(en, q_up) with q_up = the proxy heading + rho (the rolled centre column's
heading) mapped into the reference canvas Oriented(local_to_map(centre), beta) — the manifest's own GT construction
(`bevloc.data.ortho.gt_homography`). The rotation left in H is up_bearing_deg - assumed: -crop_rot_deg under heading
"prior" (the matcher's se2 solver must recover it), 0 under "gt" (VIGOR's known-orientation geometry).

Reference: the orthophoto of the entry's year rendered onto the canvas (224 * cfg.reference.scale px at
cfg.grid.cell_m m/px) centred at the prior (coarse pass) or at the coarse pose (the fine pass's window). Resampling as
VIGOR's reference (VigorPairs.window_reference): area-average first when the canvas GSD is coarser than the source
(Poznań 2024: 0.05 m/px), then one bilinear resampling; a finer canvas (2025: 0.25 m/px source) is bilinear. Canvas px
outside cfg.vigor.ref_window_m (fine config: 56 m = the canvas) or outside `extent_m` (coarse option: e.g. 71 m, a
VIGOR-tile-sized square) are black, like the off-tile border of VIGOR's canvas.

Panorama: resampled (INTER_AREA) to base_size = 2048 x 1024 (VIGOR's panorama size), rolled there by whole pixels
(rho quantised to 360 / 2048 deg; the exact rho is used everywhere), then resized to the erp_depth ERP size as
VigorPairs does. Depth: the UniK3D PNG (<seq>/unik3d_depth/<id>.png, scripts/unik3d_depth_poznan.py, stored at
2048 x 1024) rolled by the same shift, nearest-resized to the ERP size; rows looking more than ego_mask_deg below the
camera's horizon (the car body) get depth 0 = invalid (depth_query drops non-positive depth).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from bevloc.baselines.common import (depth_png_for, heading_setup, pose_from_homography, resolve_panorama,
                                     roll_shift, wrap180)
from bevloc.data.mapillary import rodrigues
from bevloc.data.ortho import Oriented, gt_homography


def _axes(beta_deg):
    b = np.radians(float(beta_deg))
    return np.array([np.cos(b), -np.sin(b)]), np.array([np.sin(b), np.cos(b)])     # right, up (E, N)


def map_to_local(en, origin_en, beta_deg):
    """Map EN (..., 2) -> local (x = along the reference's right, y = along its up) metres from origin_en."""
    right, up = _axes(beta_deg)
    d = np.asarray(en, np.float64) - np.asarray(origin_en, np.float64)
    return np.stack([d @ right, d @ up], -1)


def local_to_map(xy, origin_en, beta_deg):
    """Inverse of `map_to_local`."""
    right, up = _axes(beta_deg)
    xy = np.asarray(xy, np.float64)
    return np.asarray(origin_en, np.float64) + xy[..., :1] * right + xy[..., 1:2] * up


def roll_matrix(rho_deg):
    """M with d_cam(old) = M @ d_cam(new) for a panorama rolled so that the old relative bearing rho is at the new
    centre (camera x right, y down, z forward; azimuth measured from z towards x, i.e. clockwise seen from above)."""
    r = np.radians(float(rho_deg))
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rolled_R_w2c(R_w2c, rho_deg):
    """World -> camera rotation of the rolled panorama: R' = M(rho)^T R."""
    return roll_matrix(rho_deg).T @ np.asarray(R_w2c, np.float64)


def forward_bearing(R_w2c):
    """Bearing (deg clockwise from the world's +y = north) of the camera's forward axis projected on the ground;
    the world frame of R_w2c is ENU (Mapillary: topocentric, true north)."""
    f = np.asarray(R_w2c, np.float64).T[:, 2]
    return float(np.degrees(np.arctan2(f[0], f[1])) % 360.0)


def canvas_oriented(origin_en, beta_deg, centre_local, size, cell_m) -> Oriented:
    """The reference canvas: `size` px at `cell_m`, image-up = beta, centre pixel ((size - 1) / 2) at the local
    point centre_local."""
    c = local_to_map(np.asarray(centre_local, np.float64).reshape(2), origin_en, beta_deg).reshape(2)
    return Oriented((float(c[0]), float(c[1])), float(beta_deg), int(size), float(cell_m))


def ego_rows_mask(h, ego_mask_deg):
    """(h,) bool: ERP rows (centre) more than ego_mask_deg below the camera horizon."""
    if not ego_mask_deg:
        return np.zeros(int(h), bool)
    lat = (0.5 - (np.arange(int(h)) + 0.5) / int(h)) * 180.0
    return lat < -float(ego_mask_deg)


def render_area(ortho, o: Oriented):
    """Orthophoto on the oriented canvas `o` with VIGOR's reference resampling: area-average (cv2.INTER_AREA) of the
    source window to the canvas GSD when the source is finer by more than 1 %, then one bilinear resampling. Returns
    (rgb uint8 (S, S, 3), valid bool (S, S)); off-tile px are black / invalid. `ortho`: bevloc.data.mapillary.PoznanOrtho."""
    t = ortho.tile_for(o.centre_en)
    if t is None:
        return np.zeros((o.size, o.size, 3), np.uint8), np.zeros((o.size, o.size), bool)
    A = o.px_to_world
    uu, vv = np.meshgrid(np.arange(o.size, dtype=np.float64), np.arange(o.size, dtype=np.float64))
    E = A[0, 0] * uu + A[0, 1] * vv + A[0, 2]
    N = A[1, 0] * uu + A[1, 1] * vv + A[1, 2]
    col, row = ~t.transform * (E, N)
    col, row = col - 0.5, row - 0.5                              # pixel-centre indexing
    inside = (col >= 0) & (col <= t.width - 1) & (row >= 0) & (row <= t.height - 1)
    c0, r0 = int(np.floor(col.min())) - 2, int(np.floor(row.min())) - 2
    c1, r1 = int(np.ceil(col.max())) + 3, int(np.ceil(row.max())) + 3
    src = np.moveaxis(t.read(indexes=[1, 2, 3], window=Window(c0, r0, c1 - c0, r1 - r0), boundless=True,
                             fill_value=0), 0, -1)
    x, y = col - c0, row - r0
    k = float(o.gsd) / abs(float(t.transform.a))                 # source px per canvas px
    if k > 1.01:
        h0, w0 = src.shape[:2]
        nw, nh = max(1, int(round(w0 / k))), max(1, int(round(h0 / k)))
        src = cv2.resize(np.ascontiguousarray(src), (nw, nh), interpolation=cv2.INTER_AREA)
        sx, sy = nw / w0, nh / h0
        x, y = (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5
    img = cv2.remap(np.ascontiguousarray(src), x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    img[~inside] = 0
    return img, inside


class ManifestPanoPairs(Dataset):
    """Manifest entries -> VigorPairs-style samples (module doc). `labels` = the entries (the evaluators' name)."""

    def __init__(self, entries, orthos, cfg, heading="prior", ref_up="north", extent_m=None, ego_mask_deg=20.0,
                 base_size=(2048, 1024), depth=True):
        self.labels = list(entries)
        self.orthos = orthos
        self.cfg = cfg
        self.heading, self.ref_up = heading, ref_up
        self.extent_m = None if extent_m in (None, 0) else float(extent_m)
        self.ego_mask_deg = float(ego_mask_deg or 0.0)
        self.base_w, self.base_h = (int(v) for v in base_size)
        E = getattr(cfg, "erp_depth", None)
        self.erp_w, self.erp_h = (int(v) for v in (E.erp_size if E is not None else (896, 448)))
        self.depth = bool(depth)
        V = getattr(cfg, "vigor", None)
        w = getattr(V, "ref_window_m", None) if V is not None else None
        self.ref_window_m = None if w is None else float(w)
        self.row_sign, self.height = 1.0, float(getattr(getattr(cfg, "ipm", None), "height_m", 1.7))
        self._meta_cache, self._pano_cache = {}, (None, None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.item(i)

    # --- helpers -------------------------------------------------------------------------------------------------
    def pano_path(self, i) -> Path:
        return resolve_panorama(self.labels[i]["panorama"])

    def depth_path(self, i) -> Path:
        return depth_png_for(self.pano_path(i))

    def keep_with_depth(self):
        """Drop entries without a depth PNG; returns the number dropped."""
        before = len(self.labels)
        self.labels = [e for e in self.labels if depth_png_for(resolve_panorama(e["panorama"])).is_file()]
        return before - len(self.labels)

    def frame_meta(self, i) -> dict:
        p = self.pano_path(i)
        seq = p.parent.parent
        if seq not in self._meta_cache:
            self._meta_cache[seq] = {str(f["id"]): f for f in json.loads((seq / "images.json").read_text())}
        return self._meta_cache[seq][p.stem]

    def centre_guess_m(self, i):
        return float(np.hypot(*self.labels[i]["crop_offset_m"]))

    def geometry(self, i):
        """(beta, assumed, rho_requested, shift, rho_eff, R_w2c rolled, q_up) of entry i."""
        e = self.labels[i]
        beta, assumed, rho = heading_setup(e, self.heading, self.ref_up)
        shift, rho_eff = roll_shift(rho, self.base_w)
        R = rodrigues(self.frame_meta(i)["computed_rotation"])
        gamma = float(e["up_bearing_deg"]) - forward_bearing(R)                    # grid convergence (+ any offset)
        R_r = rolled_R_w2c(R, rho_eff)
        q_up = float((forward_bearing(R_r) + gamma) % 360.0)
        return beta, assumed, rho, shift, rho_eff, R_r, q_up

    def _pano(self, i, shift):
        """Base-size RGB panorama rolled by `shift` and the matching depth (or None), cached for the last entry."""
        key = (i, shift)
        if self._pano_cache[0] == key:
            return self._pano_cache[1]
        img = cv2.imread(str(self.pano_path(i)), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"unreadable panorama {self.pano_path(i)}")
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (self.base_w, self.base_h), interpolation=cv2.INTER_AREA)
        img = np.roll(img, shift, axis=1)
        d = None
        if self.depth and self.depth_path(i).is_file():
            from bevloc.data.vigor import read_depth_png
            d = read_depth_png(self.depth_path(i))
            if d.shape != (self.base_h, self.base_w):
                d = cv2.resize(d, (self.base_w, self.base_h), interpolation=cv2.INTER_NEAREST)
            d = np.roll(d, shift, axis=1)
        self._pano_cache = (key, (img, d))
        return img, d

    # --- the sample ----------------------------------------------------------------------------------------------
    def item(self, i, ref_centre_en=None):
        """Entry i with the reference centred at ref_centre_en (local metres; None = the prior, the origin)."""
        e = self.labels[i]
        g = self.cfg.grid
        n, cell = int(g.n), float(g.cell_m)
        size = int(n * self.cfg.reference.scale)
        beta, assumed, rho, shift, rho_eff, R_r, q_up = self.geometry(i)
        origin = np.asarray(e["crop_centre_en"], np.float64)
        centre = np.zeros(2) if ref_centre_en is None else \
            torch.as_tensor(ref_centre_en, dtype=torch.float64).detach().cpu().numpy().reshape(2).copy()
        ref_o = canvas_oriented(origin, beta, centre, size, cell)
        canvas, _valid = render_area(self.orthos[int(e["year"])], ref_o)
        half = None
        if self.ref_window_m is not None:
            half = self.ref_window_m / 2.0
        if self.extent_m is not None and ref_centre_en is None:
            half = self.extent_m / 2.0 if half is None else min(half, self.extent_m / 2.0)
        if half is not None:
            out_ = np.abs(np.arange(size) - (size - 1) / 2.0) * cell > half
            canvas[out_, :] = 0
            canvas[:, out_] = 0
        H = gt_homography(Oriented(tuple(e["en"]), q_up, n, cell), ref_o).astype(np.float32)
        img, d = self._pano(i, shift)
        erp = cv2.resize(img, (self.erp_w, self.erp_h), interpolation=cv2.INTER_AREA)
        out = dict(
            id=str(e["frame_id"]), city=str(int(e["year"])), year=int(e["year"]), scale=int(self.cfg.reference.scale),
            negative=False,
            erp=torch.from_numpy(np.ascontiguousarray(erp)).permute(2, 0, 1).float().div(255.0)[None],
            R_w2c=torch.from_numpy(R_r.astype(np.float32))[None], se2=torch.zeros(1, 3),
            ref=torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1).float().div(255.0),
            H=torch.from_numpy(H),
            en=torch.from_numpy(map_to_local(e["en"], origin, beta).astype(np.float64)),
            ref_centre_en=torch.from_numpy(centre.astype(np.float64)),
            beta_deg=beta, rho_deg=rho_eff, q_up_deg=q_up,
        )
        if d is not None:
            dd = cv2.resize(d, (self.erp_w, self.erp_h), interpolation=cv2.INTER_NEAREST)
            dd[ego_rows_mask(self.erp_h, self.ego_mask_deg), :] = 0.0
            out["depth"] = torch.from_numpy(dd)[None]
        return out

    def map_pose(self, i, H, centre_local=None, cfg=None):
        """(map EN, vehicle heading) of a pose H (BEV px -> canvas px) on the canvas of entry i centred at
        centre_local (None = the prior) under cfg's grid (default: this dataset's)."""
        e = self.labels[i]
        g = (cfg or self.cfg).grid
        beta, _assumed, _rho, _shift, rho_eff, _R, _q = self.geometry(i)
        size = int(int(g.n) * (cfg or self.cfg).reference.scale)
        ref_o = canvas_oriented(e["crop_centre_en"], beta, np.zeros(2) if centre_local is None else centre_local,
                                size, float(g.cell_m))
        en, yaw = pose_from_homography(H, int(g.n), ref_o)
        return en, float((yaw - rho_eff) % 360.0)


def residual_rotation_deg(entry, heading):
    """The rotation the matcher must recover (true minus assumed heading): -crop_rot under prior, 0 under gt."""
    _b, assumed, _r = heading_setup(entry, heading, "crop")
    return float(wrap180(float(entry["up_bearing_deg"]) - assumed))
