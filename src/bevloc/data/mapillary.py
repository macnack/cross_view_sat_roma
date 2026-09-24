"""Mapillary spherical sequences paired with a Poznań CS92 (EPSG:2180) orthophoto."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from pyproj import Geod, Transformer
from rasterio.windows import Window
from torch.utils.data import Dataset

from bevloc.bev.ipm_sphere import (
    above_contact_mask, contact_feet, depression_mask, ipm_erp, mosaic_ipm, paint_feet,
)
from bevloc.data.ortho import Oriented, gt_homography, sample_negative_reference, sample_reference

CS92 = "EPSG:2180"
_TO_CS92 = Transformer.from_crs("EPSG:4326", CS92, always_xy=True)


def sat_data_root() -> Path:
    """Root of the geoportal / lantmäteriet tile folders.

    Overridable with SAT_DATA_DIR for Eagle, where $HOME is not mounted on compute nodes."""
    return Path(os.environ.get("SAT_DATA_DIR", str(Path.home() / "Github/sat_data")))


def poznan_tiles(year: int) -> list[Path]:
    paths = sorted(sat_data_root().glob(f"geoportal_poznan_15km2_*/year_{int(year)}.tif"))
    if len(paths) < 9:
        raise FileNotFoundError(
            f"expected >= 9 Poznań tiles for {year} under {sat_data_root()}, found {len(paths)}")
    return paths


MAP_ROOT = Path(__file__).resolve().parents[3] / "data/mapillary"
TRAIN_SEQS = [
    MAP_ROOT / "Fixtor/iHfmEq03Tc6752Y4Ke8wlC",
    MAP_ROOT / "Fixtor/NWVA14Y83pMRsijaGFkmQS",
    MAP_ROOT / "Fixtor/gXabFhpwk2dcl0i4518mDQ",
    MAP_ROOT / "Fixtor/doQ3OhJBKe56c8UxjAFmat",
]
VAL_SEQS = [MAP_ROOT / "Fixtor/IcRzj0wTLZX874qitxVsQa"]          # checkpoint selection, dev numbers
TEST_SEQS = [MAP_ROOT / "Fixtor/irAsBUKtGCfhPHuMbmOcLd"]         # reserved: never trained on, never selected on


def rodrigues(r):
    r = np.asarray(r, np.float64)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def grid_bearing(lon, lat, true_bearing):
    geod = Geod(ellps="WGS84")
    lon2, lat2, _ = geod.fwd(lon, lat, 0.0, 200.0)
    e0, n0 = _TO_CS92.transform(lon, lat)
    e1, n1 = _TO_CS92.transform(lon2, lat2)
    gamma = np.degrees(np.arctan2(e1 - e0, n1 - n0))
    return (true_bearing + gamma) % 360.0, (float(e0), float(n0))


def en_of(fr):
    return _TO_CS92.transform(*fr["computed_geometry"]["coordinates"])


def src_in_query_se2(en_q, up_q_deg, en_s, up_s_deg):
    """SE(2) taking source-ego (x forward, y left) into the query ego frame.

    Returns (yaw_rad, tx, ty): rotate then translate. Built from Mapillary EN +
    grid up-bearing (CW from grid north), same convention as ``Oriented``.
    """
    bq = np.radians(float(up_q_deg))
    fwd = np.array([np.sin(bq), np.cos(bq)])          # (E, N)
    left = np.array([-np.cos(bq), np.sin(bq)])
    d = np.asarray(en_s, float) - np.asarray(en_q, float)
    tx = float(d @ fwd)
    ty = float(d @ left)
    yaw = np.radians(float(up_s_deg) - float(up_q_deg))
    return float(yaw), tx, ty


def _seq_cumdist(frames):
    """Per-frame path length along the sequence (metres in CS92)."""
    ens = [np.asarray(en_of(fr), float) for fr in frames]
    cum = np.zeros(len(frames), float)
    for i in range(1, len(frames)):
        cum[i] = cum[i - 1] + float(np.linalg.norm(ens[i] - ens[i - 1]))
    return cum


class PoznanOrtho:
    """One or more GeoTIFF tiles in EPSG:2180. A black crop is invalid."""

    def __init__(self, paths):
        self.tiles = [rasterio.open(p) for p in paths]

    def close(self):
        for t in self.tiles:
            t.close()

    def tile_for(self, en, margin=0.0):
        e, n = en
        for t in self.tiles:
            if (t.bounds.left + margin <= e <= t.bounds.right - margin
                    and t.bounds.bottom + margin <= n <= t.bounds.top - margin):
                return t
        return None

    def render(self, o: Oriented):
        t = self.tile_for(o.centre_en)
        if t is None:
            z = np.zeros((o.size, o.size, 3), np.uint8)
            return z, np.zeros((o.size, o.size), bool)
        b = np.radians(o.up_bearing_deg)
        right = np.array([np.cos(b), -np.sin(b)])
        up = np.array([np.sin(b), np.cos(b)])
        c = (o.size - 1) / 2.0
        uu, vv = np.meshgrid(np.arange(o.size), np.arange(o.size))
        E = o.centre_en[0] + o.gsd * ((uu - c) * right[0] - (vv - c) * up[0])
        N = o.centre_en[1] + o.gsd * ((uu - c) * right[1] - (vv - c) * up[1])
        inv = ~t.transform
        col, row = inv * (E, N)
        col, row = col - 0.5, row - 0.5
        c0, r0 = int(np.floor(col.min())) - 2, int(np.floor(row.min())) - 2
        c1, r1 = int(np.ceil(col.max())) + 3, int(np.ceil(row.max())) + 3
        src = t.read(indexes=[1, 2, 3], window=Window(c0, r0, c1 - c0, r1 - r0),
                     boundless=True, fill_value=0)
        src = np.moveaxis(src, 0, -1)
        k = o.gsd / abs(t.transform.a)
        if k > 1.5:
            s = max(1, int(round(k)))
            src = cv2.blur(src, (s, s))
        img = cv2.remap(np.ascontiguousarray(src), (col - c0).astype(np.float32),
                        (row - r0).astype(np.float32), cv2.INTER_LINEAR)
        inside = (col >= 0) & (col <= t.width - 1) & (row >= 0) & (row <= t.height - 1)
        img[~inside] = 0
        return img, inside


def load_frames(seq_dirs, ortho, margin_m=120.0):
    """All frames whose GPS sits `margin_m` inside some ortho tile.

    ``ortho`` may be a PoznanOrtho or a dict of them; coverage uses the first.
    """
    probe = next(iter(ortho.values())) if isinstance(ortho, dict) else ortho
    out = []
    for d in seq_dirs:
        d = Path(d)
        frames = json.loads((d / "images.json").read_text())
        kept = 0
        for fr in frames:
            en = en_of(fr)
            if probe.tile_for(en, margin=margin_m) is None:
                continue
            fr = dict(fr)
            fr["_seq"] = str(d)
            fr["_en"] = en
            out.append(fr)
            kept += 1
        print(f"  {d.name}: {kept}/{len(frames)} frames inside ortho (margin {margin_m:.0f} m)", flush=True)
    return out


class MapillaryPairs(Dataset):
    """Spherical Mapillary frames with a CS92 orthophoto reference crop.

    ``ortho`` is either a PoznanOrtho or a dict {year: PoznanOrtho}. When a dict
    is given, each sample draws a year (uniform, or the single --year at eval).

    Training augmentations (cfg.lift):
      - random reference scale from ``ref_scales`` (different map extent, same GSD)
      - random pose window: offset_frac and rot_deg drawn in ``offset_range`` / ``rot_range``
      - colour jitter on ERP and reference
      - small attitude noise on ``computed_rotation``
      - ``neg_frac`` of samples: reference displaced so GT pose is off the crop
        (no-match; certainty → 0, CE skipped — Sat-RoMa has no unmatched class bin)
      - ``seq_dists_m``: extra panoramas this many metres behind the query, soft-splat
        into one BEV via GPS+compass SE(2) (multi-frame density)
    """

    def __init__(self, frames, ortho, cfg, train=True, seed=0, erp_size=(896, 448),
                 years=None):
        self.frames = list(frames)
        self.ortho = ortho if isinstance(ortho, dict) else {0: ortho}
        self.years = list(years) if years is not None else list(self.ortho)
        self.cfg = cfg
        self.train = train
        self.erp_w, self.erp_h = erp_size
        self.rng = np.random.default_rng(seed)
        L = getattr(cfg, "lift", None)
        dists = list(getattr(L, "seq_dists_m", None) or [0.0]) if L else [0.0]
        self.seq_dists = [float(d) for d in dists]
        if not self.seq_dists or self.seq_dists == [0.0]:
            self.seq_dists = [0.0]
        # Per-sequence sorted order + cumulative distance for neighbour lookup
        self._seq_order = {}
        self._seq_cum = {}
        self._id_to_pos = {}  # global index -> (seq, pos_in_seq)
        by_seq = {}
        for i, fr in enumerate(self.frames):
            by_seq.setdefault(fr["_seq"], []).append(i)
        for seq, idxs in by_seq.items():
            order = sorted(idxs, key=lambda j: self.frames[j].get("captured_at", 0))
            self._seq_order[seq] = order
            self._seq_cum[seq] = _seq_cumdist([self.frames[j] for j in order])
            for pos, j in enumerate(order):
                self._id_to_pos[j] = (seq, pos)

    def __len__(self):
        return len(self.frames)

    def _neighbor_indices(self, i):
        """Frame indices at approx seq_dists metres behind query (include query at 0)."""
        seq, pos = self._id_to_pos[i]
        order, cum = self._seq_order[seq], self._seq_cum[seq]
        qcum = cum[pos]
        out = []
        for d in self.seq_dists:
            target = qcum - d
            # nearest earlier-or-equal frame
            k = int(np.searchsorted(cum, target, side="right") - 1)
            k = max(0, min(k, pos))
            out.append(order[k])
        # de-dupe while preserving order (query first if d=0)
        seen, uniq = set(), []
        for j in out:
            if j not in seen:
                seen.add(j)
                uniq.append(j)
        return uniq

    def _load_erp_R_pose(self, fr, rng, full=False):
        """Resized, jittered ERP (+ the full-resolution one when ``full``), R_w2c, up-bearing, EN."""
        path = Path(fr["_seq"]) / "images" / f"{fr['id']}.jpg"
        erp_full = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        erp = cv2.resize(erp_full, (self.erp_w, self.erp_h), interpolation=cv2.INTER_AREA)
        R = rodrigues(fr["computed_rotation"]).astype(np.float32)
        lon, lat = fr["computed_geometry"]["coordinates"]
        up, en = grid_bearing(lon, lat, fr["computed_compass_angle"])
        R = self._attitude_noise(R, rng)
        erp = self._colour_jitter(erp, rng)
        if full:
            return erp, R, up, en, erp_full
        return erp, R, up, en

    def _query_mode(self):
        L = getattr(self.cfg, "lift", None)
        return str(getattr(L, "query_mode", "lift") or "lift") if L else "lift"

    # Cityscapes train ids (scripts/semantic_precompute.py)
    _DYNAMIC_IDS = (11, 12, 13, 14, 15, 16, 17, 18)          # person, rider, car, truck, bus, train, motorcycle, bicycle
    _GROUND_IDS = (0, 1, 8, 9)                                # road, sidewalk, vegetation, terrain

    def _semantic(self, fr, erp_hw):
        """Precomputed SegFormer class map (H, W) uint8 resized to erp_hw, or None if absent."""
        p = Path(fr["_seq"]) / "semantic" / f"{fr['id']}.png"
        if not p.exists():
            return None
        sem = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if sem is None:
            return None
        if sem.shape[:2] != tuple(erp_hw):
            sem = cv2.resize(sem, (erp_hw[1], erp_hw[0]), interpolation=cv2.INTER_NEAREST)
        return sem

    def _erp_valid(self, fr, erp_hw):
        """(H, W) bool validity of the panorama for the IPM picture: ego-body cut by depression angle,
        plus dynamic-object (and optionally non-ground) pixels masked when a precomputed semantic map
        exists at <seq>/semantic/<id>.png (scripts/semantic_precompute.py)."""
        ipm = self.cfg.ipm
        dep = float(getattr(ipm, "max_depression_deg", 0.0) or 0.0)
        valid = depression_mask(erp_hw, dep) if dep > 0 else np.ones(erp_hw, bool)
        use_dyn = bool(getattr(ipm, "semantic_dynamic_mask", False))
        ground_only = bool(getattr(ipm, "semantic_ground_only", False))
        contact = bool(getattr(ipm, "contact_line", False))
        if use_dyn or ground_only or contact:
            sem = self._semantic(fr, erp_hw)
            if sem is not None:
                if use_dyn:
                    valid &= ~np.isin(sem, self._DYNAMIC_IDS)
                if ground_only:
                    valid &= np.isin(sem, self._GROUND_IDS)
                if contact:
                    # ipm_cl: no IPM at or above the wall/hedge contact row (no radial facade smear);
                    # the feet are painted afterwards by _paint_contact
                    classes = tuple(int(c) for c in getattr(ipm, "contact_classes", (2, 3, 4)))
                    if bool(getattr(ipm, "contact_vegetation", False)):
                        classes = classes + (8,)
                    valid &= above_contact_mask(sem, classes, erp_hw)
        return valid

    def _paint_contact(self, fr, erp_full, R, bev, bev_valid):
        """ipm_cl: draw the wall (and hedge) feet of this frame into its IPM picture, in place."""
        ipm = self.cfg.ipm
        if not bool(getattr(ipm, "contact_line", False)):
            return bev, bev_valid
        sem = self._semantic(fr, erp_full.shape[:2])
        if sem is None:
            return bev, bev_valid
        g = self.cfg.grid
        classes = [tuple(int(c) for c in getattr(ipm, "contact_classes", (2, 3, 4)))]
        if bool(getattr(ipm, "contact_vegetation", False)):
            classes.append((8,))
        thick = int(getattr(ipm, "contact_thickness_px", 2))
        rng_max = float(getattr(getattr(self.cfg, "erp_query", None), "max_range_m", 40.0)) if getattr(self.cfg, "erp_query", None) else 40.0
        for ids in classes:
            feet = contact_feet(sem, erp_full, ids, R, ipm.height_m, g.n, g.cell_m, max_range_m=rng_max)
            paint_feet(bev, bev_valid, feet, g.n, g.cell_m, thickness=thick)
        return bev, bev_valid

    def _pose_budget(self, rng):
        L = getattr(self.cfg, "lift", None)
        if self.train and L is not None and getattr(L, "offset_range", None):
            off = float(rng.uniform(*L.offset_range))
            rot = float(rng.uniform(*L.rot_range))
        else:
            off = float(self.cfg.reference.max_offset_frac)
            rot = float(self.cfg.reference.max_rot_deg)
        scales = list(getattr(L, "ref_scales", None) or [self.cfg.reference.scale]) if L else [self.cfg.reference.scale]
        scale = int(rng.choice(scales)) if self.train else int(self.cfg.reference.scale)
        return scale, off, rot

    def _colour_jitter(self, img, rng):
        """img uint8 RGB. Mild brightness / contrast / channel gain."""
        L = getattr(self.cfg, "lift", None)
        if not self.train or L is None or not getattr(L, "colour_jitter", True):
            return img
        x = img.astype(np.float32)
        bright = float(rng.uniform(0.85, 1.15))
        contrast = float(rng.uniform(0.85, 1.15))
        gain = rng.uniform(0.9, 1.1, size=3).astype(np.float32)
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = (x - mean) * contrast + mean
        x = x * bright * gain
        return np.clip(x, 0, 255).astype(np.uint8)

    def _attitude_noise(self, R, rng):
        L = getattr(self.cfg, "lift", None)
        deg = float(getattr(L, "attitude_noise_deg", 0.0) or 0.0) if L else 0.0
        if not self.train or deg <= 0:
            return R
        # small random axis-angle in the camera frame
        axis = rng.normal(size=3).astype(np.float64)
        axis /= np.linalg.norm(axis) + 1e-9
        ang = np.radians(float(rng.uniform(-deg, deg)))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        dR = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        return (dR.astype(np.float32) @ R)

    def _up_of(self, fr):
        lon, lat = fr["computed_geometry"]["coordinates"]
        return grid_bearing(lon, lat, fr["computed_compass_angle"])[0]

    def _one(self, i):
        fr = self.frames[i]
        rng = (self.rng if self.train
               else np.random.default_rng([self.cfg.matcher.seed, int(fr["id"]) % (2**32)]))
        erp_q, R_q, up_q, en_q, erp_full = self._load_erp_R_pose(fr, rng, full=True)
        g = self.cfg.grid
        query = Oriented(en_q, up_q, g.n, g.cell_m)
        year = int(rng.choice(self.years)) if self.train else int(self.years[0])
        scale, off, rot = self._pose_budget(rng)
        L = getattr(self.cfg, "lift", None)
        neg_frac = float(getattr(L, "neg_frac", 0.0) or 0.0) if L else 0.0
        negative = bool(self.train and neg_frac > 0 and rng.random() < neg_frac)
        if negative:
            sep = tuple(getattr(L, "neg_sep_frac", (0.70, 1.40)))
            ref_o = sample_negative_reference(query, rng, scale=scale, sep_frac=sep,
                                             max_rot_deg=rot)
        else:
            ref_o = sample_reference(query, rng, scale=scale, max_offset_frac=off, max_rot_deg=rot)
        return self._build(i, ref_o, year, rng, negative=negative, scale=scale,
                           loaded=(erp_q, R_q, up_q, en_q, erp_full))

    def sample_for(self, frame_id, ref_o, year):
        """Deterministic sample for one frame with a GIVEN reference crop (manifest evaluation)."""
        i = next(k for k, fr in enumerate(self.frames) if str(fr["id"]) == str(frame_id))
        rng = np.random.default_rng([self.cfg.matcher.seed, int(frame_id) % (2**32)])
        return self._build(i, ref_o, int(year), rng, negative=False,
                           scale=int(ref_o.size // self.cfg.grid.n))

    def _build(self, i, ref_o, year, rng, negative=False, scale=4, loaded=None):
        fr = self.frames[i]
        erp_q, R_q, up_q, en_q, erp_full = (loaded if loaded is not None
                                             else self._load_erp_R_pose(fr, rng, full=True))
        g = self.cfg.grid
        query = Oriented(en_q, up_q, g.n, g.cell_m)
        ref, valid = self.ortho[year].render(ref_o)
        if float(valid.mean()) < 0.5 or float((ref.sum(-1) > 0).mean()) < 0.5:
            raise RuntimeError(f"black / missing reference for {fr['id']} year {year} at {en_q}")
        ref = self._colour_jitter(ref, rng)
        H = gt_homography(query, ref_o).astype(np.float32)

        # Multi-frame: query + frames ~seq_dists metres behind
        neigh = self._neighbor_indices(i)
        erps, Rs, se2s = [], [], []
        fulls = {}                                   # neighbour full-res ERPs, decoded once (IPM mosaic)
        want_full = self._query_mode() == "ipm"
        for j in neigh:
            if j == i:
                erp, R, up, en = erp_q, R_q, up_q, en_q
            elif want_full:
                erp, R, up, en, fulls[j] = self._load_erp_R_pose(self.frames[j], rng, full=True)
            else:
                erp, R, up, en = self._load_erp_R_pose(self.frames[j], rng)
            yaw, tx, ty = src_in_query_se2(en_q, up_q, en, up)
            erps.append(torch.from_numpy(erp).permute(2, 0, 1).float().div(255.0))
            Rs.append(torch.from_numpy(R))
            se2s.append(torch.tensor([yaw, tx, ty], dtype=torch.float32))

        out = dict(
            id=fr["id"],
            year=year,
            scale=scale,
            negative=negative,
            erp=torch.stack(erps, 0),          # (T, 3, H, W)
            ref=torch.from_numpy(np.ascontiguousarray(ref)).permute(2, 0, 1).float().div(255.0),
            R_w2c=torch.stack(Rs, 0),          # (T, 3, 3)
            se2=torch.stack(se2s, 0),          # (T, 3)
            H=torch.from_numpy(H),
            en=torch.tensor(en_q, dtype=torch.float64),
        )
        if self._query_mode() == "ipm":
            # Camera-only flat-ground picture of the QUERY frame at full ERP resolution
            # (kick-off H2 lower bound): the decoder sees it through the frozen encoder.
            ipm = self.cfg.ipm
            bev, bev_valid = ipm_erp(erp_full, R_q, ipm.height_m, g.n, g.cell_m, ipm.blind_radius_m,
                                     erp_valid=self._erp_valid(fr, erp_full.shape[:2]))
            if len(neigh) > 1:
                # Multi-frame IPM mosaic: the neighbours' pictures warped by the proxy relative pose,
                # nearest camera wins per cell (the query keeps everything it sees close by).
                pics, vals, poses = [bev], [bev_valid], [(0.0, 0.0, 0.0)]
                for j, R_j, se2 in zip(neigh, Rs, se2s):
                    if j == i:
                        continue
                    full_j = fulls[j]                                    # same decode and same R as the ERP stack
                    pic_j, val_j = ipm_erp(full_j, R_j.numpy(), ipm.height_m, g.n, g.cell_m, ipm.blind_radius_m,
                                           erp_valid=self._erp_valid(self.frames[j], full_j.shape[:2]))
                    pics.append(pic_j)
                    vals.append(val_j)
                    poses.append(tuple(float(v) for v in se2.tolist()))
                bev, bev_valid = mosaic_ipm(pics, vals, poses, g.n, g.cell_m)
            bev, bev_valid = self._paint_contact(fr, erp_full, R_q, bev, bev_valid)   # ipm_cl (off by default)
            bev = self._colour_jitter(bev, rng)
            out["bev"] = torch.from_numpy(np.ascontiguousarray(bev)).permute(2, 0, 1).float().div(255.0)
            out["bev_valid"] = torch.from_numpy(bev_valid)
        return out

    def __getitem__(self, i):
        last = None
        for _ in range(8):
            try:
                return self._one(i)
            except RuntimeError as e:
                last = e
                i = int(self.rng.integers(len(self.frames)))
        raise RuntimeError(f"failed to load a usable pair after retries: {last}")


def collate(batch):
    # Variable reference scales → only batch size 1 is safe (or all same scale).
    sizes = {tuple(b["ref"].shape[-2:]) for b in batch}
    if len(sizes) > 1:
        raise RuntimeError(f"cannot collate mixed reference sizes {sizes}; use batch=1")
    # erp / R / se2 are (T, …); T must match across the batch
    Ts = {b["erp"].shape[0] for b in batch}
    if len(Ts) > 1:
        raise RuntimeError(f"cannot collate mixed sequence lengths {Ts}")
    out = {
        "erp": torch.stack([b["erp"] for b in batch]),       # (B, T, 3, H, W)
        "ref": torch.stack([b["ref"] for b in batch]),
        "R_w2c": torch.stack([b["R_w2c"] for b in batch]),   # (B, T, 3, 3)
        "se2": torch.stack([b["se2"] for b in batch]),       # (B, T, 3)
        "H": torch.stack([b["H"] for b in batch]),
        "en": torch.stack([b["en"] for b in batch]),
    }
    if "bev" in batch[0]:
        out["bev"] = torch.stack([b["bev"] for b in batch])
        out["bev_valid"] = torch.stack([b["bev_valid"] for b in batch])
    out["id"] = [b["id"] for b in batch]
    out["year"] = [b["year"] for b in batch]
    out["scale"] = [b["scale"] for b in batch]
    out["negative"] = torch.tensor([bool(b.get("negative", False)) for b in batch], dtype=torch.bool)
    return out
