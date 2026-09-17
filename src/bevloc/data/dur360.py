"""Frame access for Dur360BEV (ERP image, LiDAR scan, OxTS pose).

Wraps the dataset conventions of third_party/Dur360BEV without importing its Dataset
class (which needs labels, OSM maps and a full download). Layout: data/README.md.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bevloc.bev.fisheye import erp_maps

_REPO = Path(__file__).resolve().parents[3]

SCAN_ROWS, SCAN_COLS, POINT_BYTES = 128, 2048, 36
FISHEYE_APERTURE_DEG = 203          # only for lens="equidistant203" (fisheye_tools remap)


@dataclass
class Scan:
    """One Ouster OS1-128 sweep on its native (128, 2048) grid; row index = ring.

    Verified byte layout, 9 x 4-byte slots per point (see data/README.md):
      0-2 x,y,z float32 [m] (x fwd, y left, z up) | 3 intensity float32 | 4 t uint32 [ns since
      sweep start, constant per column, 0..~99.9 ms] | 5 reflectivity float32 | 6 packed, unused |
      7 ambient uint32 | 8 range uint32 [mm]
    """
    xyz: np.ndarray            # (128, 2048, 3) float64
    intensity: np.ndarray      # (128, 2048) float32
    t_ns: np.ndarray           # (128, 2048) uint32
    reflectivity: np.ndarray   # (128, 2048) float32
    ambient: np.ndarray        # (128, 2048) uint32
    range_m: np.ndarray        # (128, 2048) float64, 0 = no return

    @property
    def valid(self) -> np.ndarray:
        return self.range_m > 0.1

    def points(self, min_range=0.1) -> np.ndarray:
        """(N, 3) returns only."""
        return self.xyz[self.range_m > min_range]


def read_scan(path) -> Scan:
    raw = np.fromfile(path, np.uint8)
    if raw.size != SCAN_ROWS * SCAN_COLS * POINT_BYTES:
        raise ValueError(f"{path}: unexpected size {raw.size} (partial download?)")
    raw = raw.reshape(-1, POINT_BYTES)
    f = raw.view(np.float32).reshape(SCAN_ROWS, SCAN_COLS, 9)
    u = raw.view(np.uint32).reshape(SCAN_ROWS, SCAN_COLS, 9)
    xyz = f[..., :3].astype(np.float64)
    return Scan(xyz, f[..., 3].copy(), u[..., 4].copy(), f[..., 5].copy(), u[..., 7].copy(),
                np.linalg.norm(xyz, axis=-1))


@dataclass
class Frame:
    name: str
    erp: np.ndarray      # (H, W, 3) uint8 RGB, W = 2H, image centre = vehicle forward
    scan: Scan
    lla: np.ndarray | None   # lat [deg], lon [deg], alt [m]            (None until OxTS is downloaded)
    rpy: np.ndarray | None   # roll, pitch, yaw as stored; convention: data/README.md


class Dur360Frames:
    def __init__(self, root, erp_size=(1280, 640), lens="dur360", lens_params=None):
        """lens: 'dur360' = Dur360BEV's 196/203 deg piecewise model (verified against LiDAR);
        'equidistant203' = fisheye_tools' remap (kept for comparison; ~6 px off at 57 deg)."""
        self.root = Path(root)
        self.erp_size = tuple(erp_size)
        self.lens, self.lens_params = lens, dict(lens_params or {})
        self._maps = None

    @classmethod
    def from_config(cls, cfg, **override):
        e = cfg.erp
        kw = dict(erp_size=tuple(e.size), lens=e.lens, lens_params=vars(e.lens_params))
        kw.update(override)
        return cls(_REPO / cfg.data.root if not Path(cfg.data.root).is_absolute() else cfg.data.root, **kw)

    # ---- paths / availability -------------------------------------------------------------
    def path(self, modality, name):
        sub, ext = {"image": ("image", "png"), "scan": ("ouster_points", "bin"), "oxts": ("oxts", "txt")}[modality]
        return self.root / sub / "data" / f"{name}.{ext}"

    def has(self, name, modalities=("image", "scan")) -> bool:
        for m in modalities:
            p = self.path(m, name)
            if not p.exists() or (m == "scan" and p.stat().st_size != SCAN_ROWS * SCAN_COLS * POINT_BYTES):
                return False
        return True

    def names(self, modalities=("image", "scan"), start=None, stop=None, stride=1):
        """Sorted frame names that have all requested modalities (complete files only)."""
        first = sorted(p.stem for p in (self.root / "ouster_points/data").glob("*.bin")) \
            if "scan" in modalities else sorted(p.stem for p in (self.root / "image/data").glob("*.png"))
        out = [n for n in first
               if (start is None or int(n) >= start) and (stop is None or int(n) <= stop)
               and int(n) % stride == 0 and self.has(n, modalities)]
        return out

    # ---- modalities -----------------------------------------------------------------------
    def fisheye(self, name) -> np.ndarray:
        """Pre-rotated dual fisheye, 1280x640 RGB (both lenses upright, front lens on the right)."""
        img = cv2.imread(str(self.path("image", name)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.path("image", name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if h * 2 != w:
            h = w // 2
            img = img[:h]
        left = cv2.rotate(img[:, :h], cv2.ROTATE_90_CLOCKWISE)
        right = cv2.rotate(img[:, h:], cv2.ROTATE_90_COUNTERCLOCKWISE)
        return np.concatenate((left, right), axis=1)

    def erp(self, name) -> np.ndarray:
        img = self.fisheye(name)
        if self._maps is None:
            if self.lens == "dur360":
                self._maps = erp_maps(*self.erp_size, img.shape[1], img.shape[0], **self.lens_params)
            elif self.lens == "equidistant203":
                sys.path.insert(0, str(_REPO / "third_party/Dur360BEV/fisheye_tools"))
                from fisheyetools import getcvmap
                self._maps = getcvmap.dualfisheye2equi(img, size=self.erp_size,
                                                       aperture=FISHEYE_APERTURE_DEG, center_angle=0)
            else:
                raise ValueError(f"unknown lens {self.lens!r}")
        return cv2.remap(img, self._maps[0], self._maps[1], interpolation=cv2.INTER_CUBIC)

    def scan(self, name) -> Scan:
        return read_scan(self.path("scan", name))

    def points(self, name, min_range=0.1) -> np.ndarray:
        return self.scan(name).points(min_range)

    def oxts(self, name):
        p = self.path("oxts", name)
        if not p.exists():
            return None, None
        v = np.loadtxt(p).reshape(-1)
        return v[:3], v[3:6]

    def __getitem__(self, name) -> Frame:
        lla, rpy = self.oxts(name)
        return Frame(name, self.erp(name), self.scan(name), lla, rpy)
