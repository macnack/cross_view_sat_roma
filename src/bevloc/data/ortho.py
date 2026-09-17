"""Orthophoto reference crops in British National Grid (EPSG:27700) and GT transforms.

All oriented square images here (reference crop, BEV query) share one model:
pixel (u, v) -> world (E, N), defined by the world position of the image
centre, the bearing of image-up (degrees clockwise from grid north) and the
GSD. `Oriented.px_to_world` is that 3x3 affine; the GT query->reference
homography is  inv(ref.px_to_world) @ query.px_to_world.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

BNG = "EPSG:27700"


@dataclass(frozen=True)
class Oriented:
    centre_en: tuple      # world (E, N) of the image centre, metres
    up_bearing_deg: float  # bearing of image-up, clockwise from grid north
    size: int             # pixels per side
    gsd: float            # metres per pixel

    @property
    def px_to_world(self) -> np.ndarray:
        b = np.radians(self.up_bearing_deg)
        right = np.array([np.cos(b), -np.sin(b)])
        up = np.array([np.sin(b), np.cos(b)])
        c = (self.size - 1) / 2.0
        A = np.eye(3)
        A[:2, 0] = self.gsd * right
        A[:2, 1] = -self.gsd * up
        A[:2, 2] = np.asarray(self.centre_en) - self.gsd * (c * right - c * up)
        return A


def gt_homography(query: Oriented, ref: Oriented) -> np.ndarray:
    return np.linalg.inv(ref.px_to_world) @ query.px_to_world


def lonlat_to_bng(lon, lat):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", BNG, always_xy=True)
    return t.transform(lon, lat)


class OrthoMap:
    """A GeoTIFF or VRT in EPSG:27700 (build a VRT over EA tiles with gdalbuildvrt)."""

    def __init__(self, path):
        self.ds = rasterio.open(path)
        if self.ds.crs is None or self.ds.crs.to_epsg() != 27700:
            raise ValueError(f"{path}: expected EPSG:27700, got {self.ds.crs}")
        self.inv = ~self.ds.transform

    def render(self, o: Oriented):
        """Resample the map onto an oriented square. Returns (img uint8 RGB, valid bool)."""
        u, v = np.meshgrid(np.arange(o.size, dtype=np.float64), np.arange(o.size, dtype=np.float64))
        A = o.px_to_world
        E = A[0, 0] * u + A[0, 1] * v + A[0, 2]
        N = A[1, 0] * u + A[1, 1] * v + A[1, 2]
        col, row = self.inv * (E, N)          # raster px, pixel-corner origin
        col, row = col - 0.5, row - 0.5       # -> pixel-centre indexing for remap
        c0, r0 = int(np.floor(col.min())) - 2, int(np.floor(row.min())) - 2
        c1, r1 = int(np.ceil(col.max())) + 3, int(np.ceil(row.max())) + 3
        win = Window(c0, r0, c1 - c0, r1 - r0)
        src = self.ds.read(indexes=[1, 2, 3][: min(3, self.ds.count)], window=win,
                           boundless=True, fill_value=0)
        src = np.moveaxis(src, 0, -1)
        if src.shape[-1] == 1:
            src = np.repeat(src, 3, -1)
        inside = ((col >= 0) & (col <= self.ds.width - 1) & (row >= 0) & (row <= self.ds.height - 1))
        # area-average when the map is finer than the target GSD
        k = o.gsd / abs(self.ds.transform.a)
        if k > 1.5:
            s = max(1, int(round(k)))
            src = cv2.blur(src, (s, s))
        img = cv2.remap(np.ascontiguousarray(src), (col - c0).astype(np.float32),
                        (row - r0).astype(np.float32), cv2.INTER_LINEAR)
        img[~inside] = 0
        return img, inside


def sample_reference(query: Oriented, rng, scale=4, max_offset_frac=0.30, max_rot_deg=55.0) -> Oriented:
    """Reference crop at `scale`x the query extent, same GSD, with the query footprint
    displaced from the centre by up to max_offset_frac of the reference edge per axis
    (clipped so the footprint stays inside) and rotated by U(-max_rot, +max_rot)."""
    size = query.size * scale
    edge = size * query.gsd
    rot = rng.uniform(-max_rot_deg, max_rot_deg)
    lim = min(max_offset_frac * edge, edge / 2 - query.size * query.gsd / np.sqrt(2))
    off = rng.uniform(-lim, lim, 2)           # in the reference's own right/up axes
    up_b = query.up_bearing_deg + rot
    b = np.radians(up_b)
    right, up = np.array([np.cos(b), -np.sin(b)]), np.array([np.sin(b), np.cos(b)])
    centre = np.asarray(query.centre_en) - (off[0] * right + off[1] * up)
    return Oriented(tuple(centre), up_b, size, query.gsd)
