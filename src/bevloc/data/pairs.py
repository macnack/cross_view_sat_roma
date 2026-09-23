"""Training pairs: (ERP image, LiDAR sweep) query  <->  orthophoto reference crop with GT transform.

The query pose is the OxTS position and heading, in EPSG:27700. The INS->LiDAR lever arm is
still unknown, so the INS position stands in for the BEV origin (a constant sub-metre bias).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from bevloc.data.ortho import Oriented, OrthoMap, gt_homography, sample_reference
from bevloc.data.oxts import read_track


def fully_covered_names(ortho_path, names, en, size_m, min_frac=0.999, probe=48):
    """Names whose full query footprint (+ rotation/offset slack) has real ortho pixels
    everywhere, not just "inside the raster's bounding rectangle" (OrthoMap.render's `valid`
    is geometric only: a mosaic with gaps between tiles reads those gaps as black and still
    marks them valid, which would silently supervise the coarse loss against no data).
    `size_m`: the query's world size before the reference's `scale`x expansion; slack of 1.5x
    covers sample_reference's rotation. Cheap: reads a `probe` x probe downsample per frame."""
    import rasterio
    from rasterio.windows import from_bounds
    half = size_m * 0.75
    with rasterio.open(ortho_path) as ds:
        keep = []
        for name, p in zip(names, en):
            w = from_bounds(p[0] - half, p[1] - half, p[0] + half, p[1] + half, ds.transform)
            a = ds.read(1, window=w, boundless=True, fill_value=0, out_shape=(probe, probe))
            if (a > 0).mean() >= min_frac:
                keep.append(name)
    return keep


class FusionPairs(Dataset):
    def __init__(self, ds, cfg, names, ortho_path, train=True, seed=0, min_range=2.0, fixed_reference=False):
        self.ds, self.cfg, self.names, self.train = ds, cfg, list(names), train
        self.ortho_path, self._ortho = str(ortho_path), None          # opened per worker
        self.seed, self.min_range, self.fixed = seed, min_range, fixed_reference
        tr = read_track(ds.root, self.names)
        self.en, self.bearing = tr.en, tr.bearing(cfg.oxts.convention, grid=True)
        self.epoch = 0

    def __len__(self):
        return len(self.names)

    def query_pose(self, i) -> Oriented:
        return Oriented(tuple(self.en[i]), float(self.bearing[i]), self.cfg.grid.n, self.cfg.grid.cell_m)

    def __getitem__(self, i):
        if self._ortho is None:
            self._ortho = OrthoMap(self.ortho_path)
        name, r = self.names[i], self.cfg.reference
        # validation (and fixed_reference) draws the same crop every time
        rng = np.random.default_rng([self.seed, i] + ([self.epoch] if self.train and not self.fixed else []))
        q = self.query_pose(i)
        ref = sample_reference(q, rng, r.scale, r.max_offset_frac, r.max_rot_deg)
        img, _ = self._ortho.render(ref)
        scan = self.ds.scan(name)
        keep = scan.range_m > self.min_range                          # drops no-returns and the ego vehicle
        return dict(name=name,
                    erp=torch.from_numpy(self.ds.erp(name)).permute(2, 0, 1).float() / 255.0,
                    points=torch.from_numpy(scan.xyz[keep]).float(),
                    reflectivity=torch.from_numpy(scan.reflectivity[keep]).float(),
                    ref=torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
                    H=torch.from_numpy(gt_homography(q, ref)).float())


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("erp", "ref", "H")}
    out.update({k: [b[k] for b in batch] for k in ("name", "points", "reflectivity")})   # ragged
    return out


def write_synthetic_ortho(path, en, margin=300.0, gsd=1.0, seed=0):
    """A textured EPSG:27700 GeoTIFF covering the track. It has NOTHING to do with what the camera
    saw: it exists so the whole training path can be exercised (and overfit) without an orthophoto."""
    import cv2
    import rasterio
    from rasterio.transform import from_origin
    e0, n0 = en.min(0) - margin
    e1, n1 = en.max(0) + margin
    w, h = int((e1 - e0) / gsd), int((n1 - n0) / gsd)
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for k in (4, 16, 64):                                             # multi-scale blobs: locally distinctive
        img += cv2.resize(rng.uniform(0, 1, (h // k + 1, w // k + 1, 3)).astype(np.float32), (w, h),
                          interpolation=cv2.INTER_CUBIC)
    img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs="EPSG:27700",
                       transform=from_origin(e0, n1, gsd, gsd), compress="deflate") as dst:
        dst.write(np.moveaxis(img, -1, 0))
    return path
