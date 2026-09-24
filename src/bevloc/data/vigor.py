"""VIGOR (Zhu et al., CVPR 2021) reader producing the samples our matcher pipeline consumes.

Layout under ``root`` (scripts/fetch_vigor.py):
  <root>/<City>/panorama/<lat,lon,.jpg>   2048x1024 equirectangular, north-aligned (centre column = north)
  <root>/<City>/satellite/<lat,lon,.png>  640x640, north-up, ~0.10-0.12 m/px (CITY_RES, at 640 px)
  <root>/splits/**/<City>/{satellite_list.txt, same_area_balanced_{train,test}.txt, pano_label_balanced.txt}
Label line: ``pano sat1 dy1 dx1 sat2 dy2 dx2 sat3 dy3 dx3 sat4 dy4 dx4``. sat1 is the positive tile
(the panorama lies in its central quarter); (dy, dx) are tile pixels with dy > 0 = panorama south of the
tile centre and dx > 0 = panorama WEST of it (verified against the lat/lon in the file names). ``__corrected`` label files (SliceMatch) are preferred when present.

Protocol used here = the standard "known orientation" one (FG², CCVPE, Loc²): the query is the panorama
of one location, the reference is its positive tile, the pose to recover is the panorama's position in
that tile (yaw is 0 by construction). Metric errors use the per-city ground resolution.

Sample geometry (same conventions as MapillaryPairs so every query mode and the evaluators apply):
  * the query BEV is the 224 px / cfg.grid.cell_m ego-centred picture, row 0 = north (the panorama is
    north-aligned, ego forward = north);
  * the reference is the tile resampled to cfg.grid.cell_m and centred in a (size x size) black canvas
    (size = 224 * cfg.reference.scale, i.e. the 896 px crop the checkpoint expects); black = no data;
  * H (BEV px -> reference px) is a pure translation, so pose_errors(H_est, H) is in metres.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from bevloc.bev.ipm_sphere import ipm_erp

CITIES = ("NewYork", "Seattle", "SanFrancisco", "Chicago")
# metres per pixel of the 640 px tile (FG² config.ini [Constants])
CITY_RES = {"NewYork": 0.113248, "Seattle": 0.100817, "SanFrancisco": 0.118141, "Chicago": 0.111262}
# world ENU -> camera for a north-aligned panorama: camera x = east, y = down, z = north (forward)
R_NORTH = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], np.float32)


def find_label_root(root: Path) -> Path:
    """Directory that holds <City>/satellite_list.txt (the zip extracts to varying depths)."""
    root = Path(root)
    hits = sorted(root.glob("**/satellite_list.txt"))
    if not hits:
        raise FileNotFoundError(f"no satellite_list.txt under {root}; extract splits.zip first")
    return hits[0].parent.parent


def _label_file(label_root: Path, city: str, name: str) -> Path:
    corrected = label_root / city / name.replace(".txt", "__corrected.txt")
    return corrected if corrected.exists() else label_root / city / name


def read_labels(root, cities, split, train):
    """List of dict(city, pano, sat, dy, dx) for the positive tile of every panorama.

    split "samearea": <city>/same_area_balanced_{train,test}.txt; "crossarea": <city>/pano_label_balanced.txt."""
    root = Path(root)
    lr = find_label_root(root)
    out = []
    for city in cities:
        name = ("same_area_balanced_train.txt" if train else "same_area_balanced_test.txt") \
            if split == "samearea" else "pano_label_balanced.txt"
        f = _label_file(lr, city, name)
        if not f.exists():
            raise FileNotFoundError(f)
        for line in f.read_text().splitlines():
            d = line.split()
            if len(d) < 4:
                continue
            out.append(dict(city=city, pano=d[0], sat=d[1], dy=float(d[2]), dx=float(d[3])))
    return out


def split_cities(split: str, train: bool):
    if split == "samearea":
        return list(CITIES)
    return ["NewYork", "Seattle"] if train else ["SanFrancisco", "Chicago"]


class VigorPairs(Dataset):
    """Panorama + positive tile with a known position; yields the dict the trainers/evaluators use.

    Keys: id, city, year (0), scale, negative (False), erp (1, 3, h, w) float in [0, 1] (resized to
    ``erp_size`` for the lifted modes), R_w2c (1, 3, 3), se2 (1, 3), H (3, 3), en (2,) (offset of the
    camera from the tile centre in metres, east/north), ref (3, S, S), plus for query_mode "ipm":
    bev (3, n, n) and bev_valid (n, n)."""

    def __init__(self, root, cfg, cities=None, split="crossarea", train=False, limit=0, stride=1,
                 erp_size=(896, 448), row_sign=None, col_sign=None, height_m=None, seed=0):
        self.root = Path(root)
        self.cfg = cfg
        V = getattr(cfg, "vigor", None)
        # Verified from the file names' lat/lon (scripts/vigor_check_labels.py, median residual 0.05 m over
        # 3000 labels per city): dy > 0 = panorama SOUTH of the tile centre (row down: sign +1);
        # dx > 0 = panorama WEST of the tile centre (column right: sign -1).
        self.row_sign = float(row_sign if row_sign is not None else getattr(V, "row_sign", 1.0))
        self.col_sign = float(col_sign if col_sign is not None else getattr(V, "col_sign", -1.0))
        self.height = float(height_m if height_m is not None else getattr(V, "height_m", 2.0))
        self.erp_w, self.erp_h = erp_size
        self.labels = read_labels(self.root, cities or split_cities(split, train), split, train)
        if stride > 1:
            self.labels = self.labels[::stride]
        if limit:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(self.labels), size=min(limit, len(self.labels)), replace=False))
            self.labels = [self.labels[i] for i in keep]

    def __len__(self):
        return len(self.labels)

    def _query_mode(self):
        L = getattr(self.cfg, "lift", None)
        return str(getattr(L, "query_mode", "lift") or "lift") if L else "lift"

    def reference(self, city, sat_name):
        """Tile resampled to cfg.grid.cell_m, centred in a black (S, S) canvas. Returns (canvas, scale, half_px)."""
        g = self.cfg.grid
        size = int(g.n * self.cfg.reference.scale)
        img = cv2.imread(str(self.root / city / "satellite" / sat_name), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"unreadable tile {city}/{sat_name}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        s = CITY_RES[city] * 640.0 / w0 / float(g.cell_m)        # tile px -> canvas px
        new = (max(1, int(round(w0 * s))), max(1, int(round(h0 * s))))
        tile = cv2.resize(img, new, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), np.uint8)
        x0, y0 = (size - new[0]) // 2, (size - new[1]) // 2
        x1, y1 = min(size, x0 + new[0]), min(size, y0 + new[1])
        canvas[max(0, y0):y1, max(0, x0):x1] = tile[max(0, -y0):max(0, -y0) + (y1 - max(0, y0)),
                                                    max(0, -x0):max(0, -x0) + (x1 - max(0, x0))]
        return canvas, s, (size - 1) / 2.0, (w0, h0)

    def __getitem__(self, i):
        lab = self.labels[i]
        g = self.cfg.grid
        n = int(g.n)
        canvas, s, c, (w0, h0) = self.reference(lab["city"], lab["sat"])
        pano = cv2.imread(str(self.root / lab["city"] / "panorama" / lab["pano"]), cv2.IMREAD_COLOR)
        if pano is None:
            raise RuntimeError(f"unreadable panorama {lab['city']}/{lab['pano']}")
        pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
        # camera position on the canvas: tile centre + offset, scaled tile px -> canvas px
        cx = c + self.col_sign * lab["dx"] * s
        cy = c + self.row_sign * lab["dy"] * s
        o = (n - 1) / 2.0                                        # the camera is the BEV centre
        H = np.array([[1.0, 0.0, cx - o], [0.0, 1.0, cy - o], [0.0, 0.0, 1.0]], np.float32)
        res = CITY_RES[lab["city"]] * 640.0 / w0
        erp = cv2.resize(pano, (self.erp_w, self.erp_h), interpolation=cv2.INTER_AREA)
        out = dict(
            id=f"{lab['city']}/{lab['pano']}", city=lab["city"], year=0, scale=int(self.cfg.reference.scale),
            negative=False,
            erp=torch.from_numpy(erp).permute(2, 0, 1).float().div(255.0)[None],
            R_w2c=torch.from_numpy(R_NORTH.copy())[None], se2=torch.zeros(1, 3),
            ref=torch.from_numpy(canvas).permute(2, 0, 1).float().div(255.0),
            H=torch.from_numpy(H),
            en=torch.tensor([self.col_sign * lab["dx"] * res, -self.row_sign * lab["dy"] * res], dtype=torch.float64),
        )
        if self._query_mode() == "ipm":
            ipm = self.cfg.ipm
            bev, valid = ipm_erp(pano, R_NORTH, self.height, n, float(g.cell_m), float(ipm.blind_radius_m))
            out["bev"] = torch.from_numpy(np.ascontiguousarray(bev)).permute(2, 0, 1).float().div(255.0)
            out["bev_valid"] = torch.from_numpy(valid)
        return out

    def centre_guess_m(self, i):
        """Error of predicting the tile centre (the chance level of this protocol)."""
        lab = self.labels[i]
        res = CITY_RES[lab["city"]]
        return float(np.hypot(lab["dx"], lab["dy"]) * res)


def collate_vigor(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in
           ("erp", "R_w2c", "se2", "H", "en", "ref") + (("bev", "bev_valid") if "bev" in batch[0] else ())}
    out["id"] = [b["id"] for b in batch]
    out["city"] = [b["city"] for b in batch]
    out["year"] = [0] * len(batch)
    out["scale"] = [b["scale"] for b in batch]
    out["negative"] = torch.zeros(len(batch), dtype=torch.bool)
    return out
