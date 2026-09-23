"""Show several random reference crops / scales for one Mapillary frame.

Left: IPM. Right columns: orthophoto crops at different scales and random SE(2)
offsets (green = Mapillary-pose footprint, approximate). Training sees this kind of variety.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C
from bevloc.bev.grid import BevGrid
from bevloc.data.mapillary import PoznanOrtho, grid_bearing, load_frames, rodrigues, poznan_tiles
from bevloc.data.ortho import Oriented, sample_negative_reference, sample_reference

_spec = importlib.util.spec_from_file_location("ipm_mapillary", C.REPO / "scripts/ipm_mapillary.py")
_ipm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ipm)
ipm = _ipm.ipm

OUT = C.REPO / "experiments/05_lift_splat/viz"
VAL = C.REPO / "data/mapillary/Fixtor/IcRzj0wTLZX874qitxVsQa"


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--index", type=int, default=400)
    ap.add_argument("--year", type=int, default=2025)
    a = ap.parse_args()
    cfg = C.load(a.config)
    L = cfg.lift
    OUT.mkdir(parents=True, exist_ok=True)

    ortho = PoznanOrtho(poznan_tiles(a.year))
    frames = load_frames([VAL], ortho, margin_m=L.margin_m)
    fr = frames[min(a.index, len(frames) - 1)]
    path = Path(fr["_seq"]) / "images" / f"{fr['id']}.jpg"
    erp = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    R = rodrigues(fr["computed_rotation"])
    lon, lat = fr["computed_geometry"]["coordinates"]
    up, en = grid_bearing(lon, lat, fr["computed_compass_angle"])
    query = Oriented(en, up, cfg.grid.n, cfg.grid.cell_m)

    bev = ipm(erp, BevGrid(n=448, cell=0.125), R, 1.7)
    left = cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)
    left = cv2.resize(left, (336, 336))

    scales = list(L.ref_scales)
    rng = np.random.default_rng(0)
    tiles = []
    for scale in scales:
        for k in range(2):
            off = float(rng.uniform(*L.offset_range))
            rot = float(rng.uniform(*L.rot_range))
            ref_o = sample_reference(query, rng, scale=scale, max_offset_frac=off, max_rot_deg=rot)
            ref, _ = ortho.render(ref_o)
            img = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
            # Mapillary-pose query footprint in this crop (approx; not a precise GT box)
            H = np.linalg.inv(ref_o.px_to_world) @ query.px_to_world
            n = cfg.grid.n
            corners = np.array([[0, 0, 1], [n - 1, 0, 1], [n - 1, n - 1, 1], [0, n - 1, 1]], float) @ H.T
            pts = np.ascontiguousarray(np.round(corners[:, :2] / corners[:, 2:3]).astype(np.int32)).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 220, 0), 2, cv2.LINE_AA)
            img = cv2.resize(img, (336, 336))
            bar = np.zeros((28, 336, 3), np.uint8)
            cv2.putText(bar, f"scale {scale}  off {off:.2f}  rot {rot:.0f}",
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, img]))

    # No-match negatives: GT pose off the crop (red box may be outside / clipped)
    neg_tiles = []
    for k in range(4):
        scale = int(scales[k % len(scales)])
        rot = float(rng.uniform(*L.rot_range))
        ref_o = sample_negative_reference(query, rng, scale=scale,
                                          sep_frac=tuple(L.neg_sep_frac), max_rot_deg=rot)
        ref, _ = ortho.render(ref_o)
        img = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
        H = np.linalg.inv(ref_o.px_to_world) @ query.px_to_world
        n = cfg.grid.n
        corners = np.array([[0, 0, 1], [n - 1, 0, 1], [n - 1, n - 1, 1], [0, n - 1, 1]], float) @ H.T
        pts = np.ascontiguousarray(np.round(corners[:, :2] / corners[:, 2:3]).astype(np.int32)).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
        # vehicle centre in ref px
        c = np.array([[(n - 1) / 2, (n - 1) / 2, 1.0]]) @ H.T
        cx, cy = c[0, 0] / c[0, 2], c[0, 1] / c[0, 2]
        cv2.circle(img, (int(round(cx)), int(round(cy))), 6, (0, 0, 255), -1, cv2.LINE_AA)
        img = cv2.resize(img, (336, 336))
        bar = np.zeros((28, 336, 3), np.uint8)
        cv2.putText(bar, f"NEG scale {scale}  GT off map",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA)
        neg_tiles.append(np.vstack([bar, img]))

    # 2 rows of 4 positives + 1 row of 4 negatives
    row1 = np.hstack(tiles[:4])
    row2 = np.hstack(tiles[4:8])
    row3 = np.hstack(neg_tiles)
    right = np.vstack([row1, row2, row3])
    # pad left to match height
    pad = np.zeros((right.shape[0] - left.shape[0], left.shape[1], 3), np.uint8)
    left_col = np.vstack([left, pad])
    head = np.zeros((36, left_col.shape[1] + 8 + right.shape[1], 3), np.uint8)
    cv2.putText(head, f"aug preview  id {fr['id']}  year {a.year}  green=pose approx  red=NEG",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    sheet = np.vstack([head, np.hstack([left_col, np.zeros((right.shape[0], 8, 3), np.uint8), right])])
    out = OUT / "aug_preview.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", out, sheet.shape, flush=True)
    ortho.close()


if __name__ == "__main__":
    main()
