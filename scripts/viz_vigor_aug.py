"""Visual check of the VIGOR training augmentation (bevloc.data.augment): one sample, unaugmented + N augmented draws.

  make vigor-aug-viz SPLIT=samearea CITIES=Chicago INDEX=0 N=5 AUG_ARGS="--aug-geometric rot90,flip,shift --aug-rot-deg 10 --aug-photometric 1"

Each column: the reference canvas with every depth-placed panorama token drawn through the sample's H (the training
label; colour = the token's column in the ERP shown below it, i.e. its azimuth), the camera (white ring) and
the canvas north (arrow); under it the ERP with the same colour bar. If the augmentation is label-consistent, the
coloured token points lie on the same scene structures (building faces, road edges) in every column, while the
canvas, the panorama roll and the camera position change. Written to <out>/aug_<city>_<index>.jpg.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C
from bevloc.data import augment as AUG
from bevloc.data.vigor import VigorPairs
from bevloc.model.depth_query import depth_placement


def panel(s, cfg, max_depth, width=448):
    ref = (s["ref"].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[..., ::-1].copy()
    S = ref.shape[0]
    erp = s["erp"][0]
    H_, W_ = erp.shape[-2:]
    n, cell = int(cfg.grid.n), float(cfg.grid.cell_m)
    xy, valid = depth_placement(s["depth"][None], H_ // 16, W_ // 16, s["R_w2c"][None, 0], n, cell, max_depth)
    xy, valid = xy[0].reshape(-1, 2).double().numpy(), valid[0].reshape(-1).numpy()
    Hm = s["H"].numpy().astype(np.float64)
    p = np.c_[xy, np.ones(len(xy))] @ Hm.T
    p = p[:, :2] / p[:, 2:]
    cols = np.tile(np.arange(W_ // 16), H_ // 16)
    hue = (cols * 180 // (W_ // 16)).astype(np.uint8)
    bgr = cv2.cvtColor(np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], -1)[None], cv2.COLOR_HSV2BGR)[0]
    for (u, v), ok, c in zip(p, valid, bgr):
        if ok and 0 <= u < S and 0 <= v < S:
            cv2.circle(ref, (int(round(u)), int(round(v))), 3, tuple(int(x) for x in c), -1)
    o = (n - 1) / 2.0
    cam = Hm @ np.array([o, o, 1.0])
    cam = cam[:2] / cam[2]
    cv2.circle(ref, (int(round(cam[0])), int(round(cam[1]))), 9, (255, 255, 255), 2)
    cv2.arrowedLine(ref, (40, 90), (40, 20), (255, 255, 255), 3, tipLength=0.3)
    cv2.putText(ref, "N", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    ref = cv2.resize(ref, (width, width), interpolation=cv2.INTER_AREA)
    e = (erp.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[..., ::-1]
    e = cv2.resize(e, (width, width // 2), interpolation=cv2.INTER_AREA)
    bar_h = np.repeat((np.arange(width) * 180 // width).astype(np.uint8)[None], 10, 0)
    bar = cv2.cvtColor(np.stack([bar_h, np.full_like(bar_h, 255), np.full_like(bar_h, 255)], -1), cv2.COLOR_HSV2BGR)
    return np.vstack([ref, bar, e])


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--split", default="samearea", choices=("samearea", "crossarea"))
    ap.add_argument("--cities", nargs="*", default=["Chicago"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug-photometric", type=float, default=None)
    ap.add_argument("--aug-geometric", default=None)
    ap.add_argument("--aug-rot-deg", type=float, default=None)
    ap.add_argument("--out", default="experiments/14_regularisation")
    a = ap.parse_args()
    cfg = C.load(a.config)
    cfg.lift.query_mode = "erp_depth"
    ds = VigorPairs(a.root, cfg, cities=a.cities, split=a.split, train=True)
    aug = AUG.from_config(cfg, a.aug_photometric, a.aug_geometric, a.aug_rot_deg)
    if aug is None:
        raise SystemExit("no augmentation given (--aug-geometric / --aug-rot-deg / --aug-photometric)")
    max_depth = float(getattr(cfg.erp_depth, "max_depth_m", 35.0))
    torch.manual_seed(a.seed)
    cols = [panel(ds[a.index], cfg, max_depth)]
    ds.set_aug(aug)
    for _ in range(a.n):
        s = ds[a.index]
        p = panel(s, cfg, max_depth)
        d = s.get("_aug")
        if d is not None:
            txt = f"rot {d['theta']:.1f} flip {int(d['flip'])} shift {d['shift'][0]},{d['shift'][1]}"
            cv2.putText(p, txt, (8, p.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cols.append(p)
    sheet = np.hstack([np.pad(c, ((0, 0), (0, 6), (0, 0))) for c in cols])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lab = ds.labels[a.index]
    f = out / f"aug_{lab['city']}_{a.index}.jpg"
    cv2.imwrite(str(f), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {f} ({sheet.shape[1]}x{sheet.shape[0]}; column 1 = no augmentation) aug {aug.record()}")


if __name__ == "__main__":
    main()
