"""Sheet of precomputed SegFormer maps: panorama + class overlay + the masked contact-line IPM picture.

  make semantic-sheet CONFIG=configs/ipm_cl3.yaml SEQ=Fixtor/irAsBUKtGCfhPHuMbmOcLd INDEX="100 400 700 1000"

Needs data/mapillary/<seq>/semantic/ (make semantic-precompute). One row per frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C
from bevloc.data.mapillary import MAP_ROOT, MapillaryPairs, PoznanOrtho, load_frames, poznan_tiles

PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                    [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], np.uint8)
NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "t.light", "t.sign", "vegetation", "terrain",
         "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def label(img, text, h=26):
    bar = np.full((h, img.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--seq", default="Fixtor/irAsBUKtGCfhPHuMbmOcLd")
    ap.add_argument("--index", type=int, nargs="+", default=[100, 400, 700, 1000])
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out", default="experiments/08_semantic")
    a = ap.parse_args()
    cfg = C.load(a.config)
    L = cfg.lift
    L.query_mode = "ipm"
    ortho = PoznanOrtho(poznan_tiles(a.year))
    frames = load_frames([MAP_ROOT / a.seq], ortho, margin_m=L.margin_m)
    ds = MapillaryPairs(frames, {a.year: ortho}, cfg, train=False, seed=cfg.train.seed,
                        erp_size=tuple(L.erp_size), years=[a.year])
    rows = []
    for i in a.index:
        fr = ds.frames[i]
        sem = ds._semantic(fr, (896, 1792))
        if sem is None:
            print(f"no semantic map for {fr['id']}", flush=True)
            continue
        s = ds[i]
        erp = cv2.resize((s["erp"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8), (1792, 896), interpolation=cv2.INTER_AREA)
        over = cv2.addWeighted(erp, 0.5, PALETTE[np.minimum(sem, 18)], 0.5, 0)
        present = [(n, float((sem == k).mean())) for k, n in enumerate(NAMES)]
        present = sorted([p for p in present if p[1] > 0.01], key=lambda p: -p[1])[:6]
        h = 300
        pan = cv2.resize(over, (int(1792 * h / 896), h), interpolation=cv2.INTER_AREA)
        bev = (s["bev"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        valid = s["bev_valid"].numpy()
        bev = cv2.resize(bev, (h, h), interpolation=cv2.INTER_NEAREST)
        vm = cv2.resize(valid.astype(np.uint8), (h, h), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(vm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bev, cnts, -1, (255, 255, 0), 1)
        txt = "  ".join(f"{n} {100 * v:.0f}%" for n, v in present)
        row = np.hstack([label(pan, f"id {fr['id']}   {txt}"), label(bev, f"picture ({Path(a.config).stem}), valid {valid.mean():.2f}")])
        rows.append(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"semantic_sheet_{Path(a.config).stem}.jpg"
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {path}", flush=True)
    ortho.close()


if __name__ == "__main__":
    main()
