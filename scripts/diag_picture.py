"""Dump the query picture a config produces for a few frames (what the encoder actually sees).

  make ipm-picture CONFIG=configs/ipm_cl.yaml SEQ=Fixtor/irAsBUKtGCfhPHuMbmOcLd INDEX=400

Writes experiments/08_semantic/picture_<config-stem>_<id>.jpg (panorama above, picture below, valid
cells outlined) and prints the valid-cell fraction. Use it to confirm a picture variant (ego cut,
semantic mask, contact feet, mosaic) is really in effect before reading its evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C
from bevloc.data.mapillary import MAP_ROOT, MapillaryPairs, PoznanOrtho, load_frames, poznan_tiles


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--seq", default="Fixtor/irAsBUKtGCfhPHuMbmOcLd")
    ap.add_argument("--index", type=int, nargs="+", default=[400])
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--seq-dists", default="")
    ap.add_argument("--out", default="experiments/08_semantic")
    a = ap.parse_args()
    cfg = C.load(a.config)
    L = cfg.lift
    L.query_mode = "ipm"
    if a.seq_dists.strip():
        L.seq_dists_m = [float(x) for x in a.seq_dists.split(",") if x.strip()]
    ortho = PoznanOrtho(poznan_tiles(a.year))
    frames = load_frames([MAP_ROOT / a.seq], ortho, margin_m=L.margin_m)
    ds = MapillaryPairs(frames, {a.year: ortho}, cfg, train=False, seed=cfg.train.seed,
                        erp_size=tuple(L.erp_size), years=[a.year])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.config).stem
    for i in a.index:
        s = ds[i]
        bev = (s["bev"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        valid = s["bev_valid"].numpy()
        erp = (s["erp"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        big = cv2.resize(bev, (672, 672), interpolation=cv2.INTER_NEAREST)
        vmask = cv2.resize(valid.astype(np.uint8), (672, 672), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(vmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(big, cnts, -1, (0, 255, 255), 1)
        top = cv2.resize(erp, (672, 336), interpolation=cv2.INTER_AREA)
        sheet = np.vstack([top, big])
        cv2.putText(sheet, f"{stem}  id {s['id']}  valid {valid.mean():.2f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        path = out / f"picture_{stem}_{s['id']}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{stem} id {s['id']} valid {valid.mean():.3f} wrote {path}", flush=True)
    ortho.close()


if __name__ == "__main__":
    main()
