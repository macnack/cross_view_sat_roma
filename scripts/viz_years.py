"""The same reference crop in every available Poznań orthophoto year: which years are leaf-off?

  make years-viz INDEX=60          # a manifest_test entry; writes experiments/08_semantic/years_<id>.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C
from bevloc.baselines.common import load_manifest
from bevloc.data.mapillary import PoznanOrtho, sat_data_root
from bevloc.data.ortho import Oriented


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--manifest", default="experiments/06_fg2_bevsplat/manifest_test.json")
    ap.add_argument("--index", type=int, default=60)
    ap.add_argument("--size", type=int, default=448, help="crop edge in px at 0.25 m (448 = 112 m)")
    ap.add_argument("--out", default="experiments/08_semantic")
    a = ap.parse_args()
    man = load_manifest(a.manifest)
    e = [x for x in man["frames"] if int(x["year"]) == 2025][a.index]
    years = sorted({int(p.name[5:9]) for p in sat_data_root().glob("geoportal_poznan_15km2_*/year_*.tif")})
    tiles = []
    for y in years:
        paths = sorted(sat_data_root().glob(f"geoportal_poznan_15km2_*/year_{y}.tif"))
        o = PoznanOrtho(paths)
        crop = Oriented(tuple(e["en"]), float(e["up_bearing_deg"]), a.size, float(e["gsd_m"]))
        img, valid = o.render(crop)
        o.close()
        tile = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.rectangle(tile, (a.size // 2 - 112, a.size // 2 - 112), (a.size // 2 + 112, a.size // 2 + 112), (0, 220, 0), 2)
        bar = np.full((28, a.size, 3), 24, np.uint8)
        cv2.putText(bar, f"{y}   valid {valid.mean():.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile]))
    cols = 3
    rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"years_{e['frame_id']}.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"years {years}; wrote {path}")


if __name__ == "__main__":
    main()
