"""Camera height implied by the UniK3D depth maps on VIGOR, by the same rule as the Poznań check
(scripts/unik3d_depth_poznan.py): pixels whose ray looks cfg.poznan.height_band_deg below the horizon give an implied
camera height -z of their 3-D point. VIGOR panoramas are gravity-levelled Google Street View images (R = identity);
the driving direction is not known and there are no semantic maps, so every azimuth is used and the Poznań numbers are
recomputed the same way (all azimuths, no road mask) next to it, so the two ratios compare like for like.

  make vigor-depth-height            # -> experiments/12_poznan_three_way/depth_check/vigor_height_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from bevloc import config as C
from bevloc.data.vigor import read_depth_png

import importlib.util

_spec = importlib.util.spec_from_file_location("bevloc_scripts_unik3d_depth_poznan",
                                               Path(__file__).with_name("unik3d_depth_poznan.py"))
PZ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PZ)


def per_image(paths, band, az, R):
    out = []
    for p in paths:
        d = read_depth_png(p)
        h = PZ.implied_heights(d, R, band, az)
        if h.size:
            out.append(float(np.median(h)))
    return np.array(out)


def summary(hs):
    return dict(n=int(hs.size), median_m=float(np.median(hs)), p10_m=float(np.percentile(hs, 10)),
                p90_m=float(np.percentile(hs, 90)))


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--per-city", type=int, default=300)
    ap.add_argument("--poznan-glob", default="data/mapillary/Fixtor/*/unik3d_depth/*.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/12_poznan_three_way/depth_check")
    a = ap.parse_args()
    cfg = C.load(a.config)
    band = tuple(cfg.poznan.height_band_deg)
    rng = random.Random(a.seed)
    R = np.eye(3)                                            # levelled panorama, azimuth-agnostic check
    res = {"band_deg": list(band), "azimuth": "all (az_deg = 180)", "road_mask": False}
    allv = []
    for city in ("Chicago", "NewYork", "SanFrancisco", "Seattle"):
        files = sorted((Path(a.root) / city / "unik3d_depth").glob("*.png"))
        pick = rng.sample(files, min(a.per_city, len(files))) if files else []
        hs = per_image(pick, band, 180.0, R)
        res[city] = summary(hs) if hs.size else None
        allv.append(hs)
        print(f"VIGOR {city}: {res[city]}", flush=True)
    res["VIGOR_all"] = summary(np.concatenate(allv))
    pz = sorted(Path().glob(a.poznan_glob))
    hs = per_image(pz[:: max(1, len(pz) // 300)], band, 180.0, R)
    res["Poznan_same_rule"] = summary(hs) if hs.size else None
    print(f"VIGOR all: {res['VIGOR_all']}\nPoznań, same rule (levelled, all azimuths, no road mask): "
          f"{res['Poznan_same_rule']}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vigor_height_check.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out / 'vigor_height_check.json'}", flush=True)


if __name__ == "__main__":
    main()
