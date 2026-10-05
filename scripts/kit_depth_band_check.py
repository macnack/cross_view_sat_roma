"""Compare UniK3D depth of a CROPPED KITScenes dataset (a band) with the depth of the same panoramas uncropped.

  python scripts/kit_depth_band_check.py --full <kit_vigor/name> --band <kit_vigor/name_crop> [--limit 100]

Both datasets come from `kitscenes_to_vigor.py` (same scene, frames, seed), the band one with --crop. The band depth map
covers the full-map rows row0..row1 (Chicago/erp_band.json): its pixel (v, u) is the full map's (row0 + v, u). Prints the
ratio band / full over the shared pixels (median, p10, p90) and the share of pixels within 10 % of each other, overall
and by depression angle below the horizon, so a field-of-view problem of the narrow spherical camera shows up as a bias
in a part of the band.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True)
    ap.add_argument("--band", required=True)
    ap.add_argument("--city", default="Chicago")
    ap.add_argument("--depth-dir", default="unik3d_depth_v2")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    band = json.loads((Path(a.band) / a.city / "erp_band.json").read_text())
    r0, r1 = band["row0"], band["row1"]
    files = sorted((Path(a.band) / a.city / a.depth_dir).glob("*.png"))
    if a.limit:
        files = files[:a.limit]
    ratios, rows = [], []
    for f in files:
        db = np.array(Image.open(f)).astype(np.float32) / 1000.0
        df = np.array(Image.open(Path(a.full) / a.city / a.depth_dir / f.name)).astype(np.float32) / 1000.0
        full = df[r0:r1]
        ok = (db > 0.5) & (full > 0.5) & (db < 35) & (full < 35)
        ratios.append(db[ok] / full[ok])
        rows.append(np.nonzero(ok)[0])
    r = np.concatenate(ratios)
    rr = np.concatenate(rows)
    h = r1 - r0
    el = band["top_deg"] + (band["bottom_deg"] - band["top_deg"]) * (rr + 0.5) / h
    print(f"{len(files)} depth maps, {len(r)} shared pixels: band / full median {np.median(r):.3f}, p10 {np.percentile(r, 10):.3f}, "
          f"p90 {np.percentile(r, 90):.3f}, within 10 %: {np.mean(np.abs(r - 1) < 0.1):.2f}")
    for lo, hi in ((-30, -20), (-20, -10), (-10, 0), (0, 10), (10, 20), (20, 30)):
        s = (el >= lo) & (el < hi)
        if s.sum() > 100:
            print(f"  elevation {lo:+3d}..{hi:+3d} deg: median ratio {np.median(r[s]):.3f}  within 10 %: {np.mean(np.abs(r[s] - 1) < 0.1):.2f}  (n {int(s.sum())})")


if __name__ == "__main__":
    main()
