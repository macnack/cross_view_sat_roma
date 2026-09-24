"""Verify VIGOR's (dy, dx) label convention from the geography in the file names (no images needed).

  make vigor-check-labels

Tile and panorama names carry (lat, lon). For each label the panorama's true offset from the tile centre
is computed in metres (north, east) and converted to pixels with the city resolution; then dy and dx are
regressed on north and east. The slopes are the conventions: e.g. dy = +1 x north means "tile centre is
north of the panorama, in pixels". The result decides the sign parameters of bevloc.data.vigor.
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

from bevloc.data.vigor import CITY_RES, read_labels


def latlon_pano(name):
    m = re.match(r".*?,(-?\d+\.\d+),(-?\d+\.\d+),\.jpg$", name)
    return float(m.group(1)), float(m.group(2))


def latlon_sat(name):
    m = re.match(r"satellite_(-?\d+\.\d+)_(-?\d+\.\d+)\.png$", name)
    return float(m.group(1)), float(m.group(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--cities", nargs="*", default=["Chicago"])
    ap.add_argument("--split", default="samearea")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    for city in a.cities:
        rows = read_labels(a.root, [city], a.split, train=False)[: a.n]
        north, east, dy, dx = [], [], [], []
        for r in rows:
            plat, plon = latlon_pano(r["pano"])
            slat, slon = latlon_sat(r["sat"])
            # pano position relative to the tile centre, metres
            north.append((plat - slat) * 111320.0)
            east.append((plon - slon) * 111320.0 * np.cos(np.radians(slat)))
            dy.append(r["dy"])
            dx.append(r["dx"])
        north, east, dy, dx = map(np.array, (north, east, dy, dx))
        res = CITY_RES[city]
        A = np.c_[north, east]
        cy = np.linalg.lstsq(A, dy * res, rcond=None)[0]
        cx = np.linalg.lstsq(A, dx * res, rcond=None)[0]
        resid = np.hypot(dy * res - A @ cy, dx * res - A @ cx)
        print(f"{city}: n {len(rows)}  dy*res = {cy[0]:+.3f} north {cy[1]:+.3f} east   |   "
              f"dx*res = {cx[0]:+.3f} north {cx[1]:+.3f} east   residual median {np.median(resid):.2f} m")


if __name__ == "__main__":
    main()
