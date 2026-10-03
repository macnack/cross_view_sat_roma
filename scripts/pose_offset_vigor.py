"""Constant pose offset of PanoRoMa evaluations on VIGOR, per city: residual = k * gt + b per axis (east, north).

  make vigor-pose-offset [JSONS="experiments/15_panoroma_v2/eval_*.json ..."]

k is the radial scale error (e.g. the original-vs-corrected label scale), b the constant offset in metres. Frames
with error < 5 m only; reads the en_gt / en_coarse / en_fine fields that scripts/eval_vigor.py writes for the two-pass
runs (files without them are skipped). See experiments/17_corrected_labels/OFFSET.md.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--max-err", type=float, default=5.0)
    a = ap.parse_args()
    for f in a.jsons:
        fr = json.load(open(f))["frames"]
        for key in ("en_coarse", "en_fine"):
            rows = [r for r in fr if r.get(key) is not None and r.get("en_gt") is not None]
            if not rows:
                continue
            out = []
            for c in sorted({r["city"] for r in rows}):
                g = np.array([r["en_gt"] for r in rows if r["city"] == c])
                d = np.array([r[key] for r in rows if r["city"] == c]) - g
                m = np.linalg.norm(d, axis=1) < a.max_err
                fit = [np.linalg.lstsq(np.c_[g[m][:, i], np.ones(m.sum())], d[m][:, i], rcond=None)[0] for i in (0, 1)]
                out.append(f"{c} k {fit[0][0]:+.3f}/{fit[1][0]:+.3f} b E{fit[0][1]:+.2f} N{fit[1][1]:+.2f} (n {m.sum()})")
            print(f"{f.split('/')[-1]} {key[3:]}: " + " | ".join(out))


if __name__ == "__main__":
    main()
