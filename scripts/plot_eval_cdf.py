"""Error CDFs of every scored checkpoint on one manifest, from experiments/05_lift_splat/eval/eval_*.json.

  make pose-cdf MANIFEST_STEM=manifest_test YEAR=2025

One axis (position error), one line per checkpoint tag with a fixed hue, the centre-guess chance
curve dashed in grey, direct labels at the right edge, no number soup. Misses (failed RANSAC) are
drawn at the right edge so the curves end at their matched fraction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bevloc import config as C

# fixed hue per tag (categorical, assigned once, never cycled); anything else falls back to grey shades
HUES = {
    "ipm": "#0f766e", "erp_srt": "#0e7490", "erp_se2": "#155e75",
    "years": "#c2410c", "aug": "#ea580c", "multi": "#f59e0b", "seq": "#b45309", "seq_single": "#d97706",
    "pose_nll": "#7c2d12", "hybrid": "#6d28d9", "hybrid_warm": "#7e22ce",
}
FALLBACK = ["#4b5563", "#6b7280", "#9ca3af"]


def cdf(errors, x):
    v = np.asarray([np.inf if (e is None or not np.isfinite(e)) else e for e in errors], float)
    return np.array([(v <= t).mean() for t in x])


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--eval-dir", default="experiments/05_lift_splat/eval")
    ap.add_argument("--manifest-stem", default="manifest_test")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--row", default="peak", help="peak | means")
    ap.add_argument("--xmax", type=float, default=40.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    files = sorted(Path(a.eval_dir).glob(f"eval_*_{a.manifest_stem}.json"))
    x = np.linspace(0, a.xmax, 401)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
    series, chance = [], None
    k = 0
    for f in files:
        d = json.loads(f.read_text())
        tag = f.stem[len("eval_"):-len(a.manifest_stem) - 1]
        rows = [r for r in d["frames"] if int(r["year"]) == a.year]
        if not rows:
            continue
        y = cdf([r[f"pose_{a.row}_m"] for r in rows], x)
        med = float(np.median([np.inf if r[f"pose_{a.row}_m"] is None else r[f"pose_{a.row}_m"] for r in rows]))
        col = HUES.get(tag) or FALLBACK[k % len(FALLBACK)]
        k += tag not in HUES
        series.append((tag, y, med, col))
        if chance is None:
            chance = cdf([r["centre_guess_m"] for r in rows], x)
    series.sort(key=lambda s: s[2])
    for tag, y, med, col in series:
        lw = 2.6 if tag in ("ipm", "erp_srt", "erp_se2") else 1.6
        ax.plot(x, y, color=col, lw=lw, label=f"{tag}  (median {med:.1f} m)")
    if chance is not None:
        ax.plot(x, chance, color="#9ca3af", lw=1.6, ls="--", label="centre guess (chance)")
    # the legend (sorted by median) carries identity; right-edge labels collided where the curves converge
    for t in (5, 10):
        ax.axvline(t, color="#e5e7eb", lw=1, zorder=0)
    ax.set_xlim(0, a.xmax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("position error vs Mapillary pose proxy [m]  (RANSAC pose, %s row)" % a.row)
    ax.set_ylabel("fraction of frames within error")
    ax.set_title(f"{a.manifest_stem} · orthophoto {a.year} · n = {len(rows)} frames · local window ±22 m / ±10°",
                 fontsize=10, color="#374151")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#f3f4f6")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    out = Path(a.out) if a.out else Path(a.eval_dir).parent / f"cdf_{a.manifest_stem}_y{a.year}_{a.row}.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
