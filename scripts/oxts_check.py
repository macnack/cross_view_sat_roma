"""Settle the stored OxTS conventions on data: yaw zero/handedness, roll/pitch, sync.

  python scripts/oxts_check.py

Writes oxts_check.json and oxts_check.png (trajectory + heading arrows + residuals).
The yaw convention is decided by direction of travel, not by oxts/dataformat.txt.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bevloc import config as C
from bevloc.data.oxts import CONVENTIONS, convention_test, read_timestamps, read_track, wrap180

MODALITIES = ("image", "ouster_points", "oxts")


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--out", default="experiments/00_calib")
    a = ap.parse_args()
    cfg = C.load(a.config)
    root = C.REPO / cfg.data.root
    out = Path(a.out)
    C.snapshot(cfg, out)

    tr = read_track(root)
    rep = {"n_frames": len(tr)}

    # --- 1. are roll/pitch actually populated? ------------------------------------------
    rp = np.degrees(tr.rpy[:, :2])
    rep["roll_pitch"] = {"roll_abs_max_deg": float(np.abs(rp[:, 0]).max()),
                         "pitch_abs_max_deg": float(np.abs(rp[:, 1]).max()),
                         "n_unique": [int(len(np.unique(rp[:, 0]))), int(len(np.unique(rp[:, 1])))]}
    dz = np.diff(tr.lla[:, 2])
    ds = np.linalg.norm(np.diff(tr.en, axis=0), axis=1)
    m = ds > 0.5
    slope = np.degrees(np.arctan2(dz[m], ds[m]))
    rep["slope_from_altitude_deg"] = {"p5": float(np.percentile(slope, 5)), "p50": float(np.median(slope)),
                                      "p95": float(np.percentile(slope, 95)),
                                      "alt_range_m": float(np.ptp(tr.lla[:, 2]))}

    # --- 2. yaw convention vs direction of travel ---------------------------------------
    scores, use = convention_test(tr, cfg.oxts.min_speed_mps, cfg.oxts.smooth)
    best = min(scores, key=lambda k: (scores[k]["std_deg"], abs(scores[k]["mean_deg"])))
    rep["yaw"] = {"n_moving_frames": int(use.sum()), "scores": scores, "best": best,
                  "header_claims": "ccw_from_east (oxts/dataformat.txt)"}

    # --- 3. modality sync ---------------------------------------------------------------
    ts = {m: read_timestamps(root / m / "timestamps.txt") for m in MODALITIES}
    rep["sync_ms"] = {m: {"median": float(np.median(ts[m] - ts["oxts"]) * 1e3),
                          "abs_max": float(np.abs(ts[m] - ts["oxts"]).max() * 1e3)}
                      for m in MODALITIES if m != "oxts"}
    rep["oxts_rate_hz"] = float(1.0 / np.median(np.diff(ts["oxts"])))

    (out / "oxts_check.json").write_text(json.dumps(rep, indent=2))
    plot(tr, scores, best, use, out / "oxts_check.png")

    print(f"frames {len(tr)}  oxts {rep['oxts_rate_hz']:.1f} Hz  "
          f"sync vs oxts [ms]: " + ", ".join(f"{m} {v['median']:+.0f} (max |{v['abs_max']:.0f}|)"
                                             for m, v in rep["sync_ms"].items()))
    print(f"roll/pitch: {rep['roll_pitch']['n_unique'][0]}/{rep['roll_pitch']['n_unique'][1]} unique values, "
          f"|max| {rp[:, 0].max():.3f}/{rp[:, 1].max():.3f} deg   "
          f"(altitude implies road slope p5..p95 {rep['slope_from_altitude_deg']['p5']:+.1f}"
          f"..{rep['slope_from_altitude_deg']['p95']:+.1f} deg over {rep['slope_from_altitude_deg']['alt_range_m']:.0f} m)")
    print(f"yaw vs direction of travel on {use.sum()} moving frames:")
    for k, v in sorted(scores.items(), key=lambda kv: kv[1]["std_deg"]):
        print(f"  {k:15s} mean {v['mean_deg']:+8.2f} deg   std {v['std_deg']:7.2f} deg   "
              f"within 10 deg of mean {v['within_10deg']:.1%}{'   <-- best' if k == best else ''}")


def plot(tr, scores, best, use, path):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    e0, n0 = tr.en[0]
    E, N = tr.en[:, 0] - e0, tr.en[:, 1] - n0

    ax[0].plot(E, N, "-", lw=0.8, c="0.6")
    s = slice(None, None, 120)
    b = np.radians(tr.bearing(best)[s])
    ax[0].quiver(E[s], N[s], np.sin(b), np.cos(b), color="tab:red", width=0.004, scale=30,
                 label=f"stored yaw as {best}")
    ax[0].plot(E[0], N[0], "ko", ms=6, label="start")
    ax[0].set(xlabel=f"easting - {e0:.0f} [m]", ylabel=f"northing - {n0:.0f} [m]",
              title="OxTS track, EPSG:27700 (arrows every 12 s)")
    ax[0].axis("equal"); ax[0].legend(); ax[0].grid(alpha=.3)

    trav = tr.travel_bearing(2)
    for k in sorted(CONVENTIONS, key=lambda k: scores[k]["std_deg"]):
        r = wrap180(tr.bearing(k)[use] - trav[use])
        ax[1].hist(r, bins=180, range=(-180, 180), histtype="step", lw=1.6 if k == best else 1.0,
                   label=f"{k}: std {scores[k]['std_deg']:.1f} deg")
    ax[1].set(xlabel="stored yaw - direction of travel [deg]", ylabel="moving frames",
              title="yaw convention test"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    i = np.arange(len(tr))[use]
    ax[2].plot(tr.t[use], wrap180(tr.bearing(best)[use] - trav[use]), ".", ms=1.5)
    ax[2].axhline(scores[best]["mean_deg"], c="tab:red", lw=1,
                  label=f"mean {scores[best]['mean_deg']:+.2f} deg (INS yaw mounting offset)")
    ax[2].set(xlabel="time [s]", ylabel="residual [deg]", ylim=(-30, 30),
              title=f"residual of the best reading ({best}), {len(i)} moving frames")
    ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
