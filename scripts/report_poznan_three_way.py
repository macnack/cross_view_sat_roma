"""Poznań three-way comparison report (decision 2026-09-29): IPM + Sat-RoMa, FG², Loc² (flat-ground and UniK3D
depth) and PanoRoMa (coarse, two-pass) on the same 400 manifest entries (200 held-out IcRzj frames x 2025 / 2024),
one scorer (bevloc.eval.manifest_eval), one table per heading protocol, per year, an error CDF.

  make poznan-three-way          -> experiments/12_poznan_three_way/{REPORT.md, three_way.json, three_way.csv, cdf.png}

Sources (missing ones are listed as "not run"):
  * IPM + Sat-RoMa: experiments/05_lift_splat/eval/eval_ipm_manifest.json (make eval-pose, checkpoint
    05_lift_splat_fixtor_ipm_best.pt, srt solver, peak row). Its reference was rotated by crop_rot, so its heading
    is the noisy prior, like protocol "prior". Heading errors from pose_errors.
  * reported task-02 rows (heading gt, imported read-only from the main checkout into imported/*.csv):
    FG² native, Loc² native flat-ground depth. Their heading column measured |crop_rot| (see bevloc.baselines.fixtor)
    and is dropped.
  * <out>/fg2_<heading>, loc2_<depth>_<heading>, panoroma_<heading>_<ref_up>: this task's runs (frames.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bevloc import config as C
from bevloc.eval import manifest_eval as ME

IPM_JSON = "experiments/05_lift_splat/eval/eval_ipm_manifest.json"


def ipm_rows(path, years):
    d = json.loads(Path(path).read_text())
    rows = []
    for r in d["frames"]:
        if int(r["year"]) not in years:
            continue
        ok = r.get("pose_peak_m") is not None
        rows.append(dict(frame_id=str(r["frame_id"]), year=int(r["year"]), ok=ok, err_m=r.get("pose_peak_m"),
                         err_deg=r.get("yaw_peak_deg"), centre_guess_m=r.get("centre_guess_m")))
    return rows, d["meta"]


def imported_rows(path, mode="native"):
    rows = []
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            if r.get("mode", mode) != mode:
                continue
            ok = r.get("ok") == "True"
            rows.append(dict(frame_id=str(r["frame_id"]), year=int(r["year"]), ok=ok,
                             err_m=float(r["err_m"]) if ok and r.get("err_m") else None, err_deg=None))
    return rows


def run_rows(path, method=None):
    rows = ME.read_rows(path)
    return [r for r in rows if method is None or r["method"] == method]


def fmt(s, heading=True):
    ci = s["median_ci"]
    md = "–" if (not heading or s.get("median_deg") is None) else f"{s['median_deg']:.1f}"
    return (f"{s['n']} | {s['median_m']:.2f} ({ci[0]:.2f}–{ci[1]:.2f}) | {s['mean_capped_m']:.2f} | "
            f"{s['recall@1m']:.2f} | {s['recall@5m']:.2f} | {s['recall@10m']:.2f} | {s['frac_gt_30m']:.2f} | {md}")


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--out", default="experiments/12_poznan_three_way")
    ap.add_argument("--ipm-json", default=IPM_JSON)
    a = ap.parse_args()
    cfg = C.load(a.config)
    years = [int(y) for y in cfg.poznan.years]
    out = Path(a.out)
    table = []                    # (protocol, label, rows, note)

    if Path(a.ipm_json).is_file():
        r, meta = ipm_rows(a.ipm_json, years)
        table.append(("prior", "IPM + Sat-RoMa (existing)", r, f"{meta.get('ckpt')}, solver {meta.get('solver')}"))
    for lab, p in (("FG² native (reported, task 02)", out / "imported" / "fg2_zero_frames.csv"),
                   ("Loc² native flat depth (reported, task 02)", out / "imported" / "loc2_zero_frames.csv")):
        if p.is_file():
            table.append(("gt", lab, imported_rows(p), "heading column dropped (measured |crop_rot|)"))
    runs = [("FG² native", "fg2_{h}", None), ("Loc² native flat depth", "loc2_flat_{h}", None),
            ("Loc² native UniK3D", "loc2_unik3d_{h}", None)]
    for h in ("prior", "gt"):
        for lab, d, m in runs:
            p = out / d.format(h=h) / "frames.csv"
            if p.is_file():
                table.append((h, lab, run_rows(p, m), str(p.parent)))
        for p in sorted(out.glob(f"panoroma_{h}_*/frames.csv")):      # panoroma_<heading>_<ref_up><tag>
            variant = p.parent.name[len(f"panoroma_{h}_"):]
            for m, lab in (("panoroma_coarse", "PanoRoMa coarse"), ("panoroma_two_pass", "PanoRoMa two-pass"),
                           ("panoroma_fine", "PanoRoMa fine (ungated)")):
                table.append((h, f"{lab} ({variant})", run_rows(p, m), str(p.parent)))

    # identity check: every row set on the same (frame, year) pairs
    key_sets = {lab + "/" + prot: {(r["frame_id"], r["year"]) for r in rows} for prot, lab, rows, _ in table}
    ref_keys = next(iter(key_sets.values())) if key_sets else set()
    same = {k: v == ref_keys for k, v in key_sets.items()}

    summary, lines, csv_rows = {}, [], []
    lines += ["# Poznań three-way comparison: IPM + Sat-RoMa, FG², Loc², PanoRoMa", "",
              "Written by `make poznan-three-way` (scripts/report_poznan_three_way.py). Manifest "
              f"`{cfg.poznan.manifest}`, years {years}: 200 held-out frames of route IcRzj × 2 orthophoto years; "
              "reference = the manifest's crop (224 m at 0.25 m/px for IPM; each method's native extent) centred "
              "within ±22 m of the position proxy, zero-shot everywhere. Error = distance to the Mapillary pose "
              "proxy (not survey GT). Median with 95 % bootstrap interval, mean capped at 1 km, recalls over all "
              "entries (failures count as misses), heading error median over the entries with a pose.", "",
              "Heading protocols: **prior** = the panorama is oriented by the manifest's noisy heading (crop_up, "
              "U(−10°, 10°) off the proxy) and the method must recover the rest (what the IPM row always did); "
              "**gt** = oriented by the proxy heading itself (what the reported FG² / Loc² 'native' rows did: their "
              "panorama was rolled by crop_rot_deg, which uses the true heading).", ""]
    for prot in ("prior", "gt"):
        sub = [t for t in table if t[0] == prot]
        if not sub:
            continue
        lines += [f"## Protocol `{prot}`", "",
                  "| method | year | n | median m (95 % CI) | mean m | R@1 | R@5 | R@10 | > 30 m | heading median ° |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _p, lab, rows, note in sub:
            sy = ME.summarise_by_year(rows)
            summary[f"{prot}/{lab}"] = dict(summary=sy, source=note, same_entries=same[lab + "/" + prot])
            for y, s in sy.items():
                lines.append(f"| {lab} | {y} | {fmt(s, heading=any(r.get('err_deg') is not None for r in rows))} |")
                csv_rows.append(dict(protocol=prot, method=lab, year=y, n=s["n"], median_m=s["median_m"],
                                     median_lo=s["median_ci"][0], median_hi=s["median_ci"][1],
                                     mean_capped_m=s["mean_capped_m"], r1=s["recall@1m"], r5=s["recall@5m"],
                                     r10=s["recall@10m"], gt30=s["frac_gt_30m"], heading_median_deg=s["median_deg"]))
        lines.append("")
    not_same = [k for k, v in same.items() if not v]
    lines += ["## Entry identity", "",
              ("Every row is on the same (frame id, year) entries." if not not_same else
               "Rows NOT on the reference entry set: " + ", ".join(not_same)), ""]
    missing = [f"{d.format(h=h)}" for h in ("prior", "gt") for _l, d, _m in runs
               if not (out / d.format(h=h) / "frames.csv").is_file()]
    missing += [f"panoroma_{h}_north" for h in ("prior", "gt") if not (out / f"panoroma_{h}_north" / "frames.csv").is_file()]
    if missing:
        lines += ["## Not run yet", "", ", ".join(missing), ""]
    hc = out / "depth_check" / "height_check.json"
    if hc.is_file():
        s = json.loads(hc.read_text())["summary"]
        lines += ["## UniK3D camera-height check", "",
                  f"Implied camera height from road points {s['band_deg']}° below the horizon, ±{s['azimuth_deg']}° "
                  f"fore/aft ({s['n_with_depth']} panoramas, {s['n_road_only']} restricted to Cityscapes road): median "
                  f"{s['height_median_m']:.2f} m (p10 {s['height_p10_m']:.2f}, p90 {s['height_p90_m']:.2f}); flat-ground "
                  f"proxy of the reported Loc² row {s['loc2_flat_proxy_m']} m, IPM height {s['ipm_height_m']} m.", ""]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, prot in zip(axes, ("prior", "gt")):
            for _p, lab, rows, _n in [t for t in table if t[0] == prot]:
                v = np.sort([np.inf if r.get("err_m") is None else float(r["err_m"]) for r in rows])
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=lab, lw=1.6)
            ax.set_xlim(0, 30)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.set_xlabel("position error (m)")
            ax.set_title(f"heading protocol: {prot}")
            ax.legend(fontsize=7, loc="lower right")
        axes[0].set_ylabel("fraction of entries")
        fig.tight_layout()
        fig.savefig(out / "cdf.png", dpi=130)
        plt.close(fig)
        lines += ["## Error CDF", "", "![CDF](cdf.png)", ""]
    except Exception as e:  # noqa: BLE001
        lines += [f"(CDF plot failed: {e})", ""]

    out.mkdir(parents=True, exist_ok=True)
    (out / "three_way.json").write_text(json.dumps(summary, indent=2))
    with (out / "three_way.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]) if csv_rows else ["protocol"])
        w.writeheader()
        w.writerows(csv_rows)
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
