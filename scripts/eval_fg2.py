"""FG² zero-shot (released VIGOR known-orientation checkpoint) on the Fixtor × Poznań manifest (task 02 row; Poznań
three-way comparison, decision 2026-09-29).

  make fg2-eval HEADING=prior                  # the one-protocol row (panorama oriented by the noisy prior heading)
  make fg2-eval HEADING=gt                     # the reported row's orientation (proxy heading: 4.24 m / R@10 0.85)
  make fg2-smoke                               # 20 frame ids x the years

Adapter: bevloc.baselines.fixtor (native 714 x 1428 panorama, 630 px / 71 m crop at the manifest's crop centre and
crop_up bearing, Procrustes t -> map EN as audited in task 02). --heading prior|gt: see common.heading_setup. Rows are
scored by bevloc.eval.manifest_eval (the scorer of every method in experiments/12_poznan_three_way).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C
from bevloc.baselines import fg2 as fg2_wrap
from bevloc.baselines import fixtor as fx
from bevloc.baselines.common import HEADINGS, heading_setup, load_manifest, resolve_panorama, select_entries, vehicle_yaw
from bevloc.eval import manifest_eval as ME


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--manifest", default=None, help="default cfg.poznan.manifest")
    ap.add_argument("--years", default=None, help="comma list (default cfg.poznan.years)")
    ap.add_argument("--heading", default=None, choices=HEADINGS, help="default cfg.poznan.heading")
    ap.add_argument("--n", type=int, default=0, help="first N unique frame ids (0 = all)")
    ap.add_argument("--smoke", action="store_true", help="= --n 20")
    ap.add_argument("--area", default="samearea", choices=("samearea", "crossarea"))
    ap.add_argument("--seed", type=int, default=0, help="torch seed (Procrustes draws matches with multinomial)")
    ap.add_argument("--viz", type=int, default=6, help="save this many sat | panorama overlays")
    ap.add_argument("--out", default=None, help="default experiments/12_poznan_three_way/fg2_<heading>")
    a = ap.parse_args()
    cfg = C.load(a.config)
    P = cfg.poznan
    manifest = a.manifest or P.manifest
    years = [int(y) for y in (a.years.split(",") if a.years else P.years)]
    heading = a.heading or P.heading
    out = Path(a.out or f"experiments/12_poznan_three_way/fg2_{heading}")
    entries = select_entries(load_manifest(manifest), years, a.n or (20 if a.smoke else 0))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    print(f"FG² on {manifest}: {len(entries)} entries, years {years}, heading {heading}, device {dev}", flush=True)

    model, meta = fg2_wrap.load_cvm(dev, area=a.area, orientation="known_ori")
    dino = fg2_wrap.load_dino(dev)
    orthos = ME.open_orthos(years)
    rows, t0 = [], time.time()
    try:
        for i, e in enumerate(entries):
            beta, assumed, rho = heading_setup(e, heading, "crop")
            try:
                grd, shift, rho_eff = fx.roll_erp(fx.load_erp(resolve_panorama(e["panorama"]),
                                                              fg2_wrap.NATIVE["ground_image_size"]), rho)
                sat, o, valid = fx.render_sat(orthos[int(e["year"])], e["crop_centre_en"], beta)
                if float(valid.mean()) < 0.5:
                    rows.append(ME.failed_row(e, "fg2", "ortho_coverage"))
                    continue
                pred = fx.fg2_localize(model, dino, grd, sat, dev, e["crop_centre_en"], beta)
            except Exception as ex:  # noqa: BLE001 — a failed entry is a scored miss, never a silent drop
                rows.append(ME.failed_row(e, "fg2", f"{type(ex).__name__}: {ex}"))
                continue
            if not pred["ok"]:
                rows.append(ME.failed_row(e, "fg2", pred["error"]))
                continue
            row = ME.pose_row(e, "fg2", pred["en"], vehicle_yaw(pred["virtual_yaw_deg"], rho_eff),
                              rho_deg=rho_eff, yaw_r_deg=pred["yaw_r_deg"], heading=heading)
            rows.append(row)
            if i < a.viz:
                A = np.linalg.inv(o.px_to_world)
                uv = lambda en: (A @ np.array([en[0], en[1], 1.0]))[:2]          # noqa: E731
                img = fx.side_by_side(sat, grd, [(uv(e["en"]), (0, 255, 0), cv2.MARKER_CROSS),
                                                 (uv(pred["en"]), (255, 0, 0), cv2.MARKER_TILTED_CROSS)],
                                      f"FG2 {e['frame_id']} y{e['year']} {heading} err {row['err_m']:.1f} m")
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f"viz_{i:02d}_{e['frame_id']}_y{e['year']}.jpg"), img[..., ::-1])
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  {i + 1}/{len(entries)}  last {row['err_m']:.2f} m  {(time.time() - t0) / (i + 1):.2f} s/entry",
                      flush=True)
    finally:
        for o_ in orthos.values():
            o_.close()
    summary = ME.summarise_by_year(rows)
    ME.write_rows(out, rows)
    (out / "metrics.json").write_text(json.dumps(dict(
        method="FG2", manifest=manifest, years=years, heading=heading, n=len(rows), checkpoint=meta,
        yaw_sign=fx.FG2_YAW_SIGN, summary=summary), indent=2))
    C.snapshot(cfg, out, dict(manifest=manifest, heading=heading, years=years))
    ME.print_summary(f"fg2 {heading}", summary)
    print(f"wrote {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
