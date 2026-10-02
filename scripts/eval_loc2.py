"""Loc² zero-shot (released VIGOR known-orientation checkpoint) on the Fixtor × Poznań manifest (task 02 row, ported
from the main checkout; Poznań three-way comparison, decision 2026-09-29).

  make loc2-eval DEPTH=flat HEADING=gt         # the reported row (flat-ground depth proxy at 1.65 m): 4.03 m / R@10 0.83
  make loc2-eval DEPTH=unik3d HEADING=prior    # UniK3D metric depth (make poznan-depth first), the one-protocol row
  make loc2-smoke DEPTH=unik3d                 # 20 frame ids x the years

--depth flat | unik3d: the 1.65 m flat-ground ray proxy of the reported row, or UniK3D depth
(scripts/unik3d_depth_poznan.py; <seq>/unik3d_depth_v2/<id>.png) with the car body (rows steeper than
cfg.poznan.ego_mask_deg below the horizon) masked. The depth is rolled with the panorama. --heading prior|gt:
common.heading_setup. Solver: Loc²'s RANSAC Procrustes (the reported row), --no-ransac for plain Procrustes.
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
from bevloc.baselines import fixtor as fx
from bevloc.baselines import loc2 as loc2_wrap
from bevloc.baselines.common import (HEADINGS, depth_png_for, heading_setup, load_manifest, resolve_panorama,
                                     select_entries, vehicle_yaw)
from bevloc.eval import manifest_eval as ME


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--manifest", default=None, help="default cfg.poznan.manifest")
    ap.add_argument("--years", default=None, help="comma list (default cfg.poznan.years)")
    ap.add_argument("--heading", default=None, choices=HEADINGS, help="default cfg.poznan.heading")
    ap.add_argument("--depth", default="unik3d", choices=("unik3d", "flat"))
    ap.add_argument("--ego-mask-deg", type=float, default=None, help="unik3d: default cfg.poznan.ego_mask_deg")
    ap.add_argument("--no-ransac", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="first N unique frame ids (0 = all)")
    ap.add_argument("--smoke", action="store_true", help="= --n 20")
    ap.add_argument("--area", default="samearea", choices=("samearea", "crossarea"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viz", type=int, default=6)
    ap.add_argument("--out", default=None, help="default experiments/12_poznan_three_way/loc2_<depth>_<heading>")
    a = ap.parse_args()
    cfg = C.load(a.config)
    P = cfg.poznan
    manifest = a.manifest or P.manifest
    years = [int(y) for y in (a.years.split(",") if a.years else P.years)]
    heading = a.heading or P.heading
    ego = float(P.ego_mask_deg if a.ego_mask_deg is None else a.ego_mask_deg)
    ransac = not a.no_ransac
    out = Path(a.out or f"experiments/12_poznan_three_way/loc2_{a.depth}_{heading}")
    entries = select_entries(load_manifest(manifest), years, a.n or (20 if a.smoke else 0))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    hw = tuple(loc2_wrap.NATIVE["ground_image_size"])
    if a.depth == "unik3d":
        missing = [e for e in entries if not depth_png_for(resolve_panorama(e["panorama"])).is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} of {len(entries)} entries have no UniK3D depth (e.g. "
                             f"{depth_png_for(resolve_panorama(missing[0]['panorama']))}): make poznan-depth first")
    print(f"Loc² on {manifest}: {len(entries)} entries, years {years}, heading {heading}, depth {a.depth}"
          f"{f' (ego mask {ego} deg)' if a.depth == 'unik3d' else ''}, ransac {ransac}, device {dev}", flush=True)

    matcher, meta = loc2_wrap.load_matcher(dev, area=a.area, orientation="known_ori")
    dino = loc2_wrap.load_dino(dev)
    flat = fx.loc2_flat_depth(*hw)
    orthos = ME.open_orthos(years)
    rows, t0 = [], time.time()
    try:
        for i, e in enumerate(entries):
            beta, assumed, rho = heading_setup(e, heading, "crop")
            pano = resolve_panorama(e["panorama"])
            try:
                grd, shift, rho_eff = fx.roll_erp(fx.load_erp(pano, hw), rho)
                depth = flat if a.depth == "flat" else fx.loc2_unik3d_depth(depth_png_for(pano), hw, ego)
                depth = torch.roll(depth, shift, dims=-1) if shift else depth
                sat, o, valid = fx.render_sat(orthos[int(e["year"])], e["crop_centre_en"], beta)
                if float(valid.mean()) < 0.5:
                    rows.append(ME.failed_row(e, "loc2", "ortho_coverage"))
                    continue
                pred = fx.loc2_localize(matcher, dino, grd, sat, depth, dev, e["crop_centre_en"], beta, ransac=ransac)
            except Exception as ex:  # noqa: BLE001 — a failed entry is a scored miss
                rows.append(ME.failed_row(e, "loc2", f"{type(ex).__name__}: {ex}"))
                continue
            if not pred["ok"]:
                rows.append(ME.failed_row(e, "loc2", pred["error"]))
                continue
            row = ME.pose_row(e, "loc2", pred["en"], vehicle_yaw(pred["virtual_yaw_deg"], rho_eff),
                              rho_deg=rho_eff, yaw_r_deg=pred["yaw_r_deg"], scale=pred["scale"],
                              n_valid_tokens=pred["n_valid_tokens"], heading=heading, depth=a.depth)
            rows.append(row)
            if i < a.viz:
                A = np.linalg.inv(o.px_to_world)
                uv = lambda en: (A @ np.array([en[0], en[1], 1.0]))[:2]          # noqa: E731
                img = fx.side_by_side(sat, grd, [(uv(e["en"]), (0, 255, 0), cv2.MARKER_CROSS),
                                                 (uv(pred["en"]), (255, 0, 0), cv2.MARKER_TILTED_CROSS)],
                                      f"Loc2 {a.depth} {e['frame_id']} y{e['year']} {heading} err {row['err_m']:.1f} m")
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
        method="Loc2", manifest=manifest, years=years, heading=heading, depth=a.depth,
        ego_mask_deg=ego if a.depth == "unik3d" else None, camera_height_m=fx.LOC2_CAMERA_HEIGHT_M if a.depth == "flat" else None,
        ransac=ransac, n=len(rows), checkpoint=meta, yaw_sign=fx.LOC2_YAW_SIGN, summary=summary), indent=2))
    C.snapshot(cfg, out, dict(manifest=manifest, heading=heading, years=years, depth=a.depth))
    ME.print_summary(f"loc2 {a.depth} {heading}", summary)
    print(f"wrote {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
