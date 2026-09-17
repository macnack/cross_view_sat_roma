"""Estimate the ERP validity mask (ego vehicle + stitching seams) from moving frames.

Writes cfg.erp.valid_mask (.npy bool, True = usable), extents JSON and an overlay image.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C, viz
from bevloc.bev.mask import blind_extents, ego_mask, seam_mask, temporal_stats
from bevloc.run import context


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--out", default="experiments/00_calib")
    ap.add_argument("--show", nargs="*", default=["0", "2000"], help="frames for the overlay image")
    a = ap.parse_args()
    cfg = C.load(a.config)
    ds, calib, _ = context(cfg, need_mask=False)
    m, out = cfg.mask, Path(a.out)
    C.snapshot(cfg, out / "erp_mask")

    names = [n for lo, hi in m.frames.ranges for n in ds.names(("image",), lo, hi, m.frames.stride)]
    std, mean_hp, n = temporal_stats(ds.erp(x) for x in names)
    ego, top = ego_mask(std, mean_hp, m.std_thresh, m.meanhp_thresh, m.min_blob_px, m.margin_px, m.horizon_frac)
    seam = seam_mask(std.shape, m.seam_halfwidth_px)
    valid = ~(ego | seam)
    path = C.REPO / cfg.erp.valid_mask
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, valid)

    H, W = std.shape
    info = dict(erp_size=[W, H], frames_used=n, ego_fraction_of_erp=float(ego.mean()),
                seam_halfwidth_deg=(m.seam_halfwidth_px + 0.5) * 360 / W, camera_height_m=calib.camera_height,
                sectors=blind_extents(top, std.shape, calib.camera_height))
    json.dump(info, open(out / "erp_mask_extents.json", "w"), indent=1)
    rows = []
    for f in a.show:
        e = viz.enhance(ds.erp(f"{int(f):010d}"))
        e[~valid] = (0.35 * e[~valid] + 0.65 * np.array([200, 0, 200])).astype(np.uint8)
        rows.append(e[H // 2 - 40:])
    cv2.imwrite(str(out / "erp_mask_overlay.jpg"), np.vstack(rows))
    print(f"{n} frames -> {path.relative_to(C.REPO)}: ego {ego.mean():.1%} of ERP")
    for k, v in info["sectors"].items():
        print(f"  {k:5s} ground first visible at {v['blind_radius_m']:.1f} m")


if __name__ == "__main__":
    main()
