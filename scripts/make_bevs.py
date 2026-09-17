"""Render the kick-off BEV variants for some frames (readable panels + raw query PNGs).

  python scripts/make_bevs.py --frames 1000 2000            # or ranges: 200:210, 500:1400:100
"""
import argparse
from pathlib import Path

import cv2

from bevloc import config as C, viz
from bevloc.bev.variants import build_variants
from bevloc.run import context, frame_names

TITLES = {"ipm_raw": "IPM raw", "ipm_cl": "IPM below contact line (primary)",
          "ipm_edge": "IPM + wall edges", "oracle_a": "Oracle A: LiDAR only",
          "oracle_b": "Oracle B: ground IPM + LiDAR objects", "ipm_fp": "IPM + collapsed facades (rejected)"}


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/01_kickoff/bev_preview")
    ap.add_argument("--save-queries", action="store_true", help="also save each variant as <out>/queries/<variant>/<frame>.png + mask")
    a = ap.parse_args()
    cfg = C.load(a.config)
    ds, calib, erp_valid = context(cfg)
    out = Path(a.out)
    C.snapshot(cfg, out)
    for n in frame_names(ds, a.frames):
        # roll/pitch = 0 until OxTS is downloaded and its convention verified (data/README.md)
        b = build_variants(ds.erp(n), ds.points(n), erp_valid, calib, cfg, name=n)
        tiles = [viz.bev_tile(*b.get(k), TITLES[k], cfg.grid.cell_m, edge=b.edge if k == "ipm_edge" else None)
                 for k in TITLES]
        body = viz.grid_of(tiles, 3)
        strip = viz.erp_strip(b.erp, erp_valid, b.contact.row, body.shape[1],
                              f"{n}   yellow = LiDAR contact line, grey = ego/seam mask")
        cv2.imwrite(str(out / f"{n}.jpg"), cv2.vconcat([strip, body]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if a.save_queries:
            for k in b.images:
                d = out / "queries" / k
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{n}.png"), cv2.cvtColor(b.images[k], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(d / f"{n}_valid.png"), b.valid[k].astype("uint8") * 255)
        print(n, f"ground pts {b.ground_fraction:.0%}", {k: f"{v.mean():.0%}" for k, v in b.valid.items()})


if __name__ == "__main__":
    main()
