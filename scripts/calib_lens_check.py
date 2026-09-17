"""Visual checks behind the lens-model and camera<-LiDAR decisions (docs/decisions.md).

  lens : LiDAR on the ERP built with each lens model, zoomed crops (dur360 should sit on objects)
  tz   : near LiDAR points (< 12 m) for several camera heights above the LiDAR
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C, viz
from bevloc.bev.spherical import lidar_to_erp
from bevloc.data.calib import Calib
from bevloc.data.dur360 import Dur360Frames

W, H = 2560, 1280


def crops_with_points(erp_bgr, u, v, r, boxes, zoom, span):
    row = []
    for (x, y, w, h) in boxes:
        c = cv2.resize(erp_bgr[y:y + h, x:x + w], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        m = (u >= x) & (u < x + w) & (v >= y) & (v < y + h)
        ov = viz.overlay_points(c.copy(), u[m], v[m], viz.range_colours(r[m], span), zoom=zoom, origin=(x, y))
        row.append(cv2.addWeighted(ov, 0.6, c, 0.4, 0))
    hmin = min(t.shape[0] for t in row)
    return np.hstack([t[:hmin] for t in row])


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--frame", default="0")
    ap.add_argument("--boxes", nargs="+", default=["100,560,300,150", "500,410,150,225", "1560,560,300,150"],
                    help="x,y,w,h crops in the 2560x1280 ERP")
    ap.add_argument("--tz", nargs="+", type=float, default=[0.0, -0.13, -0.27])
    ap.add_argument("--out", default="experiments/00_calib")
    a = ap.parse_args()
    cfg = C.load(a.config)
    calib, name = Calib.from_config(cfg), f"{int(a.frame):010d}"
    boxes = [tuple(int(t) for t in b.split(",")) for b in a.boxes]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for lens in ("dur360", "equidistant203"):
        ds = Dur360Frames.from_config(cfg, erp_size=(W, H), lens=lens)
        p = ds.points(name, 1.0)
        u, v, r = lidar_to_erp(p, W, H, **calib.kw)
        k = r < 60
        row = crops_with_points(cv2.cvtColor(ds.erp(name), cv2.COLOR_RGB2BGR), u[k], v[k], r[k], boxes, 4, 30)
        cv2.putText(row, f"ERP lens = {lens}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        rows.append(row)
    cv2.imwrite(str(out / f"lens_check_{name}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])

    ds = Dur360Frames.from_config(cfg, erp_size=(W, H))
    erp, p, rows = cv2.cvtColor(ds.erp(name), cv2.COLOR_RGB2BGR), ds.points(name, 1.0), []
    for tz in a.tz:
        u, v, r = lidar_to_erp(p, W, H, R_cl=calib.R_cl, t_cl=[calib.t_cl[0], calib.t_cl[1], tz])
        k = r < 12
        row = crops_with_points(erp, u[k], v[k], r[k], boxes[:1] + boxes[2:], 3, 12)
        cv2.putText(row, f"tz = {tz:+.2f} m (points < 12 m)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        rows.append(row)
    cv2.imwrite(str(out / f"tz_check_{name}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out / f"lens_check_{name}.jpg", "and", out / f"tz_check_{name}.jpg")


if __name__ == "__main__":
    main()
