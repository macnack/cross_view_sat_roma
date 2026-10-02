"""KITScenes ring cameras -> ERP panorama (PanoRoMa query format) + LiDAR overlay + calibration checks.

  python scripts/kitscenes_pano.py [--config configs/default.yaml] [--frames 0 50] [--out experiments/15_kitscenes]

Per frame: <out>/pano_<frame>.jpg (stitched ERP), overlay_<frame>.jpg (LiDAR range drawn on it), and one metrics.json
(coverage, LiDAR-depth-edge alignment curve, pose heading vs travel). Calibration comes from the scene's calib.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C
from bevloc.bev.spherical import lidar_to_erp
from bevloc.data import kitscenes as K


def overlay(erp, pts, centre, min_range, max_range=40.0):
    """LiDAR returns as turbo-coloured dots (red = near) on the panorama."""
    h, w = erp.shape[:2]
    u, v, r = lidar_to_erp(pts, w, h, t_cl=-np.asarray(centre))
    m = (r > min_range) & (r < max_range)
    col = cv2.applyColorMap((255 * (1 - np.clip(r[m] / max_range, 0, 1))).astype(np.uint8)[:, None],
                            cv2.COLORMAP_TURBO)[:, 0]
    out = erp.copy()
    for (x, y), c in zip(np.stack([u[m], v[m]], 1).astype(int), col):
        cv2.circle(out, (int(x) % w, int(y)), 1, (int(c[2]), int(c[1]), int(c[0])), -1)   # col is BGR, out is RGB
    return out


def sheet(erp, ov, valid, cams, title):
    """Readable figure: both panoramas cropped to the valid elevation band, azimuth ticks, ring-camera centres."""
    h, w = erp.shape[:2]
    rows = np.where(valid.any(1))[0]
    r0, r1 = max(rows.min() - 4, 0), min(rows.max() + 5, h)
    panels = []
    for img, name in ((erp, "stitched ERP (6 ring cameras)"), (ov, "+ LiDAR range (red = near, blue = 40 m)")):
        p = img[r0:r1].copy()                                            # a real copy: the labels must not touch `img`
        for az in range(-180, 181, 45):                                  # azimuth to the right of forward
            x = int((az / 360 + 0.5) * w) % w
            cv2.line(p, (x, 0), (x, 12), (255, 255, 255), 2)
            cv2.putText(p, f"{az:+d}", (min(max(x - 18, 2), w - 44), 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        for n, c in cams.items():
            x = int((-c.yaw_deg / 360 + 0.5) * w) % w                    # yaw is ccw (left); azimuth is cw (right)
            cv2.putText(p, n.replace("camera_ring_", ""), (min(max(x - 60, 2), w - 190), p.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(p, name, (10, p.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        panels.append(p)
    out = np.concatenate(panels, 0)
    cv2.putText(out, title, (w // 2 - 330, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--out", default="experiments/15_kitscenes")
    ap.add_argument("--crop", action="store_true", help="also write pano_crop_<frame>.jpg (no black pixel)")
    ap.add_argument("--ego-mask-deg", type=float, help="crop: drop rows steeper than this below the horizon")
    ap.add_argument("--max-elevation-deg", type=float, help="crop: drop rows above this elevation")
    a = ap.parse_args()
    cfg = C.load(a.config)
    k = cfg.kitscenes
    crop = k.crop
    do_crop = a.crop or bool(crop.enabled)
    ego_deg = a.ego_mask_deg if a.ego_mask_deg is not None else crop.ego_mask_deg
    max_el = a.max_elevation_deg if a.max_elevation_deg is not None else crop.max_elevation_deg
    out = Path(a.out)
    C.snapshot(cfg, out)
    sc = K.KitScene(C.REPO / k.root / k.split / k.scene)
    st = K.ErpStitcher(sc.cams, size=k.erp_size, pano_centre=k.pano_centre, scale=k.prefilter_scale,
                       blend_power=k.blend_power)
    rep = {"scene": k.scene, "n_frames": len(sc), "pano_centre_ref_m": st.pano_centre.tolist(),
           "camera_yaw_deg": {n: round(c.yaw_deg, 2) for n, c in sc.cams.items()}, "frames": {}}
    deg_per_px = 360.0 / k.erp_size[0]
    for f in (a.frames if a.frames else k.frames):
        erp, valid = st(sc.images(f))
        xyz, _, ring = sc.lidar(f, k.lidar, with_ring=True)
        al = K.depth_edge_alignment(erp, xyz, ring, st.pano_centre)
        cv2.imwrite(str(out / f"pano_{f:04d}.jpg"), cv2.cvtColor(erp, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        ov = overlay(erp, xyz, st.pano_centre, k.overlay_min_range_m)
        cv2.imwrite(str(out / f"overlay_{f:04d}.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        sh = sheet(erp, ov, valid, sc.cams, f"KITScenes {k.scene[:8]} frame {f}: azimuth (deg, + = right of forward)")
        cv2.imwrite(str(out / f"sheet_{f:04d}.jpg"), cv2.cvtColor(sh, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out / f"mask_{f:04d}.png"), valid.astype(np.uint8) * 255)          # white = image available
        rep["frames"][f] = {"valid_frac": float(valid.mean()), "n_lidar": int(len(xyz)), **al}
        if do_crop:
            ce, cm, info = K.crop_clean(erp, valid, ego_deg, max_el)
            assert cm.all()
            cv2.imwrite(str(out / f"pano_crop_{f:04d}.jpg"), cv2.cvtColor(ce, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            rep["frames"][f]["crop"] = info
            print(f"  crop: rows {info['row0']}..{info['row1']} of {k.erp_size[1]}, elevation "
                  f"{info['elevation_top_deg']:+.1f}..{info['elevation_bottom_deg']:+.1f} deg, {ce.shape[1]}x{ce.shape[0]} px, "
                  f"no black pixel")
        print(f"frame {f}: valid {valid.mean():.3f}, {al['n_edge_points']} depth-edge pts, gradient peaks at "
              f"shift {al['peak_shift_px']} px = {al['peak_shift_px'] * deg_per_px:.2f} deg")
    hd = sc.heading_vs_travel()
    rep["heading_vs_travel_deg"] = {"median": float(np.median(hd)), "std": float(hd.std()), "n": int(len(hd))}
    print("pose heading - travel direction: median %.2f deg, std %.2f deg (n=%d)" % (np.median(hd), hd.std(), len(hd)))
    (out / "metrics.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
