"""One sheet: what the local and global search windows are, and what the matcher did inside them.

Green box = where the vehicle's 56 m view actually sits in the aerial photo.
Red box = where the coarse decoder plus RANSAC put it.
The two windows are the same matcher. Only the aerial crop's allowed shift and spin change.
"""
import argparse
import importlib.util

import cv2
import numpy as np
import torch

from bevloc import config as C
from bevloc.data.ortho import Oriented
from bevloc.eval.metrics import pose_errors
from bevloc.match.satroma import SatRoMa
from bevloc.run import context
from bevloc.viz import enhance

_spec = importlib.util.spec_from_file_location("train_roma_rgb", C.REPO / "scripts/train_roma_rgb.py")
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
covered_frames, frame = _train.covered_frames, _train.frame

OUT = C.REPO / "experiments/03_roma_rgb/viz"


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    a = ap.parse_args()
    cfg = C.load(a.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds, calib, erp_valid = context(cfg)
    names, track, ea = covered_frames(cfg, ds)
    OUT.mkdir(parents=True, exist_ok=True)

    name = "0000002110" if "0000002110" in names else names[len(names) // 2]
    miss = "0000002040" if "0000002040" in names else names[len(names) // 3]
    print("frames", name, miss, flush=True)

    matcher = SatRoMa.from_config(cfg, min_valid_frac=0.99)
    frozen = {k: v.detach().cpu().clone() for k, v in matcher.m.model.decoder.state_dict().items()}
    rows = [schematic(cfg, track, ea, name)]
    rows.append(match_row(cfg, ds, calib, erp_valid, track, ea, matcher, frozen, name, "oracle_b",
                          0.30, 55, "frozen", "Global window, frozen decoder"))
    rows.append(match_row(cfg, ds, calib, erp_valid, track, ea, matcher, frozen, name, "oracle_b",
                          0.30, 55, C.REPO / "checkpoints/03_roma_rgb_oracle_b.pt",
                          "Global window, trained coarse decoder"))
    rows.append(match_row(cfg, ds, calib, erp_valid, track, ea, matcher, frozen, name, "oracle_b",
                          0.10, 10, "frozen", "Local window, frozen decoder"))
    rows.append(match_row(cfg, ds, calib, erp_valid, track, ea, matcher, frozen, name, "oracle_b",
                          0.10, 10, C.REPO / "checkpoints/03_roma_rgb_oracle_b_track_neigh.pt",
                          "Local window, trained decoder + neighbour hinge"))
    rows.append(match_row(cfg, ds, calib, erp_valid, track, ea, matcher, frozen, miss, "oracle_b",
                          0.10, 10, C.REPO / "checkpoints/03_roma_rgb_oracle_b_track_neigh.pt",
                          f"Same local matcher, a miss ({miss})"))
    rows.append(match_row(cfg, ds, calib, erp_valid, track, ea, matcher, frozen, name, "ipm_cl",
                          0.10, 10, C.REPO / "checkpoints/03_roma_rgb_ipm_cl_track.pt",
                          "Local window, camera IPM only"))

    w = max(r.shape[1] for r in rows)
    sheet = np.vstack([fit_width(r, w) for r in rows])
    path = OUT / "local_vs_global.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", path, sheet.shape, flush=True)


def schematic(cfg, track, ea, name):
    """The aerial window is always 224 m. Local/global is how far its centre may sit from the car."""
    i = track.names.index(name)
    bearing = float(track.bearing(cfg.oxts.convention, grid=True)[i])
    gsd = 0.5
    span = 420.0
    view = Oriented(tuple(track.en[i]), bearing, int(span / gsd), gsd)
    img, _ = ea.render(view)
    canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H = np.linalg.inv(view.px_to_world)

    def poly(centre_off, rot, half, color, thick):
        b = np.radians(bearing + rot)
        right, up = np.array([np.cos(b), -np.sin(b)]), np.array([np.sin(b), np.cos(b)])
        c = np.asarray(view.centre_en) + centre_off[0] * right + centre_off[1] * up
        corners = [c + sx * half * right + sy * half * up for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
        pts = np.c_[np.stack(corners), np.ones(4)] @ H.T
        px = np.ascontiguousarray(np.round(pts[:, :2]).astype(np.int32)).reshape(-1, 1, 2)
        cv2.polylines(canvas, [px], True, color, thick, cv2.LINE_AA)

    # Allowed places for the centre of the 224 m window. Offset is per axis, so a square.
    poly((0, 0), 0, 22.4, (80, 180, 80), 1)
    poly((0, 0), 0, 67.2, (40, 140, 220), 1)
    poly((0, 0), 0, 112, (255, 255, 255), 2)          # the 224 m window, perfectly centred
    poly((50, 40), 55, 112, (40, 140, 220), 2)        # one global extreme
    poly((16, -12), 10, 112, (80, 180, 80), 2)        # one local extreme
    poly((0, 0), 0, 28, (255, 220, 0), 2)             # the 56 m query

    lines = [
        ("Same matcher. Local and global are two sizes of aerial crop.", (255, 255, 255), 0.65),
        ("White: 224 m photo centred on the car.  Cyan: the 56 m view we match.", (230, 230, 230), 0.5),
        ("Small green square: the photo centre may sit within 22 m. That is local.", (80, 200, 80), 0.5),
        ("Small orange square: the photo centre may sit within 67 m. That is global.", (40, 170, 240), 0.5),
        ("Large green square: one local crop, spun 10 deg. Large orange: one global crop, spun 55 deg.", (210, 210, 210), 0.5),
    ]
    bar = np.zeros((36 + 28 * len(lines), canvas.shape[1], 3), np.uint8)
    for i, (text, color, scale) in enumerate(lines):
        put(bar, text, (16, 32 + 28 * i), scale, color)
    return np.vstack([bar, canvas])


def match_row(cfg, ds, calib, erp_valid, track, ea, matcher, frozen, name, variant, offset, rot, ckpt, title):
    cfg.reference.max_offset_frac = offset
    cfg.reference.max_rot_deg = rot
    load_decoder(matcher, frozen, ckpt)
    bev, ref, H, valid = frame(cfg, ds, calib, erp_valid, track, ea, name, variant)
    m = matcher.match(bev, ref, mask=valid, H_gt=H)
    if m.H is None:
        err = None
    else:
        err = pose_errors(m.H, H, 224, cfg.grid.cell_m)
    print(f"  {title}: pose {None if err is None else round(err['position_m'], 1)} m", flush=True)
    return compose(bev, ref, H, None if err is None else m.H, title, err, name, variant)


def load_decoder(matcher, frozen, ckpt):
    if ckpt == "frozen":
        matcher.m.model.decoder.load_state_dict(frozen)
        return
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    matcher.m.model.decoder.load_state_dict(state["decoder"])


def compose(bev, ref, H_gt, H_est, title, err, name, variant):
    ref_bgr = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
    s = 223
    box(ref_bgr, H_gt, (0, 220, 0), 3)
    if H_est is not None:
        box(ref_bgr, H_est, (0, 0, 255), 2)
    q = cv2.resize(enhance(bev), (448, 448), interpolation=cv2.INTER_NEAREST)
    r = cv2.resize(ref_bgr, (672, 672), interpolation=cv2.INTER_AREA)
    q = cv2.copyMakeBorder(q, 112, 112, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    body = np.hstack([q, r])
    head = np.zeros((78, body.shape[1], 3), np.uint8)
    put(head, f"{title}    frame {name}    {variant}", (12, 28), 0.65, (255, 255, 255))
    if err is None:
        put(head, "RANSAC found no pose. Green = truth.", (12, 58), 0.6, (80, 80, 255))
    else:
        put(head, f"position {err['position_m']:.1f} m    yaw {err['yaw_deg']:.0f} deg    "
                  "green = truth, red = estimate", (12, 58), 0.6, (80, 220, 80))
    return np.vstack([head, body])


def box(img, H, color, thick):
    s = 223
    corners = np.c_[[[0, 0], [s, 0], [s, s], [0, s]], np.ones(4)]
    p = corners @ np.asarray(H, float).T
    p = np.round(p[:, :2] / p[:, 2:3])
    p = np.ascontiguousarray(p.astype(np.int32)).reshape(-1, 1, 2)
    cv2.polylines(img, [p], True, color, thick, cv2.LINE_AA)


def put(img, text, xy, scale, color):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def fit_width(img, w):
    if img.shape[1] == w:
        return img
    out = np.zeros((img.shape[0], w, 3), np.uint8)
    out[:, :img.shape[1]] = img
    return out


if __name__ == "__main__":
    main()
