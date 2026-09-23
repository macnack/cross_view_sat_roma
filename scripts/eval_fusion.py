"""Qualitative + quantitative check of the fusion pipeline on real frames: how does Sat-RoMa's
decoder actually behave when the query comes from our LiDAR-painted BEV features and the
reference is a real map (EA RGB or NLP intensity)? Optionally builds the query from a short
DRIVEN SEQUENCE (odometry-accumulated LiDAR, each frame's own image features) instead of one
frame, to see whether denser LiDAR coverage helps.

  python scripts/eval_fusion.py --checkpoint checkpoints/02_fusion_real_nlp2021_run2_patchmask.pt \
      --ortho data/ortho/durham/2021_nlp/durham_2021_nlp_intensity_1m_uint8.tif --frames 200

  # same query pose, but the BEV is built from 0/2/5/10 m of accumulated driving:
  python scripts/eval_fusion.py --checkpoint ... --ortho ... --frames 200 --seq-dists 0 2 5 10

For each (frame, sequence length): runs BEV encoder -> Sat-RoMa decoder -> RANSAC homography
(the same estimate_homography used at inference in bevloc.match.satroma), then renders
[IPM-RGB BEV, for human orientation only -- NOT the network's actual input]
[reference, green = ground truth, red = predicted, yellow = every query patch's raw top-1 guess]
and reports corner/position/yaw error, plus the raw per-patch argmax median error (meaningful
long before RANSAC succeeds -- see docs/decisions.md).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C, viz
from bevloc.bev.mosaic import se2_from_pose
from bevloc.bev.variants import build_variants
from bevloc.data.calib import Calib
from bevloc.data.dur360 import Dur360Frames
from bevloc.data.ortho import Oriented, OrthoMap, gt_homography, sample_reference
from bevloc.data.oxts import read_track
from bevloc.eval.metrics import pose_errors
from bevloc.match.satroma import PACKAGE_DIR
from bevloc.model.coarse import FeatureQueryMatcher
from bevloc.model.fusion_bev import FusionBEV
from bevloc.run import frame_names

import sys
sys.path.insert(0, PACKAGE_DIR)
from sat_roma.ransac import estimate_homography            # noqa: E402
from sat_roma.ransac.transforms import convert_to_pixel_homography  # noqa: E402


def load(ckpt_path, cfg, calib, dev):
    ck = torch.load(ckpt_path, map_location=dev)
    bev = FusionBEV.from_config(cfg, calib).to(dev).eval()
    bev.load_state_dict(ck["bev"])
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=True).eval()
    matcher.model.decoder.load_state_dict(ck["decoder"])
    print(f"loaded {ckpt_path} (trained to step {ck.get('step', '?')})")
    return bev, matcher


def sequence_frames(ds, names, poses, i0, max_dist, dev, min_range=2.0):
    """Frames from `names`/`poses` (an odometry npz: names, poses 4x4, as in
    scripts/lidar_odometry.py / make_mosaic.py) within `max_dist` m of travel from i0, each as a
    dict ready for FusionBEV.forward_sequence: its own ERP, its own LiDAR points twice over --
    once untouched (xyz_cam, for sampling that frame's own image) and once transformed by the
    relative pose into frame i0's level frame (xyz_bev, for BEV placement). Same forward-looking
    convention as make_mosaic.py (accumulates frames AFTER i0, not before)."""
    j0 = names.index(i0)
    travelled = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poses[j0:, :2, 3], axis=0), axis=1))]
    js = [j0 + k for k, t in enumerate(travelled) if t <= max_dist]
    T0inv = np.linalg.inv(poses[j0])
    out = []
    for j in js:
        n = names[j]
        erp = torch.from_numpy(ds.erp(n)).permute(2, 0, 1).float().to(dev) / 255.0
        scan = ds.scan(n)
        keep = scan.range_m > min_range
        xyz_cam = torch.from_numpy(scan.xyz[keep]).float().to(dev)
        refl = torch.from_numpy(scan.reflectivity[keep]).float().to(dev)
        R, t = se2_from_pose(T0inv @ poses[j])
        xy_bev = xyz_cam[:, :2].cpu().numpy() @ R.T + t
        xyz_bev = torch.from_numpy(np.c_[xy_bev, scan.xyz[keep][:, 2]]).float().to(dev)
        Rt, tt = torch.from_numpy(R).float().to(dev), torch.from_numpy(t).float().to(dev)
        out.append(dict(erp=erp, xyz_cam=xyz_cam, xyz_bev=xyz_bev, refl=refl,
                        to_bev=(lambda p, Rt=Rt, tt=tt: torch.cat([p[:, :2] @ Rt.T + tt, p[:, 2:3]], 1))))
    return out, [names[j] for j in js], travelled[js[-1] - j0] if len(js) > 1 else 0.0


@torch.no_grad()
def match(f_q, matcher, ref, dev, cfg):
    """Returns (H_est or None, argmax_xy (196, 2) reference px of every query patch's raw top-1
    guess -- a graded view of "how close is it", useful long before training is far enough along
    for RANSAC's strict multi-hypothesis consensus to ever succeed (see docs/decisions.md)."""
    gm = matcher(f_q, ref[None].to(dev))[0]
    k = int(round(gm.shape[0] ** 0.5))
    am = gm.reshape(k * k, -1).argmax(0).cpu().numpy()          # per-patch top-1 cell, flat index
    s = ref.shape[-1] / k
    argmax_xy = np.c_[(am % k + 0.5) * s - 0.5, (am // k + 0.5) * s - 0.5]

    r = estimate_homography(gm.float().cpu(), backend="numpy", model="sRT", use_means_for_ransac=False,
                            ransac_method=cv2.RANSAC, ransac_reproj_threshold=cfg.matcher.reproj_cells,
                            ransac_max_iters=5000, ransac_confidence=0.995, refine=False, return_details=True)
    if int(r.pts_A.shape[0]) < 4 or np.array_equal(np.asarray(r.H), np.eye(3)):
        return None, argmax_xy
    H = np.asarray(convert_to_pixel_homography(np.asarray(r.H, dtype=np.float64), in_patch_dim=int(gm.shape[-1]),
                   out_patch_dim=k, crop_res=(224, 224), map_res=(896, 896),
                   cell_convention="center"), dtype=np.float64)
    return H, argmax_xy


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ortho", required=True)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--seq-dists", nargs="+", type=float, default=[0.0],
                    help="build the query from odometry-accumulated frames up to this many metres of "
                        "travel past each --frames entry; 0 = single frame only (default)")
    ap.add_argument("--poses", default="experiments/00_calib/kiss_icp_poses_130_299.npz")
    ap.add_argument("--out", default="experiments/02_fusion/eval")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = C.load(a.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ds, calib = Dur360Frames.from_config(cfg), Calib.from_config(cfg)
    bev, matcher = load(a.checkpoint, cfg, calib, dev)
    erp_valid = torch.from_numpy(np.load(C.REPO / cfg.erp.valid_mask)).float()[None, None].to(dev)
    ortho_map = OrthoMap(a.ortho)
    tr = read_track(ds.root)
    rng = np.random.default_rng(a.seed)

    Z = np.load(a.poses)
    pose_names = [str(x) for x in Z["names"]]
    if max(a.seq_dists) > 0 and not set(frame_names(ds, a.frames)) <= set(pose_names):
        print(f"NOTE: some --frames are outside {a.poses}'s range ({pose_names[0]}-{pose_names[-1]}); "
              f"those will only run at seq-dist 0")

    p14 = (np.arange(14) + 0.5) * 16 - 0.5
    gv, gu = np.meshgrid(p14, p14, indexing="ij")
    results = {}
    for n in frame_names(ds, a.frames):
        i = int(n)
        q = Oriented(tuple(tr.en[i]), float(tr.bearing(cfg.oxts.convention, grid=True)[i]), cfg.grid.n, cfg.grid.cell_m)
        r = sample_reference(q, rng, cfg.reference.scale, cfg.reference.max_offset_frac, cfg.reference.max_rot_deg)
        ref_img, ref_ok = ortho_map.render(r)
        if not ref_ok.any() or (ref_img.sum(-1) > 0).mean() < 0.5:
            print(f"{n}: skipped, reference mostly black at this pose"); continue
        H_gt = gt_homography(q, r)
        gt_xy = (np.c_[gu.ravel(), gv.ravel(), np.ones(196)] @ H_gt.T)
        gt_xy = gt_xy[:, :2] / gt_xy[:, 2:3]
        ref_t = torch.from_numpy(ref_img).permute(2, 0, 1).float().to(dev) / 255.0

        erp0 = ds.erp(n)
        b = build_variants(erp0, ds.points(n), np.load(C.REPO / cfg.erp.valid_mask), calib, cfg, name=n,
                           variants=("ipm_cl",))

        dists = a.seq_dists if n in pose_names else [0.0]
        for d in dists:
            if d == 0.0:
                scan = ds.scan(n)
                keep = scan.range_m > 2.0
                frames = [dict(erp=torch.from_numpy(erp0).permute(2, 0, 1).float().to(dev) / 255.0,
                              xyz_cam=torch.from_numpy(scan.xyz[keep]).float().to(dev),
                              xyz_bev=torch.from_numpy(scan.xyz[keep]).float().to(dev),
                              refl=torch.from_numpy(scan.reflectivity[keep]).float().to(dev))]
                used, actual_d = [n], 0.0
            else:
                frames, used, actual_d = sequence_frames(ds, pose_names, Z["poses"], n, d, dev)
            with torch.no_grad():
                f_q, _, _ = bev.forward_sequence(frames, erp_valid)
            H_est, argmax_xy = match(f_q, matcher, ref_t, dev, cfg)
            err = pose_errors(H_est, H_gt, 224, cfg.grid.cell_m) if H_est is not None else None
            argmax_err_m = float(np.median(np.linalg.norm(argmax_xy - gt_xy, axis=1))) * cfg.grid.cell_m
            results[(n, d)] = (err, argmax_err_m, len(used))
            ransac_msg = "RANSAC FAILED" if err is None else f"RANSAC ok, corner {err['corner_m']:.2f} m"
            print(f"{n} +{d:g}m ({len(used)} frames, actually {actual_d:.1f} m): {ransac_msg}   |   "
                  f"raw argmax median error {argmax_err_m:.1f} m (chance ~ {896 * cfg.grid.cell_m / 2:.0f} m)")
            write_overlay(out / f"{n}_seq{d:g}m.jpg", b.images["ipm_cl"], ref_img, H_gt, H_est, argmax_xy, gt_xy,
                         err, argmax_err_m, a.checkpoint, cfg.grid.cell_m, d, len(used))

    print("\nsummary (mean raw argmax error, RANSAC success rate) by sequence length:")
    for d in a.seq_dists:
        rows = [v for (n, dd), v in results.items() if dd == d]
        if not rows: continue
        ok = sum(1 for err, *_ in rows if err is not None)
        print(f"  +{d:g} m: argmax err {np.mean([r[1] for r in rows]):.1f} m   "
              f"RANSAC {ok}/{len(rows)}   mean frames used {np.mean([r[2] for r in rows]):.1f}")


def write_overlay(path, bev_rgb, ref, H_gt, H_est, argmax_xy, gt_xy, err, argmax_err_m, checkpoint, cell_m,
                  seq_dist, n_frames, zoom=4):
    def txt(im, s, xy, c, sc=0.6):
        cv2.putText(im, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, c, 2, cv2.LINE_AA)

    n = ref.shape[0]
    o = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
    s = 224 - 1
    corners = np.c_[[[0, 0], [s, 0], [s, s], [0, s]], np.ones(4)]
    gt = (corners @ H_gt.T); gt = gt[:, :2] / gt[:, 2:3]
    cv2.polylines(o, [np.round(gt).astype(np.int32)], True, (0, 255, 0), 3)
    if H_est is not None:
        est = (corners @ H_est.T); est = est[:, :2] / est[:, 2:3]
        cv2.polylines(o, [np.round(est).astype(np.int32)], True, (0, 0, 255), 2)
    for (ax, ay), (gx, gy) in zip(argmax_xy, gt_xy):
        cv2.line(o, (int(gx), int(gy)), (int(ax), int(ay)), (0, 120, 0), 1, cv2.LINE_AA)
        cv2.circle(o, (int(ax), int(ay)), 3, (0, 230, 255), -1)

    half = n // (2 * zoom)
    cx, cy = np.clip(np.round(gt[0]).astype(int), half, n - half)
    z = cv2.resize(o[cy - half:cy + half, cx - half:cx + half], (n, n), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(o, (cx - half, cy - half), (cx + half, cy + half), (255, 255, 255), 2)

    q = cv2.resize(viz.enhance(bev_rgb), (n, n), interpolation=cv2.INTER_NEAREST)
    body = np.hstack([q, o, z])
    head = np.zeros((172, body.shape[1], 3), np.uint8)
    txt(head, f"Fusion BEV -> Sat-RoMa, checkpoint {Path(checkpoint).name}   |   sequence +{seq_dist:g} m "
             f"({n_frames} frame{'s' if n_frames != 1 else ''})", (12, 30), (255, 255, 255), 0.7)
    txt(head, "left: IPM-RGB BEV of the QUERY frame only, for human orientation -- NOT the network's actual "
             "input (LiDAR-painted DINOv3 features, possibly from several frames)", (12, 60), (190, 190, 190), 0.48)
    txt(head, "reference: green box = ground truth, red box = RANSAC estimate   |   yellow dots = each of the "
             "196 query patches' own top-1 guess, green line to where it should be", (12, 84), (190, 190, 190), 0.48)
    if err is None:
        txt(head, "RANSAC FAILED (< 4 inlier modes)", (12, 118), (0, 0, 255))
    else:
        txt(head, f"RANSAC: corner error {err['corner_m']:.2f} m   position error {err['position_m']:.2f} m   "
                  f"yaw error {err['yaw_deg']:.2f} deg", (12, 118), (0, 255, 0))
    txt(head, f"raw per-patch argmax: median error {argmax_err_m:.1f} m over 196 patches "
             f"(chance ~ {896 * cell_m / 2:.0f} m)", (12, 148), (0, 230, 255))
    cv2.imwrite(str(path), np.vstack([head, body]), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main()
