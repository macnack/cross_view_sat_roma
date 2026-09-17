"""LiDAR odometry (KISS-ICP) over consecutive Dur360BEV scans.

Runs in the `kissslam` conda env (kiss-icp is not in the main env, and that env has no
OpenCV), so this script is deliberately standalone and does not import bevloc.
Scan byte layout: see bevloc.data.dur360.Scan / data/README.md.

Output npz: names (N,), poses (N, 4, 4) = LiDAR frame k -> LiDAR frame of the first scan.
Stand-in for OxTS poses until they are downloaded; afterwards a cross-check of them.
"""
import argparse
from pathlib import Path

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("--root", default="data/dur360bev")
ap.add_argument("--start", type=int, default=130)
ap.add_argument("--end", type=int, default=299)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or f"experiments/00_calib/kiss_icp_poses_{a.start}_{a.end}.npz"

cfg = KISSConfig()
cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = 80.0, 2.5, True
cfg.mapping.voxel_size = 0.5
odo, names, poses = KissICP(cfg), [], []
for k in range(a.start, a.end + 1):
    f = Path(a.root) / "ouster_points/data" / f"{k:010d}.bin"
    if not f.exists() or f.stat().st_size != 128 * 2048 * 36:
        continue
    raw = np.fromfile(f, np.uint8).reshape(-1, 36)
    xyz = raw.view(np.float32).reshape(-1, 9)[:, :3].astype(np.float64)
    t = raw.view(np.uint32).reshape(-1, 9)[:, 4].astype(np.float64)      # ns since sweep start
    keep = np.linalg.norm(xyz, axis=1) > 0.5
    odo.register_frame(xyz[keep], t[keep] / max(t.max(), 1.0))
    names.append(f.stem)
    poses.append(odo.last_pose.copy())
poses = np.array(poses)
Path(out).parent.mkdir(parents=True, exist_ok=True)
np.savez(out, names=np.array(names), poses=poses)
step = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
print(f"{len(names)} scans -> {out}: path {step.sum():.1f} m, median speed {np.median(step) * 10:.1f} m/s, "
      f"z drift {poses[-1, 2, 3] - poses[0, 2, 3]:+.2f} m")
