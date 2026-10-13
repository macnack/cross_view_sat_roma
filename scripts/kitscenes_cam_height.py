"""Camera (panorama centre) height above the road of a KITScenes scene, from the LiDAR ground plane
(`bevloc.data.kitscenes.camera_height_above_ground`): the `pillar.cam_height_m` the pillar lift needs on this rig.

    make kitscenes-cam-height [FRAMES="0 50 99"] [SCENE_DIR=...]   -> experiments/15_kitscenes/cam_height.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bevloc import config as C  # noqa: E402
from bevloc.data import kitscenes as K  # noqa: E402


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--scene-dir", help="scene directory (default: cfg.kitscenes root/split/scene)")
    ap.add_argument("--frames", type=int, nargs="*", help="frame ids (default: every frame)")
    ap.add_argument("--out", default="experiments/15_kitscenes")
    a = ap.parse_args()
    cfg = C.load(a.config)
    k = cfg.kitscenes
    sc = K.KitScene(Path(a.scene_dir) if a.scene_dir else C.REPO / k.root / k.split / k.scene)
    centre = (np.mean([c.centre for c in sc.cams.values()], 0) if isinstance(k.pano_centre, str)
              else np.asarray(k.pano_centre, np.float64))
    frames = a.frames if a.frames else list(sc.frame_ids)
    rows = {}
    for f in frames:
        xyz, _ = sc.lidar(f, k.lidar)
        try:
            h, n = K.camera_height_above_ground(xyz, centre)
        except ValueError as e:
            print(f"frame {f}: {e}")
            continue
        rows[f] = {"height_m": h, "n_ground": n}
    hs = np.array([r["height_m"] for r in rows.values()])
    rep = {"scene": str(sc.path), "pano_centre_ref_m": centre.tolist(), "n_frames": len(rows),
           "height_median_m": float(np.median(hs)), "height_p10_m": float(np.percentile(hs, 10)),
           "height_p90_m": float(np.percentile(hs, 90)), "frames": rows}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cam_height.json").write_text(json.dumps(rep, indent=1))
    print(f"pano centre {centre.round(3).tolist()} m in the reference frame; camera height above the road over "
          f"{len(rows)} frames: median {rep['height_median_m']:.2f} m (p10 {rep['height_p10_m']:.2f}, "
          f"p90 {rep['height_p90_m']:.2f}); wrote {out / 'cam_height.json'}")


if __name__ == "__main__":
    main()
