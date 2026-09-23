"""FG² evaluation on the shared Fixtor×Poznań manifest (task 02).

Step-1 gate: verify the released checkpoint loads. Full zero-shot matching
requires a working DINOv2+mmcv FG² env and is blocked until native VIGOR
reproduction succeeds (see experiments/06_fg2_bevsplat/STATUS.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bevloc import config as C
from bevloc.baselines import fg2 as fg2_wrap
from bevloc.baselines.common import load_manifest


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--manifest", default="experiments/06_fg2_bevsplat/manifest.json")
    ap.add_argument("--out", default="experiments/06_fg2_bevsplat/fg2_zero")
    ap.add_argument("--n", type=int, default=0, help="limit frames (0=all); smoke uses 20")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--area", default="samearea", choices=("samearea", "crossarea"))
    ap.add_argument("--orientation", default="known_ori",
                    choices=("known_ori", "unknown_ori"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    man = load_manifest(a.manifest)
    n = a.n or (20 if a.smoke else 0)
    frames = man["frames"]
    if n:
        # keep year pairs together: take first n unique frame_ids
        seen, keep = set(), []
        for fr in frames:
            if fr["frame_id"] in seen:
                if fr["frame_id"] in {k["frame_id"] for k in keep}:
                    keep.append(fr)
                continue
            if len(seen) >= n:
                break
            seen.add(fr["frame_id"])
            keep.append(fr)
        # also include the matching other-year rows already collected above
        frames = [fr for fr in man["frames"] if fr["frame_id"] in seen]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- Step-1 gate: checkpoint load ---
    try:
        model, meta = fg2_wrap.load_cvm(device, area=a.area, orientation=a.orientation)
    except Exception as e:
        payload = {"status": "blocked", "stage": "load_cvm", "error": str(e)}
        (out / "metrics.json").write_text(json.dumps(payload, indent=2))
        raise SystemExit(f"FG² checkpoint load failed: {e}")

    # Probe: one forward with zeros at native sizes to confirm graph runs.
    H, W = fg2_wrap.NATIVE["ground_image_size"]
    S = fg2_wrap.NATIVE["satellite_image_size"][0]
    try:
        dino = fg2_wrap.load_dino(device)
        with torch.no_grad():
            grd = torch.zeros(1, 3, H, W, device=device)
            sat = torch.zeros(1, 3, S, S, device=device)
            gf = dino(grd)
            sf = dino(sat)
            score, score_orig, _ = model(gf, sf)
        probe = {
            "matching_score_shape": list(score.shape),
            "ok": True,
        }
    except Exception as e:
        probe = {"ok": False, "error": str(e)}
        (out / "metrics.json").write_text(json.dumps({
            "status": "blocked",
            "stage": "forward_probe",
            "checkpoint": meta,
            "probe": probe,
            "reason": (
                "Released FG² weights load, but the native forward pass failed. "
                "Likely missing mmcv / FG² env deps. Do not debug the Fixtor adapter "
                "until native VIGOR eval reproduces (docs/tasks/02_fg2_bevsplat.md §1)."
            ),
        }, indent=2))
        print(json.dumps({"status": "blocked", "probe": probe}, indent=2))
        raise SystemExit(1)

    metrics = {
        "status": "checkpoint_ok_native_eval_blocked",
        "checkpoint": meta,
        "probe": probe,
        "manifest": a.manifest,
        "n_requested": n or len({f["frame_id"] for f in man["frames"]}),
        "blocker": (
            "VIGOR dataset not present on this machine; cannot reproduce a published "
            "native metric. Fixtor zero-shot deferred per task §1 stop rule."
        ),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.yaml").write_text(Path(a.config).read_text())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
