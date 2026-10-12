"""Selected-height overlay of the pillar query (task 07; FG² Fig. 4): panorama above, the BEV map of the slice each
cell selected below (metres above ground, ego at the centre, forward up), for N frames of a VIGOR-layout root.

    make pillar-height-viz CKPT=checkpoints/vigor_pillar_smoke_B_last.pt VIGOR_ROOT=data/vigor_smoke TAG=smoke_B
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bevloc import config as C  # noqa: E402
from bevloc.data.vigor import VigorPairs, collate_vigor  # noqa: E402
from bevloc.model.coarse import FeatureQueryMatcher  # noqa: E402
from bevloc.model.query import apply_query_cfg, build_query, load_query_state  # noqa: E402


def height_panel(z_m, z_max, cell_px=12):
    """(n, n) metres above ground (nan = no selection) -> BGR panel, FG² Fig. 4 colours (viridis), nan black."""
    n = z_m.shape[0]
    v = np.nan_to_num(z_m / max(z_max, 1e-6), nan=0.0)
    img = cv2.applyColorMap((np.clip(v, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    img[np.isnan(z_m)] = 0
    img = cv2.resize(img, (n * cell_px, n * cell_px), interpolation=cv2.INTER_NEAREST)
    c = n * cell_px // 2
    cv2.drawMarker(img, (c, c), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
    return img


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--split", default="samearea")
    ap.add_argument("--cities", nargs="*", default=["Chicago"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="experiments/22_pillar_lift")
    a = ap.parse_args()
    cfg = C.load(a.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = state.get("mode")
    if mode != "pillar":
        raise SystemExit(f"{a.ckpt} is a {mode!r} checkpoint, not pillar")
    apply_query_cfg(cfg, state)
    cfg.lift.query_mode = mode
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False).to(dev)
    query = build_query(cfg, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    ds = VigorPairs(a.root, cfg, cities=a.cities, split=a.split, train=False, limit=a.n, seed=0)
    rows = []
    z_max = max(query.z)
    for i in range(min(a.n, len(ds))):
        s = ds[i]
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_vigor([s]).items()}
        with torch.no_grad():
            query(batch, matcher)
            _, z_m = query.selected_height()
        pano = (batch["erp"][0, 0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        panel = height_panel(z_m[0].cpu().numpy(), z_max)
        h = panel.shape[0]
        pano = cv2.resize(pano, (int(pano.shape[1] * h / pano.shape[0]), h))
        rows.append(np.concatenate([pano, np.full((h, 8, 3), 255, np.uint8), panel], 1))
    sheet = np.concatenate(rows, 0)
    out = Path(a.out) / f"height_{a.tag}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {out}  (colour = selected slice height above ground, 0 .. {z_max:g} m; black = no image)")


if __name__ == "__main__":
    main()
