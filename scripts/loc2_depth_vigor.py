"""UniK3D metric depth for VIGOR panoramas, in the layout Loc² reads (<City>/unik3d_depth/<stem>.png, uint16
millimetres clipped at 65 m): the same model, spherical camera and post-processing as
third_party/Loc2/preprocess/infer_depth_vigor.py, restricted to the panoramas a run needs.

  make loc2-depth SPLIT=samearea CITIES="Chicago" LIMIT=3000        # the panoramas of the eval draw (seed 0)
  make loc2-depth SPLIT=samearea CITIES="Chicago" TRAIN=1           # every training label of the split
  make loc2-depth SPLIT=crossarea LIMIT=6000                        # cross-area eval draw (SF + Chicago)

Existing depth files are skipped, so reruns only fill gaps. Needs the HF cache to hold lpiccinelli/unik3d-vitl
(downloaded on first use when HF_HUB_OFFLINE=0).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bevloc import config as C
from bevloc.baselines import loc2 as loc2_wrap
from bevloc.data.vigor import read_labels, split_cities


def draw(labels, limit, seed):
    """The same subset VigorPairs(limit=, seed=) scores, so depth exists exactly for the evaluated panoramas."""
    if not limit:
        return labels
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(labels), size=min(limit, len(labels)), replace=False))
    return [labels[i] for i in keep]


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--split", default="samearea", choices=("samearea", "crossarea"))
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--train", action="store_true", help="the split's training labels instead of the test labels")
    ap.add_argument("--limit", type=int, default=0, help="same draw as eval_vigor.py --limit (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="unik3d-vitl")
    ap.add_argument("--resolution-level", type=int, default=9)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cities = a.cities or split_cities(a.split, a.train)
    labels = draw(read_labels(a.root, cities, a.split, a.train), a.limit, a.seed)
    todo = []
    seen = set()
    for lab in labels:
        key = (lab["city"], lab["pano"])
        if key in seen:
            continue
        seen.add(key)
        if a.overwrite or not loc2_wrap.depth_png_path(a.root, *key).is_file():
            todo.append(key)
    print(f"{len(seen)} panoramas in the draw ({cities}, split {a.split}, {'train' if a.train else 'test'}), "
          f"{len(todo)} without depth", flush=True)
    if not todo:
        print("done", flush=True)
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loc2_wrap.load_unik3d(dev, a.name, a.resolution_level)
    cams = {}
    t0 = time.time()
    max_mm = loc2_wrap.NATIVE["depth_png_max_m"] * 1000.0
    for k, (city, pano) in enumerate(todo, 1):
        img = np.array(Image.open(Path(a.root) / city / "panorama" / pano).convert("RGB"))
        h, w = img.shape[:2]
        if (w, h) not in cams:
            cams[(w, h)] = loc2_wrap.spherical_camera(w, h)
        rgb = torch.from_numpy(img).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(rgb=rgb, camera=cams[(w, h)], normalize=True, rays=None)
        depth = out["points"][0].norm(dim=0).detach().cpu().numpy()          # metres along the ray
        png = np.clip(depth * 1000.0, 0.0, max_mm).astype(np.uint16)
        path = loc2_wrap.depth_png_path(a.root, city, pano)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(png).save(path)
        if k % 100 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)} done  {(time.time() - t0) / k:.2f} s/img  last {city}/{pano} "
                  f"median depth {np.median(depth):.1f} m", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
