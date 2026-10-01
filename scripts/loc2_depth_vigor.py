"""UniK3D metric depth for VIGOR panoramas, in the layout Loc² reads (<City>/<DEPTH_DIR>/<stem>.png, uint16
millimetres clipped at 65 m; DEPTH_DIR = unik3d_depth_v2, bevloc.data.vigor): the same model, spherical camera and
post-processing as third_party/Loc2/preprocess/infer_depth_vigor.py, restricted to the panoramas a run needs.
v2: a fresh camera per panorama (the v1 folder unik3d_depth reused one camera, which UniK3D mutates in place, and
has row stripes); each map is checked with depth_stripe_score before it is written (rejects are listed
at the end and in <root>/rejects_<DEPTH_DIR>_<draw>.json, the job then exits non-zero; the rest is written).

  make loc2-depth SPLIT=samearea CITIES="Chicago" LIMIT=3000        # the panoramas of the eval draw (seed 0)
  make loc2-depth SPLIT=samearea CITIES="Chicago" TRAIN=1           # every training label of the split
  make loc2-depth SPLIT=crossarea LIMIT=6000                        # cross-area eval draw (SF + Chicago)

Existing depth files are skipped (writes are atomic: an existing file is complete), so reruns only fill gaps; jobs
over disjoint draws (one per city x train/test) never write the same file. Needs the HF cache to hold
lpiccinelli/unik3d-vitl (downloaded on first use when HF_HUB_OFFLINE=0).
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
from bevloc.data.vigor import DEPTH_DIR, STRIPE_MAX_M, read_labels, split_cities


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
    ap.add_argument("--stripe-max", type=float, default=STRIPE_MAX_M,
                    help="row-stripe guard (m); maps above it are not written but listed (rejects_*.json)")
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
          f"{len(todo)} without depth in {DEPTH_DIR}/", flush=True)
    if not todo:
        print("done", flush=True)
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loc2_wrap.load_unik3d(dev, a.name, a.resolution_level)
    t0 = time.time()
    writer = loc2_wrap.DepthWriter(a.stripe_max)
    for k, (city, pano) in enumerate(todo, 1):
        img = np.array(Image.open(Path(a.root) / city / "panorama" / pano).convert("RGB"))
        depth = loc2_wrap.infer_distance(model, img)                         # metres along the ray, fresh camera
        writer.write(depth, loc2_wrap.depth_png_path(a.root, city, pano), f"{city}/{pano}")
        if k % 100 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)} done  {(time.time() - t0) / k:.2f} s/img  last {city}/{pano} "
                  f"median depth {np.median(depth):.1f} m", flush=True)
    tag = f"{a.split}_{'train' if a.train else 'test'}_{'-'.join(cities)}_limit{a.limit}_seed{a.seed}"
    n_rej = writer.finish(Path(a.root) / f"rejects_{DEPTH_DIR}_{tag}.json")
    if n_rej:
        raise SystemExit(f"{n_rej} panoramas rejected by the stripe guard (listed above); every other map is written")
    print("done", flush=True)


if __name__ == "__main__":
    main()
