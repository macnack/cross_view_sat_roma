"""FG² (released VIGOR checkpoint, native sizes, third_party/FG2 unchanged) on the same VIGOR samples that
eval_vigor.py scores, so the two rows are like-for-like.

  make fg2-vigor SPLIT=samearea CITIES="Chicago" LIMIT=3000 TAG=chicago_same

Ours: the city filter, the sample draw (VigorPairs' limit/seed, i.e. the same panoramas eval_vigor.py
scored), the metres conversion (FG²'s own: 630 px tile, city GSD of the 640 px original) and the bootstrap
summary. FG²'s: dataloader, DINOv2 extractor, CVM, both test-time solvers (weighted Procrustes = the
paper's default, and its RANSAC option). Known orientation. When mmcv is not installed, the two mmcv
layers FG² imports come from bevloc.baselines.mmcv_shim.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # FG²'s vigor_eval.py sets this for determinism

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from bevloc import config as C  # noqa: E402
from bevloc.baselines import fg2 as fg2_wrap  # noqa: E402
from bevloc.data.vigor import CITY_RES, VigorPairs, find_label_root, split_cities  # noqa: E402
from bevloc.eval.report import summarise_pose  # noqa: E402


def fg2_modules():
    """FG²'s dataloader and solvers, imported from its own directory (they read ./config.ini at import)."""
    fg2_wrap.ensure_fg2_on_path()
    prev = os.getcwd()
    try:
        os.chdir(fg2_wrap.FG2_ROOT)
        from dataloaders.dataloader_vigor import VIGORDataset  # noqa: WPS433
        from utils.utils import create_metric_grid, e2eProbabilisticProcrustesSolver, weighted_procrustes_2d  # noqa: WPS433
    finally:
        os.chdir(prev)
    ini = configparser.ConfigParser()
    ini.read(fg2_wrap.FG2_ROOT / "config.ini")
    return VIGORDataset, create_metric_grid, e2eProbabilisticProcrustesSolver, weighted_procrustes_2d, ini


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--split", default="samearea", choices=("samearea", "crossarea"))
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="same draw as eval_vigor.py --limit (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=None, help="default cfg.vigor.fg2_batch")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--solver", default="both", choices=("both", "procrustes", "ransac"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="experiments/09_vigor")
    a = ap.parse_args()
    cfg = C.load(a.config)
    batch = a.batch or int(getattr(cfg.vigor, "fg2_batch", 24))
    cities = a.cities or split_cities(a.split, False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    VIGORDataset, create_metric_grid, Solver, procrustes, ini = fg2_modules()

    class CityDataset(VIGORDataset):
        """FG²'s dataset restricted to the cities on disk (its same-area list is all four)."""

        def __init__(self, cities, **kw):
            self._cities = list(cities)
            super().__init__(**kw)

        def _get_city_list(self):
            return self._cities

    root = Path(a.root)
    ds = CityDataset(cities, root=str(root), label_root=str(find_label_root(root)), split=a.split, train=False,
                     random_orientation=False, first_run=False)
    ours = VigorPairs(a.root, cfg, cities=cities, split=a.split, train=False, limit=a.limit, seed=a.seed)
    by_name = {os.path.basename(p): i for i, p in enumerate(ds.grd_list)}
    idx = [by_name[lab["pano"]] for lab in ours.labels]
    names = [lab["pano"] for lab in ours.labels]
    loader = DataLoader(Subset(ds, idx), batch_size=batch, shuffle=False, num_workers=a.workers)
    N = fg2_wrap.NATIVE
    print(f"{len(idx)} samples, split {a.split}, cities {cities}, FG² native ground {N['ground_image_size']} "
          f"satellite {N['satellite_image_size']}, batch {batch}", flush=True)

    model, meta = fg2_wrap.load_cvm(dev, area=a.split, orientation="known_ori")
    dino = fg2_wrap.load_dino(dev)
    V = ini["VIGOR"]
    grid_h, sat_px = float(V["grid_size_h"]), N["satellite_image_size"][0]
    n_samples = int(ini["Model"]["num_samples_matches"])
    sat_grid = create_metric_grid(grid_h, N["sat_bev_res"], batch).to(dev)
    grd_grid = create_metric_grid(grid_h, N["grd_bev_res"], batch).to(dev)
    ransac_args = (int(V["it_RANSAC_procrustes"]), int(V["it_matches"]), int(V["num_samples_matches_ransac"]),
                   int(V["num_corr_2d_2d"]), int(V["num_ref_steps"]), float(V["th_inlier"]), float(V["th_soft_inlier"]))

    def to_m(t, tgt, city):
        """FG² vigor_eval.py: metres in the BEV grid -> pixels of the 630 px tile -> metres with the city GSD."""
        off = (t / grid_h * sat_px - tgt).abs()[:, 0]
        px = torch.sqrt(off[:, 0] ** 2 + off[:, 1] ** 2).cpu().numpy()
        return [float(p * CITY_RES[c] * 640 / sat_px) for p, c in zip(px, city)]

    rows, t_model, t_solve, done = [], 0.0, {"procrustes": 0.0, "ransac": 0.0}, 0
    with torch.no_grad():
        for grd, sat, tgt, _Rgt, city in loader:
            B = grd.shape[0]
            grd, sat, tgt = grd.to(dev), sat.to(dev), tgt.to(dev)
            t0 = time.time()
            score, _, _ = model(dino(grd), dino(sat))
            t_model += time.time() - t0
            _, _n_sat, n_grd = score.shape
            res = {}
            if a.solver in ("both", "procrustes"):
                t0 = time.time()
                flat = score.flatten(1)
                bidx = torch.arange(B, device=dev)[:, None].expand(B, n_samples)
                s = torch.multinomial(flat, n_samples)
                X = sat_grid[:B][bidx, torch.div(s, n_grd, rounding_mode="trunc")]
                Y = grd_grid[:B][bidx, s % n_grd]
                _R, t, _ok = procrustes(X, Y, use_weights=True, use_mask=True, w=flat[bidx, s])
                t_solve["procrustes"] += time.time() - t0
                res["procrustes"] = to_m(t, tgt, city) if t is not None else [None] * B
            if a.solver in ("both", "ransac"):
                t0 = time.time()
                solver = Solver(*ransac_args, sat_grid[:B], grd_grid[:B])
                _R, t, _, _ = solver.estimate_pose(score, return_inliers=False)
                t_solve["ransac"] += time.time() - t0
                res["ransac"] = to_m(t, tgt, city) if t is not None else [None] * B
            centre = to_m(torch.zeros_like(tgt), tgt, city)
            for b in range(B):
                rows.append(dict(pano=names[done + b], city=city[b], centre_guess_m=centre[b],
                                 **{f"{k}_m": v[b] for k, v in res.items()}))
            done += B
            if (done // batch) % 10 == 0 or done == len(idx):
                msg = "  ".join(f"{k} median so far {np.median([r[f'{k}_m'] for r in rows if r[f'{k}_m'] is not None]):.1f} m"
                                for k in res)
                print(f"  {done} done  {msg}", flush=True)

    solvers = [k for k in ("procrustes", "ransac") if f"{k}_m" in rows[0]]
    summary = {}
    for name in ["all"] + sorted({r["city"] for r in rows}):
        sub = rows if name == "all" else [r for r in rows if r["city"] == name]
        summary[name] = {k: summarise_pose([r[f"{k}_m"] for r in sub]) for k in solvers}
        summary[name]["centre_guess"] = summarise_pose([r["centre_guess_m"] for r in sub])
        for k in solvers:
            summary[name][f"mean_{k}_m"] = float(np.mean([min(1e3 if r[f"{k}_m"] is None else r[f"{k}_m"], 1e3) for r in sub]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_fg2_{a.tag}_{a.split}.json"
    path.write_text(json.dumps(dict(meta=dict(method="FG2", checkpoint=meta, split=a.split, cities=cities, n=len(rows),
                                              limit=a.limit, seed=a.seed, orientation="known_ori", city_res=CITY_RES,
                                              sec_per_sample=dict(model=t_model / len(rows),
                                                                  **{k: v / len(rows) for k, v in t_solve.items()})),
                                    frames=rows, summary=summary), indent=2))
    with (out / f"eval_fg2_{a.tag}_{a.split}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name, s in summary.items():
        for k in solvers:
            p = s[k]
            print(f"{name:13s} n {p['n']:5d}  {k:10s} median {p['median_m']:.2f} m {tuple(round(v, 2) for v in p['median_ci'])}  "
                  f"mean {s[f'mean_{k}_m']:.2f} m  R@5 {p['recall@5m']:.2f}  R@10 {p['recall@10m']:.2f}  "
                  f"| centre guess median {s['centre_guess']['median_m']:.2f} m", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
