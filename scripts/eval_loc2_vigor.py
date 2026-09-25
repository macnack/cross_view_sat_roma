"""Loc² (released VIGOR checkpoint, native sizes, third_party/Loc2 unchanged) on the same VIGOR samples that
eval_vigor.py and eval_fg2_vigor.py score, so the rows are like-for-like.

  make loc2-depth SPLIT=samearea CITIES="Chicago" LIMIT=3000          # first: depth for the draw
  make loc2-vigor SPLIT=samearea CITIES="Chicago" LIMIT=3000 TAG=chicago_same

Ours: the city filter, the sample draw (VigorPairs' limit/seed), the bootstrap summary. Loc²'s: dataloader
(panorama + UniK3D depth), DINOv2 extractor, matcher, depth lifting of the ground tokens, scale-aware
Procrustes (paper default) and its RANSAC option. Known orientation. Errors are metric (Loc² solves in metres).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # Loc²'s eval_vigor.py sets this for determinism

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset, default_collate  # noqa: E402

from bevloc import config as C  # noqa: E402
from bevloc.baselines import loc2 as loc2_wrap  # noqa: E402
from bevloc.data.vigor import CITY_RES, VigorPairs, find_label_root, split_cities  # noqa: E402
from bevloc.eval.report import summarise_pose  # noqa: E402


def loc2_modules():
    def imp():
        from dataloaders.dataloader_vigor_with_depth import VIGORDataset  # noqa: WPS433
        from models.utils import e2eProbabilisticProcrustesSolver, weighted_procrustes_2d_with_scale  # noqa: WPS433
        return VIGORDataset, e2eProbabilisticProcrustesSolver, weighted_procrustes_2d_with_scale
    return loc2_wrap._import_from_loc2(imp)


def metric_grid(size_m, res, device):
    """Loc² eval_vigor.py create_metric_grid: aerial point coordinates (1, res*res, 2) in metres."""
    axis = torch.linspace(-size_m / 2, size_m / 2, res, device=device)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x.reshape(-1), y.reshape(-1)), -1)[None]


def spherical_grids(ground_hw, batch, device):
    """Loc² eval_vigor.py create_spherical_grids: ray angles of the ground tokens (batch, 1, h, w)."""
    phi = torch.linspace(0, 2 * np.pi, int(ground_hw[1] / 14), device=device)
    theta = torch.linspace(0, np.pi, int(ground_hw[0] / 14), device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    return theta[None, None].repeat(batch, 1, 1, 1), phi[None, None].repeat(batch, 1, 1, 1)


def safe_collate(batch):
    batch = [s for s in batch if s is not None]
    return default_collate(batch) if batch else None


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--split", default="samearea", choices=("samearea", "crossarea"))
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="same draw as eval_vigor.py --limit (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=None, help="default cfg.vigor.loc2_batch")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--solver", default="both", choices=("both", "procrustes", "ransac"))
    ap.add_argument("--max-depth", type=float, default=None, help="default Loc²'s 35 m")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="experiments/09_vigor")
    a = ap.parse_args()
    cfg = C.load(a.config)
    N = loc2_wrap.NATIVE
    batch = a.batch or int(getattr(cfg.vigor, "loc2_batch", 24))
    max_depth = a.max_depth or N["max_depth_m"]
    cities = a.cities or split_cities(a.split, False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    VIGORDataset, Solver, procrustes = loc2_modules()

    class CityDataset(VIGORDataset):
        def __init__(self, cities, **kw):
            self._cities = list(cities)
            super().__init__(**kw)

        def _get_city_list(self):
            return self._cities

    root = Path(a.root)
    ds = CityDataset(cities, root=str(root), label_root=str(find_label_root(root)), split=a.split, train=False,
                     random_orientation=0)
    ours = VigorPairs(a.root, cfg, cities=cities, split=a.split, train=False, limit=a.limit, seed=a.seed)
    by_name = {os.path.basename(p): i for i, p in enumerate(ds.grd_list)}
    idx = [by_name[lab["pano"]] for lab in ours.labels]
    names = [lab["pano"] for lab in ours.labels]
    missing = [n for lab in ours.labels for n in [lab["pano"]] if not loc2_wrap.depth_png_path(root, lab["city"], n).is_file()]
    print(f"{len(idx)} samples, split {a.split}, cities {cities}, batch {batch}, max depth {max_depth} m, "
          f"{len(missing)} without a depth file (skipped)", flush=True)
    loader = DataLoader(Subset(ds, idx), batch_size=batch, shuffle=False, num_workers=a.workers, collate_fn=safe_collate)

    model, meta = loc2_wrap.load_matcher(dev, area=a.split, orientation="known_ori")
    dino = loc2_wrap.load_dino(dev)
    city_grid = {c: metric_grid(640.0 * CITY_RES[c], N["sat_bev_res"], dev) for c in CITY_RES}
    theta, phi = spherical_grids(N["ground_image_size"], batch, dev)
    n_samples = N["num_samples_matches"]
    ransac_args = (100, 20, 8192, 3, 4, 2.5, 5.0)      # Loc² eval_vigor.py defaults

    rows, t_model, t_solve, done = [], 0.0, {"procrustes": 0.0, "ransac": 0.0}, 0
    with torch.no_grad():
        for data in loader:
            if data is None:
                continue
            grd, depth, sat, tgt, _Rgt, city, resolution = data
            B = grd.shape[0]
            grd, depth, sat, tgt = grd.to(dev), depth.to(dev), sat.to(dev), tgt.to(dev)
            sat_grid = torch.cat([city_grid[c] for c in city], 0)
            t0 = time.time()
            gf, sf = dino(grd), dino(sat)
            d = torch.clip(depth, 0, max_depth)
            d_low = F.interpolate(d, size=gf.shape[-2:], mode="nearest")
            mask = ~(d_low == d.max()).flatten(1)
            th, ph = theta[:B], phi[:B]
            gx = d_low * torch.sin(th) * torch.cos(ph)
            gy = d_low * torch.sin(th) * (-torch.sin(ph))
            grd_xy = torch.cat((gx.flatten(2), gy.flatten(2)), 1).permute(0, 2, 1)
            score, _ = model(gf, sf, mask)
            t_model += time.time() - t0
            _, _n_sat, n_grd = score.shape
            res = {}
            if a.solver in ("both", "procrustes"):
                t0 = time.time()
                flat = score.flatten(1)
                bidx = torch.arange(B, device=dev)[:, None].expand(B, n_samples)
                s = torch.multinomial(flat, n_samples)
                X = sat_grid[bidx, torch.div(s, n_grd, rounding_mode="trunc")]
                Y = grd_xy[bidx, s % n_grd]
                _R, t, _scale, _ = procrustes(Y, X, use_weights=True, use_mask=True, w=flat[bidx, s])
                t_solve["procrustes"] += time.time() - t0
                res["procrustes"] = t
            if a.solver in ("both", "ransac"):
                t0 = time.time()
                _R, t, _scale, _, _ = Solver(*ransac_args, sat_grid, grd_xy).estimate_pose(score, return_inliers=False)
                t_solve["ransac"] += time.time() - t0
                res["ransac"] = t
            for b in range(B):
                gt_m = tgt[b] * resolution[b]
                row = dict(pano=names[done + b], city=city[b], centre_guess_m=float(torch.norm(gt_m, dim=-1).item()))
                for k, t in res.items():
                    ok = t is not None and bool(torch.isfinite(t[b]).all())
                    row[f"{k}_m"] = float(torch.norm(t[b] - gt_m, dim=-1).item()) if ok else None
                rows.append(row)
            done += B
            if (done // batch) % 10 == 0 or done >= len(idx) - len(missing):
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
    path = out / f"eval_loc2_{a.tag}_{a.split}.json"
    path.write_text(json.dumps(dict(meta=dict(method="Loc2", checkpoint=meta, split=a.split, cities=cities, n=len(rows),
                                              limit=a.limit, seed=a.seed, orientation="known_ori", max_depth_m=max_depth,
                                              skipped_no_depth=len(missing), city_res=CITY_RES,
                                              sec_per_sample=dict(model=t_model / max(1, len(rows)),
                                                                  **{k: v / max(1, len(rows)) for k, v in t_solve.items()})),
                                    frames=rows, summary=summary), indent=2))
    with (out / f"eval_loc2_{a.tag}_{a.split}.csv").open("w", newline="") as f:
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
