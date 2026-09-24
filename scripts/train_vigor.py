"""Fine-tune a query checkpoint (decoder + any query parameters) on VIGOR, known orientation.

  make vigor-train CKPT=checkpoints/05_lift_splat_fixtor_ipm_long_best.pt SPLIT=samearea CITIES="Chicago" STEPS=3000 TAG=chicago
  make vigor-train CKPT= QUERY=ipm SPLIT=samearea CITIES="Chicago" STEPS=30000 TAG=chicago_noposnan   # no warm start

Same recipe as train_lift_splat.py (coarse CE over reference cells + certainty + neighbour hinge + pose NLL),
on VigorPairs samples. The training set is the split's train labels for the given cities; validation is a
fixed random subset of the matching test labels, scored with the windowed loss and the argmax/pose proxies
every --val-every steps (the RANSAC evaluation is scripts/eval_vigor.py). Saves `<tag>_best.pt` by validation
pose error in the same {"query", "mode", "decoder"} layout every evaluator loads.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lift_splat import step, validate  # noqa: E402

from bevloc import config as C  # noqa: E402
from bevloc.data.vigor import VigorPairs, collate_vigor, split_cities  # noqa: E402
from bevloc.model.coarse import FeatureQueryMatcher  # noqa: E402
from bevloc.model.query import build_query, load_query_state  # noqa: E402


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", default=None, help="warm start; omit to start from the released Sat-RoMa decoder "
                    "(the no-Poznań ablation) with the query mode given by --query")
    ap.add_argument("--query", default="ipm", choices=("lift", "ipm", "hybrid", "erp"), help="query mode when no --ckpt")
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--split", default="samearea", choices=("samearea", "crossarea"))
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--val-cities", nargs="*", default=None, help="default = the training cities (same-area) or the split's test cities")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (the IPM picture is built on the CPU per sample)")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-samples", type=int, default=200)
    ap.add_argument("--neighbour-radius", type=int, default=4)
    ap.add_argument("--neighbour-weight", type=float, default=0.5)
    ap.add_argument("--pose-nll-weight", type=float, default=0.5)
    ap.add_argument("--local-radius", type=int, default=0, help="CE window in cells (0 = full 56x56 map: the tile IS the search area)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="experiments/09_vigor")
    a = ap.parse_args()
    cfg = C.load(a.config)
    L = cfg.lift
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    state = torch.load(a.ckpt, map_location=dev, weights_only=False) if a.ckpt else None
    mode = state.get("mode", "lift") if state else a.query
    L.query_mode = mode
    tr_cities = a.cities or split_cities(a.split, True)
    va_cities = a.val_cities or (tr_cities if a.split == "samearea" else split_cities(a.split, False))
    tr = VigorPairs(a.root, cfg, cities=tr_cities, split=a.split, train=True)
    va = VigorPairs(a.root, cfg, cities=va_cities, split=a.split, train=False, limit=a.val_samples, seed=1)
    print(f"train {len(tr)} ({tr_cities})  val {len(va)} ({va_cities})  mode {mode}  batch {a.batch}", flush=True)
    tr_loader = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate_vigor,
                           drop_last=True, persistent_workers=a.workers > 0)
    va_loader = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate_vigor)

    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=True)
    for n, p in matcher.model.decoder.named_parameters():
        if "conv_refiner" in n:
            p.requires_grad = False
    if state:
        matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    query = build_query(cfg, mode).to(dev)
    if state:
        load_query_state(query, state)
    else:
        print("no --ckpt: decoder = released Sat-RoMa checkpoint, query untrained", flush=True)
    groups = []
    qp = [p for p in query.parameters() if p.requires_grad]
    if qp:
        groups.append({"params": qp, "lr": L.lr_lift})
    dec = [p for p in matcher.model.decoder.parameters() if p.requires_grad]
    groups.append({"params": dec, "lr": cfg.train.lr_decoder})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C.snapshot(cfg, out, dict(tag=a.tag, ckpt=a.ckpt, split=a.split, train_cities=tr_cities, val_cities=va_cities))
    ckpt_path = C.REPO / "checkpoints" / f"vigor_{a.tag}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log = out / f"train_{a.tag}.csv"
    log.write_text("step,split,ce,top1,cell_err_m,pose_nll,pose_err_m,n,sec\n")

    def forever():
        while True:
            for b in tr_loader:
                yield b
    it = forever()
    m_per_cell = (cfg.grid.n * cfg.reference.scale / 56.0) * cfg.grid.cell_m
    query.train()
    best, t0 = float("inf"), time.time()
    for k in range(1, a.steps + 1):
        batch = {kk: (v.to(dev) if torch.is_tensor(v) else v) for kk, v in next(it).items()}
        opt.zero_grad(set_to_none=True)
        loss, st = step(query, matcher, batch, cfg, L.min_patch_valid, a.local_radius,
                        a.neighbour_radius, a.neighbour_weight, certainty_weight=0.01,
                        pose_nll_weight=a.pose_nll_weight)
        loss.backward()
        opt.step()
        with log.open("a") as f:
            f.write(f"{k},train,{st['ce']:.4f},{st['acc']:.4f},{st['cell_err'] * m_per_cell:.2f},"
                    f"{st['pose_nll']:.4f},{st['pose_err'] * m_per_cell:.2f},{st['n']},{time.time() - t0:.1f}\n")
        if k == 1 or k % 25 == 0:
            print(f"step {k} TRAIN  CE {st['ce']:.3f}  poseNLL {st['pose_nll']:.3f}  top1 {st['acc']:.1%}  "
                  f"arg {st['cell_err'] * m_per_cell:.1f} m  n {st['n']}  ({time.time() - t0:.0f}s)", flush=True)
        if k % a.val_every == 0 or k == a.steps:
            v = validate(query, matcher, va_loader, cfg, L.min_patch_valid, a.local_radius, device=dev,
                         max_batches=max(1, a.val_samples // a.batch), certainty_weight=0.01,
                         pose_nll_weight=a.pose_nll_weight)
            with log.open("a") as f:
                f.write(f"{k},val,{v['ce']:.4f},{v['top1']:.4f},{v['cell_err_m'] / (cfg.grid.cell_m * 16) * m_per_cell:.2f},"
                        f"{v['pose_nll']:.4f},{v['pose_err_m'] / (cfg.grid.cell_m * 16) * m_per_cell:.2f},{v['n']},{time.time() - t0:.1f}\n")
            pose_m = v["pose_err_m"] / (cfg.grid.cell_m * 16) * m_per_cell
            print(f"step {k} VAL    CE {v['ce']:.3f}  poseNLL {v['pose_nll']:.3f}  top1 {v['top1']:.1%}  pose {pose_m:.1f} m  n {v['n']}", flush=True)
            if pose_m < best:
                best = pose_m
                torch.save({"query": query.state_dict(), "mode": mode,
                            "decoder": {kk: vv for kk, vv in matcher.model.decoder.state_dict().items() if "conv_refiner" not in kk},
                            "step": k, "val": v}, ckpt_path)
                print(f"  best -> {ckpt_path}", flush=True)
            query.train()
    print(f"wrote {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
