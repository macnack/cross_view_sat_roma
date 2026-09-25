"""Fine-tune a query checkpoint (decoder + any query parameters) on VIGOR, known orientation.

  make vigor-train CKPT=checkpoints/05_lift_splat_fixtor_ipm_long_best.pt SPLIT=samearea CITIES="Chicago" STEPS=3000 TAG=chicago
  make vigor-train CKPT= QUERY=ipm SPLIT=samearea CITIES="Chicago" STEPS=30000 TAG=chicago_noposnan   # no warm start
  make vigor-train CKPT= QUERY=erp_depth SPLIT=samearea CITIES="Chicago" STEPS=10000 TAG=chicago_same_10k_erp_depth

Same recipe as train_lift_splat.py (coarse CE over reference cells + certainty + neighbour hinge + pose NLL),
on VigorPairs samples. The training set is the split's train labels for the given cities minus a held-out
--val-frac (FG²'s protocol: the train list shuffled with seed 0, 80/20); validation = the first --val-samples
of the held-out part, scored with the windowed loss and the argmax/pose proxies every --val-every steps (the
RANSAC evaluation is scripts/eval_vigor.py). --val-frac 0 restores the earlier draw from the TEST labels of
--val-cities (checkpoint selection then peeks at the test cities: cross-area runs before 2026-09-25 did this).
Saves `<tag>_best.pt` by validation pose error in the same {"query", "mode", "decoder"} layout every evaluator loads.

--query erp_depth (task 04, Step 2): ERP tokens placed from UniK3D depth (`make loc2-depth ... TRAIN=1` first; labels
whose panorama has no depth file are dropped and counted), plus Loc²'s VCE pose loss (cfg.train.vce_weight, auto = 1
for this mode; --vce-weight overrides). With VCE on, the checkpoint is selected by the validation Procrustes pose
error (vce_pose_m, fixed match draw) instead of the heat-map proxy.
"""
from __future__ import annotations

import argparse
import copy
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
from bevloc.model.coarse import FeatureQueryMatcher, resolve_vce_weight, vce_options  # noqa: E402
from bevloc.model.query import apply_query_cfg, build_query, load_query_state  # noqa: E402


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", default=None, help="warm start; omit to start from the released Sat-RoMa decoder "
                    "(the no-Poznań ablation) with the query mode given by --query")
    ap.add_argument("--query", default="ipm", choices=("lift", "ipm", "hybrid", "erp", "erp_depth"),
                    help="query mode when no --ckpt")
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--split", default="samearea", choices=("samearea", "crossarea"))
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--val-cities", nargs="*", default=None, help="default = the training cities (same-area) or the split's test cities")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (the IPM picture is built on the CPU per sample)")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-samples", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="hold out this fraction of the TRAINING labels for validation (seed 0); 0 = draw from the test labels")
    ap.add_argument("--neighbour-radius", type=int, default=4)
    ap.add_argument("--neighbour-weight", type=float, default=0.5)
    ap.add_argument("--pose-nll-weight", type=float, default=0.5)
    ap.add_argument("--head", action="store_true",
                    help="erp_depth: train the projection head (sets cfg.erp_depth.head; default off = ablation 4b)")
    ap.add_argument("--vce-weight", type=float, default=None,
                    help="Loc² VCE pose loss weight (default cfg.train.vce_weight: auto = 1 for erp_depth, else 0)")
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
    if state:
        apply_query_cfg(cfg, state)                 # e.g. an erp_depth warm start keeps its head setting
    if a.head:
        cfg.erp_depth.head = True
    L.query_mode = mode
    vce_w = resolve_vce_weight(cfg, mode) if a.vce_weight is None else float(a.vce_weight)
    vce_opts = vce_options(cfg)
    tr_cities = a.cities or split_cities(a.split, True)
    va_cities = a.val_cities or (tr_cities if a.split == "samearea" else split_cities(a.split, False))
    tr = VigorPairs(a.root, cfg, cities=tr_cities, split=a.split, train=True)
    if a.val_frac > 0:
        idx = np.random.default_rng(0).permutation(len(tr.labels))
        n_tr = int(len(idx) * (1.0 - a.val_frac))
        va = copy.copy(tr)
        va.labels = [tr.labels[i] for i in idx[n_tr:n_tr + a.val_samples]]
        tr.labels = [tr.labels[i] for i in idx[:n_tr]]
        va_cities = tr_cities
        print(f"train {len(tr)} ({tr_cities})  val {len(va)} of {len(idx) - n_tr} held out from the train list "
              f"(val_frac {a.val_frac})  mode {mode}  batch {a.batch}", flush=True)
    else:
        va = VigorPairs(a.root, cfg, cities=va_cities, split=a.split, train=False, limit=a.val_samples, seed=1)
        print(f"train {len(tr)} ({tr_cities})  val {len(va)} from the TEST labels ({va_cities})  mode {mode}  batch {a.batch}", flush=True)
    n_no_depth = {}
    if tr.depth:                                    # after the split, so the held-out part is the same as without depth
        n_no_depth = dict(train=tr.keep_with_depth(), val=va.keep_with_depth())
        print(f"depth: dropped {n_no_depth['train']} train / {n_no_depth['val']} val labels without a depth file "
              f"-> train {len(tr)}  val {len(va)}", flush=True)
        if not len(tr) or not len(va):
            raise SystemExit("no depth files: run `make loc2-depth SPLIT=... CITIES=... TRAIN=1` first")
    print(f"VCE weight {vce_w}  {vce_opts if vce_w else ''}", flush=True)
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
        E = getattr(cfg, "erp_depth", None)
        lr_q = float(getattr(E, "lr_head", L.lr_lift)) if mode == "erp_depth" and E is not None else L.lr_lift
        groups.append({"params": qp, "lr": lr_q})
        print(f"query {mode}: {sum(p.numel() for p in qp) / 1e6:.2f} M trainable parameters (lr {lr_q})", flush=True)
    dec = [p for p in matcher.model.decoder.parameters() if p.requires_grad]
    groups.append({"params": dec, "lr": cfg.train.lr_decoder})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C.snapshot(cfg, out, dict(tag=a.tag, ckpt=a.ckpt, split=a.split, train_cities=tr_cities, val_cities=va_cities,
                              val_frac=a.val_frac, val_samples=a.val_samples, query_mode=mode,
                              vce_weight=vce_w, no_depth=n_no_depth))
    ckpt_path = C.REPO / "checkpoints" / f"vigor_{a.tag}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log = out / f"train_{a.tag}.csv"
    log.write_text("step,split,ce,top1,cell_err_m,pose_nll,pose_err_m,n,sec,vce_m,vce_pose_m\n")

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
                        pose_nll_weight=a.pose_nll_weight, vce_weight=vce_w, vce_opts=vce_opts)
        loss.backward()
        opt.step()
        with log.open("a") as f:
            f.write(f"{k},train,{st['ce']:.4f},{st['acc']:.4f},{st['cell_err'] * m_per_cell:.2f},"
                    f"{st['pose_nll']:.4f},{st['pose_err'] * m_per_cell:.2f},{st['n']},{time.time() - t0:.1f},"
                    f"{st['vce_m']:.3f},{st['vce_pose_m']:.3f}\n")
        if k == 1 or k % 25 == 0:
            print(f"step {k} TRAIN  CE {st['ce']:.3f}  poseNLL {st['pose_nll']:.3f}  top1 {st['acc']:.1%}  "
                  f"arg {st['cell_err'] * m_per_cell:.1f} m  n {st['n']}"
                  + (f"  VCE {st['vce_m']:.2f} m  procrustes {st['vce_pose_m']:.2f} m" if vce_w else "")
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
        if k % a.val_every == 0 or k == a.steps:
            v = validate(query, matcher, va_loader, cfg, L.min_patch_valid, a.local_radius, device=dev,
                         max_batches=max(1, a.val_samples // a.batch), certainty_weight=0.01,
                         pose_nll_weight=a.pose_nll_weight, vce_weight=vce_w, vce_opts=vce_opts)
            with log.open("a") as f:
                f.write(f"{k},val,{v['ce']:.4f},{v['top1']:.4f},{v['cell_err_m'] / (cfg.grid.cell_m * 16) * m_per_cell:.2f},"
                        f"{v['pose_nll']:.4f},{v['pose_err_m'] / (cfg.grid.cell_m * 16) * m_per_cell:.2f},{v['n']},{time.time() - t0:.1f},"
                        f"{v['vce_m']:.3f},{v['vce_pose_m']:.3f}\n")
            pose_m = v["pose_err_m"] / (cfg.grid.cell_m * 16) * m_per_cell
            print(f"step {k} VAL    CE {v['ce']:.3f}  poseNLL {v['pose_nll']:.3f}  top1 {v['top1']:.1%}  pose {pose_m:.1f} m  n {v['n']}"
                  + (f"  VCE {v['vce_m']:.2f} m  procrustes {v['vce_pose_m']:.2f} m" if vce_w else ""), flush=True)
            score = v["vce_pose_m"] if vce_w and v["vce_pose_m"] == v["vce_pose_m"] else pose_m
            if score < best:
                best = score
                torch.save({"query": query.state_dict(), "mode": mode,
                            "erp_depth": vars(cfg.erp_depth) if mode == "erp_depth" else None,
                            "decoder": {kk: vv for kk, vv in matcher.model.decoder.state_dict().items() if "conv_refiner" not in kk},
                            "step": k, "val": v}, ckpt_path)
                print(f"  best -> {ckpt_path}", flush=True)
            query.train()
    print(f"wrote {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
