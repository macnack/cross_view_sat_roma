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

--refine-weight W (cfg.train.refine_weight, default 0 = the refiner frozen and left out of the checkpoint, as before):
RoMa's fine loss on the decoder's conv refiner (its only refiner, stride 16; train_lift_splat.refine_step_loss): the
refiner is unfrozen, trained on detached coarse inputs, and saved in the checkpoint's decoder dict, from which
eval_vigor.py --refine then reads it. Validation logs the refined vs input-warp end-point error (fine_epe_px).

Second-pass decoder (coarse-to-fine, task 04): `--config configs/vigor_cell00625_fine.yaml` trains unchanged except
for the reference, which VigorPairs then builds as a 56 m window at 0.0625 m/px centred on the true position plus a
uniform disc jitter of cfg.vigor.ref_jitter_m (a fresh draw every time in training; the --val-frac validation copy
uses the same distribution with one fixed draw per sample, so the validation curve is comparable across steps).

Numerics (after the 2026-09-26 run went NaN at step 23,425 under float16 autocast): the refiner trains in
cfg.train.refine_precision (float32), its gradient norm is clipped to cfg.train.refine_grad_clip, a batch whose
refined warp / fine loss is non-finite trains the coarse terms only (counted), a step whose total loss or refiner
gradient is non-finite does not update (counted), and more than cfg.train.refine_max_nonfinite such steps in a row
stop the run with an error. A validation with any non-finite logged value is never saved as `_best.pt`;
`_last_finite.pt` is the last checkpoint whose validation was fully finite.
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
from train_lift_splat import refine_options, step, validate  # noqa: E402

from bevloc import config as C  # noqa: E402
from bevloc.data.vigor import VigorPairs, collate_vigor, split_cities  # noqa: E402
from bevloc.model.coarse import FeatureQueryMatcher, resolve_vce_weight, vce_options  # noqa: E402
from bevloc.model.query import apply_query_cfg, build_query, load_query_state  # noqa: E402
from bevloc.model.refine import (  # noqa: E402
    REFINER_KEY, NonFiniteGuard, best_candidate, clip_refiner_grads, decoder_state, refiner_parameters,
    set_refiner_trainable,
)


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
    ap.add_argument("--pose-nll-weight", type=float, default=None,
                    help="heat-map pose NLL weight (default 0.5; 0 for erp_depth, whose placed tokens do not "
                         "target the camera cell the NLL rewards)")
    ap.add_argument("--head", action="store_true",
                    help="erp_depth: train the projection head (sets cfg.erp_depth.head; default off = ablation 4b)")
    ap.add_argument("--vce-weight", type=float, default=None,
                    help="Loc² VCE pose loss weight (default cfg.train.vce_weight: auto = 1 for erp_depth, else 0)")
    ap.add_argument("--refine-weight", type=float, default=None,
                    help="RoMa fine loss on the decoder's conv refiner (default cfg.train.refine_weight, 0 = off: "
                         "refiner frozen and not saved)")
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
    if a.pose_nll_weight is None:
        a.pose_nll_weight = 0.0 if mode == "erp_depth" else 0.5
    refine_w = float(getattr(cfg.train, "refine_weight", 0.0) or 0.0) if a.refine_weight is None else float(a.refine_weight)
    refine_opts = refine_options(cfg)
    grad_clip = float(getattr(cfg.train, "refine_grad_clip", 1.0))
    guard = NonFiniteGuard(int(getattr(cfg.train, "refine_max_nonfinite", 20)))
    placed = mode == "erp_depth"          # heat-map "pose" = argmax of the token votes, not a pose for placed tokens
    train_meta = dict(pose_nll_weight=a.pose_nll_weight, vce_weight=vce_w, vce_opts=vce_opts if vce_w else None,
                      neighbour_radius=a.neighbour_radius, neighbour_weight=a.neighbour_weight, steps=a.steps,
                      batch=a.batch, local_radius=a.local_radius, refine_weight=refine_w,
                      refine_opts=refine_opts if refine_w else None,
                      grid=dict(cell_m=float(cfg.grid.cell_m),
                                ref_window_m=getattr(getattr(cfg, "vigor", None), "ref_window_m", None),
                                ref_jitter_m=float(getattr(getattr(cfg, "vigor", None), "ref_jitter_m", 0.0) or 0.0)),
                      refine_grad_clip=grad_clip if refine_w else None,
                      refine_max_nonfinite=guard.max if refine_w else None)
    print(f"pose NLL weight {a.pose_nll_weight}  refine weight {refine_w}" + (f" {refine_opts}" if refine_w else ""),
          flush=True)
    tr_cities = a.cities or split_cities(a.split, True)
    va_cities = a.val_cities or (tr_cities if a.split == "samearea" else split_cities(a.split, False))
    tr = VigorPairs(a.root, cfg, cities=tr_cities, split=a.split, train=True)
    if a.val_frac > 0:
        idx = np.random.default_rng(0).permutation(len(tr.labels))
        n_tr = int(len(idx) * (1.0 - a.val_frac))
        va = copy.copy(tr)
        va.jitter_seed = 0          # window mode: the training jitter distribution, one fixed draw per val sample
        va.labels = [tr.labels[i] for i in idx[n_tr:n_tr + a.val_samples]]
        tr.labels = [tr.labels[i] for i in idx[:n_tr]]
        va_cities = tr_cities
        print(f"train {len(tr)} ({tr_cities})  val {len(va)} of {len(idx) - n_tr} held out from the train list "
              f"(val_frac {a.val_frac})  mode {mode}  batch {a.batch}", flush=True)
    else:
        va = VigorPairs(a.root, cfg, cities=va_cities, split=a.split, train=False, limit=a.val_samples, seed=1)
        print(f"train {len(tr)} ({tr_cities})  val {len(va)} from the TEST labels ({va_cities})  mode {mode}  batch {a.batch}", flush=True)
    if tr.ref_window_m is not None:
        print(f"reference: {tr.ref_window_m} m window at {cfg.grid.cell_m} m/px centred on the true position + a "
              f"uniform disc jitter of radius {tr.ref_jitter_m} m (train: fresh per draw; val: fixed per sample, "
              f"jitter_seed {va.jitter_seed})", flush=True)
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
    n_ref = set_refiner_trainable(matcher.model.decoder, refine_w > 0)
    if refine_w:
        print(f"conv refiner unfrozen: {n_ref / 1e6:.2f} M parameters (fine loss weight {refine_w})", flush=True)
    if state:
        matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    # a warm start that carries a trained refiner keeps it in every checkpoint, even with the fine loss off
    # (it is then frozen at those weights): silently dropping trained weights would revert eval to the released ones
    keep_refiner = refine_w > 0 or bool(state and any(REFINER_KEY in k for k in state["decoder"]))
    if keep_refiner and not refine_w:
        print("warm start carries a trained conv refiner: kept (frozen) and saved in the checkpoints", flush=True)
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
                              vce_weight=vce_w, no_depth=n_no_depth, pose_nll_weight=a.pose_nll_weight,
                              refine_weight=refine_w))
    ckpt_path = C.REPO / "checkpoints" / f"vigor_{a.tag}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log = out / f"train_{a.tag}.csv"
    log.write_text("step,split,ce,top1,cell_err_m,pose_nll,pose_err_m,n,sec,vce_m,vce_pose_m,fine_epe_px,fine_epe_in_px,"
                   "fine_skipped\n")
    val_keys = (["ce", "top1"] + (["pose_nll", "pose_err_m"] if a.pose_nll_weight else [])
                + (["vce_m", "vce_pose_m"] if vce_w else []) + (["fine_epe_px", "fine_epe_in_px"] if refine_w else []))
    ref_params = refiner_parameters(matcher.model.decoder) if refine_w else []
    last_finite_path = ckpt_path.with_name(f"vigor_{a.tag}_last_finite.pt")

    def ckpt_state(k, v):
        return {"query": query.state_dict(), "mode": mode,
                "erp_depth": vars(cfg.erp_depth) if mode == "erp_depth" else None,
                "train": train_meta,
                "decoder": decoder_state(matcher.model.decoder, include_refiner=keep_refiner),
                "step": k, "val": v}

    def forever():
        while True:
            for b in tr_loader:
                yield b
    it = forever()
    m_per_cell = (cfg.grid.n * cfg.reference.scale / 56.0) * cfg.grid.cell_m
    query.train()
    best, best_step, t0 = float("inf"), None, time.time()
    v = None
    n_skip_step = 0
    for k in range(1, a.steps + 1):
        batch = {kk: (v.to(dev) if torch.is_tensor(v) else v) for kk, v in next(it).items()}
        opt.zero_grad(set_to_none=True)
        loss, st = step(query, matcher, batch, cfg, L.min_patch_valid, a.local_radius,
                        a.neighbour_radius, a.neighbour_weight, certainty_weight=0.01,
                        pose_nll_weight=a.pose_nll_weight, vce_weight=vce_w, vce_opts=vce_opts,
                        refine_weight=refine_w, refine_opts=refine_opts)
        bad = not bool(torch.isfinite(loss.detach()))
        if bad:                                        # no update at all from a non-finite total loss
            n_skip_step += 1
            print(f"step {k}: non-finite loss, no update ({n_skip_step} so far)", flush=True)
        else:
            loss.backward()
            if ref_params:
                _, g_ok = clip_refiner_grads(ref_params, grad_clip)
                if not g_ok:
                    print(f"step {k}: non-finite refiner gradient, refiner not updated", flush=True)
                bad = not g_ok
            opt.step()
        bad = bad or bool(st.get("fine_skipped", 0))
        guard.update(bad, k)                           # raises after refine_max_nonfinite bad steps in a row
        with log.open("a") as f:
            f.write(f"{k},train,{st['ce']:.4f},{st['acc']:.4f},{st['cell_err'] * m_per_cell:.2f},"
                    f"{st['pose_nll']:.4f},{st['pose_err'] * m_per_cell:.2f},{st['n']},{time.time() - t0:.1f},"
                    f"{st['vce_m']:.3f},{st['vce_pose_m']:.3f},{st['fine_epe_px']:.3f},{st['fine_epe_in_px']:.3f},"
                    f"{st.get('fine_skipped', 0)}\n")
        if k == 1 or k % 25 == 0:
            print(f"step {k} TRAIN  CE {st['ce']:.3f}  poseNLL {st['pose_nll']:.3f}  top1 {st['acc']:.1%}  "
                  f"arg {st['cell_err'] * m_per_cell:.1f} m  n {st['n']}"
                  + (f"  VCE {st['vce_m']:.2f} m  procrustes {st['vce_pose_m']:.2f} m" if vce_w else "")
                  + (f"  fine EPE {st['fine_epe_px']:.2f} px (input {st['fine_epe_in_px']:.2f})"
                     f"  non-finite steps {guard.total}" if refine_w else "")
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
        if k % a.val_every == 0 or k == a.steps:
            v = validate(query, matcher, va_loader, cfg, L.min_patch_valid, a.local_radius, device=dev,
                         max_batches=max(1, a.val_samples // a.batch), certainty_weight=0.01,
                         pose_nll_weight=a.pose_nll_weight, vce_weight=vce_w, vce_opts=vce_opts,
                         refine_weight=refine_w, refine_opts=refine_opts)
            with log.open("a") as f:
                f.write(f"{k},val,{v['ce']:.4f},{v['top1']:.4f},{v['cell_err_m'] / (cfg.grid.cell_m * 16) * m_per_cell:.2f},"
                        f"{v['pose_nll']:.4f},{v['pose_err_m'] / (cfg.grid.cell_m * 16) * m_per_cell:.2f},{v['n']},{time.time() - t0:.1f},"
                        f"{v['vce_m']:.3f},{v['vce_pose_m']:.3f},{v['fine_epe_px']:.3f},{v['fine_epe_in_px']:.3f},"
                        f"{v.get('fine_skipped', 0)}\n")
            pose_m = v["pose_err_m"] / (cfg.grid.cell_m * 16) * m_per_cell
            print(f"step {k} VAL    CE {v['ce']:.3f}  poseNLL {v['pose_nll']:.3f}  top1 {v['top1']:.1%}  "
                  + (f"{'heatmap-argmax (not a pose)' if placed else 'pose'} {pose_m:.1f} m  " if a.pose_nll_weight else "")
                  + f"n {v['n']}"
                  + (f"  VCE {v['vce_m']:.2f} m  procrustes {v['vce_pose_m']:.2f} m" if vce_w else "")
                  + (f"  fine EPE {v['fine_epe_px']:.2f} px (input {v['fine_epe_in_px']:.2f})" if refine_w else ""),
                  flush=True)
            score = v["vce_pose_m"] if vce_w and v["vce_pose_m"] == v["vce_pose_m"] else pose_m
            finite, is_best = best_candidate(v, score, best, val_keys)
            if not finite:
                print(f"  WARNING step {k}: validation not finite ({ {kk: v.get(kk) for kk in val_keys} }, "
                      f"fine batches skipped {v.get('fine_skipped', 0)}): not a checkpoint candidate", flush=True)
            else:
                torch.save(ckpt_state(k, v), last_finite_path)
            if is_best:
                best, best_step = score, k
                torch.save(ckpt_state(k, v), ckpt_path)
                print(f"  best -> {ckpt_path}", flush=True)
            query.train()
    last_path = ckpt_path.with_name(f"vigor_{a.tag}_last.pt")   # a late (multimodal) checkpoint stays evaluable
    torch.save(ckpt_state(a.steps, v if a.steps else None), last_path)
    print(f"wrote {ckpt_path} (best, step {best_step}), {last_finite_path} (last finite validation) and {last_path} "
          f"(last, step {a.steps}); non-finite steps {guard.total}, no-update steps {n_skip_step}", flush=True)


if __name__ == "__main__":
    main()
