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

Long runs (2026-09-29, PanoRoMa 100 epochs; bevloc.model.trainrun): --epochs E sets steps = E x (len(train) // batch)
and validates every epoch; --save-every-epochs K adds `vigor_<tag>_ep{e:03d}.pt` (same lean layout as `_best.pt`:
the query module = projection head, and the decoder; the frozen DINOv3 encoder lives in the matcher and is never
saved - asserted at start-up). After every validation `vigor_<tag>_resume.pt` (weights + AdamW state + RNG +
counters, written atomically) is refreshed; --resume auto continues from it (same sample order: the training order
is a permutation seeded by (seed, epoch)), so SLURM requeues and 7-day segments chained with afterany continue
instead of restarting; a job killed mid-epoch loses that epoch only, its CSV rows past the resume step are dropped on
resume, and a different batch / training-set size / tag is refused. Exact on resume: weights, AdamW, sample order,
the main-process RNG (VCE draws); NOT bitwise: the DataLoader workers' RNG (the fine config's fresh reference jitter
per draw; same distribution, other draws) and cuDNN nondeterminism. A run found finished rewrites `_last.pt` and exits
0 (afterok chains). All checkpoints are written atomically (tmp + fsync + rename). --profile N times N steps (data
wait / host-to-device / compute, peak memory, GPU util, a per-phase breakdown) and exits without checkpoints (refused
with --resume / --save-every-epochs): `make eagle-probe`. --pin-memory, --prefetch, --tf32 (default off; recorded in
the checkpoints' train dict) tune throughput.
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
from bevloc.model import trainrun as R  # noqa: E402
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
    ap.add_argument("--epochs", type=int, default=0,
                    help="> 0 overrides --steps: steps = epochs x (len(train) // batch) (drop_last); validation "
                         "every epoch unless --val-every is given")
    ap.add_argument("--save-every-epochs", type=int, default=0,
                    help="with --epochs: also save checkpoints/vigor_<tag>_ep{e:03d}.pt every K epochs (lean: the "
                         "trainable weights only, same layout as _best.pt)")
    ap.add_argument("--resume", default=None,
                    help="path to a vigor_<tag>_resume.pt (decoder, head, optimizer, step, RNG, best) to continue from; "
                         "'auto' = checkpoints/vigor_<tag>_resume.pt if it exists, else start fresh (chained jobs)")
    ap.add_argument("--profile", type=int, default=0,
                    help="probe: run this many timed training steps (after --profile-warmup), print one PROFILE json "
                         "line (peak memory, data-wait vs compute, samples/s, GPU util, per-sample CPU cost), no "
                         "validation, no checkpoints; exit code 3 on CUDA OOM")
    ap.add_argument("--profile-warmup", type=int, default=3)
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 for float32 matmuls (the frozen encoder runs in float32; default off = the "
                         "precision of every run so far; cuDNN convolutions already default to TF32 in PyTorch). "
                         "Recorded in the checkpoints' train dict")
    ap.add_argument("--train-limit", type=int, default=0, help="smoke tests: keep only the first N training labels")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (the IPM picture is built on the CPU per sample)")
    ap.add_argument("--prefetch", type=int, default=2, help="DataLoader prefetch_factor (batches per worker; needs workers > 0)")
    ap.add_argument("--pin-memory", action="store_true", help="DataLoader pin_memory + non-blocking host-to-device copies")
    ap.add_argument("--val-every", type=int, default=None, help="default 500 steps, or one epoch with --epochs")
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
    if a.profile and (a.resume or a.save_every_epochs):
        raise SystemExit("--profile is a throughput probe (exits after the timed steps, writes no checkpoint): "
                         "refusing it together with --resume / --save-every-epochs (a real run's flags)")
    cfg = C.load(a.config)
    L = cfg.lift
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.tf32:                                     # off by default: every run so far used full float32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True     # already PyTorch's default; set explicitly for the record
        print("TF32 matmuls/convolutions ON (--tf32)", flush=True)
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
    # the held-out split, so eval_vigor.py --calib can reproduce it and skip the frames used for selection (task 06)
    train_meta.update(split=a.split, cities=list(tr_cities), val_frac=a.val_frac, val_samples=a.val_samples)
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
    if a.train_limit:                               # smoke tests of the epoch / resume logic only
        tr.labels = tr.labels[:a.train_limit]
        print(f"--train-limit: training on the first {len(tr)} labels only (smoke test)", flush=True)
    # epoch-seeded order (seed, epoch): a resumed run continues the same permutation mid-epoch (bevloc.model.trainrun)
    sampler = R.EpochSampler(len(tr), seed=cfg.train.seed)
    # the loader's own generator (worker base seeds): starting an epoch's iterator must not consume the global RNG,
    # or a resumed run's global stream (VCE draws) would drift from the uninterrupted one
    g_loader = torch.Generator().manual_seed(int(cfg.train.seed))
    wkw = dict(prefetch_factor=a.prefetch) if a.workers > 0 else {}
    tr_loader = DataLoader(tr, batch_size=a.batch, sampler=sampler, num_workers=a.workers, collate_fn=collate_vigor,
                           drop_last=True, persistent_workers=a.workers > 0, pin_memory=a.pin_memory, generator=g_loader,
                           **wkw)
    va_loader = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate_vigor,
                           pin_memory=a.pin_memory, **wkw)
    spe = R.steps_per_epoch(len(tr), a.batch)
    if a.epochs > 0:
        a.steps = R.epoch_steps(len(tr), a.batch, a.epochs)
    if a.val_every is None:
        a.val_every = spe if a.epochs > 0 else 500
    train_meta.update(steps=a.steps, epochs=a.epochs or None, steps_per_epoch=spe, n_train=len(tr),
                      lr_decoder=float(cfg.train.lr_decoder), tf32=bool(a.tf32),
                      matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                      cudnn_tf32=bool(torch.backends.cudnn.allow_tf32))
    print(f"{spe} steps per epoch (batch {a.batch}, drop_last)  total {a.steps} steps"
          + (f" = {a.epochs} epochs" if a.epochs else f" = {a.steps / spe:.2f} epochs")
          + f"  validation every {a.val_every} steps  workers {a.workers} prefetch {a.prefetch if a.workers else '-'} "
            f"pin_memory {a.pin_memory}", flush=True)

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
        train_meta.update(lr_query=float(lr_q))
        print(f"query {mode}: {sum(p.numel() for p in qp) / 1e6:.2f} M trainable parameters (lr {lr_q})", flush=True)
    dec = [p for p in matcher.model.decoder.parameters() if p.requires_grad]
    groups.append({"params": dec, "lr": cfg.train.lr_decoder})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = C.REPO / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"vigor_{a.tag}_best.pt"
    log = out / f"train_{a.tag}.csv"
    val_keys = (["ce", "top1"] + (["pose_nll", "pose_err_m"] if a.pose_nll_weight else [])
                + (["vce_m", "vce_pose_m"] if vce_w else []) + (["fine_epe_px", "fine_epe_in_px"] if refine_w else []))
    ref_params = refiner_parameters(matcher.model.decoder) if refine_w else []
    last_finite_path = ckpt_path.with_name(f"vigor_{a.tag}_last_finite.pt")

    def ckpt_state(k, v):
        # lean: the query module (projection head; the frozen encoder lives in the matcher, not here) and the decoder
        return {"query": query.state_dict(), "mode": mode,
                "erp_depth": vars(cfg.erp_depth) if mode == "erp_depth" else None,
                "train": train_meta,
                "decoder": decoder_state(matcher.model.decoder, include_refiner=keep_refiner),
                "step": k, "epoch": k / spe, "val": v}

    sd0 = ckpt_state(0, None)
    R.assert_no_frozen_encoder(sd0, matcher.model.encoder)
    rep = R.param_groups_report({"query": sd0["query"], "decoder": sd0["decoder"]})
    n_enc = sum(p.numel() for p in matcher.model.encoder.parameters())
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    print("checkpoint contents: " + "  ".join(f"{g} {n / 1e6:.2f} M tensors ({b / 2 ** 20:.0f} MiB)"
                                              for g, (n, b) in rep.items())
          + f"  | optimised {n_opt / 1e6:.2f} M params | frozen encoder {n_enc / 1e6:.1f} M NOT saved", flush=True)
    step_kw = dict(certainty_weight=0.01, pose_nll_weight=a.pose_nll_weight, vce_weight=vce_w, vce_opts=vce_opts,
                   refine_weight=refine_w, refine_opts=refine_opts)

    def to_dev(b):
        return {kk: (x.to(dev, non_blocking=a.pin_memory) if torch.is_tensor(x) else x) for kk, x in b.items()}

    def train_step(batch, tick=lambda name: None):
        """One optimizer step; returns (loss, stats, bad, g_ok) as the main loop needs them. tick(name) is called
        after the forward, backward and optimizer phases (the --profile breakdown synchronises there)."""
        opt.zero_grad(set_to_none=True)
        loss, st = step(query, matcher, batch, cfg, L.min_patch_valid, a.local_radius,
                        a.neighbour_radius, a.neighbour_weight, **step_kw)
        bad = not bool(torch.isfinite(loss.detach()))
        tick("forward")
        g_ok = True
        if not bad:
            loss.backward()
            tick("backward")
            if ref_params:
                _, g_ok = clip_refiner_grads(ref_params, grad_clip)
            opt.step()
            tick("optimizer")
        return loss, st, bad, g_ok

    if a.profile:
        run_profile(a, tr, tr_loader, sampler, dev, to_dev, train_step, matcher)
        return

    # ---- resume ----
    res_path = R.resume_path(ckpt_dir, a.tag)
    rp = None
    if a.resume == "auto":
        rp = res_path if res_path.is_file() else None
        print(f"--resume auto: {'resuming from ' + str(rp) if rp else 'no ' + str(res_path) + ', starting fresh'}",
              flush=True)
    elif a.resume:
        rp = Path(a.resume)
    k0, best, best_step, n_skip_step, elapsed0, v = 0, float("inf"), None, 0, 0.0, None
    if rp is not None:
        cnt = R.load_resume(rp, query, matcher.model.decoder, opt, loader_gen=g_loader)
        R.check_resume(cnt, a.batch, spe, len(tr), a.tag)
        k0, best, best_step = int(cnt["step"]), float(cnt["best"]), cnt["best_step"]
        n_skip_step, guard.total, elapsed0 = int(cnt["n_skip_step"]), int(cnt["guard_total"]), float(cnt["elapsed"])
        v = cnt.get("val")
        if bool(cnt.get("tf32", False)) != bool(a.tf32):
            print(f"WARNING: resume file trained with tf32={cnt.get('tf32', False)}, this segment --tf32={a.tf32}",
                  flush=True)
        print(f"resumed {rp}: step {k0} (epoch {k0 / spe:.2f}), best {best:.3f} at step {best_step}", flush=True)
        if k0 >= a.steps:
            # finished: make sure `_last.pt` exists and is whole (a kill between the final resume save and the
            # `_last.pt` write would otherwise leave a chained fine run without its --ckpt), then exit 0 (afterok)
            last_path = ckpt_path.with_name(f"vigor_{a.tag}_last.pt")
            R.atomic_save(ckpt_state(k0, v), last_path)
            print(f"already at step {k0} >= {a.steps}: nothing to do (rewrote {last_path})", flush=True)
            return
        n_drop = R.truncate_log(log, k0)
        if n_drop:
            print(f"{log}: dropped {n_drop} rows after step {k0} (re-run by this segment)", flush=True)
    else:
        snap = dict(tag=a.tag, ckpt=a.ckpt, split=a.split, train_cities=tr_cities, val_cities=va_cities,
                    val_frac=a.val_frac, val_samples=a.val_samples, query_mode=mode,
                    vce_weight=vce_w, no_depth=n_no_depth, pose_nll_weight=a.pose_nll_weight,
                    refine_weight=refine_w, epochs=a.epochs, steps=a.steps, batch=a.batch, tf32=a.tf32)
        p_snap = C.snapshot(cfg, out, snap)
        # several runs may share --out (the coarse and fine long runs do) and overwrite config.yaml: keep a per-tag copy
        (out / f"config_{a.tag}.yaml").write_text(Path(p_snap).read_text())
    if rp is None or not log.is_file():
        log.write_text("step,split,ce,top1,cell_err_m,pose_nll,pose_err_m,n,sec,vce_m,vce_pose_m,fine_epe_px,"
                       "fine_epe_in_px,fine_skipped\n")

    def save_resume(k, v):
        R.atomic_save(R.make_resume(ckpt_state(k, v), opt, loader_gen=g_loader, counters=dict(
            step=k, best=best, best_step=best_step, n_skip_step=n_skip_step, guard_total=guard.total,
            elapsed=time.time() - t0, batch=a.batch, steps_per_epoch=spe, n_train=len(tr), tag=a.tag,
            tf32=bool(a.tf32), val=v)), res_path)

    def batches(k_start):
        e, skip = divmod(k_start, spe)
        while True:
            sampler.set_epoch(e, skip * a.batch)           # the rest of epoch e in its seeded order
            for b in tr_loader:
                yield b
            e, skip = e + 1, 0
    it = batches(k0)
    m_per_cell = (cfg.grid.n * cfg.reference.scale / 56.0) * cfg.grid.cell_m
    query.train()
    t0 = time.time() - elapsed0
    for k in range(k0 + 1, a.steps + 1):
        batch = to_dev(next(it))
        loss, st, bad, g_ok = train_step(batch)
        if bad:                                        # no update at all from a non-finite total loss
            n_skip_step += 1
            print(f"step {k}: non-finite loss, no update ({n_skip_step} so far)", flush=True)
        elif not g_ok:
            print(f"step {k}: non-finite refiner gradient, refiner not updated", flush=True)
            bad = True
        bad = bad or bool(st.get("fine_skipped", 0))
        guard.update(bad, k)                           # raises after refine_max_nonfinite bad steps in a row
        with log.open("a") as f:
            f.write(f"{k},train,{st['ce']:.4f},{st['acc']:.4f},{st['cell_err'] * m_per_cell:.2f},"
                    f"{st['pose_nll']:.4f},{st['pose_err'] * m_per_cell:.2f},{st['n']},{time.time() - t0:.1f},"
                    f"{st['vce_m']:.3f},{st['vce_pose_m']:.3f},{st['fine_epe_px']:.3f},{st['fine_epe_in_px']:.3f},"
                    f"{st.get('fine_skipped', 0)}\n")
        if k == k0 + 1 or k % 25 == 0:
            print(f"step {k} TRAIN  CE {st['ce']:.3f}  poseNLL {st['pose_nll']:.3f}  top1 {st['acc']:.1%}  "
                  f"arg {st['cell_err'] * m_per_cell:.1f} m  n {st['n']}"
                  + (f"  VCE {st['vce_m']:.2f} m  procrustes {st['vce_pose_m']:.2f} m" if vce_w else "")
                  + (f"  fine EPE {st['fine_epe_px']:.2f} px (input {st['fine_epe_in_px']:.2f})"
                     f"  non-finite steps {guard.total}" if refine_w else "")
                  + f"  ({time.time() - t0:.0f}s, epoch {k / spe:.2f})", flush=True)
        e_save = R.is_epoch_save(k, spe, a.save_every_epochs) if a.epochs else 0
        if k % a.val_every == 0 or k == a.steps or e_save:          # an epoch save always validates first
            v = validate(query, matcher, va_loader, cfg, L.min_patch_valid, a.local_radius, device=dev,
                         max_batches=max(1, -(-a.val_samples // a.batch)), certainty_weight=0.01,
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
                R.atomic_save(ckpt_state(k, v), last_finite_path)
            if is_best:
                best, best_step = score, k
                R.atomic_save(ckpt_state(k, v), ckpt_path)
                print(f"  best -> {ckpt_path}", flush=True)
            if e_save:
                p_ep = R.epoch_ckpt_path(ckpt_dir, a.tag, e_save)
                R.atomic_save(ckpt_state(k, v), p_ep)
                print(f"  epoch {e_save} -> {p_ep} ({p_ep.stat().st_size / 2 ** 20:.0f} MiB)", flush=True)
            save_resume(k, v)
            query.train()
    last_path = ckpt_path.with_name(f"vigor_{a.tag}_last.pt")   # a late (multimodal) checkpoint stays evaluable
    R.atomic_save(ckpt_state(a.steps, v if a.steps else None), last_path)
    print(f"wrote {ckpt_path} (best, step {best_step}), {last_finite_path} (last finite validation) and {last_path} "
          f"(last, step {a.steps}, {last_path.stat().st_size / 2 ** 20:.0f} MiB); resume state {res_path}; "
          f"non-finite steps {guard.total}, no-update steps {n_skip_step}", flush=True)


def run_profile(a, tr, tr_loader, sampler, dev, to_dev, train_step, matcher=None):
    """--profile N: time --profile-warmup + N real training steps (no validation, no checkpoints) and print one
    PROFILE json line: peak CUDA memory, per-step data wait (blocked on next(loader)) / host-to-device / compute
    (forward + backward + optimizer, synchronised), samples/s, nvidia-smi GPU utilisation, per-sample CPU cost."""
    cuda = dev == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    cost = R.sample_cost(tr, n=16)
    print(f"per-sample CPU cost (main process, s): {cost}", flush=True)
    base = dict(workers=a.workers, prefetch=a.prefetch if a.workers else None, pin_memory=a.pin_memory,
                cpus=len(os.sched_getaffinity(0)), config=a.config, sample_cost_s=cost)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    total = torch.cuda.get_device_properties(0).total_memory if cuda else 0
    sampler.set_epoch(0)
    it = iter(tr_loader)
    t_data, t_h2d, t_comp = [], [], []
    gu = R.GpuUtil()
    try:
        for k in range(a.profile_warmup + a.profile):
            if k == a.profile_warmup:
                gu.__enter__()
            t = time.perf_counter()
            b = next(it)
            t1 = time.perf_counter()
            batch = to_dev(b)
            sync()
            t2 = time.perf_counter()
            train_step(batch)
            sync()
            t3 = time.perf_counter()
            if k >= a.profile_warmup:
                t_data.append(t1 - t)
                t_h2d.append(t2 - t1)
                t_comp.append(t3 - t2)
            print(f"profile step {k + 1}: data {t1 - t:.3f}s h2d {t2 - t1:.3f}s compute {t3 - t2:.3f}s", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        gu.__exit__(None, None, None)
        R.print_profile(dict(batch=a.batch, oom=True, error=str(e).splitlines()[0][:200],
                             peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3, **base))
        sys.exit(3)
    gu.__exit__(None, None, None)
    if matcher is not None and a.profile:
        # breakdown on the last batch, 3 repeats, medians: frozen encoder (no grad) on the reference and on the
        # panorama separately, then the full step split at synchronised forward / backward / optimizer boundaries
        # (forward includes both encoder passes; decoder forward = forward - encoder)
        rows = []
        for _ in range(3):
            r = {}
            with torch.no_grad():
                sync()
                t = time.perf_counter()
                matcher.reference_features(batch["ref"])
                sync()
                r["enc_ref"] = time.perf_counter() - t
                t = time.perf_counter()
                matcher.model.encoder(batch["erp"][:, 0])
                sync()
                r["enc_pano"] = time.perf_counter() - t
            last = [time.perf_counter()]

            def tick(name, r=r, last=last):
                sync()
                now = time.perf_counter()
                r[name] = now - last[0]
                last[0] = now
            train_step(batch, tick)
            rows.append(r)
        bd = {k: float(np.median([r.get(k, float("nan")) for r in rows])) for k in rows[0]}
        bd["decoder_fwd_etc"] = bd["forward"] - bd["enc_ref"] - bd["enc_pano"]
        base.update(breakdown_s=bd, tf32=a.tf32,
                    shapes=dict(ref=list(batch["ref"].shape), erp=list(batch["erp"].shape)))
    R.print_profile(R.profile_summary(
        a.batch, t_data, t_h2d, t_comp, torch.cuda.max_memory_allocated() if cuda else 0,
        torch.cuda.max_memory_reserved() if cuda else 0, total, util=gu.mean(), extra=dict(oom=False, **base)))


if __name__ == "__main__":
    main()
