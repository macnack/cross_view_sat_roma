"""Train the camera + LiDAR fusion BEV encoder against the Sat-RoMa decoder (coarse loss).

  python scripts/train_fusion.py --check-targets            # class layout vs the released checkpoint
  python scripts/train_fusion.py --overfit 4 --steps 300    # plumbing test on a SYNTHETIC reference
  python scripts/train_fusion.py                            # real run; needs data.ortho in the config

Without an orthophoto only the first two modes work: the synthetic reference is random texture,
so its numbers say "gradients flow and targets are consistent", nothing about localization.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from bevloc import config as C
from bevloc.data.calib import Calib
from bevloc.data.dur360 import Dur360Frames
from bevloc.data.oxts import read_track
from bevloc.data.pairs import FusionPairs, collate, fully_covered_names, write_synthetic_ortho
from bevloc.model.coarse import FeatureQueryMatcher, coarse_loss, coarse_targets, ref_cell_validity, to_tensor
from bevloc.model.fusion_bev import FusionBEV


def check_targets(cfg, dev):
    """The released checkpoint on an aerial pair must agree with coarse_targets (and not with its transpose)."""
    from bevloc.match import satroma
    ref = cv2.cvtColor(cv2.imread(str(Path(satroma.PACKAGE_DIR) / "examples/reference.png")), cv2.COLOR_BGR2RGB)
    th, q0, cx, cy = np.radians(25), 111.5, 560.0, 330.0
    c, s = np.cos(th), np.sin(th)
    H = np.array([[c, -s, cx - (c * q0 - s * q0)], [s, c, cy - (s * q0 + c * q0)], [0, 0, 1]])
    q = cv2.warpPerspective(ref, np.linalg.inv(H), (224, 224))
    m = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False).eval()
    with torch.no_grad():
        gm = m(m.image_query_features(to_tensor(q, dev)), to_tensor(ref, dev))
    idx, use = coarse_targets(torch.tensor(H, dtype=torch.float32, device=dev)[None],
                              torch.ones(1, 14, 14, dtype=torch.bool, device=dev))
    k = int(round(gm.shape[1] ** 0.5))
    for name, t in (("row*K+col", idx), ("col*K+row (transposed)", (idx % k) * k + idx // k)):
        loss, st = coarse_loss(gm, t, use)
        print(f"  {name:24s} CE {loss:7.3f}   top-1 {st['acc']:6.1%}   argmax distance {st['cell_err']:5.2f} cells")
    ok = coarse_loss(gm, idx, use)[1]["acc"] > 0.5
    print("TARGET LAYOUT", "OK" if ok else "WRONG")
    raise SystemExit(0 if ok else 1)


def step(bev, matcher, erp_valid, batch, cfg, dev):
    f_q, patch_frac, _ = bev(batch["erp"].to(dev), erp_valid, [p.to(dev) for p in batch["points"]],
                             [r.to(dev) for r in batch["reflectivity"]])
    ref = batch["ref"].to(dev)
    gm = matcher(f_q, ref)
    # ref_valid: mask OUT individual query patches whose match target lands on a black/no-data
    # cell of THIS reference crop, rather than requiring the whole frame's footprint to be clean
    # (fully_covered_names below is now only a cheap "has some real content at all" pre-filter).
    rv = ref_cell_validity(ref, min_frac=cfg.train.min_ref_cell_valid)
    H = batch["H"].to(dev)
    idx, use = coarse_targets(H, patch_frac >= cfg.train.min_patch_valid, ref_valid=rv)
    loss, st = coarse_loss(gm, idx, use)
    st["ce"] = float(loss)
    return loss, st


@torch.no_grad()
def validate(bev, matcher, erp_valid, loader, cfg, dev):
    bev.eval(); matcher.eval()
    rows = [(s["ce"], s) for _, s in (step(bev, matcher, erp_valid, b, cfg, dev) for b in loader)]
    bev.train(); matcher.train()
    w = np.array([s["n"] for _, s in rows], float)
    avg = lambda v: float(np.nansum(np.array(v) * w) / max(w.sum(), 1))
    return dict(loss=avg([l for l, _ in rows]), acc=avg([s["acc"] for _, s in rows]),
                cell_err=avg([s["cell_err"] for _, s in rows]))


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    ap.add_argument("--out", default="experiments/02_fusion/run")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--overfit", type=int, default=0, help="train AND validate on the first N frames, synthetic reference")
    ap.add_argument("--check-targets", action="store_true")
    ap.add_argument("--ortho", help="override cfg.data.ortho for this run only (compare reference sources)")
    ap.add_argument("--val-range", nargs=2, type=int, metavar=("START", "END"),
                    help="override cfg.train.val_ranges with a single [START, END] block for this run")
    ap.add_argument("--lr-decoder", type=float, default=None,
                    help="override train.lr_decoder for this run (0 keeps the released decoder frozen)")
    ap.add_argument("--min-coverage", type=float, default=0.05,
                    help="fully_covered_names' min_frac, used ONLY as a cheap existence pre-filter (skip frames "
                        "with ~no ortho overlap at all, so we don't waste a training step on them); the real "
                        "correctness work is per-patch ref_cell_validity masking in step(), which excludes only "
                        "the query patches whose target actually lands on black/no-data, not the whole frame. "
                        "Raise this back towards 1.0 only if you want the old whole-frame-must-be-clean behaviour.")
    a = ap.parse_args()
    cfg = C.load(a.config)
    if a.lr_decoder is not None:
        cfg.train.lr_decoder = a.lr_decoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.preferred_linalg_library("cusolver")    # MAGMA's batched LU floods the log with warnings
    if a.check_targets:
        check_targets(cfg, dev)
    t = cfg.train
    steps = a.steps or t.steps
    torch.manual_seed(t.seed); np.random.seed(t.seed)

    ds, calib = Dur360Frames.from_config(cfg), Calib.from_config(cfg)
    names = ds.names(("image", "scan", "oxts"))
    out = Path(a.out)
    if a.overfit:
        names = names[:: max(1, len(names) // a.overfit)][: a.overfit]
        out.mkdir(parents=True, exist_ok=True)
        ortho = out.parent / "synthetic_ortho.tif"
        if not ortho.exists():
            write_synthetic_ortho(ortho, read_track(ds.root).en)
        tr_names, va_names = names * 50, names        # every repeat draws its own crop; val crops are never trained on
    else:
        ortho_cfg = a.ortho or cfg.data.ortho
        if not ortho_cfg:
            raise SystemExit("data.ortho is not set: no reference map to train against. "
                             "Use --overfit N for the synthetic plumbing test (docs/decisions.md).")
        ortho = Path(ortho_cfg) if Path(ortho_cfg).is_absolute() else C.REPO / ortho_cfg
        tr_full = read_track(ds.root, names)
        edge_m = cfg.grid.n * cfg.grid.cell_m * cfg.reference.scale         # reference edge before rotation slack
        covered = set(fully_covered_names(ortho, names, tr_full.en, edge_m, min_frac=a.min_coverage))
        dropped = len(names) - len(covered)
        if dropped:
            print(f"ortho pre-filter (min_frac={a.min_coverage}): {len(covered)}/{len(names)} frames kept "
                  f"({dropped} dropped -- ~no overlap at all with {ortho.name}); remaining partial gaps are "
                  f"handled per-patch by ref_cell_validity, see docs/decisions.md")
        names = [n for n in names if n in covered]
        if not names:
            raise SystemExit(f"no frame has full ortho coverage under {ortho} -- check data.ortho / the mosaic extent")
        val_ranges = [tuple(a.val_range)] if a.val_range else t.val_ranges
        in_val = lambda n: any(lo <= int(n) <= hi for lo, hi in val_ranges)
        tr_names = [n for n in names if not in_val(n)]
        va_names = [n for n in names if in_val(n)][:: max(1, sum(map(in_val, names)) // t.val_frames)][: t.val_frames]
        if not tr_names or not va_names:
            raise SystemExit(f"train/val split is empty after coverage filtering (train {len(tr_names)}, "
                             f"val {len(va_names)}) -- widen train.val_ranges or re-check ortho coverage")
    C.snapshot(cfg, out, dict(ortho=str(ortho), n_train=len(tr_names), n_val=len(va_names), overfit=a.overfit,
                              val_ranges=val_ranges if not a.overfit else None))

    # num_workers=0: __getitem__ is ~0.1 s (a 224 px ERP decode + a few 10^4 LiDAR points + one ortho window
    # read), far under a training step's compute time, so worker processes buy nothing here -- and DO NOT
    # help: CUDA is already initialized in this process by the time the loaders are first iterated (models
    # are built below with .to(dev)), and fork()-ing a process with a live CUDA context hangs unpredictably
    # in worker processes rather than erroring, which is exactly what an earlier run of this script did.
    mk = lambda n, train: DataLoader(FusionPairs(ds, cfg, n, ortho, train=train, seed=t.seed), batch_size=t.batch,
                                     shuffle=train, num_workers=0, collate_fn=collate, drop_last=train)
    tr, va = mk(tr_names, True), mk(va_names, False)
    erp_valid = torch.from_numpy(np.load(C.REPO / cfg.erp.valid_mask)).float()[None, None].to(dev)

    bev = FusionBEV.from_config(cfg, calib).to(dev).train()
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=t.lr_decoder > 0).train()
    groups = [dict(params=[p for p in bev.parameters() if p.requires_grad], lr=t.lr_bev)]
    if t.lr_decoder > 0:
        groups.append(dict(params=list(matcher.model.decoder.parameters()), lr=t.lr_decoder))
    opt = torch.optim.AdamW(groups, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.05)
    n_par = lambda ps: sum(p.numel() for p in ps if p.requires_grad) / 1e6
    print(f"train {len(tr_names)} / val {len(va_names)} frames   trainable: BEV {n_par(bev.parameters()):.1f} M, "
          f"decoder {n_par(matcher.model.decoder.parameters()):.1f} M   reference: {ortho}")

    log = csv.writer(open(out / "log.csv", "w", newline=""))
    log.writerow(["step", "split", "ce", "acc", "cell_err", "sec_per_step", "gpu_gb"])
    k, t0, run = 0, time.time(), []
    while k < steps:
        tr.dataset.epoch += 1
        for batch in tr:
            loss, st = step(bev, matcher, erp_valid, batch, cfg, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step(); sched.step()
            k += 1
            run.append((st["ce"], st["acc"], st["cell_err"]))
            if k % 25 == 0 or k == steps:
                m = np.nanmean(run, 0); run = []
                dt, gb = (time.time() - t0) / k, torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else 0
                log.writerow([k, "train", *m[:3], dt, gb])
                print(f"step {k:5d}  train CE {m[0]:6.3f}  top-1 {m[1]:6.1%}  argmax dist {m[2]:5.2f} cells   "
                      f"{dt:.2f} s/step  {gb:.1f} GB", flush=True)
            if k % t.val_every == 0 or k == steps:
                v = validate(bev, matcher, erp_valid, va, cfg, dev)
                log.writerow([k, "val", v["loss"], v["acc"], v["cell_err"], "", ""])
                print(f"step {k:5d}  VAL   CE {v['loss']:6.3f}  top-1 {v['acc']:6.1%}  argmax dist "
                      f"{v['cell_err']:5.2f} cells   (chance: CE {np.log(3136):.2f}, 1 cell = 4 m)", flush=True)
            if k >= steps:
                break
    ck = C.REPO / "checkpoints" / f"{out.parent.name}_{out.name}.pt"
    ck.parent.mkdir(exist_ok=True)
    torch.save(dict(bev=bev.state_dict(), decoder=matcher.model.decoder.state_dict(), step=k), ck)
    print("saved", ck)


if __name__ == "__main__":
    main()
