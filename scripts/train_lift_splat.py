"""Train spherical Lift-Splat against Sat-RoMa coarse CE on Mapillary + Poznań ortho.

  make lift-overfit
  make lift-splat          # same-year 2025
  make lift-splat-years    # cross-year refs as appearance augmentation
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bevloc import config as C
from bevloc.data.mapillary import MapillaryPairs, PoznanOrtho, collate, load_frames
from bevloc.model.coarse import (
    FeatureQueryMatcher, coarse_targets, pose_heatmap_nll, ref_cell_validity, roma_coarse_loss,
)
from bevloc.model.lift_splat import SphericalLiftSplat

MAP_ROOT = C.REPO / "data/mapillary"
TRAIN_SEQS = [
    MAP_ROOT / "Fixtor/iHfmEq03Tc6752Y4Ke8wlC",
    MAP_ROOT / "Fixtor/NWVA14Y83pMRsijaGFkmQS",
    MAP_ROOT / "Fixtor/gXabFhpwk2dcl0i4518mDQ",
    MAP_ROOT / "Fixtor/doQ3OhJBKe56c8UxjAFmat",
]
VAL_SEQS = [
    MAP_ROOT / "Fixtor/IcRzj0wTLZX874qitxVsQa",
]


def encode_erp(matcher, erp):
    with torch.no_grad():
        return matcher.model.encoder(erp)[16]


def step(lift, matcher, batch, cfg, min_patch, local_radius, neighbour_radius, neighbour_weight,
         certainty_weight=0.01, pose_nll_weight=0.0):
    erp, ref = batch["erp"], batch["ref"]
    # erp: (B, 3, H, W) legacy or (B, T, 3, H, W) multi-frame
    if erp.ndim == 4:
        f_erp = encode_erp(matcher, erp)
        f_q, patch_frac = lift(f_erp, batch["R_w2c"], erp_hw=erp.shape[-2:])
    else:
        B, T = erp.shape[:2]
        feats = [encode_erp(matcher, erp[:, t]) for t in range(T)]
        f_erp = torch.stack(feats, 1)  # (B, T, C, h, w)
        f_q, patch_frac = lift.forward_multiframe(
            f_erp, batch["R_w2c"], batch["se2"], erp_hw=erp.shape[-2:])
    out = matcher.model.decoder({16: f_q}, matcher.reference_features(ref),
                                scale_factor=matcher.wrapper.im_a_size / 560.0)
    gm = out[16]["gm_cls"]
    cert = out[16].get("gm_certainty")
    # Decoder always classifies over a fixed K×K grid (K=56 → 3136 for sat493m),
    # independent of reference pixel size. Cell size in pixels = ref_size / K.
    ref_size = int(ref.shape[-1])
    cells = int(round(gm.shape[1] ** 0.5))
    rv = ref_cell_validity(ref, cells=cells, min_frac=cfg.train.min_ref_cell_valid)
    idx, use = coarse_targets(batch["H"], patch_frac >= min_patch, ref_valid=rv,
                              ref_size=ref_size, cells=cells)
    # No-match pairs: GT pose is off the crop. Sat-RoMa has no unmatched class bin —
    # drop CE for the whole sample and train certainty toward "not matchable".
    neg = batch.get("negative")
    if neg is not None and bool(neg.any()):
        use = use.clone()
        use[neg] = False
    loss, st = roma_coarse_loss(gm, idx, use, gm_certainty=cert, certainty_weight=certainty_weight,
                                local_radius=local_radius, neighbour_radius=neighbour_radius,
                                neighbour_weight=neighbour_weight)
    st = dict(st)
    st["pose_nll"] = 0.0
    st["pose_err"] = float("nan")
    if pose_nll_weight and (neg is None or not bool(neg.all())):
        pnll, pst = pose_heatmap_nll(
            gm, batch["H"], matchable=use, gm_certainty=cert,
            local_radius=local_radius, ref_size=ref_size,
        )
        loss = loss + float(pose_nll_weight) * pnll
        st["pose_nll"] = pst["pose_nll"]
        st["pose_err"] = pst["pose_err"]
    return loss, st


@torch.no_grad()
def validate(lift, matcher, loader, cfg, min_patch, local_radius, device, max_batches=32,
             certainty_weight=0.01, pose_nll_weight=0.0):
    lift.eval()
    rows = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        _, st = step(lift, matcher, batch, cfg, min_patch, local_radius, 0, 0.0,
                     certainty_weight=certainty_weight, pose_nll_weight=pose_nll_weight)
        rows.append(st)
    lift.train()
    if not rows:
        return dict(ce=float("nan"), top1=float("nan"), top5=float("nan"), cell_err_m=float("nan"),
                    pose_nll=float("nan"), pose_err_m=float("nan"), n=0)
    w = np.array([r["n"] for r in rows], float)

    def avg(key):
        return float(np.nansum([r.get(key, float("nan")) * r["n"] for r in rows]) / max(w.sum(), 1))

    m_per = cfg.grid.cell_m * 16  # val fixed scale-4
    return dict(ce=avg("ce"), top1=avg("acc"), top5=avg("top5"),
                cell_err_m=avg("cell_err") * m_per,
                pose_nll=avg("pose_nll"), pose_err_m=avg("pose_err") * m_per,
                n=int(w.sum()))


def open_years(years):
    out = {}
    for y in years:
        paths = sorted(Path.home().glob(f"Github/sat_data/geoportal_poznan_15km2_*/year_{y}.tif"))
        if len(paths) < 9:
            raise SystemExit(f"expected 9 Poznań tiles for {y}, found {len(paths)}")
        out[int(y)] = PoznanOrtho(paths)
    return out


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--out", default="experiments/05_lift_splat/fixtor_multi")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--local-radius", type=int, default=-1)
    ap.add_argument("--max-offset", type=float, default=-1.0)
    ap.add_argument("--max-rot", type=float, default=-1.0)
    ap.add_argument("--years", default="2025", help="comma-separated ortho years; train samples one per step")
    ap.add_argument("--val-year", type=int, default=2025, help="fixed year for the held-out route")
    ap.add_argument("--ckpt", default="", help="warm-start lift+decoder from this checkpoint")
    ap.add_argument("--neighbour-radius", type=int, default=-1)
    ap.add_argument("--neighbour-weight", type=float, default=-1.0)
    ap.add_argument("--pose-nll-weight", type=float, default=-1.0,
                    help="weight on pose-heatmap NLL (0=off; default from cfg.lift.pose_nll_weight)")
    ap.add_argument("--seq-dists", default="",
                    help="comma metres behind query for multi-frame splat, e.g. 0,2,5")
    ap.add_argument("--val-every", type=int, default=0)
    a = ap.parse_args()
    cfg = C.load(a.config)
    L = cfg.lift
    if a.seq_dists.strip():
        L.seq_dists_m = [float(x) for x in a.seq_dists.split(",") if x.strip()]
    if a.max_offset >= 0:
        cfg.reference.max_offset_frac = a.max_offset
    elif not a.overfit:
        cfg.reference.max_offset_frac = L.max_offset_frac
    if a.max_rot >= 0:
        cfg.reference.max_rot_deg = a.max_rot
    elif not a.overfit:
        cfg.reference.max_rot_deg = L.max_rot_deg
    local_radius = L.local_radius if a.local_radius < 0 else a.local_radius
    neigh_r = (L.neighbour_radius if a.neighbour_radius < 0 else a.neighbour_radius)
    neigh_w = (L.neighbour_weight if a.neighbour_weight < 0 else a.neighbour_weight)
    cert_w = float(getattr(L, "certainty_weight", 0.01) or 0.01)
    pose_w = float(getattr(L, "pose_nll_weight", 0.0) or 0.0)
    if a.pose_nll_weight >= 0:
        pose_w = float(a.pose_nll_weight)
    steps = a.steps or L.steps
    batch = a.batch or L.batch
    val_every = a.val_every or L.val_every
    years = [int(y) for y in a.years.split(",") if y.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    orthos = open_years(sorted(set(years + [a.val_year])))
    print(f"ortho years {sorted(orthos)}  train-sample {years}  val-year {a.val_year}", flush=True)

    print("train sequences:", flush=True)
    train_frames = load_frames(TRAIN_SEQS, orthos[years[0]], margin_m=L.margin_m)
    print("val sequences:", flush=True)
    val_frames = load_frames(VAL_SEQS, orthos[a.val_year], margin_m=L.margin_m)
    if a.overfit:
        idx = [int(round(i * (len(train_frames) - 1) / max(a.overfit - 1, 1))) for i in range(a.overfit)]
        train_frames = [train_frames[i] for i in idx]
        val_frames = train_frames
        cfg.reference.max_offset_frac = 0.10 if a.max_offset < 0 else cfg.reference.max_offset_frac
        cfg.reference.max_rot_deg = 10.0 if a.max_rot < 0 else cfg.reference.max_rot_deg
        local_radius = 6 if a.local_radius < 0 else local_radius
        print(f"overfit {a.overfit} frames", flush=True)
    if not train_frames:
        raise SystemExit("no training frames inside the orthophoto")

    val_stride = max(1, len(val_frames) // max(L.val_frames, 1))
    val_frames = val_frames[::val_stride][: L.val_frames]
    print(f"train {len(train_frames)}  val {len(val_frames)}  local_radius {local_radius}  "
          f"neigh {neigh_r}/{neigh_w}  offset {cfg.reference.max_offset_frac}  "
          f"rot {cfg.reference.max_rot_deg}  neg_frac {getattr(L, 'neg_frac', 0)}  "
          f"cert_w {cert_w}  pose_nll_w {pose_w}", flush=True)

    tr = MapillaryPairs(train_frames, orthos, cfg, train=True, seed=cfg.train.seed,
                        erp_size=tuple(L.erp_size), years=years)
    va = MapillaryPairs(val_frames, orthos, cfg, train=False, seed=cfg.train.seed + 1,
                        erp_size=tuple(L.erp_size), years=[a.val_year])
    tr_loader = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0, collate_fn=collate)
    va_loader = DataLoader(va, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate)

    def forever():
        while True:
            for b in tr_loader:
                yield b
    it = forever()

    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=L.train_decoder)
    for n, p in matcher.model.decoder.named_parameters():
        if "conv_refiner" in n:
            p.requires_grad = False
    lift = SphericalLiftSplat(
        dim=L.dim, depth_bins=L.depth_bins, d_min=L.d_min, d_max=L.d_max,
        n=cfg.grid.n, cell=cfg.grid.cell_m, max_elev_deg=L.max_elev_deg,
    ).to(dev)
    start_step = 0
    if a.ckpt:
        state = torch.load(a.ckpt, map_location=dev, weights_only=False)
        lift.load_state_dict(state["lift"])
        matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
        start_step = int(state.get("step", 0))
        print(f"warm-start {a.ckpt} (step {start_step})", flush=True)

    groups = [{"params": lift.parameters(), "lr": L.lr_lift}]
    dec = [p for p in matcher.model.decoder.parameters() if p.requires_grad]
    if dec:
        groups.append({"params": dec, "lr": cfg.train.lr_decoder})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)

    out = Path(a.out)
    C.snapshot(cfg, out, dict(
        overfit=a.overfit, years=years, val_year=a.val_year, n_train=len(train_frames),
        n_val=len(val_frames), local_radius=local_radius, neighbour_radius=neigh_r,
        neighbour_weight=neigh_w, pose_nll_weight=pose_w, ckpt=a.ckpt,
        train_seqs=[str(p) for p in TRAIN_SEQS], val_seqs=[str(p) for p in VAL_SEQS],
    ))
    log = out / "log.csv"
    if not log.exists():
        log.write_text("step,split,ce,cert,top1,top5,cell_err_m,pose_nll,pose_err_m,n,sec\n")

    lift.train()
    t0 = time.time()
    best_val = float("inf")
    for step_i in range(1, steps + 1):
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in next(it).items()}
        opt.zero_grad(set_to_none=True)
        loss, st = step(lift, matcher, batch, cfg, L.min_patch_valid, local_radius, neigh_r, neigh_w,
                        certainty_weight=cert_w, pose_nll_weight=pose_w)
        loss.backward()
        opt.step()
        ref_px = int(batch["ref"].shape[-1])
        m_per_cell = (ref_px / 56.0) * cfg.grid.cell_m
        cell_m = st["cell_err"] * m_per_cell if st["n"] else float("nan")
        pose_m = st["pose_err"] * m_per_cell if st["pose_err"] == st["pose_err"] else float("nan")
        global_step = start_step + step_i
        neg_tag = " NEG" if (batch.get("negative") is not None and bool(batch["negative"].any())
                             and st["n"] == 0) else ""
        with log.open("a") as f:
            f.write(f"{global_step},train,{st['ce']:.4f},{st['cert']:.4f},{st['acc']:.4f},"
                    f"{st['top5']:.4f},{cell_m:.2f},{st['pose_nll']:.4f},{pose_m:.2f},"
                    f"{st['n']},{time.time()-t0:.1f}\n")
        if step_i == 1 or step_i % 20 == 0 or step_i == steps:
            print(f"step {global_step} TRAIN  CE {st['ce']:.3f}  cert {st['cert']:.3f}  "
                  f"poseNLL {st['pose_nll']:.3f}  top1 {st['acc']:.1%}  "
                  f"arg {cell_m:.1f} m  pose {pose_m:.1f} m  n {st['n']}{neg_tag}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if val_frames and (step_i % val_every == 0 or step_i == steps):
            vst = validate(lift, matcher, va_loader, cfg, L.min_patch_valid, local_radius,
                           device=dev, max_batches=L.val_frames, certainty_weight=cert_w,
                           pose_nll_weight=pose_w)
            with log.open("a") as f:
                f.write(f"{global_step},val,{vst['ce']:.4f},,{vst['top1']:.4f},{vst['top5']:.4f},"
                        f"{vst['cell_err_m']:.2f},{vst['pose_nll']:.4f},{vst['pose_err_m']:.2f},"
                        f"{vst['n']},{time.time()-t0:.1f}\n")
            print(f"step {global_step} VAL    CE {vst['ce']:.3f}  poseNLL {vst['pose_nll']:.3f}  "
                  f"top1 {vst['top1']:.1%}  arg {vst['cell_err_m']:.1f} m  "
                  f"pose {vst['pose_err_m']:.1f} m  n {vst['n']}", flush=True)
            score = vst["pose_err_m"] if pose_w and vst["pose_err_m"] == vst["pose_err_m"] else vst["ce"]
            if score < best_val:
                best_val = score
                ckpt = C.REPO / "checkpoints" / f"{out.parent.name}_{out.name}_best.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"lift": lift.state_dict(),
                            "decoder": {k: v for k, v in matcher.model.decoder.state_dict().items()
                                        if "conv_refiner" not in k},
                            "step": global_step, "val": vst}, ckpt)
                print(f"  best -> {ckpt}", flush=True)
        t0 = time.time()

    ckpt = C.REPO / "checkpoints" / f"{out.parent.name}_{out.name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lift": lift.state_dict(),
                "decoder": {k: v for k, v in matcher.model.decoder.state_dict().items()
                            if "conv_refiner" not in k},
                "step": start_step + steps}, ckpt)
    print(f"wrote {ckpt}", flush=True)
    for o in orthos.values():
        o.close()


if __name__ == "__main__":
    main()
