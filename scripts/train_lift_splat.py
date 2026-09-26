"""Train spherical Lift-Splat against Sat-RoMa coarse CE on Mapillary + Poznań ortho.

  make lift-overfit
  make lift-splat          # same-year 2025
  make lift-splat-years    # cross-year refs as appearance augmentation
"""
from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bevloc import config as C
from bevloc.data.mapillary import (
    TRAIN_SEQS, VAL_SEQS, MapillaryPairs, PoznanOrtho, collate, load_frames, poznan_tiles,
)
from bevloc.model.coarse import (
    FeatureQueryMatcher, coarse_targets, pose_heatmap_nll, ref_cell_validity, roma_coarse_loss, vce_pose_loss,
)
from bevloc.model.query import build_query, load_query_state
from bevloc.model.refine import (
    NonFiniteGuard, RefinerTap, clip_refiner_grads, decoder_state, fine_loss, gt_warp, refiner_parameters,
    set_refiner_trainable, token_centres_px,
)


def refine_options(cfg):
    """RoMa fine-loss settings from cfg.train (refine_alpha, refine_c, refine_cert_cells), with RoMa's defaults, and
    the refiner's training precision (refine_precision, default float32)."""
    T = cfg.train
    return dict(alpha=float(getattr(T, "refine_alpha", 0.5)), c=float(getattr(T, "refine_c", 1e-4)),
                cert_cells=float(getattr(T, "refine_cert_cells", 0.5)),
                precision=str(getattr(T, "refine_precision", "float32")))


FINE_KEYS = ("fine_epe_px", "fine_epe_in_px", "fine_epe_med_px", "fine_epe_in_med_px", "fine_reg", "fine_cert")


def refine_step_loss(out16, tap, H, use, tok_valid, query_xy, ref_size, cells, certainty_weight, opts):
    """RoMa's fine loss on the decoder's stride-16 conv refiner (its only refiner; bevloc.model.refine).

    The refiner ran on detached inputs (`RefinerTap(detach_inputs=True)`), so its loss trains the refiner only.
    Refined warp = detached W_in + the refiner's displacement; certainty logit = detached gm_certainty + the
    refiner's delta. Ground truth per token: H applied to its query point (token centre for the picture modes, the
    placed point for erp / erp_depth); `use` = the coarse CE's supervised tokens.
    Returns (loss, stats), or (None, stats with fine_skipped = 1) when the refined warp, the certainty or the loss is
    non-finite: the batch then trains the coarse terms only."""
    h, w = tap.x.shape[-2:]
    if query_xy is None:
        q = token_centres_px(h, w, device=H.device, dtype=torch.float64)[None].expand(H.shape[0], h, w, 2)
    else:
        q = query_xy.double()
    gt = gt_warp(H.double(), q, ref_size).float()
    warp = tap.refined_warp()
    cert = (out16["gm_certainty"].detach().float() + tap.delta_certainty.float())[:, 0]
    skipped = dict(fine_skipped=1, fine_reg=float("nan"), fine_cert=float("nan"), fine_epe=float("nan"),
                   fine_epe_in=float("nan"), fine_epe_med=float("nan"), fine_epe_in_med=float("nan"), fine_n=0)
    if not (bool(torch.isfinite(warp).all()) and bool(torch.isfinite(cert).all())):
        return None, skipped
    opts = {k: v for k, v in opts.items() if k != "precision"}
    loss, st = fine_loss(warp, cert, tap.warp_in.detach().float(), gt, use, tok_valid, cell_norm=2.0 / cells,
                         stride=16, certainty_weight=certainty_weight, **opts)
    if not bool(torch.isfinite(loss)):
        return None, skipped
    st["fine_skipped"] = 0
    return loss, st


def step(query, matcher, batch, cfg, min_patch, local_radius, neighbour_radius, neighbour_weight,
         certainty_weight=0.01, pose_nll_weight=0.0, vce_weight=0.0, vce_opts=None, generator=None,
         refine_weight=0.0, refine_opts=None):
    """vce_weight > 0 adds Loc²'s VCE pose loss (bevloc.model.coarse.vce_pose_loss) on the placed query points;
    needs a query with `placement` (erp, erp_depth). vce_opts: its keyword arguments (coarse.vce_options).
    refine_weight > 0 adds RoMa's fine loss on the decoder's conv refiner (`refine_step_loss`; refine_opts =
    `refine_options(cfg)`); the caller unfreezes the refiner (`set_refiner_trainable`)."""
    ref = batch["ref"]
    f_q, patch_frac = query(batch, matcher)          # lift | ipm | hybrid | erp | erp_depth, see bevloc.model.query
    # scale_factor = sqrt(query px area) / 560: 0.4 for the 224 px BEV queries, 1.13 for a 448x896 ERP grid
    sf = float(((f_q.shape[-2] * 16) * (f_q.shape[-1] * 16)) ** 0.5 / 560.0)
    tap = (RefinerTap(matcher.model.decoder, detach_inputs=True,
                      precision=(refine_opts or {}).get("precision", "float32"))
           if refine_weight else contextlib.nullcontext())
    with tap:
        out = matcher.model.decoder({16: f_q}, matcher.reference_features(ref), scale_factor=sf)
    gm = out[16]["gm_cls"]
    cert = out[16].get("gm_certainty")
    # Decoder always classifies over a fixed K×K grid (K=56 → 3136 for sat493m),
    # independent of reference pixel size. Cell size in pixels = ref_size / K.
    ref_size = int(ref.shape[-1])
    cells = int(round(gm.shape[1] ** 0.5))
    rv = ref_cell_validity(ref, cells=cells, min_frac=cfg.train.min_ref_cell_valid)
    # ERP-token query: tokens are placed on the virtual BEV after matching, so their GT cell is
    # where the placed point lands (plan Task 6); other queries use the patch-centre grid.
    placed = query.placement(batch) if hasattr(query, "placement") else None
    query_xy = placed[0] if placed is not None else None
    idx, use = coarse_targets(batch["H"], patch_frac >= min_patch, ref_valid=rv,
                              ref_size=ref_size, cells=cells, query_xy=query_xy)
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
    st["vce_m"] = float("nan")
    st["vce_pose_m"] = float("nan")
    if vce_weight:
        if placed is None:
            raise ValueError("vce_weight > 0 needs a query with placed tokens (erp, erp_depth)")
        tok_ok = placed[1] & (patch_frac >= min_patch)
        if neg is not None and bool(neg.any()):
            tok_ok = tok_ok & ~neg[:, None, None]
        vloss, vst = vce_pose_loss(gm, query_xy, tok_ok, batch["H"], float(cfg.grid.cell_m), int(cfg.grid.n),
                                   ref_size=ref_size, gm_certainty=cert, generator=generator, ref_valid=rv,
                                   **(vce_opts or {}))
        loss = loss + float(vce_weight) * vloss
        st["vce_m"] = vst["vce_m"]
        st["vce_pose_m"] = vst["vce_pose_m"]
    for k in FINE_KEYS:
        st[k] = float("nan")
    st["fine_skipped"] = 0
    if refine_weight:
        tok_valid = patch_frac >= min_patch
        if placed is not None:
            tok_valid = tok_valid & placed[1]
        rloss, rst = refine_step_loss(out[16], tap, batch["H"], use, tok_valid, query_xy, ref_size, cells,
                                      certainty_weight, refine_opts or {})
        st["fine_skipped"] = rst["fine_skipped"]
        if rloss is not None:
            loss = loss + float(refine_weight) * rloss
        px = ref_size / 2.0                                          # normalised -> reference px
        st.update(fine_epe_px=rst["fine_epe"] * px, fine_epe_in_px=rst["fine_epe_in"] * px,
                  fine_epe_med_px=rst["fine_epe_med"] * px, fine_epe_in_med_px=rst["fine_epe_in_med"] * px,
                  fine_reg=rst["fine_reg"], fine_cert=rst["fine_cert"])
    return loss, st


@torch.no_grad()
def validate(query, matcher, loader, cfg, min_patch, local_radius, device, max_batches=32,
             certainty_weight=0.01, pose_nll_weight=0.0, vce_weight=0.0, vce_opts=None, refine_weight=0.0,
             refine_opts=None):
    query.eval()
    rows = []
    # a fixed generator: the VCE's match draw is the same at every validation, so the curve is comparable
    gen = torch.Generator(device=device).manual_seed(0) if vce_weight else None
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        _, st = step(query, matcher, batch, cfg, min_patch, local_radius, 0, 0.0,
                     certainty_weight=certainty_weight, pose_nll_weight=pose_nll_weight,
                     vce_weight=vce_weight, vce_opts=vce_opts, generator=gen,
                     refine_weight=refine_weight, refine_opts=refine_opts)
        rows.append(st)
    query.train()
    if not rows:
        return dict(ce=float("nan"), top1=float("nan"), top5=float("nan"), cell_err_m=float("nan"),
                    pose_nll=float("nan"), pose_err_m=float("nan"), vce_m=float("nan"), vce_pose_m=float("nan"),
                    fine_epe_px=float("nan"), fine_epe_in_px=float("nan"), fine_skipped=0, n=0)
    w = np.array([r["n"] for r in rows], float)

    def avg(key):
        return float(np.nansum([r.get(key, float("nan")) * r["n"] for r in rows]) / max(w.sum(), 1))

    def mean(key):                                   # per-batch quantities (already in metres)
        v = [r.get(key, float("nan")) for r in rows]
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    m_per = cfg.grid.cell_m * 16  # val fixed scale-4
    return dict(ce=avg("ce"), top1=avg("acc"), top5=avg("top5"),
                cell_err_m=avg("cell_err") * m_per,
                pose_nll=avg("pose_nll"), pose_err_m=avg("pose_err") * m_per,
                vce_m=mean("vce_m"), vce_pose_m=mean("vce_pose_m"),
                fine_epe_px=mean("fine_epe_px"), fine_epe_in_px=mean("fine_epe_in_px"),
                fine_skipped=int(sum(int(r.get("fine_skipped", 0)) for r in rows)),
                n=int(w.sum()))


def open_years(years):
    out = {}
    for y in years:
        out[int(y)] = PoznanOrtho(poznan_tiles(y))
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
    ap.add_argument("--query", default="", help="lift | ipm | hybrid (default cfg.lift.query_mode)")
    a = ap.parse_args()
    cfg = C.load(a.config)
    L = cfg.lift
    if a.query:
        L.query_mode = a.query
    mode = str(getattr(L, "query_mode", "lift") or "lift")
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
    refine_w = float(getattr(cfg.train, "refine_weight", 0.0) or 0.0)
    refine_opts = refine_options(cfg)
    set_refiner_trainable(matcher.model.decoder, refine_w > 0)     # frozen (and not saved) unless the fine loss is on
    ref_params = refiner_parameters(matcher.model.decoder) if refine_w > 0 else []
    guard = NonFiniteGuard(int(getattr(cfg.train, "refine_max_nonfinite", 20)))
    query = build_query(cfg, mode).to(dev)
    start_step = 0
    if a.ckpt:
        state = torch.load(a.ckpt, map_location=dev, weights_only=False)
        load_query_state(query, state)
        matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
        start_step = int(state.get("step", 0))
        print(f"warm-start {a.ckpt} (step {start_step}, mode {state.get('mode', 'lift')})", flush=True)

    groups = []
    qp = [p for p in query.parameters() if p.requires_grad]
    if qp:
        groups.append({"params": qp, "lr": L.lr_lift})
    dec = [p for p in matcher.model.decoder.parameters() if p.requires_grad]
    if dec:
        groups.append({"params": dec, "lr": cfg.train.lr_decoder})
    if not groups:
        raise SystemExit("nothing to train: query has no parameters and the decoder is frozen")
    opt = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    print(f"query mode {mode}: {sum(p.numel() for p in qp) / 1e6:.2f} M trainable query params, "
          f"{sum(p.numel() for p in dec) / 1e6:.1f} M decoder params", flush=True)

    out = Path(a.out)
    C.snapshot(cfg, out, dict(
        overfit=a.overfit, query_mode=mode, years=years, val_year=a.val_year, n_train=len(train_frames),
        n_val=len(val_frames), local_radius=local_radius, neighbour_radius=neigh_r,
        neighbour_weight=neigh_w, pose_nll_weight=pose_w, ckpt=a.ckpt,
        train_seqs=[str(p) for p in TRAIN_SEQS], val_seqs=[str(p) for p in VAL_SEQS],
    ))
    log = out / "log.csv"
    if not log.exists():
        log.write_text("step,split,ce,cert,top1,top5,cell_err_m,pose_nll,pose_err_m,n,sec\n")

    query.train()
    t0 = time.time()
    best_val = float("inf")
    for step_i in range(1, steps + 1):
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in next(it).items()}
        opt.zero_grad(set_to_none=True)
        loss, st = step(query, matcher, batch, cfg, L.min_patch_valid, local_radius, neigh_r, neigh_w,
                        certainty_weight=cert_w, pose_nll_weight=pose_w, refine_weight=refine_w,
                        refine_opts=refine_opts)
        bad = not bool(torch.isfinite(loss.detach()))
        if not bad:
            loss.backward()
            if ref_params:
                bad = not clip_refiner_grads(ref_params, float(getattr(cfg.train, "refine_grad_clip", 1.0)))[1]
            opt.step()
        guard.update(bad or bool(st.get("fine_skipped", 0)), step_i)
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
            vst = validate(query, matcher, va_loader, cfg, L.min_patch_valid, local_radius,
                           device=dev, max_batches=L.val_frames, certainty_weight=cert_w,
                           pose_nll_weight=pose_w, refine_weight=refine_w, refine_opts=refine_opts)
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
                torch.save({"query": query.state_dict(), "mode": mode,
                            "decoder": decoder_state(matcher.model.decoder, refine_w > 0),
                            "step": global_step, "val": vst}, ckpt)
                print(f"  best -> {ckpt}", flush=True)
        t0 = time.time()

    ckpt = C.REPO / "checkpoints" / f"{out.parent.name}_{out.name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"query": query.state_dict(), "mode": mode,
                "decoder": decoder_state(matcher.model.decoder, refine_w > 0),
                "step": start_step + steps}, ckpt)
    print(f"wrote {ckpt}", flush=True)
    for o in orthos.values():
        o.close()


if __name__ == "__main__":
    main()
