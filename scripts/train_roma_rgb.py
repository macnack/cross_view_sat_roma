"""RoMa coarse loss on an RGB BEV, both images through the frozen sat493m encoder.

The BEV is a picture. The map is a picture. Sat-RoMa already classifies, per query
patch, which reference cell it belongs to, and predicts whether the patch is
matchable. That is the loss. Cosine between pooled descriptors is not.

  --probe N     zero-shot posterior on N frames (top-1/5/16, GT mass, entropy). No training.
  default       fine-tune the coarse decoder (the fine refiner stays frozen) on EA tiles.

Classification is taken only on patches whose valid fraction is >= --min-valid
(RoMa uses 0.99). Certainty is a BCE against that mask, weight 0.01.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bevloc import config as C
from bevloc.bev.variants import build_variants
from bevloc.data.ortho import OrthoMap, gt_homography, sample_reference
from bevloc.data.oxts import read_track
from bevloc.data.pairs import fully_covered_names
from bevloc.match.satroma import SatRoMa, SatRoMaMatcher
from bevloc.model.coarse import FeatureQueryMatcher, coarse_targets, ref_cell_validity, roma_coarse_loss
from bevloc.run import context

CELL_M = 4.0


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--out", default="experiments/03_roma_rgb/oracle_b")
    ap.add_argument("--variant", default="oracle_b", choices=("ipm_cl", "oracle_a", "oracle_b"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--probe", type=int, default=0, help="zero-shot posterior on this many frames, then exit")
    ap.add_argument("--probe-tag", default="", help="extra suffix on the probe json, so a rerun does not overwrite")
    ap.add_argument("--min-valid", type=float, default=0.99, help="RoMa supervises patches above this valid fraction")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--head", default="decoder", choices=("decoder", "cosine", "cross"),
                    help="decoder: Sat-RoMa coarse head. cosine: learned cosine. cross: 2-layer cross-attention, from scratch")
    ap.add_argument("--neighbour-radius", type=int, default=0,
                    help="hinge the true cell above every cell inside this radius (0 = RoMa CE only)")
    ap.add_argument("--neighbour-weight", type=float, default=0.1)
    ap.add_argument("--ckpt", default="", help="warm-start the decoder from this checkpoint")
    ap.add_argument("--distinctive", action="store_true",
                    help="weight each patch by how unique its true reference cell is")
    ap.add_argument("--max-offset", type=float, default=-1.0, help="override reference.max_offset_frac; -1 keeps the config")
    ap.add_argument("--max-rot", type=float, default=-1.0, help="override reference.max_rot_deg; -1 keeps the config")
    ap.add_argument("--eval", action="store_true", help="score --ckpt with RANSAC pose error, no training")
    ap.add_argument("--local-radius", type=int, default=0,
                    help="if >0, CE only over cells within this Chebyshev radius of the GT cell")
    ap.add_argument("--color-match", action="store_true",
                    help="shift the BEV's per-channel mean and std onto the reference crop before the encoder")
    a = ap.parse_args()
    cfg = C.load(a.config)
    if a.max_offset >= 0:
        cfg.reference.max_offset_frac = a.max_offset
    if a.max_rot >= 0:
        cfg.reference.max_rot_deg = a.max_rot
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds, calib, erp_valid = context(cfg)
    names, track, ea = covered_frames(cfg, ds)
    print(f"{len(names)} EA-covered frames, variant {a.variant}, min_valid {a.min_valid}", flush=True)

    if a.probe:
        probe(cfg, ds, calib, erp_valid, names, track, ea, dev, a)
        return
    if a.eval:
        evaluate(cfg, ds, calib, erp_valid, names, track, ea, dev, a)
        return
    train(cfg, ds, calib, erp_valid, names, track, ea, dev, a)


def covered_frames(cfg, ds):
    ortho = C.REPO / cfg.data.ortho
    names = ds.names(("image", "scan", "oxts"))
    track = read_track(ds.root, names)
    edge = cfg.grid.n * cfg.grid.cell_m * cfg.reference.scale
    keep = set(fully_covered_names(ortho, names, track.en, edge, min_frac=0.999))
    names = [n for n in names if n in keep]
    if not names:
        raise SystemExit(f"no frame fully covered by {ortho}")
    return names, track, OrthoMap(ortho)


def probe(cfg, ds, calib, erp_valid, names, track, ea, dev, a):
    pick = np.linspace(0, len(names) - 1, min(a.probe, len(names))).astype(int)
    names = [names[i] for i in dict.fromkeys(pick.tolist())]
    matcher = SatRoMa.from_config(cfg)
    sf = matcher.m.im_a_size / 560.0
    rows = []
    t0 = time.time()
    for name in names:
        bev, ref, H, valid = frame(cfg, ds, calib, erp_valid, track, ea, name, a.variant,
                                   color_match=a.color_match)
        f_q = matcher.encode(bev, matcher.m.im_a_size)
        f_s = matcher.encode(ref, matcher.m.im_b_size)
        with matcher.m.model.exposed_intermediates(), torch.no_grad():
            gm = matcher.m.model.decoder({16: f_q}, {16: f_s}, scale_factor=sf)[16]["gm_cls"]
        st = posterior_stats(gm, valid, H, a.min_valid, dev)
        st.update(patch_cosine(f_q, f_s, valid, H, a.min_valid))
        rows.append(st)
        print(f"  {name}  top1 {st['top1']:.2f}  top5 {st['top5']:.2f}  "
              f"top16 {st['top16']:.2f}  mass {st['gt_mass']:.4f}  "
              f"H {st['entropy']:.2f}  cos {st['cos_gt']:.3f}/{st['cos_rnd']:.3f}  "
              f"cos_c {st['cos_gt_c']:.3f}/{st['cos_rnd_c']:.3f}  n {st['n']}", flush=True)
    m = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    tag = a.variant + ("_colormatch" if a.color_match else "") + (f"_{a.probe_tag}" if a.probe_tag else "")
    out = Path("experiments/03_roma_rgb") / f"probe_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": m, "frames": rows, "names": names}, indent=2))
    print(f"PROBE {a.variant}  n_frames {len(rows)}  {m}  ({time.time() - t0:.0f}s)  -> {out}", flush=True)


def train(cfg, ds, calib, erp_valid, names, track, ea, dev, a):
    out = Path(a.out)
    C.snapshot(cfg, out, dict(variant=a.variant, min_valid=a.min_valid, steps=a.steps,
                              local_radius=a.local_radius, color_match=a.color_match))
    in_val = lambda n: any(lo <= int(n) <= hi for lo, hi in cfg.train.val_ranges)
    tr = [n for n in names if not in_val(n)]
    va = [n for n in names if in_val(n)]
    if len(va) < 8 or not tr:
        n_va = max(16, len(names) // 8)
        va, tr = names[-n_va:], names[:-n_va]
    va = va[:: max(1, len(va) // 16)][:16]
    print(f"train {len(tr)} / val {len(va)}", flush=True)

    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=(a.head == "decoder")).train()
    head = None
    params = []
    if a.head == "decoder":
        for n, p in matcher.model.decoder.named_parameters():
            # The fine refiner is what Sat-RoMa turns off under appearance change.
            p.requires_grad = "conv_refiner" not in n
            if p.requires_grad:
                params.append(p)
        lr = cfg.train.lr_decoder
        print(f"trainable coarse decoder {sum(p.numel() for p in params) / 1e6:.2f} M, fine refiner frozen", flush=True)
    else:
        for p in matcher.model.decoder.parameters():
            p.requires_grad = False
        head = (TokenCosine() if a.head == "cosine" else CrossMatch()).to(dev)
        params = list(head.parameters())
        lr = 1e-3
        print(f"trainable {a.head} head {sum(p.numel() for p in params) / 1e6:.3f} M, decoder frozen", flush=True)
    if a.ckpt and a.head == "decoder":
        state = torch.load(a.ckpt, map_location=dev, weights_only=False)
        matcher.model.decoder.load_state_dict(state["decoder"])
        print(f"loaded decoder from {a.ckpt}", flush=True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.train.weight_decay)
    sf = 224 / 560.0
    rng = np.random.default_rng(cfg.train.seed)
    log_path = out / "log.csv"
    log_f = log_path.open("w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "split", "ce", "cert", "top1", "top5", "cell_err_m", "sec_per_step"])
    t0 = time.time()
    for step in range(1, a.steps + 1):
        batch = [frame(cfg, ds, calib, erp_valid, track, ea, n, a.variant, rng, a.color_match)
                 for n in rng.choice(tr, a.batch)]
        loss, st = roma_step(matcher, batch, dev, sf, a.min_valid, a.local_radius, head,
                             a.neighbour_radius, a.neighbour_weight, a.distinctive)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 25 == 0 or step == a.steps:
            dt = (time.time() - t0) / step
            print(f"step {step:4d}  CE {st['ce']:.3f}  cert {st['cert']:.3f}  "
                  f"top1 {st['acc']:.1%}  top5 {st['top5']:.1%}  argmax {st['cell_err'] * CELL_M:.1f} m  "
                  f"{dt:.2f} s/step", flush=True)
            log.writerow([step, "train", f"{st['ce']:.4f}", f"{st['cert']:.4f}",
                          f"{st['acc']:.4f}", f"{st['top5']:.4f}", f"{st['cell_err'] * CELL_M:.2f}", f"{dt:.3f}"])
            log_f.flush()
        if step % 100 == 0 or step == a.steps:
            stats = [roma_step(matcher, [frame(cfg, ds, calib, erp_valid, track, ea, n, a.variant,
                                                color_match=a.color_match)],
                                dev, sf, a.min_valid, a.local_radius, head,
                                a.neighbour_radius, a.neighbour_weight, a.distinctive)[1]
                     for n in va]
            vce, v1, v5, verr = (np.nanmean([s[k] for s in stats]) for k in ("ce", "acc", "top5", "cell_err"))
            print(f"step {step:4d}  VAL  CE {vce:.3f}  top1 {v1:.1%}  top5 {v5:.1%}  "
                  f"argmax {verr * CELL_M:.1f} m", flush=True)
            log.writerow([step, "val", f"{vce:.4f}", "", f"{v1:.4f}", f"{v5:.4f}", f"{verr * CELL_M:.2f}", ""])
            log_f.flush()
    ck = C.REPO / "checkpoints" / f"{out.parent.name}_{out.name}.pt"
    ck.parent.mkdir(exist_ok=True)
    payload = {"step": a.steps, "variant": a.variant, "head": a.head}
    if head is None:
        payload["decoder"] = matcher.model.decoder.state_dict()
    else:
        payload[a.head] = head.state_dict()
    torch.save(payload, ck)
    print("saved", ck)


def evaluate(cfg, ds, calib, erp_valid, names, track, ea, dev, a):
    """RANSAC pose of a trained coarse decoder. Argmax cell distance is not a pose."""
    from bevloc.eval.metrics import pose_errors, recall
    if not a.ckpt:
        raise SystemExit("--eval needs --ckpt (or --ckpt frozen)")
    matcher = SatRoMa.from_config(cfg, min_valid_frac=a.min_valid)
    if a.ckpt != "frozen":
        state = torch.load(a.ckpt, map_location=dev, weights_only=False)
        matcher.m.model.decoder.load_state_dict(state["decoder"])
    in_val = lambda n: any(lo <= int(n) <= hi for lo, hi in cfg.train.val_ranges)
    va = [n for n in names if in_val(n)]
    tr = [n for n in names if not in_val(n)]
    pick = [(n, "val") for n in va[:: max(1, len(va) // 24)][:24]]
    pick += [(n, "train") for n in tr[:: max(1, len(tr) // 12)][:12]]
    rows = []
    for name, split in pick:
        bev, ref, H, valid = frame(cfg, ds, calib, erp_valid, track, ea, name, a.variant, color_match=a.color_match)
        m = matcher.match(bev, ref, mask=valid, H_gt=H)
        if m.H is None:
            pos = yaw = None
        else:
            e = pose_errors(m.H, H, 224, cfg.grid.cell_m)
            pos, yaw = e["position_m"], e["yaw_deg"]
        arg = None if m.argmax_cells is None else m.argmax_cells * CELL_M
        rows.append(dict(name=name, split=split, position_m=pos, yaw_deg=yaw, argmax_m=arg, modes=m.n_modes))
        print(f"  {split:5s} {name}  pose {pos if pos is None else round(pos, 1)} m  "
              f"yaw {yaw if yaw is None else round(yaw, 1)}  argmax {arg if arg is None else round(arg, 1)} m  "
              f"modes {m.n_modes}", flush=True)
    for split in ("val", "train"):
        sub = [r for r in rows if r["split"] == split]
        pos = [r["position_m"] for r in sub]
        arg = [r["argmax_m"] for r in sub if r["argmax_m"] is not None]
        print(f"EVAL {split}  n {len(sub)}  {recall(pos)}  "
              f"median_pose {np.nanmedian([np.inf if p is None else p for p in pos]):.1f} m  "
              f"median_argmax {np.median(arg):.1f} m", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "frozen" if a.ckpt == "frozen" else "trained"
    (out / f"pose_eval_{tag}.json").write_text(json.dumps(rows, indent=2))


def frame(cfg, ds, calib, erp_valid, track, ea, name, variant, rng=None, color_match=False):
    i = track.names.index(name)
    from bevloc.data.ortho import Oriented
    bearing = float(track.bearing(cfg.oxts.convention, grid=True)[i])
    q = Oriented(tuple(track.en[i]), bearing, cfg.grid.n, cfg.grid.cell_m)
    rng = np.random.default_rng([cfg.matcher.seed, int(name)]) if rng is None else rng
    r = cfg.reference
    ref_o = sample_reference(q, rng, r.scale, r.max_offset_frac, r.max_rot_deg)
    img, _ = ea.render(ref_o)
    bev = build_variants(ds.erp(name), ds.points(name), erp_valid, calib, cfg, name=name, variants=(variant,))
    image = bev.images[variant]
    if color_match:
        image = match_color(image, img)
    return image, img, gt_homography(q, ref_o), bev.valid[variant]


def match_color(bev, ref):
    """Per-channel mean/std of the BEV onto the reference. Same pixels, satellite-like contrast."""
    b = bev.astype(np.float32)
    r = ref.astype(np.float32)
    for c in range(b.shape[-1]):
        bs = float(b[..., c].std())
        if bs < 1.0:
            continue
        b[..., c] = (b[..., c] - b[..., c].mean()) / bs * float(r[..., c].std()) + float(r[..., c].mean())
    return np.clip(b, 0, 255).astype(bev.dtype)


class TokenCosine(torch.nn.Module):
    """Per-patch categorical from a learned cosine. This is the BEV-Patch-PF objective, at patch level.

    Two linear maps (query, reference) then temperature-scaled cosine over the 56x56 cells.
    No Gaussian-process matcher, no certainty head.
    """

    def __init__(self, dim=1024, width=256):
        super().__init__()
        self.query = torch.nn.Linear(dim, width, bias=False)
        self.ref = torch.nn.Linear(dim, width, bias=False)
        self.log_temp = torch.nn.Parameter(torch.zeros(()))

    def forward(self, f_q, f_s):
        B, _, h, w = f_q.shape
        q = F.normalize(self.query(f_q.flatten(2).transpose(1, 2)), dim=-1)
        r = F.normalize(self.ref(f_s.flatten(2).transpose(1, 2)), dim=-1)
        temp = self.log_temp.exp().clamp(min=0.05)
        logits = torch.matmul(q, r.transpose(1, 2)) / temp
        return logits.transpose(1, 2).reshape(B, -1, h, w)


class CrossMatch(torch.nn.Module):
    """Two cross-attention layers, then a cosine over reference cells. No pretrained decoder.

    This is the RoMa idea (a query patch looks at the reference, then classifies a cell)
    without the released 67M Gaussian-process head. Compared with TokenCosine it can mix
    patches; compared with the decoder it has no aerial-matching prior.
    """

    def __init__(self, dim=1024, width=256, layers=2, heads=4):
        super().__init__()
        self.query = torch.nn.Linear(dim, width)
        self.ref = torch.nn.Linear(dim, width)
        layer = torch.nn.TransformerDecoderLayer(
            width, heads, dim_feedforward=width * 4, batch_first=True, norm_first=True, dropout=0.0)
        self.mix = torch.nn.TransformerDecoder(layer, layers)
        self.log_temp = torch.nn.Parameter(torch.zeros(()))

    def forward(self, f_q, f_s):
        B, _, h, w = f_q.shape
        q = self.query(f_q.flatten(2).transpose(1, 2))
        r = self.ref(f_s.flatten(2).transpose(1, 2))
        hdn = self.mix(q, r)
        temp = self.log_temp.exp().clamp(min=0.05)
        logits = torch.matmul(F.normalize(hdn, dim=-1), F.normalize(r, dim=-1).transpose(1, 2)) / temp
        return logits.transpose(1, 2).reshape(B, -1, h, w)


def cell_uniqueness(f_s):
    """(B, cells*cells) in [0, 1]: 1 when a reference cell matches no other cell.

    Mean-centred first. Raw DINOv3 cosine is ~0.9 between every pair, so uniqueness
    computed on it is ~0 everywhere and the weight does nothing.
    """
    r = f_s.float().flatten(2).transpose(1, 2)
    r = F.normalize(r - r.mean(1, keepdim=True), dim=-1)
    sim = torch.matmul(r, r.transpose(1, 2))
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool)
    return (1 - sim.masked_fill(eye[None], -1).amax(-1)).clamp(0, 1)


def roma_step(matcher, batch, dev, sf, min_valid, local_radius=0, head=None,
              neighbour_radius=0, neighbour_weight=0.1, distinctive=False):
    qs, refs, Hs, valids = zip(*batch)
    q = torch.stack([SatRoMaMatcher_image(x, 224, dev) for x in qs])
    ref = torch.stack([SatRoMaMatcher_image(x, 896, dev) for x in refs])
    H = torch.tensor(np.stack(Hs), dtype=torch.float32, device=dev)
    with torch.no_grad():
        f_q = matcher.model.encoder(q)[16]
        f_s = matcher.model.encoder(ref)[16]
    if head is None:
        out = matcher.model.decoder({16: f_q}, {16: f_s}, scale_factor=sf)[16]
        gm, cert = out["gm_cls"], out.get("gm_certainty")
    else:
        gm, cert = head(f_q, f_s), None
    frac = patch_fraction(valids, dev)
    rv = ref_cell_validity(ref, min_frac=0.5)
    idx, inside = coarse_targets(H, torch.ones_like(frac, dtype=torch.bool), ref_valid=rv)
    matchable = inside & (frac >= min_valid)
    weight = None
    if distinctive:
        u = cell_uniqueness(f_s)
        # idx is row-major, the same order cell_uniqueness flattened in.
        weight = u.gather(1, idx.reshape(idx.shape[0], -1)).reshape_as(idx)
    return roma_coarse_loss(gm, idx, matchable, cert, certainty_weight=0.01,
                            local_radius=local_radius, neighbour_radius=neighbour_radius,
                            neighbour_weight=neighbour_weight, patch_weight=weight)


def patch_fraction(valids, dev):
    t = torch.from_numpy(np.stack([v.astype(np.float32) for v in valids]))[:, None].to(dev)
    return F.avg_pool2d(t, 16)[:, 0]


def SatRoMaMatcher_image(img, size, dev):
    return SatRoMaMatcher.load_image(img, size).to(dev)


def patch_cosine(f_q, f_s, valid, H, min_valid):
    """Cosine of each matchable query token to its GT reference cell, and to a random cell.

    If these two are the same, the frozen sat493m tokens do not rank the true cell,
    and no classifier on top of them can localise.
    """
    frac = patch_fraction([valid], f_q.device)
    H_t = torch.as_tensor(H, dtype=torch.float32, device=f_q.device)[None]
    idx, inside = coarse_targets(H_t, torch.ones_like(frac, dtype=torch.bool))
    use = inside & (frac >= min_valid)
    q = f_q.float().permute(0, 2, 3, 1)[use]
    r = idx[use]
    if r.numel() == 0:
        return dict(cos_gt=np.nan, cos_rnd=np.nan)
    ref = f_s[0].float().flatten(1).T
    gt = F.normalize(ref[r], dim=-1)
    rnd = F.normalize(ref[torch.randint(0, ref.shape[0], (r.shape[0],), device=ref.device)], dim=-1)
    qn = F.normalize(q, dim=-1)
    # DINOv3 tokens share a dominant direction, so raw cosine is ~0.92 to every cell.
    # Subtract each side's mean before the second cosine; that is the signal a cosine loss sees.
    q_c = F.normalize(q - q.mean(0, keepdim=True), dim=-1)
    ref_c = F.normalize(ref - ref.mean(0, keepdim=True), dim=-1)
    return dict(
        cos_gt=float((qn * gt).sum(-1).mean()),
        cos_rnd=float((qn * rnd).sum(-1).mean()),
        cos_gt_c=float((q_c * ref_c[r]).sum(-1).mean()),
        cos_rnd_c=float((q_c * ref_c[torch.randint(0, ref.shape[0], (r.shape[0],), device=ref.device)]).sum(-1).mean()),
    )


def posterior_stats(gm, valid, H, min_valid, dev):
    """Zero-shot: where the frozen categorical puts the ground-truth cell."""
    frac = patch_fraction([valid], dev)
    H_t = torch.as_tensor(H, dtype=torch.float32, device=gm.device)[None]
    idx, inside = coarse_targets(H_t, torch.ones_like(frac, dtype=torch.bool))
    use = inside & (frac >= min_valid)
    logits = gm.float().permute(0, 2, 3, 1)[use]
    tgt = idx[use]
    if tgt.numel() == 0:
        return dict(top1=np.nan, top5=np.nan, top16=np.nan, gt_mass=np.nan, entropy=np.nan, n=0)
    log_p = torch.log_softmax(logits, dim=-1)
    p = log_p.exp()
    rank = (logits > logits.gather(1, tgt[:, None])).sum(1)
    return dict(
        top1=float((rank < 1).float().mean()),
        top5=float((rank < 5).float().mean()),
        top16=float((rank < 16).float().mean()),
        gt_mass=float(p.gather(1, tgt[:, None]).mean()),
        entropy=float((-(p * log_p).sum(-1)).mean()),
        n=int(tgt.numel()),
    )


if __name__ == "__main__":
    main()
