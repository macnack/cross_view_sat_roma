"""Train the pose-correctness head on a calibration cache (task 06, step 4; docs/tasks/06_certainty.md "Head").

  make vigor-cert-head CACHE=experiments/10_loc2_matcher/cert_cache/erpd4city_calib.pkl TAG=erpd4city \
       [STREAMS="map tokens frame"] [HEAD_ARGS="--epochs 60 --seed 0"]

Data: a `--draw calib` cache of scripts/certainty_cache_vigor.py (held-out training frames of the frozen matcher; a
test-draw cache is refused). The LAST cfg.certainty_head.val_frames frames of the cache (draw order) are the
validation frames: early stopping and the temperature are the only things fitted on them; the rest trains the head.
Targets: BCE with logits on the events "coarse error < tau" for every tau of cfg.certainty.targets_m (a frame
without a pose is wrong for every tau); the tau = cfg.certainty.rank_target_m logit is the one the abstention curve
ranks by, and its validation NLL is the early-stopping criterion (best epoch restored). AdamW, batch, epochs, lr,
weight decay, patience, hidden sizes and the augmentation switch come from the config block (CLI overrides); the
augmentation is the D4 action of `bevloc.model.certainty_head.augment` (random quarter turns + mirror of the maps
with the token grid positions moved consistently, the label unchanged). After training, one temperature per target
is fitted on the validation frames (minimum validation NLL of sigmoid(logit / T)); the frame statistics are
standardised on the training frames (buffers in the state dict).
--streams map|tokens|frame (several): the ablation rows; the frame stream alone is the pipeline check (it must
reproduce the logistic of scripts/certainty_vigor.py within noise).
Writes <out>/certainty_head_<tag>_<streams>.pt (state dict + head config + standardisation + temperature + the
training curve + the cache header) and certainty_head_<tag>_<streams>.json (the curve and validation metrics).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bevloc import config as C
from bevloc.eval import calibration as K
from bevloc.model.certainty_head import (
    CacheArrays, CertaintyHead, augment, canonical_streams, load_cache, n_parameters, save_head, streams_tag,
)

HEAD_KEYS = ("lr", "weight_decay", "batch", "epochs", "patience", "val_frames", "seed")


def head_cfg(cfg):
    """The `certainty_head:` block of the run's config, else configs/default.yaml's (the VIGOR configs are full
    copies written before the block existed; the same fallback for `certainty:` in main)."""
    return getattr(cfg, "certainty_head", None) or C.load().certainty_head


def bce(logits, y):
    """Mean binary cross-entropy of sigmoid(logits) against y, numpy, per column: (n_targets,)."""
    z = np.asarray(logits, np.float64)
    y = np.asarray(y, np.float64)
    return (np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1.0 - y)).mean(0)


def fit_temperature(logits, y, lo=-4.0, hi=4.0, iters=60):
    """One temperature per target: T = argmin over [e^lo, e^hi] (golden section on log T) of the BCE of
    sigmoid(logit / T); 1 when the target has one class only (nothing to calibrate)."""
    z = np.asarray(logits, np.float64)
    y = np.asarray(y, np.float64)
    out = np.ones(z.shape[1])
    g = (np.sqrt(5.0) - 1.0) / 2.0
    for j in range(z.shape[1]):
        if y[:, j].min() == y[:, j].max():
            continue

        def f(t):
            return bce(z[:, j:j + 1] / np.exp(t), y[:, j:j + 1])[0]
        a, b = lo, hi
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = f(d)
        t = 0.5 * (a + b)
        out[j] = np.exp(t) if f(t) <= f(0.0) else 1.0            # never worse than no scaling
    return out


def val_metrics(logits, y, temperature=None, n_bins=10):
    """Per target: nll, auroc, ece (after the temperature when given)."""
    z = np.asarray(logits, np.float64)
    if temperature is not None:
        z = z / np.asarray(temperature, np.float64)[None]
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    nll = bce(z, y)
    return [dict(nll=float(nll[j]), auroc=K.auroc(p[:, j], y[:, j] > 0.5), ece=K.reliability(p[:, j], y[:, j], n_bins)[1])
            for j in range(z.shape[1])]


@torch.no_grad()
def logits_of(head, D, idx, batch):
    head.eval()
    out = []
    for lo in range(0, len(idx), batch):
        sl = idx[lo:lo + batch]
        out.append(head(maps=D["maps"][sl].float(), tokens=D["tokens"][sl].float(), token_valid=D["valid"][sl],
                        frame=D["frame"][sl]).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, head.n_targets))


def train(arrays, streams, hc, targets_m, rank_target_m, dev, epochs, lr, weight_decay, batch, patience, val_frames,
          seed, aug, log=print):
    """Train on all but the last val_frames frames of `arrays` (CacheArrays), early stop on the validation NLL at
    the rank target, fit the temperature. Returns (head on cpu, record dict)."""
    n = len(arrays)
    if val_frames < 1 or n - val_frames < 1:
        raise SystemExit(f"{n} frames cannot be split into {n - val_frames} training and {val_frames} validation frames")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    rng = np.random.default_rng(int(seed))
    tr, va = np.arange(n - val_frames), np.arange(n - val_frames, n)
    y_all = arrays.labels(targets_m)
    j_rank = [float(t) for t in targets_m].index(float(rank_target_m))
    head = CertaintyHead(streams=streams, n_targets=len(targets_m), n_frame=arrays.frame.shape[1],
                         n_token=arrays.tokens.shape[-1], map_channels=tuple(hc.map_channels),
                         token_dim=int(hc.token_dim), frame_dim=int(hc.frame_dim), fusion_dim=int(hc.fusion_dim),
                         dropout=float(hc.dropout))
    head.set_standardisation(arrays.frame[tr], arrays.frame_log)
    head.to(dev)
    D = dict(maps=torch.as_tensor(arrays.maps).to(dev), tokens=torch.as_tensor(arrays.tokens).to(dev),
             valid=torch.as_tensor(arrays.token_valid).to(dev), frame=torch.as_tensor(arrays.frame).to(dev),
             y=torch.as_tensor(y_all).to(dev))
    opt = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    log(f"head {streams_tag(streams)}: {n_parameters(head)} parameters; {len(tr)} training / {len(va)} validation "
        f"frames; base rates {[round(float(v), 3) for v in y_all[tr].mean(0)]} (train) "
        f"{[round(float(v), 3) for v in y_all[va].mean(0)]} (val); aug {'on' if aug else 'off'}; device {dev}")
    curve, best, best_state, best_epoch, since = [], np.inf, copy.deepcopy(head.state_dict()), 0, 0
    t0 = time.time()
    for ep in range(1, int(epochs) + 1):
        head.train()
        perm = rng.permutation(tr)
        tot, nb = 0.0, 0
        for lo in range(0, len(perm), int(batch)):
            sl = torch.as_tensor(perm[lo:lo + int(batch)], device=dev)
            maps, tokens = D["maps"][sl].float(), D["tokens"][sl].float()
            if aug:
                k = torch.as_tensor(rng.integers(0, 4, len(sl)), device=dev)
                flip = torch.as_tensor(rng.random(len(sl)) < 0.5, device=dev)
                maps, tokens = augment(maps, tokens, k, flip, arrays.grid)
            logits = head(maps=maps, tokens=tokens, token_valid=D["valid"][sl], frame=D["frame"][sl])
            loss = F.binary_cross_entropy_with_logits(logits, D["y"][sl])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        zv = logits_of(head, D, va, int(batch))
        vm = val_metrics(zv, y_all[va])
        nll = vm[j_rank]["nll"]
        curve.append(dict(epoch=ep, train_loss=tot / max(1, nb), val=vm, val_nll_rank=nll, sec=time.time() - t0))
        improved = nll < best - 1e-7
        if improved:
            best, best_state, best_epoch, since = nll, copy.deepcopy(head.state_dict()), ep, 0
        else:
            since += 1
        log(f"  epoch {ep:3d}  train {tot / max(1, nb):.4f}  val nll@{rank_target_m:g}m {nll:.4f}  "
            f"auroc {[round(m['auroc'], 3) if m['auroc'] == m['auroc'] else None for m in vm]}"
            f"{'  *' if improved else ''}")
        if since >= int(patience):
            log(f"  early stop: no improvement for {patience} epochs (best epoch {best_epoch})")
            break
    head.load_state_dict(best_state)
    zv = logits_of(head, D, va, int(batch))
    T = fit_temperature(zv, y_all[va])
    with torch.no_grad():
        head.temperature.copy_(torch.as_tensor(T, dtype=torch.float32))
    rec = dict(best_epoch=best_epoch, epochs_run=len(curve), curve=curve, temperature=[float(t) for t in T],
               val_before=val_metrics(zv, y_all[va]), val_after=val_metrics(zv, y_all[va], T),
               n_train=int(len(tr)), n_val=int(len(va)), train_ids=[arrays.ids[i] for i in tr],
               val_ids=[arrays.ids[i] for i in va], n_params=n_parameters(head), seconds=float(time.time() - t0))
    return head.cpu(), rec


def main(argv=None):
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--cache", required=True, help="calibration cache (certainty_cache_vigor.py --draw calib)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="experiments/10_loc2_matcher")
    ap.add_argument("--streams", nargs="+", default=None, help="map tokens frame (default: the config's)")
    for k in HEAD_KEYS:
        ap.add_argument(f"--{k.replace('_', '-')}", type=(int if k in ("batch", "epochs", "patience", "val_frames", "seed")
                                                         else float), default=None)
    ap.add_argument("--no-aug", action="store_true", help="no rotation / mirror augmentation (config: augment)")
    ap.add_argument("--allow-test-draw", action="store_true", help="train on a test-draw cache (never for a table)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = C.load(a.config)
    hc, cc = head_cfg(cfg), getattr(cfg, "certainty", None) or C.load().certainty
    kw = {k: (getattr(hc, k) if getattr(a, k) is None else getattr(a, k)) for k in HEAD_KEYS}
    streams = canonical_streams(a.streams or list(hc.streams))
    aug = bool(hc.augment) and not a.no_aug
    targets = [float(t) for t in cc.targets_m]
    if float(cc.rank_target_m) not in targets:
        sys.exit(f"certainty.rank_target_m {cc.rank_target_m} must be one of targets_m {targets}")
    cache = load_cache(a.cache)
    if cache["meta"].get("draw") != "calib" and not a.allow_test_draw:
        sys.exit(f"{a.cache} is a {cache['meta'].get('draw')!r} draw, not 'calib': the head trains on held-out training "
                 f"frames only (--allow-test-draw for a smoke test)")
    arrays = CacheArrays(cache)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cache {a.cache}: {len(arrays)} frames, cities {cache['meta'].get('cities')}, checkpoint "
          f"{cache['meta'].get('ckpt')}, consensus {cache['meta'].get('consensus')}, gate "
          f"{(cache['meta'].get('gate') or {}).get('mismatches')} mismatches", flush=True)
    head, rec = train(arrays, streams, hc, targets, float(cc.rank_target_m), dev, epochs=kw["epochs"], lr=kw["lr"],
                      weight_decay=kw["weight_decay"], batch=kw["batch"], patience=kw["patience"],
                      val_frames=kw["val_frames"], seed=kw["seed"], aug=aug, log=lambda s: print(s, flush=True))
    m = cache["meta"]
    extra = dict(tag=a.tag, streams=list(streams), targets_m=targets, rank_target_m=float(cc.rank_target_m),
                 frame_keys=arrays.frame_keys, frame_log=[bool(x) for x in arrays.frame_log], token_grid=arrays.grid,
                 token_keys=m.get("token_keys"), map_keys=m.get("map_keys"),
                 cache=dict(path=str(a.cache), ckpt=m.get("ckpt"), config=m.get("config"), solver=m.get("solver"),
                            consensus=m.get("consensus"), draw=m.get("draw"), draw_info=m.get("draw_info"),
                            cities=m.get("cities"), n=m.get("n"), gate=m.get("gate")),
                 train_opts=dict(kw, augment=aug, map_channels=list(hc.map_channels), token_dim=int(hc.token_dim),
                                 frame_dim=int(hc.frame_dim), fusion_dim=int(hc.fusion_dim), dropout=float(hc.dropout)),
                 command=" ".join(sys.argv), **rec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"certainty_head_{a.tag}_{streams_tag(streams)}"
    pt = save_head(out / f"{name}.pt", head, extra)
    js = out / f"{name}.json"
    js.write_text(json.dumps({k: v for k, v in extra.items() if k not in ("train_ids", "val_ids")}, indent=2,
                             default=float))
    j = targets.index(float(cc.rank_target_m))
    b, c = rec["val_before"][j], rec["val_after"][j]
    print(f"best epoch {rec['best_epoch']} of {rec['epochs_run']}; validation at {cc.rank_target_m:g} m: nll "
          f"{b['nll']:.4f} -> {c['nll']:.4f} (T = {rec['temperature'][j]:.3f}), auroc {c['auroc']:.4f}, ece "
          f"{b['ece']:.4f} -> {c['ece']:.4f}; {rec['n_params']} parameters, {rec['seconds']:.0f} s", flush=True)
    print(f"wrote {pt} and {js}", flush=True)
    return pt


if __name__ == "__main__":
    main()
