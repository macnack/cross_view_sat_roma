"""Sweep the Sat-RoMa consensus (mode extraction + multi-hypothesis RANSAC) on VIGOR from a decoder cache.

  make vigor-sweep CKPT=... CONFIG=configs/vigor_cell0125.yaml SPLIT=samearea CITIES=Chicago LIMIT=3000 TAG=...
       [CALIB_CITIES="Chicago NewYork SanFrancisco Seattle" CALIB_LIMIT=1000] [FINE_CONFIG=... FINE_CKPT=...]
       [SWEEP="--solvers se2 --workers 16"]

Why: the consensus parameters are the Sat-RoMa paper's (4 m cells, 224 m crops, 14 x 14 query tokens); we run 2 m cells
on a 112 m canvas, 1 m cells in a 56 m fine window and 56 x 28 panorama tokens. The parameters and their package
values are listed in `bevloc.match.consensus`.

Stage 1, cache (`--stage cache`, GPU): the decoder runs once per frame (the coarse pass; with --fine-config/--fine-ckpt
also the fine pass, the machinery of eval_vigor.py: window centred on the coarse pose of the caching consensus) and
stores per frame what the consensus needs:
  - the mode list of the package's own extraction (`extract_modes` = find_gaussians, fixed 4 px window) at the LOOSEST
    threshold (--cache-thr, 0.004): token (col, row), means, peaks, window mass, blurred height. The per-token categorical
    itself (K*K x h*w = 3136 x 1568 floats per frame) is too large, and the top-K cells are NOT enough to rebuild the
    modes: the threshold is tested on the 5 x 5 Gaussian-blurred categorical, whose value at a cell depends on up to 25
    raw cells, and a mode's mean is a moment over a 5 x 5 window of that blurred map. So the cache keeps the modes, and
    every threshold >= the cache threshold is an exact filter of them (height > thr; the local-maximum test and the
    means do not depend on the threshold).
  - also the top-8 cells (index, probability) of every valid token, the token certainty logits, the validity mask,
    the placed BEV points (erp / erp_depth), the reference-cell validity, the GT homography and the online pose error
    of the caching consensus (the sweep checks that it reproduces it exactly).
  Not sweepable from the cache: thresholds below --cache-thr, the peak window size (fixed_window_size), the adaptive
  extraction (adaptive_gauss_fit), the patch-validity rule (min_valid_frac), anything upstream of the logits, and --
  for the fine pass -- the coarse consensus (the fine window is centred on the caching run's coarse pose).
Stage 2, sweep (`--stage sweep`, CPU): from the cache, the consensus for every configuration of a one-factor-at-a-time
pass around the caching configuration (the current default), then a joint grid over the --joint-top most sensitive
factors (range of the calibration objective), with median (bootstrap CI), mean (capped at 1 km), R@5, R@10, gross-miss
fraction (> --gross-m m or no pose), modes used per frame and inlier ratio per configuration.
Protocol: selection happens on the CALIBRATION draw only (held-out training frames, `eval_vigor.calib_split`; with a
fine pass the frames held out for BOTH checkpoints); the test draw (e.g. the 3000 Chicago samples) is scored for the
default and the chosen configuration only, reported separately. Selecting on a test cache is refused. The best
calibration configuration replaces the default only if its paired bootstrap interval (objective change against the
default on the same frames) lies below zero (--no-guard: always). Output:
<out>/sweep_consensus_<tag>.json (+ .md): `chosen` (coarse) and `chosen_fine` blocks with the config and its calib / test
numbers, which `eval_vigor.py --consensus-json` applies.
Determinism: every RANSAC is seeded with cfg.matcher.seed exactly as eval_vigor (cv2.setRNGSeed / numpy seed per call).
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from bevloc import config as C
from bevloc.data.vigor import VigorPairs, pose_en, read_labels, split_cities
from bevloc.eval.metrics import pose_errors
from bevloc.eval.report import summarise_pose
from bevloc.match.consensus import (
    CERTS, PACKAGE_WINDOW, RANSAC_MULT, SOLVERS, TARGETS, ConsensusCfg, apply_settings, extract_modes, mode_weights,
    select_modes, settings_of,
)
from bevloc.match.satroma import SatRoMa, consensus_for_query, query_tokens
from bevloc.match.vote_stats import ref_cell_valid


def _ev():
    """scripts/eval_vigor.py under a unique name (third_party/Loc2 also ships an eval_vigor.py)."""
    name = "bevloc_scripts_eval_vigor"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "eval_vigor.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FACTORS = ("reproj_cells", "target", "max_modes", "mode_thr", "cert", "solver", "ransac")
TOPK = 8
SWEEP_KEYS = ("cache_thr", "calib_limit", "select", "joint_top", "gross_m", "n_boot")


def sweep_cfg(config):
    """The `consensus_sweep:` block of the run's config, else configs/default.yaml's (the VIGOR configs are full copies
    written before the block existed)."""
    return getattr(C.load(config), "consensus_sweep", None) or C.load().consensus_sweep


def fill_defaults(a):
    """CLI values left at None take the config block's."""
    sc = sweep_cfg(a.config)
    for k in SWEEP_KEYS:
        if getattr(a, k) is None:
            setattr(a, k, getattr(sc, k))
    return sc


# ---- stage 1: cache ------------------------------------------------------------------------------------------------

def frame_record(cons, gm, query, batch, frac, n, cert, thr, topk=TOPK):
    """What the consensus of one decoded sample needs (see module doc). gm (K*K, h, w) logits, cert (h, w) or None."""
    valid, xy = query_tokens(cons, query, batch, frac, n, min_frac=0.05)
    valid = torch.as_tensor(valid, dtype=torch.bool, device=gm.device)
    gmm = gm.detach().float().clone()
    gmm[:, ~valid] = 0.0                                             # as consensus_for_query: masked tokens vote nothing
    md = extract_modes(gmm, thr)
    k = int(round(gm.shape[0] ** 0.5))
    p = torch.softmax(gmm[:, valid], dim=0)                           # (K*K, n_valid)
    kk = min(int(topk), p.shape[0])
    tp, ti = torch.topk(p, kk, dim=0) if p.shape[1] else (p[:kk], p[:kk].long())
    return dict(
        placed=xy is not None,
        valid=valid.cpu().numpy(),
        xy=None if xy is None else torch.as_tensor(xy).detach().float().cpu().numpy(),
        cert=None if cert is None else cert.detach().float().cpu().numpy(),
        modes=md, in_dim=int(gm.shape[-1]), out_dim=k,
        topk_idx=ti.T.cpu().numpy().astype(np.int16), topk_p=tp.T.cpu().numpy().astype(np.float16))


def solve(rec, c: ConsensusCfg, stub):
    """The consensus c on a cached frame record: Match (as `consensus_for_query` would give with c)."""
    md = rec["modes"]
    idx = select_modes(md, c, rec["cert"], rec["valid"])
    w = mode_weights(md, idx, rec["cert"]) if c.cert == "weight" else None
    apply_settings(stub, c)
    tok, peaks, means = md["tok"][idx], md["peaks"][idx], md["means"][idx]
    if rec["placed"]:
        return SatRoMa.placed_fit(stub, tok, peaks, means, rec["xy"], rec["out_dim"], weights=w)
    return SatRoMa.package_fit(stub, tok, peaks, means, rec["in_dim"], rec["out_dim"], None, weights=w)


def consensus_stub(meta):
    from types import SimpleNamespace as NS
    return NS(m=NS(im_a_size=int(meta["im_a_size"]), im_b_size=int(meta["im_b_size"])), use_means=False,
              reproj=3.0, seed=int(meta["seed"]), solver="srt")


def coarse_error(H, rec, meta):
    return None if H is None else pose_errors(H, rec["H_gt"], int(meta["n"]), float(meta["cell_m"]))["position_m"]


def fine_errors(H, rec, meta_f, gate_m):
    """(fine error, gated error) in the tile frame, the rule of eval_vigor.fine_pass_row."""
    en_f = None if H is None else pose_en(H, rec["centre_en"], int(meta_f["n"]), float(meta_f["cell_m"]), int(rec["S"]))
    en_c, en_gt = rec["en_coarse"], rec["en_gt"]
    shift = None if en_f is None or en_c is None else float(np.linalg.norm(en_f - en_c))
    fallback = en_f is None or (shift is not None and shift > float(gate_m))
    en_g = en_c if fallback else en_f

    def err(en):
        return None if en is None else float(np.linalg.norm(np.asarray(en) - en_gt))
    return err(en_f), err(en_g)


def heldout_ids(root, split, train_meta, val_frac, assume, what):
    """Ordered ids (city/pano) of a checkpoint's held-out training frames after its val_samples (calib_split, all of
    them), over the checkpoint's own training cities."""
    EV = _ev()
    tm = dict(train_meta or {})
    cities = tm.get("cities") or (assume or {}).get("cities")
    if not cities:
        raise SystemExit(f"--calib: {what} records no training cities; pass --assume-train-split (--train-cities / "
                         f"--fine-train-cities)")
    labels = read_labels(Path(root), list(cities), split, True)
    try:
        held, info = EV.calib_split(len(labels), train_meta, val_frac, list(cities), 0, assume)
    except ValueError as e:
        raise SystemExit(f"{what}: {e}")
    return [f"{labels[i]['city']}/{labels[i]['pano']}" for i in held], info


def build_draw(a, cfg, train_meta, draw, fine_train=None):
    """VigorPairs of the calibration or the test draw; (ds, info)."""
    if draw == "test":
        cities = a.cities or split_cities(a.split, a.train_split)
        ds = VigorPairs(a.root, cfg, cities=cities, split=a.split, train=a.train_split, limit=a.limit, stride=a.stride,
                        row_sign=a.row_sign, height_m=a.height)
        return ds, dict(draw="test", cities=list(cities), limit=a.limit)
    assume = None
    if a.assume_train_split:
        if a.val_samples is None:
            raise SystemExit("--assume-train-split needs --val-samples (the training run's)")
        assume = dict(val_frac=a.val_frac, val_samples=a.val_samples,
                      cities=list(a.train_cities or split_cities(a.split, True)))
    ids, info = heldout_ids(a.root, a.split, train_meta, a.val_frac, assume, "--ckpt")
    info = dict(draw="calib", coarse=info)
    if fine_train is not None:
        fassume = None
        if a.fine_assume_train_split:
            if a.fine_val_samples is None or not a.fine_train_cities:
                raise SystemExit("--fine-assume-train-split needs --fine-val-samples and --fine-train-cities")
            fassume = dict(val_frac=a.fine_val_frac, val_samples=a.fine_val_samples, cities=list(a.fine_train_cities))
        fids, finfo = heldout_ids(a.root, a.split, fine_train, a.fine_val_frac, fassume, "--fine-ckpt")
        fset = set(fids)
        n0 = len(ids)
        ids = [i for i in ids if i in fset]                          # held out for BOTH checkpoints
        info.update(fine=finfo, n_heldout_both=len(ids), n_heldout_coarse=n0)
    cities = a.calib_cities or sorted({i.split("/")[0] for i in ids})
    ids = [i for i in ids if i.split("/")[0] in set(cities)]
    if a.calib_limit:
        ids = ids[:a.calib_limit]
    if not ids:
        raise SystemExit("--calib: no held-out frames left")
    ds = VigorPairs(a.root, cfg, cities=sorted({i.split("/")[0] for i in ids}), split=a.split, train=True,
                    row_sign=a.row_sign, height_m=a.height)
    by_id = {f"{lab['city']}/{lab['pano']}": lab for lab in ds.labels}
    ds.labels = [by_id[i] for i in ids]
    info.update(cities=list(cities), limit=a.calib_limit, n=len(ids),
                rule="calib_split of each checkpoint (permutation seed 0 of its train list, last val_frac held out, "
                     "its val_samples skipped), intersected, filtered to the cities, first --calib-limit")
    return ds, info


def cache_draw(a, draw, M):
    """Run the decoder over one draw and write the cache; returns its path."""
    EV = _ev()
    fill_defaults(a)
    cfg, dev = M["cfg"], M["dev"]
    ds, info = build_draw(a, cfg, M["train_meta"], draw, M["fine_train"])
    n_no_depth = ds.keep_with_depth() if ds.depth else 0
    fine = None
    if M["fine_args"] is not None:
        import copy
        fa = copy.copy(M["fine_args"])
        fa.calib = draw == "calib"                                    # the fine dataset reads the train lists then
        fine, _ = EV.decoder_fine(fa, cfg, ds, dev)                   # filters ds to depth in place when needed
    cons = M["cons"]
    n, cell = int(cfg.grid.n), float(cfg.grid.cell_m)
    cc = EV.certainty_cfg(cfg)
    frames, fine_frames = [], []
    print(f"cache {draw}: {len(ds)} frames ({n_no_depth} without depth dropped)", flush=True)
    for i in range(len(ds)):
        try:
            s = ds[i]
        except RuntimeError as e:
            print(f"  skip {i}: {e}", flush=True)
            continue
        batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = M["query"](batch, M["matcher"])
            f_s = M["matcher"].reference_features(batch["ref"])
            sf = float(((f_q.shape[-2] * 16) * (f_q.shape[-1] * 16)) ** 0.5 / 560.0)
            with M["matcher"].model.exposed_intermediates():
                o16 = M["matcher"].model.decoder({16: f_q}, f_s, scale_factor=sf)[16]
        gm = o16["gm_cls"][0]
        cert = o16["gm_certainty"][0, 0] if o16.get("gm_certainty") is not None else None
        rec = frame_record(cons, gm, M["query"], batch, frac, n, cert, a.cache_thr)
        rec.update(id=s["id"], city=s["city"], centre_guess_m=ds.centre_guess_m(i), H_gt=s["H"].numpy().astype(float),
                   ref_valid=ref_cell_valid(batch["ref"], rec["out_dim"], min_frac=float(cc.ref_cell_min_frac)))
        m0 = consensus_for_query(cons, gm, M["query"], batch, frac, n, min_frac=0.05, certainty=cert, stats=False)
        rec["pose_online_m"] = coarse_error(m0.H, rec, dict(n=n, cell_m=cell))
        frames.append(rec)
        if fine is not None:
            S = int(s["ref"].shape[-1])
            centre_c = EV._centre_of(s)
            en_gt = pose_en(rec["H_gt"], centre_c, n, cell, S)
            en_c = pose_en(m0.H, centre_c, n, cell, S) if m0.H is not None else None
            centre_f = en_c if en_c is not None else np.zeros(2)
            try:
                gm_f, cert_f, batch_f, frac_f, s_f = fine.decode(i, torch.from_numpy(np.asarray(centre_f, np.float64)))
                nf = int(fine.cfg.grid.n)
                rf = frame_record(fine.cons, gm_f, fine.query, batch_f, frac_f, nf, cert_f, a.cache_thr)
                rf.update(id=s["id"], city=s["city"], H_gt=s_f["H"].numpy().astype(float),
                          centre_en=EV._centre_of(s_f), S=int(s_f["ref"].shape[-1]), en_gt=en_gt, en_coarse=en_c,
                          coarse_m=rec["pose_online_m"])
                mf = consensus_for_query(fine.cons, gm_f, fine.query, batch_f, frac_f, nf, min_frac=0.05,
                                         certainty=cert_f, stats=False)
                rf["pose_online_m"], rf["pose_online_gated_m"] = fine_errors(
                    mf.H, rf, dict(n=nf, cell_m=float(fine.cfg.grid.cell_m)), a.fine_gate)
            except Exception as e:                                   # keep the coarse record (eval_vigor's rule)
                if EV._is_oom(e):
                    raise
                print(f"  fine pass failed on {i}: {type(e).__name__}: {e}", flush=True)
                rf = dict(id=s["id"], city=s["city"], error=f"{type(e).__name__}: {e}", en_gt=en_gt, en_coarse=en_c,
                          coarse_m=rec["pose_online_m"])
            fine_frames.append(rf)
        if len(frames) % 100 == 0:
            print(f"  {len(frames)} cached", flush=True)
    base = settings_of(cons).to_dict()
    common = dict(ckpt=a.ckpt, config=a.config, split=a.split, draw=draw, draw_info=info, cache_thr=float(a.cache_thr),
                  window=PACKAGE_WINDOW, topk=TOPK, seed=int(cfg.matcher.seed), skipped_no_depth=n_no_depth,
                  ids=[f["id"] for f in frames], mode=M["mode"], train=M["train_meta"])
    out = dict(meta=dict(common, n=n, cell_m=cell, im_a_size=int(cons.m.im_a_size), im_b_size=int(cons.m.im_b_size),
                         consensus=base), frames=frames)
    if fine is not None:
        out["fine"] = dict(meta=dict(n=int(fine.cfg.grid.n), cell_m=float(fine.cfg.grid.cell_m),
                                     im_a_size=int(fine.cons.m.im_a_size), im_b_size=int(fine.cons.m.im_b_size),
                                     seed=int(fine.cfg.matcher.seed), consensus=settings_of(fine.cons).to_dict(),
                                     gate_m=float(a.fine_gate), config=a.fine_config, ckpt=a.fine_ckpt,
                                     train=M["fine_train"]),
                           frames=fine_frames)
    path = cache_path(a, draw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return path


def cache_path(a, draw):
    return Path(a.cache_dir or (Path(a.out) / "sweep_cache")) / f"{a.tag}_{draw}.pkl"


def load_models(a):
    """The coarse matcher / query / consensus as eval_vigor.run builds them (+ --consensus-json on the coarse pass)."""
    from bevloc.model.coarse import FeatureQueryMatcher
    from bevloc.model.query import apply_query_cfg, build_query, load_query_state
    EV = _ev()
    cfg = C.load(a.config)
    if a.solver:
        cfg.matcher.solver = a.solver
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = state.get("mode", "lift")
    cfg.lift.query_mode = mode
    apply_query_cfg(cfg, state)
    print(f"checkpoint {a.ckpt}: mode {mode}, step {state.get('step')}, training {state.get('train')}", flush=True)
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    query = build_query(cfg, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    cons = SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=False, min_valid_frac=0.05)
    coarse_c = EV.chosen_consensus(a)[0]
    if coarse_c is not None:
        apply_settings(cons, coarse_c)
    fine_train = None
    if a.fine_ckpt:
        fine_train = torch.load(a.fine_ckpt, map_location="cpu", weights_only=False).get("train")
        print(f"fine checkpoint {a.fine_ckpt}: training {fine_train}", flush=True)
    return dict(cfg=cfg, dev=dev, mode=mode, train_meta=state.get("train"), matcher=matcher, query=query, cons=cons,
                fine_train=fine_train, fine_args=a if a.fine_ckpt else None)


# ---- stage 2: sweep --------------------------------------------------------------------------------------------------

_W = {}


def _eval_one(args):
    """Worker: one configuration over every frame of one pass. -> dict(errors, gated, modes, inliers)."""
    c, which = args
    recs, meta, gate = _W[which]
    stub = consensus_stub(meta)
    errs, gated, modes, inl = [], [], [], []
    for r in recs:
        if r.get("error"):
            errs.append(None)
            gated.append(None)
            modes.append(0)
            inl.append(0.0)
            continue
        m = solve(r, c, stub)
        if which == "fine":
            e, g = fine_errors(m.H, r, meta, gate)
        else:
            e = g = coarse_error(m.H, r, meta)
        errs.append(e)
        gated.append(g)
        modes.append(int(m.n_modes))
        inl.append(float(m.inlier_ratio))
    return dict(errors=errs, gated=gated, modes=modes, inliers=inl)


def run_configs(configs, which, workers):
    import cv2
    cv2.setNumThreads(1)
    jobs = [(c, which) for c in configs]
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            return pool.map(_eval_one, jobs, chunksize=1)
    return [_eval_one(j) for j in jobs]


def _arr(errors):
    return np.array([np.inf if e is None else float(e) for e in errors], float)


def metrics(res, gross_m, n_boot, key="errors"):
    e = res[key]
    s = summarise_pose(e, n_boot=n_boot)
    v = _arr(e)
    return dict(n=s["n"], median_m=s["median_m"], median_ci=s["median_ci"],
                mean_m=float(np.mean(np.minimum(v, 1e3))), r5=s["recall@5m"], r10=s["recall@10m"],
                gross=float((v > float(gross_m)).mean()) if len(v) else float("nan"),
                no_pose=int(np.isinf(v).sum()), modes_mean=float(np.mean(res["modes"])) if res["modes"] else 0.0,
                inlier_mean=float(np.mean(res["inliers"])) if res["inliers"] else 0.0)


def objective(mt, select):
    return (mt["mean_m"], mt["median_m"]) if select == "mean" else (mt["median_m"], mt["mean_m"])


def ofat_configs(base: ConsensusCfg, grid):
    out, seen = [("baseline", None, base)], {base}
    for f in FACTORS:
        for v in grid[f]:
            c = ConsensusCfg.from_dict(dict(base.to_dict(), **{f: v}))
            if c not in seen:
                seen.add(c)
                out.append((f, v, c))
    return out


def paired_delta(e_new, e_base, n_boot, select="median", seed=0):
    """objective(new) - objective(base) on the same frames (median, or mean capped at 1 km), with a paired bootstrap
    95 % interval (frames resampled together)."""
    a, b = np.minimum(_arr(e_new), 1e3), np.minimum(_arr(e_base), 1e3)
    f = np.mean if select == "mean" else np.median

    def stat(idx):
        return float(f(a[idx]) - f(b[idx]))
    d = stat(np.arange(len(a))) if len(a) else float("nan")
    rng = np.random.default_rng(seed)
    bs = [stat(rng.integers(0, len(a), len(a))) for _ in range(int(n_boot))] if len(a) else []
    return dict(stat=select, delta_m=d, n=int(len(a)),
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None)


def sweep_pass(which, recs, meta, gate, grid, a):
    """OFAT then joint grid on one pass of the calibration cache. Returns (report dict, chosen ConsensusCfg)."""
    _W[which] = (recs, meta, gate)
    base = ConsensusCfg.from_dict(meta["consensus"])
    if any(r.get("cert") is None for r in recs if not r.get("error")) and set(grid["cert"]) - {"off"}:
        print(f"[{which}] the decoder gave no certainty for some frames: cert factor restricted to 'off'", flush=True)
        grid = dict(grid, cert=["off"])
    ofat = ofat_configs(base, grid)
    res = run_configs([c for _, _, c in ofat], which, a.workers)
    rows, by_cfg = [], {}
    for (f, v, c), r in zip(ofat, res):
        mt = metrics(r, a.gross_m, a.n_boot)
        if which == "fine":
            mt["gated"] = metrics(r, a.gross_m, a.n_boot, key="gated")
        rows.append(dict(factor=f, value=v, config=c.to_dict(), key=c.key(), **{"calib": mt}))
        by_cfg[c] = (mt, r)
    # reproduction check: the baseline (= the caching consensus) must equal the online pose of every frame
    online = [x.get("pose_online_m") for x in recs]
    base_err = by_cfg[base][1]["errors"]
    mism = int(sum(not (o is None and e is None or (o is not None and e is not None and abs(o - e) < 1e-9))
                   for o, e in zip(online, base_err)))
    sens = {}
    for f in FACTORS:
        vals = [objective(by_cfg[ConsensusCfg.from_dict(dict(base.to_dict(), **{f: v}))][0], a.select)[0]
                for v in grid[f]]
        vals = [x for x in vals if np.isfinite(x)]
        sens[f] = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
    top = [f for f, _ in sorted(sens.items(), key=lambda kv: -kv[1])[:a.joint_top] if sens[f] > 0]
    joint = []
    if len(top) > 1:
        combos = [ConsensusCfg.from_dict(dict(base.to_dict(), **dict(zip(top, vs))))
                  for vs in itertools.product(*[grid[f] for f in top])]
        combos = [c for c in dict.fromkeys(combos) if c not in by_cfg]
        for c, r in zip(combos, run_configs(combos, which, a.workers)):
            mt = metrics(r, a.gross_m, a.n_boot)
            if which == "fine":
                mt["gated"] = metrics(r, a.gross_m, a.n_boot, key="gated")
            joint.append(dict(factors=top, config=c.to_dict(), key=c.key(), calib=mt))
            by_cfg[c] = (mt, r)
    best = min(by_cfg, key=lambda c: objective(by_cfg[c][0], a.select))
    delta = paired_delta(by_cfg[best][1]["errors"], base_err, a.n_boot, a.select)
    # guard against picking noise out of ~100 configurations: the best one replaces the baseline only when its paired
    # calibration interval lies entirely below zero (--no-guard: always the best)
    adopt = best == base or a.no_guard or (delta["ci"] is not None and delta["ci"][1] < 0)
    chosen = best if adopt else base
    rep = dict(baseline=base.to_dict(), reproduces_online=dict(mismatches=mism, n=len(recs)), ofat=rows,
               sensitivity=sens, joint_factors=top, joint=joint, n_configs=len(by_cfg),
               best=dict(config=best.to_dict(), key=best.key(), calib=by_cfg[best][0], baseline_calib=by_cfg[base][0],
                         paired_delta_vs_baseline_calib=delta),
               chosen=dict(config=chosen.to_dict(), key=chosen.key(), calib=by_cfg[chosen][0],
                           paired_delta_vs_baseline_calib=paired_delta(by_cfg[chosen][1]["errors"], base_err,
                                                                       a.n_boot, a.select),
                           rule="best on calib" if adopt else "baseline kept: the best configuration's paired "
                                                             "calibration interval includes zero"))
    print(f"[{which}] reproduction of the online pose by the cached baseline: {mism} mismatches of {len(recs)}",
          flush=True)
    print(f"[{which}] sensitivity ({a.select}, m): " + ", ".join(f"{f} {s:.3f}" for f, s in sens.items()), flush=True)
    print(f"[{which}] joint grid over {top}: {len(joint)} configurations", flush=True)
    return rep, chosen


def assert_calib(cache, what):
    d = cache["meta"].get("draw")
    if d != "calib":
        raise SystemExit(f"refusing to select on {what}: its draw is {d!r}, not 'calib' (selection happens on held-out "
                         f"training frames only; the test draw is scored for the chosen configuration afterwards)")


def score_test(cache, which, configs, a):
    recs = cache["frames"] if which == "coarse" else cache["fine"]["frames"]
    meta = cache["meta"] if which == "coarse" else cache["fine"]["meta"]
    _W[which] = (recs, meta, meta.get("gate_m", a.fine_gate))
    out = []
    for c, r in zip(configs, run_configs(configs, which, a.workers)):
        mt = metrics(r, a.gross_m, a.n_boot)
        if which == "fine":
            mt["gated"] = metrics(r, a.gross_m, a.n_boot, key="gated")
        out.append((mt, r))
    return out


def sweep(a, calib_path, test_path):
    with open(calib_path, "rb") as f:
        cal = pickle.load(f)
    assert_calib(cal, calib_path)
    test = None
    if test_path is not None and Path(test_path).exists():
        with open(test_path, "rb") as f:
            test = pickle.load(f)
        if test["meta"].get("draw") == "calib":
            raise SystemExit(f"{test_path} is a calibration cache, not a test draw")
        overlap = set(cal["meta"]["ids"]) & set(test["meta"]["ids"])
        if overlap:
            raise SystemExit(f"calibration and test draws share {len(overlap)} frames: refusing")
    sc = fill_defaults(a)
    grid = {f: list(getattr(a, f"grid_{f}") or getattr(sc.grid, f)) for f in FACTORS}
    grid["cert"] = ["off" if v is False else v for v in grid["cert"]]      # YAML 1.1 reads a bare off as False
    grid["max_modes"] = [0 if v in ("all", 0, "0") else int(v) for v in grid["max_modes"]]
    grid["reproj_cells"] = [float(v) for v in grid["reproj_cells"]]
    grid["mode_thr"] = [float(v) for v in grid["mode_thr"]]
    low = [t for t in grid["mode_thr"] if t < cal["meta"]["cache_thr"] - 1e-12]
    if low:
        raise SystemExit(f"mode thresholds {low} are below the cache threshold {cal['meta']['cache_thr']}: re-cache "
                         f"with --cache-thr {min(low)}")
    passes = [p for p in a.passes if p == "coarse" or "fine" in cal]
    result = dict(meta=dict(tag=a.tag, calib_cache=str(calib_path), test_cache=None if test is None else str(test_path),
                            select=a.select, gross_m=a.gross_m, n_boot=a.n_boot, joint_top=a.joint_top, grid=grid,
                            calib=cal["meta"]["draw_info"], calib_n=len(cal["frames"]), ckpt=cal["meta"]["ckpt"],
                            config=cal["meta"]["config"], cache_thr=cal["meta"]["cache_thr"],
                            fine=cal.get("fine", {}).get("meta"),
                            test=None if test is None else test["meta"]["draw_info"],
                            test_n=None if test is None else len(test["frames"]),
                            not_sweepable=["mode thresholds below the cache threshold", "fixed_window_size (4)",
                                           "adaptive_gauss_fit", "min_valid_frac", "the decoder / logits",
                                           "fine pass: the coarse consensus that centred the window"]))
    for which in passes:
        recs = cal["frames"] if which == "coarse" else cal["fine"]["frames"]
        meta = cal["meta"] if which == "coarse" else cal["fine"]["meta"]
        rep, best = sweep_pass(which, recs, meta, meta.get("gate_m", a.fine_gate), grid, a)
        block = dict(config=best.to_dict(), key=best.key(), calib=rep["chosen"]["calib"], rule=rep["chosen"]["rule"],
                     best_calib=dict(key=rep["best"]["key"], calib=rep["best"]["calib"]),
                     baseline=dict(config=rep["baseline"], calib=rep["best"]["baseline_calib"]),
                     paired_delta_vs_baseline_calib=rep["chosen"]["paired_delta_vs_baseline_calib"], test=None)
        if test is not None and (which == "coarse" or "fine" in test):
            base = ConsensusCfg.from_dict(rep["baseline"])
            (mt_b, r_b), (mt_c, r_c) = score_test(test, which, [base, best], a)
            block["test"] = mt_c
            block["baseline"]["test"] = mt_b
            block["paired_delta_vs_baseline_test"] = paired_delta(r_c["errors"], r_b["errors"], a.n_boot, a.select)
        result[which] = rep
        result["chosen" if which == "coarse" else "chosen_fine"] = block
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sweep_consensus_{a.tag}.json"
    path.write_text(json.dumps(result, indent=2, default=_json_default))
    (out / f"sweep_consensus_{a.tag}.md").write_text(markdown(result))
    print(markdown(result), flush=True)
    print(f"wrote {path}", flush=True)
    return result


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _fmt(mt):
    if mt is None:
        return "| – | – | – | – | – | – |"
    ci = mt["median_ci"]
    return (f"| {mt['median_m']:.2f} ({ci[0]:.2f}–{ci[1]:.2f}) | {mt['mean_m']:.2f} | {mt['r5']:.3f} | {mt['r10']:.3f} "
            f"| {mt['gross']:.3f} | {mt['modes_mean']:.0f} |")


def _delta(d):
    ci = "" if d.get("ci") is None else f" (95 % CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f})"
    return f"{d['delta_m']:+.3f} m{ci}"


def markdown(res):
    L = [f"# Consensus sweep `{res['meta']['tag']}`", "",
         f"Selection on the calibration draw (n = {res['meta']['calib_n']}, objective {res['meta']['select']}); "
         f"test draw n = {res['meta']['test_n']}. Gross = error > {res['meta']['gross_m']} m or no pose. "
         f"Modes = mean modes used per frame.", ""]
    for which, key in (("coarse", "chosen"), ("fine", "chosen_fine")):
        if which not in res:
            continue
        rep, ch = res[which], res[key]
        L += [f"## {which} pass", "",
              f"Cached baseline reproduces the online pose: {rep['reproduces_online']['mismatches']} mismatches of "
              f"{rep['reproduces_online']['n']}.", "",
              "| draw | configuration | median (95 % CI) | mean | R@5 | R@10 | gross | modes |",
              "|---|---|---|---|---|---|---|---|",
              f"| calib | baseline: {ConsensusCfg.from_dict(ch['baseline']['config']).key()} {_fmt(ch['baseline']['calib'])}",
              f"| calib | **chosen**: {ch['key']} {_fmt(ch['calib'])}"]
        if ch.get("test"):
            L += [f"| test | baseline {_fmt(ch['baseline']['test'])}", f"| test | **chosen** {_fmt(ch['test'])}"]
        d, dt = ch["paired_delta_vs_baseline_calib"], ch.get("paired_delta_vs_baseline_test")
        bd = rep["best"]["paired_delta_vs_baseline_calib"]
        L += ["", f"Chosen: {ch['rule']}. Best on calib: {rep['best']['key']}, paired {bd['stat']} change "
                  f"{_delta(bd)}. Chosen − baseline, paired {d['stat']} change: calib {_delta(d)}"
              + (f"; test {_delta(dt)}" if dt else "") + ".",
              "", "One factor at a time (calibration draw):", "",
              "| factor | value | median (95 % CI) | mean | R@5 | R@10 | gross | modes |", "|---|---|---|---|---|---|---|---|"]
        for r in rep["ofat"]:
            v = "all" if r["factor"] == "max_modes" and r["value"] == 0 else r["value"]
            L.append(f"| {r['factor']} | {'' if v is None else v} {_fmt(r['calib'])}")
        L += ["", "Sensitivity (range of the objective over each factor's values, m): "
              + ", ".join(f"{f} {s:.3f}" for f, s in rep["sensitivity"].items()), ""]
        if rep["joint"]:
            L += [f"Joint grid over {rep['joint_factors']} (best 10 of {len(rep['joint'])}):", "",
                  "| configuration | median (95 % CI) | mean | R@5 | R@10 | gross | modes |", "|---|---|---|---|---|---|---|"]
            key_ = (lambda r: (r["calib"]["mean_m"], r["calib"]["median_m"])) if res["meta"]["select"] == "mean" \
                else (lambda r: (r["calib"]["median_m"], r["calib"]["mean_m"]))
            for r in sorted(rep["joint"], key=key_)[:10]:
                L.append(f"| {r['key']} {_fmt(r['calib'])}")
            L.append("")
    return "\n".join(L) + "\n"


# ---- CLI --------------------------------------------------------------------------------------------------------------

def _cpus():
    try:
        return max(1, len(os.sched_getaffinity(0)))                 # the SLURM allocation, not the node
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def build_parser():
    EV = _ev()
    ap = EV.build_parser(__doc__)
    for act in ap._actions:
        if act.dest == "ckpt":
            act.required = False                                     # --stage sweep needs no checkpoint
    ap.set_defaults(out="experiments/10_loc2_matcher")
    ap.add_argument("--stage", default="all", choices=("all", "cache", "sweep"))
    ap.add_argument("--draws", nargs="+", default=["calib", "test"], choices=("calib", "test"),
                    help="draws to cache: calib = held-out training frames (selection), test = --cities / --limit")
    ap.add_argument("--calib-cities", nargs="*", default=None,
                    help="calibration draw: keep these cities (default: the checkpoint's training cities)")
    ap.add_argument("--calib-limit", type=int, default=None,
                    help="calibration frames (first N after the skip; default consensus_sweep.calib_limit)")
    ap.add_argument("--train-cities", nargs="*", default=None,
                    help="with --assume-train-split: the coarse checkpoint's training cities (default: the split's "
                         "training cities)")
    ap.add_argument("--fine-val-frac", type=float, default=0.2, help="the fine checkpoint's --val-frac")
    ap.add_argument("--fine-assume-train-split", action="store_true",
                    help="fine checkpoint without a recorded split: assert --fine-val-frac/--fine-val-samples/"
                         "--fine-train-cities")
    ap.add_argument("--fine-val-samples", type=int, default=None)
    ap.add_argument("--fine-train-cities", nargs="*", default=None)
    ap.add_argument("--cache-thr", type=float, default=None,
                    help="mode threshold of the cache = the loosest one that can be swept (consensus_sweep.cache_thr)")
    ap.add_argument("--cache-dir", default=None, help="default <out>/sweep_cache (git-ignored)")
    ap.add_argument("--calib-cache", default=None, help="--stage sweep: calibration cache (default from --tag)")
    ap.add_argument("--test-cache", default=None, help="--stage sweep: test cache (default from --tag, if present)")
    ap.add_argument("--passes", nargs="+", default=["coarse", "fine"], choices=("coarse", "fine"))
    ap.add_argument("--select", default=None, choices=("median", "mean"),
                    help="calibration objective (ties broken by the other)")
    ap.add_argument("--no-guard", action="store_true",
                    help="choose the best calibration configuration even when its paired interval against the "
                         "baseline includes zero")
    ap.add_argument("--joint-top", type=int, default=None, help="joint grid over this many most sensitive factors")
    ap.add_argument("--gross-m", type=float, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.epilog = "Defaults of --cache-thr, --calib-limit, --select, --joint-top, --gross-m, --n-boot and the grid: " \
                "the config's consensus_sweep block (configs/default.yaml)."
    ap.add_argument("--workers", type=int, default=_cpus(),
                    help="sweep processes (default: the CPUs this job may use; request them with --cpus-per-task)")
    ap.add_argument("--grid-reproj_cells", "--reproj", dest="grid_reproj_cells", nargs="+", type=float, default=None)
    ap.add_argument("--grid-target", "--targets", dest="grid_target", nargs="+", choices=TARGETS, default=None)
    ap.add_argument("--grid-max_modes", "--max-modes", dest="grid_max_modes", nargs="+", default=None,
                    help="integers, 'all' (= 0)")
    ap.add_argument("--grid-mode_thr", "--mode-thr", dest="grid_mode_thr", nargs="+", type=float, default=None)
    ap.add_argument("--grid-cert", "--cert", dest="grid_cert", nargs="+", choices=CERTS, default=None)
    ap.add_argument("--grid-solver", "--solvers", dest="grid_solver", nargs="+", choices=SOLVERS, default=None)
    ap.add_argument("--grid-ransac", "--ransac", dest="grid_ransac", nargs="+", choices=tuple(RANSAC_MULT), default=None)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if bool(a.fine_config) != bool(a.fine_ckpt):
        raise SystemExit("--fine-config and --fine-ckpt go together")
    if a.stage in ("all", "cache"):
        if not a.ckpt:
            raise SystemExit("--ckpt is required to cache")
        if a.calib:
            raise SystemExit("use --draws calib (not --calib): the sweep caches the calibration draw itself")
        fill_defaults(a)
        M = load_models(a)
        for draw in a.draws:
            cache_draw(a, draw, M)
    if a.stage in ("all", "sweep"):
        cal = Path(a.calib_cache) if a.calib_cache else cache_path(a, "calib")
        tst = Path(a.test_cache) if a.test_cache else cache_path(a, "test")
        sweep(a, cal, tst if tst.exists() else None)


if __name__ == "__main__":
    main()
