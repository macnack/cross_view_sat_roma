"""Cache the frozen matcher's evidence per VIGOR frame for the pose-correctness head (task 06, step 4).

  make vigor-cert-cache CKPT=... CONFIG=configs/vigor_cell0125.yaml SPLIT=samearea DRAW=calib TAG=erpd4city \
       CALIB_LIMIT=8000 VIGOR_ARGS="--solver se2 --assume-train-split --val-samples 400 --train-cities Chicago NewYork SanFrancisco Seattle"
  make vigor-cert-cache CKPT=... CONFIG=... SPLIT=samearea DRAW=test CITIES=Chicago LIMIT=3000 TAG=erpd4city VIGOR_ARGS="--solver se2"
  make vigor-cert-cache CKPT=... CONFIG=... SPLIT=samearea DRAW=test LIMIT=12000 TAG=erpd4city VIGOR_ARGS="--solver se2"
       (all cities: leave CITIES unset; the seed-0 draw is over the concatenated city lists, so the tables' 12000-sample
       rows, drawn in the split's default order NewYork Seattle SanFrancisco Chicago, are reproduced only in that order)

One pass of the frozen matcher (checkpoint, config, solver and consensus exactly as eval_vigor.py's default path: the
peak and means rows, the task-06 statistics of the peak row) over one draw:
  --draw calib   the checkpoint's held-out training frames after the val_samples that selected it (the rule of
                 eval_vigor.py --calib / sweep_consensus_vigor.py: `calib_split`; --assume-train-split --val-samples
                 --train-cities for a checkpoint whose train dict predates the split record), first --calib-limit
                 frames, restricted to --calib-cities (default: the checkpoint's training cities);
  --draw test    the draw every table uses: VigorPairs(--cities, --split, limit=--limit, seed 0).
Per frame the cache stores (docs/tasks/06_certainty.md, "Data"; bevloc.model.certainty_head for the layout):
  maps (4, K, K) float16: ego vote map, token vote map, pose footprint (+1 = a valid token with an inlier mode, -1 =
      any other valid token, at the cell the coarse H sends the token to; 0 elsewhere or without a pose), reference-
      cell validity; tokens (T, 8) float16 + token_valid (T,) bool (TOKEN_KEYS: mass at the pose-consistent cell,
      top-1 mass, normalised entropy, certainty logit, distance of the token's peak from the pose-consistent cell in
      cells (K without a pose), placed range m, normalised azimuth column and row); the frame statistics of
      certainty_vigor.py (FRAME_KEYS: the inlier ratio and STAT_KEYS, raw, None = missing); the coarse errors and
      inlier ratios of both rows; frame id and city.
Reproduction gate: after the pass, eval_vigor.score (the real default path, a second decode) runs on the first
--gate-frames frames (0 = every frame, the default) and every cached error, inlier ratio and statistic must equal it
(errors within 1e-3 m, statistics within 1e-6); a mismatch stops the script with the frame ids (--allow-mismatch:
recorded in meta.gate and the script continues). The gate doubles the GPU time; use --gate-frames for smoke tests.
Output: <cache-dir>/<tag>_<draw>.pkl (default <out>/cert_cache, git-ignored), ~50 KB per frame.
Not in scope: the fine pass (--fine-* refused), --hyp (dropped from the head comparison).
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bevloc import config as C
from bevloc.eval.metrics import pose_errors
from bevloc.match.consensus import apply_settings, extract_modes, select_modes, settings_of
from bevloc.match.satroma import SatRoMa, consensus_for_query, query_tokens
from bevloc.match.vote_stats import STAT_KEYS, apply_h, ego_vote_map, pose_px, px_to_cell, ref_cell_valid, stats_row, \
    token_centres, token_valid, vote_map
from bevloc.model.certainty_head import MAP_KEYS, TOKEN_KEYS

ROOT = Path(__file__).resolve().parent


def _load(name, file):
    """A sibling script under a unique module name (third_party/Loc2 also ships an eval_vigor.py)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ev():
    return _load("bevloc_scripts_eval_vigor", "eval_vigor.py")


def _sw():
    return _load("bevloc_scripts_sweep_consensus_vigor", "sweep_consensus_vigor.py")


def _cv():
    return _load("bevloc_scripts_certainty_vigor", "certainty_vigor.py")


def frame_keys():
    """(keys, log mask) of the frame statistics in certainty_vigor.FEATURES order (row = peak), without the --hyp
    spread (not cached), plus vote_offtile_mass (a STAT_KEY the logistic never used): 19 = inlier ratio + STAT_KEYS."""
    feats = [(k.format(row="peak"), tr) for k, tr in _cv().FEATURES if k != "spread_hyp_m"]
    keys = [k for k, _ in feats] + [k for k in STAT_KEYS if k not in {k for k, _ in feats}]
    log = {k: tr == "log" for k, tr in feats}
    return keys, [bool(log.get(k, False)) for k in keys]


ERR_TOL_M, STAT_TOL = 1e-3, 1e-6


# ---- evidence of one decoded frame ----------------------------------------------------------------------------------

def correspondences(cons, gm, cert, valid):
    """The consensus's own correspondences on logits gm (K*K, h, w) whose masked tokens are zeroed: (tok (N, 2) float32
    (col, row), tgt (N, 2) reference cells) under the instance's settings (`bevloc.match.consensus`, the modes
    `_pick_modes` selects: extraction at mode_thr, certainty rule, per-token cap; target peak / means)."""
    c = settings_of(cons)
    md = extract_modes(gm, c.mode_thr)
    cert_np = None if cert is None else np.asarray(cert.detach().float().cpu().numpy(), np.float64).reshape(gm.shape[-2:])
    idx = select_modes(md, c, cert_np if c.cert in ("weight", "filter") else None, np.asarray(valid, bool))
    return md["tok"][idx], (md["means"] if cons.use_means else md["peaks"])[idx]


def evidence(cons, gm, cert, valid, q_xy, H, ref_valid, cell_m):
    """maps (4, K, K) float16, tokens (T, 8) float16, token_valid (T,) bool, n_inlier_tokens, n_inlier_modes of one
    frame (module doc). gm (K*K, h, w) logits with the masked tokens zeroed, cert (h, w) logits or None, valid (h, w)
    bool, q_xy (h, w, 2) query px of every token (placed point, or the token centre), H (3, 3) query px -> reference
    px or None, ref_valid (K, K) bool."""
    K = int(round(gm.shape[0] ** 0.5))
    h, w = gm.shape[-2:]
    im_a, im_b = int(cons.m.im_a_size), int(cons.m.im_b_size)
    vt = token_valid(gm, valid).reshape(-1)
    T = h * w
    p = torch.softmax(gm.detach().float(), dim=0).reshape(K * K, T).T.cpu().numpy().astype(np.float32)   # (T, K*K)
    xy = np.asarray(q_xy, np.float64).reshape(T, 2)
    vote, _ = vote_map(gm, valid)
    ego = ego_vote_map(gm, np.asarray(q_xy, np.float64), valid, im_a, im_b) if vt.any() else None
    maps = np.zeros((len(MAP_KEYS), K, K), np.float32)
    maps[MAP_KEYS.index("ego")] = 0.0 if ego is None else ego
    maps[MAP_KEYS.index("vote")] = 0.0 if vote is None else vote
    maps[MAP_KEYS.index("ref_valid")] = np.asarray(ref_valid, np.float32)
    tok = np.zeros((T, len(TOKEN_KEYS)), np.float32)
    n_inl_tok = n_inl_mode = 0
    if vt.any():
        pv = p[vt]
        tok[vt, TOKEN_KEYS.index("top1")] = pv.max(1)
        tok[vt, TOKEN_KEYS.index("entropy")] = -(pv * np.log(np.clip(pv, 1e-12, None))).sum(1) / np.log(K * K)
        if cert is not None:
            tok[vt, TOKEN_KEYS.index("cert")] = cert.detach().float().cpu().numpy().reshape(-1)[vt]
        tok[vt, TOKEN_KEYS.index("depth_m")] = np.linalg.norm(xy[vt] - (im_a - 1) / 2.0, axis=1) * float(cell_m)
        rows, cols = np.divmod(np.arange(T), w)
        tok[vt, TOKEN_KEYS.index("azimuth")] = (cols[vt] + 0.5) / w
        tok[vt, TOKEN_KEYS.index("row")] = (rows[vt] + 0.5) / h
        tok[vt, TOKEN_KEYS.index("peak_pose_cells")] = K                           # replaced below when H exists
        if H is not None:
            cell = px_to_cell(apply_h(H, xy), im_b, K)                            # (T, 2) continuous (x, y)
            ci = np.rint(cell).astype(int)
            inside = (ci >= 0).all(1) & (ci < K).all(1)
            idx = np.where(inside, ci[:, 1] * K + ci[:, 0], 0)
            vi = np.flatnonzero(vt)
            tok[vt, TOKEN_KEYS.index("p_pose")] = np.where(inside[vi], p[vi, idx[vi]], 0.0)
            pk = pv.argmax(1)
            tok[vt, TOKEN_KEYS.index("peak_pose_cells")] = np.hypot(pk % K - cell[vt, 0], pk // K - cell[vt, 1])
            # the consensus's inlier set: modes within the reprojection threshold of H, in cells; a token is an
            # inlier when one of its modes is
            t_xy, tgt = correspondences(cons, gm, cert, valid)
            t_idx = t_xy[:, 1].astype(int) * w + t_xy[:, 0].astype(int)
            err = np.linalg.norm(px_to_cell(apply_h(H, xy[t_idx]), im_b, K) - np.asarray(tgt, np.float64), axis=1)
            inl_mode = err <= float(cons.reproj)
            n_inl_mode = int(inl_mode.sum())
            inl_tok = np.zeros(T, bool)
            inl_tok[t_idx[inl_mode]] = True
            n_inl_tok = int((inl_tok & vt).sum())
            val = np.where(inl_tok, 1.0, -1.0)
            sel = vt & inside
            fp = np.zeros((K, K))
            np.add.at(fp, (ci[sel, 1], ci[sel, 0]), val[sel])
            maps[MAP_KEYS.index("footprint")] = np.clip(fp, -1.0, 1.0)
    return maps.astype(np.float16), tok.astype(np.float16), vt.astype(bool), n_inl_tok, n_inl_mode


def decode(sample, query, matcher, dev):
    """The decode of eval_vigor.score for one sample: (batch, frac, gm (K*K, h, w) logits, certainty (h, w) or None)."""
    batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in sample.items()}
    with torch.no_grad():
        f_q, frac = query(batch, matcher)
        f_s = matcher.reference_features(batch["ref"])
        sf = float(((f_q.shape[-2] * 16) * (f_q.shape[-1] * 16)) ** 0.5 / 560.0)
        with matcher.model.exposed_intermediates():
            o16 = matcher.model.decoder({16: f_q}, f_s, scale_factor=sf)[16]
    cert = o16["gm_certainty"][0, 0] if o16.get("gm_certainty") is not None else None
    return batch, frac, o16["gm_cls"][0], cert


def frame_record(s, i, ds, query, matcher, cons, cfg, dev, keys):
    """One cached frame: the rows of eval_vigor.score (both coarse rows + the peak row's statistics) and the head's
    evidence. cons: {"peak", "means"} consensus instances with stats_radius set."""
    n, cell = int(cfg.grid.n), float(cfg.grid.cell_m)
    cc = _ev().certainty_cfg(cfg)
    for c in cons.values():                                        # as eval_vigor.score
        c.stats_radius = float(cc.stats_radius_cells)
    batch, frac, gm, cert = decode(s, query, matcher, dev)
    K = int(round(gm.shape[0] ** 0.5))
    rv = ref_cell_valid(batch["ref"], K, min_frac=float(cc.ref_cell_min_frac))
    H_gt = s["H"].numpy().astype(float)
    row = dict(id=s["id"], city=s["city"], centre_guess_m=ds.centre_guess_m(i))
    m = {}
    for tag, c in cons.items():
        m[tag] = consensus_for_query(c, gm, query, batch, frac, n, min_frac=0.05, certainty=cert, ref_valid=rv,
                                     stats=tag == "peak")
        e = pose_errors(m[tag].H, H_gt, n, cell) if m[tag].H is not None else None
        row[f"pose_{tag}_m"] = None if e is None else e["position_m"]
        row[f"yaw_{tag}_deg"] = None if e is None else e["yaw_deg"]
        row[f"inliers_{tag}"] = m[tag].inlier_ratio
    mh = m["means"].H
    st = stats_row(m["peak"].stats, None if mh is None else pose_px(mh, int(cons["peak"].m.im_a_size)), cell)
    st["inliers_peak"] = row["inliers_peak"]
    row["stats"] = {k: st.get(k) for k in keys}
    # the head's evidence, from the same logits: masked tokens zeroed as consensus_for_query does
    valid, xy = query_tokens(cons["peak"], query, batch, frac, n, 0.05)
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=gm.device)
    gmm = gm.clone()
    gmm[:, ~valid_t] = 0.0
    q_xy = (torch.as_tensor(xy).detach().float().cpu().numpy() if xy is not None
            else token_centres(gm.shape[-2], gm.shape[-1], int(cons["peak"].m.im_a_size)))
    maps, tok, tv, n_it, n_im = evidence(cons["peak"], gmm, cert, valid_t.cpu().numpy(), q_xy, m["peak"].H, rv, cell)
    row.update(maps=maps, tokens=tok, token_valid=tv, n_inlier_tokens=n_it, n_inlier_modes_recomputed=n_im,
               n_inliers=int(m["peak"].n_inliers), n_modes=int(m["peak"].n_modes),
               H=None if m["peak"].H is None else np.asarray(m["peak"].H, np.float64), H_gt=H_gt,
               placed=xy is not None, token_hw=tuple(int(v) for v in gm.shape[-2:]))
    return row


# ---- reproduction gate -----------------------------------------------------------------------------------------------

def _same_value(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return (np.isnan(a) and np.isnan(b)) or a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def compare_rows(frames, rows, keys, err_tol=ERR_TOL_M, stat_tol=STAT_TOL):
    """Cached frames vs eval_vigor.score rows (matched by id): the coarse errors and inlier ratios of both rows and
    every frame statistic must agree. Returns (mismatches [(id, what)], n compared)."""
    by_id = {r["id"]: r for r in rows}
    bad, n = [], 0
    for f in frames:
        r = by_id.get(f["id"])
        if r is None:
            continue
        n += 1
        for k, tol in (("pose_peak_m", err_tol), ("pose_means_m", err_tol), ("inliers_peak", stat_tol),
                       ("inliers_means", stat_tol)):
            if not _same_value(f.get(k), r.get(k), tol):
                bad.append((f["id"], f"{k}: cached {f.get(k)} vs eval_vigor {r.get(k)}"))
                break
        else:
            for k in keys:
                if not _same_value(f["stats"].get(k), r.get(k), stat_tol):
                    bad.append((f["id"], f"{k}: cached {f['stats'].get(k)} vs eval_vigor {r.get(k)}"))
                    break
    return bad, n


def check_reproduction(frames, ds, query, matcher, cons, cfg, dev, keys, a):
    """eval_vigor.score on the first --gate-frames frames of the draw (a second decode) against the cache; SystemExit
    with the mismatching ids unless --allow-mismatch (then recorded)."""
    n_max = int(a.gate_frames) if a.gate_frames else 0
    print(f"gate: eval_vigor.score on {'every frame' if not n_max else f'the first {n_max} frames'} of the draw",
          flush=True)
    rows = _ev().score(ds, query, matcher, cons, cfg, dev, n_max=n_max)
    bad, n = compare_rows(frames, rows, keys)
    print(f"gate: cached vs eval_vigor.py default path: {len(bad)} mismatches of {n} frames", flush=True)
    for fid, what in bad[:20]:
        print(f"  {fid}: {what}", flush=True)
    if bad and not a.allow_mismatch:
        raise SystemExit(f"the cache does not reproduce eval_vigor.py on {len(bad)} of {n} frames: "
                         f"{[i for i, _ in bad[:20]]}{' ...' if len(bad) > 20 else ''} (--allow-mismatch records "
                         f"the count and continues)")
    return dict(n=n, mismatches=len(bad), ids=[i for i, _ in bad[:200]], what=[w for _, w in bad[:200]],
                err_tol_m=ERR_TOL_M, stat_tol=STAT_TOL, frames_requested=n_max or "all",
                rule="eval_vigor.score (second decode, default consensus, peak + means rows, statistics) must give "
                     "the cached errors, inlier ratios and statistics")


# ---- models and draw ----------------------------------------------------------------------------------------------------

def load_models(a):
    """cfg, dev, mode, train_meta, matcher, query and the {"peak", "means"} consensus instances exactly as
    eval_vigor.run builds them (--consensus-json applied to both rows, each keeping its own target)."""
    EV = _ev()
    M = _sw().load_models(a)                          # cfg (solver applied), matcher, query, train_meta, mode
    cfg = M["cfg"]
    cons = {tag: SatRoMa.from_wrapper(M["matcher"].wrapper, cfg, use_means=means, min_valid_frac=0.05)
            for tag, means in (("peak", False), ("means", True))}
    coarse_c = EV.chosen_consensus(a)[0]
    if coarse_c is not None:
        for c in cons.values():
            apply_settings(c, coarse_c, target=False)
        print(f"coarse consensus from --consensus-json: {coarse_c.key()}", flush=True)
    cc = EV.certainty_cfg(cfg)
    for c in cons.values():
        c.stats_radius = float(cc.stats_radius_cells)
    M["cons"] = cons
    return M


def cache_path(a, draw):
    return Path(a.cache_dir or (Path(a.out) / "cert_cache")) / f"{a.tag}_{draw}.pkl"


def cache_draw(a, M, draw=None):
    """The pass over one draw; writes the cache and returns its path."""
    EV, SW = _ev(), _sw()
    draw = draw or a.draw
    cfg, dev, cons = M["cfg"], M["dev"], M["cons"]
    ds, info = SW.build_draw(a, cfg, M["train_meta"], draw)
    n_no_depth = ds.keep_with_depth() if ds.depth else 0
    keys, log = frame_keys()
    frames = []
    t0 = time.time()
    print(f"cache {draw}: {len(ds)} frames ({n_no_depth} without depth dropped), cities "
          f"{sorted({lab['city'] for lab in ds.labels})}, consensus {settings_of(cons['peak']).key()}", flush=True)
    for i in range(len(ds)):
        try:
            s = ds[i]
        except RuntimeError as e:                                   # unreadable / missing image, as eval_vigor
            print(f"  skip {i}: {e}", flush=True)
            continue
        frames.append(frame_record(s, i, ds, M["query"], M["matcher"], cons, cfg, dev, keys))
        if len(frames) % 100 == 0:
            e = [np.inf if f["pose_peak_m"] is None else f["pose_peak_m"] for f in frames]
            print(f"  {len(frames)} cached  median so far {np.median(e):.2f} m  {time.time() - t0:.0f} s", flush=True)
    if not frames:
        raise SystemExit("no frame cached")
    gate = check_reproduction(frames, ds, M["query"], M["matcher"], cons, cfg, dev, keys, a)
    n_inl_diff = int(sum(f["n_inlier_modes_recomputed"] != f["n_inliers"] for f in frames))
    if n_inl_diff:
        print(f"note: the recomputed inlier-mode count differs from the consensus's on {n_inl_diff} frames (threshold "
              f"boundary in the px -> cell round trip); the footprint uses the recomputed set", flush=True)
    hw = frames[0]["token_hw"]
    meta = dict(ckpt=a.ckpt, config=a.config, config_resolved=C._plain(cfg), split=a.split, draw=draw, draw_info=info,
                ids=[f["id"] for f in frames], cities=sorted({f["city"] for f in frames}), n=len(frames),
                mode=M["mode"], train=M["train_meta"], seed=int(cfg.matcher.seed), solver=cons["peak"].solver,
                consensus=settings_of(cons["peak"]).to_dict(), consensus_means=settings_of(cons["means"]).to_dict(),
                consensus_json=list(a.consensus_json) if a.consensus_json else None,
                grid_n=int(cfg.grid.n), cell_m=float(cfg.grid.cell_m), im_a_size=int(cons["peak"].m.im_a_size),
                im_b_size=int(cons["peak"].m.im_b_size), K=int(frames[0]["maps"].shape[-1]),
                token_grid="erp" if frames[0]["placed"] else "bev", token_hw=[int(hw[0]), int(hw[1])],
                map_keys=list(MAP_KEYS), token_keys=list(TOKEN_KEYS), frame_keys=keys, frame_log=log,
                stats=dict(radius_cells=float(EV.certainty_cfg(cfg).stats_radius_cells),
                           ref_cell_min_frac=float(EV.certainty_cfg(cfg).ref_cell_min_frac), row="peak"),
                skipped_no_depth=n_no_depth, gate=gate, inlier_count_differs=n_inl_diff,
                footprint="+1 valid token with an inlier mode, -1 other valid token, at round(cell of H(token)); "
                          "clipped sum; 0 without a pose",
                command=" ".join(sys.argv), seconds=float(time.time() - t0))
    path = cache_path(a, draw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(meta=meta, frames=frames), f, protocol=pickle.HIGHEST_PROTOCOL)
    e = np.array([np.inf if f["pose_peak_m"] is None else f["pose_peak_m"] for f in frames])
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB): {len(frames)} frames, peak median {np.median(e):.2f} m, "
          f"{int(np.isinf(e).sum())} without a pose, {(e > 10).mean() * 100:.1f} % > 10 m, {time.time() - t0:.0f} s",
          flush=True)
    return path


def build_parser():
    EV, SW = _ev(), _sw()
    ap = EV.build_parser(__doc__)
    ap.set_defaults(out="experiments/10_loc2_matcher")
    ap.add_argument("--draw", default="calib", choices=("calib", "test"),
                    help="calib = held-out training frames (the head's train / val); test = --cities / --limit draw")
    ap.add_argument("--calib-cities", nargs="*", default=None,
                    help="calibration draw: keep these cities (default: the checkpoint's training cities)")
    ap.add_argument("--calib-limit", type=int, default=None,
                    help="calibration frames (first N after the skip; default consensus_sweep.calib_limit)")
    ap.add_argument("--train-cities", nargs="*", default=None,
                    help="with --assume-train-split: the checkpoint's training cities IN THE ORDER of the training "
                         "run's --cities (the held-out permutation is over the concatenated city lists; calib_split "
                         "checks the set only); default: the split's")
    ap.add_argument("--cache-dir", default=None, help="default <out>/cert_cache (git-ignored)")
    ap.add_argument("--gate-frames", type=int, default=0,
                    help="reproduction gate on the first N frames (0 = every frame): eval_vigor.score re-decodes them")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="continue when the gate finds mismatches (count and ids recorded in meta.gate)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.fine_ckpt or a.fine_config or a.refine or a.hyp or a.calib or a.calibrate:
        raise SystemExit("certainty_cache_vigor.py caches the coarse default path only: no --fine-*, --refine, --hyp, "
                         "--calib (use --draw calib) or --calibrate")
    if a.calib_limit is None:                                      # the sweep's default (consensus_sweep.calib_limit)
        a.calib_limit = int(_sw().sweep_cfg(a.config).calib_limit)
    # the gate decodes every frame a second time: keep cuDNN on its deterministic, non-benchmarked kernels so the two
    # forward passes agree bit for bit (a float-noise change of the logits could move a mode across its threshold)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    M = load_models(a)
    return cache_draw(a, M)


if __name__ == "__main__":
    main()
