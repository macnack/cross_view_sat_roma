"""Run a query checkpoint on VIGOR (known orientation): position error in metres per city, with chance.

  make vigor-eval CKPT=checkpoints/05_lift_splat_fixtor_ipm_long_best.pt SPLIT=crossarea TAG=ipm_zeroshot LIMIT=0
  make vigor-calibrate CKPT=...           # settles row_sign / camera height on 150 samples (see below)

Per sample: IPM picture of the north-aligned panorama (or the checkpoint's own query mode) -> frozen
encoder + decoder -> Sat-RoMa consensus (peak and means rows) against the positive tile. The estimated
pose is the panorama's position in the tile; the error is the distance in metres (per-city resolution).
Also reported: the centre guess (predict the tile centre), which is this protocol's chance level.
Queries with placed tokens (erp, erp_depth) go through the placed-token consensus (`SatRoMa.consensus_from_gm`, the
consensus half of `match_placed`); an erp_depth checkpoint evaluates only the panoramas of the draw that have a depth
file (`make loc2-depth` with the same SPLIT / CITIES / LIMIT), and reports how many were dropped.
--refine S (task 04, sub-cell stage; 0 = off, the default, and then every number is what it was): also decode the
Sat-RoMa decoder's conv refiner output (its only refiner, at stride 16 on the same projected tokens, conditioned on
the coarse warp; see bevloc.model.refine), sample the refined warp on a query grid of stride S px, and add the row
`refined` (pose_refined_m, inliers_refined, ...) from the same solver on those correspondences. --refine-init
none|coarse|ransac (several = several rows, `refined_<init>`): refiner on the package's own coarse warp, RANSAC from
scratch | same warp, correspondences gated around the coarse (peak) pose | refiner re-run on the coarse-RANSAC-
consistent warp, then gated. See `bevloc.match.satroma.refined_for_query`.
--fine-config <yaml> --fine-ckpt <pt> (task 04, coarse-to-fine second pass; off by default, and then every number is
what it was): after the coarse rows, build the fine reference (a window of the tile at the fine config's GSD,
configs/vigor_cell00625_fine.yaml: 56 m at 0.0625 m/px) centred on the coarse `peak` pose (the tile centre when the
coarse RANSAC found nothing, counted as `no_coarse`), rebuild the query at the fine GSD from the same panorama (the
picture: a new ipm_erp call at the fine cell; erp_depth: the placement in fine virtual BEV px), run the fine decoder +
the same consensus (solver, threshold in cells, seed), map the fine pose back to the tile frame (`pose_en`) and add
`pose_fine_m`, `yaw_fine_deg`, `inliers_fine`, plus `pose_fine_gated_m`: the coarse pose when the fine one is more than
--fine-gate metres from it or missing (`fallback_fine`). Errors of both rows are distances in the tile frame (metres
east/north of the tile centre, the frame of the sample's `en`). scripts/loftr_fine_vigor.py reuses this path with
LoFTR as the second-pass matcher.
Per frame, also the vote-map statistics of task 06 (`bevloc.match.vote_stats.STAT_KEYS`: vote entropy / effective
support / top-1 mass / mass within 2 cells of the pose, modes seen and kept, peak-vs-means pose agreement, distance of
the pose from the tile centre, certainty logits over the valid tokens, valid tokens, mean placed range of erp_depth
tokens), computed from the logits already decoded: the inputs of `make vigor-certainty`. They are new keys; every
existing key is what it was. --hyp K (0 = off) adds `pose_hyp_med_m` / `spread_hyp_m` / `n_hyp_ok`: K RANSAC runs on
bootstrap resamples of the peak row's modes, the pose of their medoid and the median distance of the K poses from it.
--calib: score the held-out part of the TRAINING list instead (train_vigor.py's --val-frac rule: train labels of the
cities shuffled with seed 0, the last val_frac held out), skipping its first val_samples (they selected the checkpoint)
and taking the next --limit frames: the calibration set of task 06, never the test draw (`calib_split`; the split is
read from the checkpoint and checked, --assume-train-split for checkpoints that predate that record).
Calibration mode scores row_sign in {+1, -1} x height in {1.6, 2.0, 2.5, 3.0} on a small subset and prints
the table; the best pair is what `vigor:` in configs/default.yaml should carry.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from bevloc import config as C
from bevloc.data.vigor import CITY_RES, VigorPairs, pose_en, split_cities
from bevloc.eval.metrics import pose_errors
from bevloc.eval.report import summarise_pose
from bevloc.match.satroma import REFINE_INITS, SatRoMa, consensus_for_query, refined_for_query
from bevloc.match.vote_stats import pose_px, ref_cell_valid, stats_row
from bevloc.model.coarse import FeatureQueryMatcher
from bevloc.model.query import apply_query_cfg, build_query, load_query_state
from bevloc.model.refine import REFINER_KEY, RefinerTap


def refine_tags(inits):
    """Row tag per refinement init: `refined` for a single init, `refined_<init>` for several."""
    return {i: "refined" if len(inits) == 1 else f"refined_{i}" for i in inits}


def _centre_of(s):
    c = s.get("ref_centre_en")
    return np.zeros(2) if c is None else np.asarray(torch.as_tensor(c, dtype=torch.float64).cpu().numpy(), float)


class DecoderFine:
    """Second-pass matcher: a Sat-RoMa decoder (fine-tuned on jittered windows) at the fine config's GSD.

    ds: VigorPairs built with the fine config over the same labels as the coarse pass; query / matcher / cons: built
    from the fine config and checkpoint. Called with (i, centre_en) -> (Match on the window, the window sample)."""

    def __init__(self, ds, query, matcher, cons, cfg, dev):
        self.ds, self.query, self.matcher, self.cons, self.cfg, self.dev = ds, query, matcher, cons, cfg, dev

    def __call__(self, i, centre_en):
        s = self.ds.item(i, ref_centre_en=centre_en)
        batch = {k: (v[None].to(self.dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = self.query(batch, self.matcher)
            f_s = self.matcher.reference_features(batch["ref"])
            sf = float(((f_q.shape[-2] * 16) * (f_q.shape[-1] * 16)) ** 0.5 / 560.0)
            with self.matcher.model.exposed_intermediates():
                gm = self.matcher.model.decoder({16: f_q}, f_s, scale_factor=sf)[16]["gm_cls"][0]
        m = consensus_for_query(self.cons, gm, self.query, batch, frac, int(self.cfg.grid.n), min_frac=0.05,
                                stats=False)
        return m, s


FINE_KEYS = ("pose_fine_m", "yaw_fine_deg", "inliers_fine", "pose_fine_gated_m", "fine_shift_m", "no_coarse_fine",
             "en_gt", "en_coarse", "en_fine", "fine_centre_en")


def _is_oom(e):
    """CUDA out-of-memory must stop the run, not become a per-sample error row."""
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom is not None and isinstance(e, oom)) or "out of memory" in str(e).lower()


def fine_pass_row(s, i, coarse, cfg, fine, gate_m):
    """The second-pass columns of one sample. s: the coarse sample, coarse: its `peak` Match, cfg: the coarse config,
    fine: a callable (i, centre_en) -> (Match, window sample) with a `.cfg` (the fine config). Every pose is taken to
    the tile frame (east, north metres from the tile centre) with `pose_en`, and errors are distances there."""
    n, S, cell = int(cfg.grid.n), int(s["ref"].shape[-1]), float(cfg.grid.cell_m)
    centre_c = _centre_of(s)
    en_gt = pose_en(s["H"].numpy().astype(float), centre_c, n, cell, S)
    en_c = pose_en(coarse.H, centre_c, n, cell, S) if coarse.H is not None else None
    centre_f = en_c if en_c is not None else np.zeros(2)          # no coarse pose: the tile centre (chance level)
    m, sf = fine(i, torch.from_numpy(np.asarray(centre_f, np.float64)))
    nf, Sf, cell_f = int(fine.cfg.grid.n), int(sf["ref"].shape[-1]), float(fine.cfg.grid.cell_m)
    en_f = pose_en(m.H, _centre_of(sf), nf, cell_f, Sf) if m.H is not None else None
    shift = None if en_f is None or en_c is None else float(np.linalg.norm(en_f - en_c))
    # gate: keep the coarse pose when the fine pose is missing or jumped more than gate_m (no coarse pose: nothing to
    # compare with, the fine pose stands)
    fallback = en_f is None or (shift is not None and shift > float(gate_m))
    en_g = en_c if fallback else en_f

    def err(en):
        return None if en is None else float(np.linalg.norm(en - en_gt))
    yaw = pose_errors(m.H, sf["H"].numpy().astype(float), nf, cell_f)["yaw_deg"] if m.H is not None else None
    out = {"pose_fine_m": err(en_f), "yaw_fine_deg": yaw, "inliers_fine": m.inlier_ratio,
           "pose_fine_gated_m": err(en_g), "fallback_fine": bool(fallback), "fine_shift_m": shift,
           "no_coarse_fine": en_c is None,
           "en_gt": en_gt.tolist(), "en_coarse": None if en_c is None else en_c.tolist(),
           "en_fine": None if en_f is None else en_f.tolist(), "fine_centre_en": np.asarray(centre_f).tolist(),
           "fine_error": None}
    out.update(getattr(fine, "last_info", None) or {})             # matcher-specific counts (LoFTR: nmatch_fine, ...)
    return out


def certainty_cfg(cfg):
    """The `certainty:` block (statistics radius, reference-cell validity): the run's config, else configs/default.yaml
    (the VIGOR configs are full copies written before the block existed; one source for the defaults)."""
    return getattr(cfg, "certainty", None) or C.load().certainty


def calib_split(n_labels, train_meta, val_frac, cities, limit, assume=None):
    """Indices (into the full training label list of `cities`) of the --calib frames and a record of the rule.

    The held-out part is train_vigor.py's: permutation(seed 0) of the n_labels training labels, the last val_frac held
    out; its first val_samples were the validation frames that selected the checkpoint, so they are skipped:
    idx[n_tr + val_samples : n_tr + val_samples + limit] (limit 0 = the rest). The split is read from the checkpoint's
    `train` dict (train_vigor.py stores split / cities / val_frac / val_samples since task 06). Refused (ValueError):
    no val_frac in the checkpoint (old protocol, or trained before the split was stored) unless `assume` =
    dict(val_frac, val_samples, cities) is given (--assume-train-split: the user asserts the training settings), a
    val_frac different from --val-frac, cities different from the training cities, or nothing left after the skip."""
    tm = dict(train_meta or {})
    source = "checkpoint"
    if tm.get("val_frac") is None:
        if assume is None:
            raise ValueError("--calib: the checkpoint's train dict has no val_frac (old protocol, or trained before "
                             "train_vigor.py stored its split): its held-out frames cannot be reproduced. If you know "
                             "it was trained with the held-out split, pass --assume-train-split with --val-frac, "
                             "--val-samples and --cities as used in training")
        tm.update(assume)
        source = "asserted (--assume-train-split)"
    if not float(tm["val_frac"]) > 0:
        raise ValueError(f"--calib: the checkpoint was trained with val_frac {tm['val_frac']}: no held-out frames")
    if abs(float(tm["val_frac"]) - float(val_frac)) > 1e-12:
        raise ValueError(f"--calib: --val-frac {val_frac} differs from the checkpoint's val_frac {tm['val_frac']}")
    if tm.get("cities") is None or sorted(tm["cities"]) != sorted(cities):
        raise ValueError(f"--calib: cities {sorted(cities)} differ from the training cities {tm.get('cities')}")
    vs = int(tm.get("val_samples") or 0)
    idx = np.random.default_rng(0).permutation(n_labels)
    n_tr = int(n_labels * (1.0 - float(val_frac)))
    lo = n_tr + vs
    hi = n_labels if not limit else min(n_labels, lo + int(limit))
    if hi <= lo:
        raise ValueError(f"--calib: nothing left: {n_labels - n_tr} held out, {vs} used for checkpoint selection")
    info = dict(val_frac=float(val_frac), val_samples_skipped=vs, cities=sorted(cities), source=source,
                n_train_list=int(n_labels), n_heldout=int(n_labels - n_tr),
                heldout_slice=[int(lo - n_tr), int(hi - n_tr)], n=int(hi - lo),
                rule="permutation(seed 0) of the train labels; idx[n_tr + val_samples : n_tr + val_samples + limit]")
    return idx[lo:hi], info


def score(ds, query, matcher, cons, cfg, dev, n_max=0, verbose=False, refine=0, refine_inits=("coarse",),
          refine_gate=None, refine_min_cert=0.0, refine_min_corr=8, fine=None, fine_gate=6.0, hyp=0):
    """refine = 0: the coarse rows only (identical to before the sub-cell stage, plus the STAT_KEYS statistics of the
    peak row). refine = S > 0: also the refined row(s) from the decoder's stride-16 refiner, correspondences on a
    stride-S query grid (see module doc). fine (e.g. `DecoderFine`): also the second-pass rows around the coarse `peak`
    pose (`fine_pass_row`). hyp = K > 0: also the bootstrap-hypothesis keys (HYP_KEYS) of the peak row."""
    rows = []
    tags = refine_tags(refine_inits) if refine else {}
    n = int(cfg.grid.n)
    cc = certainty_cfg(cfg)
    idx = range(len(ds)) if not n_max else range(min(n_max, len(ds)))
    for i in idx:
        try:
            s = ds[i]
        except RuntimeError as e:                                   # unreadable / missing image
            print(f"  skip {i}: {e}", flush=True)
            continue
        batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = query(batch, matcher)
            f_s = matcher.reference_features(batch["ref"])
            sf = float(((f_q.shape[-2] * 16) * (f_q.shape[-1] * 16)) ** 0.5 / 560.0)
            tap = RefinerTap(matcher.model.decoder) if refine else contextlib.nullcontext()
            with matcher.model.exposed_intermediates(), tap:
                o16 = matcher.model.decoder({16: f_q}, f_s, scale_factor=sf)[16]
            gm = o16["gm_cls"][0]
        cert = o16["gm_certainty"][0, 0] if o16.get("gm_certainty") is not None else None
        ref_valid = ref_cell_valid(batch["ref"], int(round(gm.shape[0] ** 0.5)), min_frac=float(cc.ref_cell_min_frac))
        H = s["H"].numpy().astype(float)
        row = dict(id=s["id"], city=s["city"], centre_guess_m=ds.centre_guess_m(i))
        coarse = {}
        for tag, c in cons.items():
            c.stats_radius = float(cc.stats_radius_cells)
            m = coarse[tag] = consensus_for_query(c, gm, query, batch, frac, n, min_frac=0.05, certainty=cert,
                                                  ref_valid=ref_valid, stats=tag == "peak")
            err = pose_errors(m.H, H, n, float(cfg.grid.cell_m)) if m.H is not None else None
            row[f"pose_{tag}_m"] = None if err is None else err["position_m"]
            row[f"yaw_{tag}_deg"] = None if err is None else err["yaw_deg"]
            row[f"inliers_{tag}"] = m.inlier_ratio
        if "peak" in coarse:
            mh = coarse["means"].H if "means" in coarse else None
            row.update(stats_row(coarse["peak"].stats, None if mh is None else pose_px(mh, int(cons["peak"].m.im_a_size)),
                                 float(cfg.grid.cell_m)))
            if hyp:
                hs = SatRoMa.hypothesis_spread(cons["peak"], coarse["peak"], int(hyp), seed=int(cfg.matcher.seed))
                ok = hs is not None and hs["H"] is not None
                row["pose_hyp_med_m"] = pose_errors(hs["H"], H, n, float(cfg.grid.cell_m))["position_m"] if ok else None
                row["spread_hyp_m"] = (hs["spread_px"] * float(cfg.grid.cell_m)
                                       if ok and hs["spread_px"] is not None else None)        # < 2 hypotheses: None
                row["n_hyp_ok"] = 0 if hs is None else int(hs["n_ok"])
        for init, tag in tags.items():
            with torch.no_grad():
                m, info = refined_for_query(cons["peak"], o16, tap, query, batch, refine, init=init,
                                            coarse=coarse["peak"], gate_cells=refine_gate,
                                            min_cert=refine_min_cert, frac=frac, min_frac=0.05,
                                            min_corr=refine_min_corr)
            err = pose_errors(m.H, H, n, float(cfg.grid.cell_m)) if m.H is not None else None
            row[f"pose_{tag}_m"] = None if err is None else err["position_m"]
            row[f"yaw_{tag}_deg"] = None if err is None else err["yaw_deg"]
            row[f"inliers_{tag}"] = m.inlier_ratio
            row[f"ncorr_{tag}"], row[f"nused_{tag}"], row[f"fallback_{tag}"] = info["n_corr"], info["n_used"], info["fallback"]
        if fine is not None:
            try:
                row.update(fine_pass_row(s, i, coarse["peak"], cfg, fine, fine_gate))
            except Exception as e:                                  # keep the coarse row; the fine columns are None
                if _is_oom(e):
                    raise
                print(f"  fine pass failed on {i}: {type(e).__name__}: {e}", flush=True)
                row.update({k: None for k in FINE_KEYS})
                row["fallback_fine"] = None
                row["fine_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if verbose and len(rows) % 25 == 0:
            print(f"  {len(rows)} done  peak median so far "
                  f"{np.median([np.inf if r['pose_peak_m'] is None else r['pose_peak_m'] for r in rows]):.1f} m", flush=True)
    return rows


def med(rows, key="pose_peak_m"):
    return float(np.median([np.inf if r[key] is None else r[key] for r in rows])) if rows else float("nan")


def build_parser(doc=__doc__):
    ap = C.add_args(argparse.ArgumentParser(description=doc))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--split", default="crossarea", choices=("crossarea", "samearea"))
    ap.add_argument("--train-split", action="store_true", help="evaluate on the training cities/labels instead")
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="random subset per run (0 = all)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--row-sign", type=float, default=None)
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--solver", default="")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--refine", type=int, default=0, choices=(0, 1, 2, 4, 8, 16),
                    help="sub-cell stage: correspondence grid stride in query px of the decoder's (stride-16) conv "
                         "refiner output; 0 = off (default). 16 = one per token; < 16 samples the refined warp "
                         "bilinearly between tokens (the decoder has no finer refiner)")
    ap.add_argument("--refine-init", nargs="+", default=["coarse"], choices=REFINE_INITS,
                    help="none = package coarse warp, RANSAC from scratch; coarse = same, gated around the coarse "
                         "pose; ransac = refiner re-run on the coarse-RANSAC warp, gated. Several -> several rows")
    ap.add_argument("--refine-gate", type=float, default=None,
                    help="seeded inits: keep correspondences within this many reference cells of the coarse pose "
                         "(default cfg.matcher.reproj_cells)")
    ap.add_argument("--refine-min-cert", type=float, default=0.0,
                    help="keep refined correspondences with sigmoid(refined certainty) >= this (0 = all)")
    ap.add_argument("--refine-min-corr", type=int, default=8,
                    help="seeded inits: fewer correspondences than this after the gate -> fall back to the coarse "
                         "pose (counted as a fallback)")
    ap.add_argument("--fine-config", default=None,
                    help="second pass: config of the fine window (e.g. configs/vigor_cell00625_fine.yaml)")
    ap.add_argument("--fine-ckpt", default=None, help="second pass: decoder checkpoint trained with --fine-config")
    ap.add_argument("--fine-gate", type=float, default=6.0,
                    help="second pass: keep the coarse pose when the fine pose is farther than this (metres) from "
                         "it (row pose_fine_gated_m; counted as fallbacks)")
    ap.add_argument("--hyp", type=int, default=0,
                    help="multi-hypothesis spread: K RANSAC runs on bootstrap resamples of the peak row's modes -> "
                         "pose_hyp_med_m (medoid pose), spread_hyp_m; 0 = off (default)")
    ap.add_argument("--calib", action="store_true",
                    help="score the held-out --val-frac of the TRAINING list (train_vigor.py's split: seed-0 "
                         "shuffle, last val_frac held out), after the val_samples that selected the checkpoint, "
                         "--limit frames (0 = all): the calibration set for make vigor-certainty. Not --calibrate "
                         "(row_sign / height)")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="--calib: held-out fraction; must equal the checkpoint's (train_vigor.py --val-frac)")
    ap.add_argument("--val-samples", type=int, default=None,
                    help="--calib with --assume-train-split: the training run's --val-samples (skipped)")
    ap.add_argument("--assume-train-split", action="store_true",
                    help="--calib for a checkpoint that does not record its split (trained before task 06): assert "
                         "that it was trained with --val-frac / --val-samples / --cities as given here; recorded "
                         "in meta.calib.source")
    ap.add_argument("--out", default="experiments/09_vigor")
    ap.add_argument("--tag", required=True)
    return ap


def fine_config(a, cfg):
    """The fine config, with the coarse run's matcher block (solver, threshold in cells, seed: the same options)."""
    import copy
    cfg_f = C.load(a.fine_config)
    cfg_f.matcher = copy.deepcopy(cfg.matcher)
    return cfg_f


def fine_dataset(a, cfg_f, ds):
    """VigorPairs with the fine config (window, GSD, query mode) over exactly the coarse run's labels (the draw after
    any depth filtering)."""
    train_labels = bool(a.train_split or getattr(a, "calib", False))   # a calibration draw reads the train lists
    ds_f = VigorPairs(a.root, cfg_f, cities=a.cities or split_cities(a.split, train_labels), split=a.split,
                      train=train_labels, row_sign=ds.row_sign, height_m=ds.height)
    ds_f.labels = ds.labels
    return ds_f


def check_fine_grid(state, cfg_f, ckpt, config):
    """Refuse a fine checkpoint trained at another GSD than the fine config's (train_vigor.py records
    train.grid.cell_m). Checkpoints written before that record carry none: warned, not refused."""
    grid = ((state.get("train") or {}).get("grid") or {}) if isinstance(state, dict) else {}
    rec = grid.get("cell_m")
    if rec is None:
        print(f"WARNING: {ckpt} records no grid cell_m (older checkpoint); cannot check it against {config}", flush=True)
        return
    if abs(float(rec) - float(cfg_f.grid.cell_m)) > 1e-9:
        raise SystemExit(f"--fine-ckpt {ckpt} was trained at cell_m {rec} m/px, but --fine-config {config} has "
                         f"{cfg_f.grid.cell_m} m/px: the fine pass needs a checkpoint trained with that config")


def decoder_fine(a, cfg, ds, dev):
    """Build the Sat-RoMa second pass from --fine-config / --fine-ckpt. Filters ds (in place) to the panoramas with a
    depth file when the fine query needs depth, so both passes score the same samples. Returns (fine, meta)."""
    cfg_f = fine_config(a, cfg)
    state = torch.load(a.fine_ckpt, map_location=dev, weights_only=False)
    check_fine_grid(state, cfg_f, a.fine_ckpt, a.fine_config)
    mode = state.get("mode", "lift")
    cfg_f.lift.query_mode = mode
    apply_query_cfg(cfg_f, state)
    matcher = FeatureQueryMatcher(cfg_f.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    query = build_query(cfg_f, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    ds_f = fine_dataset(a, cfg_f, ds)
    n_drop = 0
    if ds_f.depth:
        n_drop = ds.keep_with_depth()
        ds_f.labels = ds.labels
    cons = SatRoMa.from_wrapper(matcher.wrapper, cfg_f, use_means=False, min_valid_frac=0.05)
    print(f"fine pass: {a.fine_ckpt} (mode {mode}, step {state.get('step')}), cell {cfg_f.grid.cell_m} m, window "
          f"{ds_f.ref_window_m} m, gate {a.fine_gate} m, solver {cfg_f.matcher.solver}"
          + (f"; {n_drop} panoramas without depth dropped from both passes" if n_drop else ""), flush=True)
    meta = dict(config=a.fine_config, ckpt=a.fine_ckpt, mode=mode, ckpt_step=state.get("step"), train=state.get("train"),
                cell_m=float(cfg_f.grid.cell_m), window_m=ds_f.ref_window_m, gate_m=a.fine_gate,
                matcher="Sat-RoMa decoder", skipped_no_depth=n_drop)
    return DecoderFine(ds_f, query, matcher, cons, cfg_f, dev), meta


def _mean_capped(rows, key):
    return float(np.mean([min(np.inf if r[key] is None else r[key], 1e3) for r in rows]))


def run(a, make_fine=None):
    """The evaluation. make_fine(a, cfg, ds, dev) -> (fine, meta) adds the second pass (None: coarse rows only)."""
    cfg = C.load(a.config)
    if a.solver:
        cfg.matcher.solver = a.solver
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = state.get("mode", "lift")
    cfg.lift.query_mode = mode
    apply_query_cfg(cfg, state)
    train_meta = state.get("train")
    print(f"checkpoint {a.ckpt}: mode {mode}, step {state.get('step')}, training {train_meta}", flush=True)
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    trained_refiner = any(REFINER_KEY in k for k in state["decoder"])
    refine_gate = float(cfg.matcher.reproj_cells if a.refine_gate is None else a.refine_gate)
    if a.refine:
        print(f"refine: stride {a.refine}, init {a.refine_init}, gate {refine_gate} cells, min cert "
              f"{a.refine_min_cert}, refiner {'from the checkpoint (trained)' if trained_refiner else 'released (frozen)'}",
              flush=True)
    query = build_query(cfg, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    cons = {tag: SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=means, min_valid_frac=0.05)
            for tag, means in (("peak", False), ("means", True))}
    cities = a.cities or split_cities(a.split, a.train_split or a.calib)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.calibrate:
        print(f"calibration on {a.limit or 150} samples, cities {cities}", flush=True)
        table = []
        for rs in (1.0, -1.0):
            for h in (1.6, 2.0, 2.5, 3.0):
                ds = VigorPairs(a.root, cfg, cities=cities, split=a.split, train=a.train_split,
                                limit=a.limit or 150, row_sign=rs, height_m=h)
                rows = score(ds, query, matcher, cons, cfg, dev)
                table.append(dict(row_sign=rs, height_m=h, median_m=med(rows), median_means_m=med(rows, "pose_means_m"),
                                  centre_guess_median_m=float(np.median([r["centre_guess_m"] for r in rows])), n=len(rows)))
                print(f"  row_sign {rs:+.0f}  height {h:.1f} m  ->  peak median {table[-1]['median_m']:.2f} m  "
                      f"means {table[-1]['median_means_m']:.2f} m  (centre guess {table[-1]['centre_guess_median_m']:.2f} m)", flush=True)
        best = min(table, key=lambda r: r["median_m"])
        (out / f"calibration_{a.tag}.json").write_text(json.dumps(dict(table=table, best=best), indent=2))
        print(f"best: row_sign {best['row_sign']:+.0f}, height {best['height_m']:.1f} m  ({best['median_m']:.2f} m)")
        return

    calib = None
    if a.calib:
        # train_vigor.py's held-out split, bit for bit: the full train list, permutation(seed 0), last val_frac held out
        ds = VigorPairs(a.root, cfg, cities=cities, split=a.split, train=True, row_sign=a.row_sign, height_m=a.height)
        assume = None
        if a.assume_train_split:
            if a.val_samples is None:
                raise SystemExit("--assume-train-split needs --val-samples (the training run's)")
            assume = dict(val_frac=a.val_frac, val_samples=a.val_samples, cities=list(cities))
        try:
            held, calib = calib_split(len(ds.labels), train_meta, a.val_frac, cities, a.limit, assume)
        except ValueError as e:
            raise SystemExit(str(e))
        ds.labels = [ds.labels[i] for i in held]
        calib["ids"] = [f"{lab['city']}/{lab['pano']}" for lab in ds.labels]      # before the depth filter
        print(f"calib: held-out frames {calib['heldout_slice'][0]}..{calib['heldout_slice'][1]} of "
              f"{calib['n_heldout']} (the first {calib['val_samples_skipped']} selected the checkpoint; split from "
              f"{calib['source']})", flush=True)
    else:
        ds = VigorPairs(a.root, cfg, cities=cities, split=a.split, train=a.train_split, limit=a.limit,
                        stride=a.stride, row_sign=a.row_sign, height_m=a.height)
    n_no_depth = ds.keep_with_depth() if ds.depth else 0      # after the draw: a subset of the same samples
    if ds.depth:
        print(f"depth: {n_no_depth} panoramas of the draw have no depth file (skipped)", flush=True)
    fine, fine_meta = make_fine(a, cfg, ds, dev) if make_fine is not None else (None, None)
    print(f"{len(ds)} samples, split {a.split}, cities {cities}, row_sign {ds.row_sign:+.0f}, height {ds.height} m", flush=True)
    rows = score(ds, query, matcher, cons, cfg, dev, verbose=True, refine=a.refine, refine_inits=tuple(a.refine_init),
                 refine_gate=refine_gate, refine_min_cert=a.refine_min_cert, refine_min_corr=a.refine_min_corr,
                 fine=fine, fine_gate=a.fine_gate, hyp=a.hyp)
    rtags = list(refine_tags(a.refine_init).values()) if a.refine else []
    ftags = ["fine", "fine_gated"] if fine is not None else []
    summary = {}
    for name in ["all"] + sorted({r["city"] for r in rows}):
        sub = rows if name == "all" else [r for r in rows if r["city"] == name]
        summary[name] = {
            "peak": summarise_pose([r["pose_peak_m"] for r in sub]),
            "means": summarise_pose([r["pose_means_m"] for r in sub]),
            "centre_guess": summarise_pose([r["centre_guess_m"] for r in sub]),
            "mean_peak_m": _mean_capped(sub, "pose_peak_m"),
        }
        for t in rtags:
            summary[name][t] = summarise_pose([r[f"pose_{t}_m"] for r in sub])
            summary[name][f"mean_{t}_m"] = _mean_capped(sub, f"pose_{t}_m")
            summary[name][f"fallback_{t}"] = int(sum(bool(r[f"fallback_{t}"]) for r in sub))
        for t in ftags:
            summary[name][t] = summarise_pose([r[f"pose_{t}_m"] for r in sub])
            summary[name][f"mean_{t}_m"] = _mean_capped(sub, f"pose_{t}_m")
        if ftags:
            summary[name]["nopose_fine"] = int(sum(r["pose_fine_m"] is None for r in sub))
            summary[name]["fallback_fine_gated"] = int(sum(bool(r["fallback_fine"]) for r in sub))
            summary[name]["no_coarse_fine"] = int(sum(bool(r["no_coarse_fine"]) for r in sub))
            summary[name]["fine_errors"] = int(sum(r.get("fine_error") is not None for r in sub))
        if a.hyp:
            summary[name]["hyp_med"] = summarise_pose([r["pose_hyp_med_m"] for r in sub])
            summary[name]["mean_hyp_med_m"] = _mean_capped(sub, "pose_hyp_med_m")
            sp = [r["spread_hyp_m"] for r in sub if r["spread_hyp_m"] is not None]
            summary[name]["spread_hyp_median_m"] = float(np.median(sp)) if sp else None
    path = out / f"eval_vigor_{a.tag}_{a.split}.json"
    path.write_text(json.dumps(dict(meta=dict(ckpt=a.ckpt, mode=mode, split=a.split, cities=cities, n=len(rows),
                                              row_sign=ds.row_sign, height_m=ds.height, solver=cfg.matcher.solver,
                                              skipped_no_depth=n_no_depth, ckpt_step=state.get("step"),
                                              train=train_meta,
                                              refine=dict(stride=a.refine, inits=a.refine_init, gate_cells=refine_gate,
                                                          min_cert=a.refine_min_cert, min_corr=a.refine_min_corr,
                                                          trained_refiner=trained_refiner,
                                                          path="decoder conv_refiner['16'] (the only refiner): input "
                                                               "warp = cls_to_flow_refine(gm_cls) for none/coarse, "
                                                               "H_coarse(query point) for ransac")
                                              if a.refine else None,
                                              fine=fine_meta,
                                              city_res=CITY_RES, hyp=a.hyp, calib=calib,
                                              stats=dict(radius_cells=float(certainty_cfg(cfg).stats_radius_cells),
                                                         ref_cell_min_frac=float(certainty_cfg(cfg).ref_cell_min_frac),
                                                         row="peak")),
                                    frames=rows, summary=summary), indent=2))
    for name, s in summary.items():
        p, c = s["peak"], s["centre_guess"]
        print(f"{name:13s} n {p['n']:5d}  peak median {p['median_m']:.2f} m {tuple(round(v, 2) for v in p['median_ci'])}  "
              f"mean {s['mean_peak_m']:.2f} m  R@5 {p['recall@5m']:.2f}  R@10 {p['recall@10m']:.2f}  "
              f"| centre guess median {c['median_m']:.2f} m", flush=True)
        for t in rtags + ftags:
            r = s[t]
            tail = {"fine": f"no pose {s.get('nopose_fine')}  (errors: {s.get('fine_errors')})",
                    "fine_gated": f"fallback {s.get('fallback_fine_gated')}  (no coarse pose: {s.get('no_coarse_fine')})"
                    }.get(t, f"fallback {s.get(f'fallback_{t}')}")
            print(f"{'':13s} {t:>14s} median {r['median_m']:.2f} m {tuple(round(v, 2) for v in r['median_ci'])}  "
                  f"mean {s[f'mean_{t}_m']:.2f} m  R@5 {r['recall@5m']:.2f}  R@10 {r['recall@10m']:.2f}  {tail}",
                  flush=True)
        if a.hyp:
            r = s["hyp_med"]
            print(f"{'':13s} {'hyp medoid':>14s} median {r['median_m']:.2f} m  R@5 {r['recall@5m']:.2f}  "
                  f"R@10 {r['recall@10m']:.2f}  median spread {s['spread_hyp_median_m']} m (K={a.hyp})", flush=True)
    print(f"wrote {path}", flush=True)


def main():
    a = build_parser().parse_args()
    if bool(a.fine_config) != bool(a.fine_ckpt):
        raise SystemExit("--fine-config and --fine-ckpt go together")
    run(a, decoder_fine if a.fine_ckpt else None)


if __name__ == "__main__":
    main()
