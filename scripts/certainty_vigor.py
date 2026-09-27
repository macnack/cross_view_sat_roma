"""Calibrated per-frame confidence from an eval_vigor.py json (task 06, docs/tasks/06_certainty.md, designs 1 + 5).

  make vigor-certainty EVAL_JSON=experiments/10_loc2_matcher/eval_vigor_<tag>_samearea.json \
                       CALIB_JSON=experiments/10_loc2_matcher/eval_vigor_<tag>_calib_samearea.json TAG=<tag>

The calibrator is fitted on CALIB_JSON (held-out training frames, `eval_vigor.py --calib`) and evaluated on EVAL_JSON
(the test draw); the two must share no frame. Without CALIB_JSON: a --folds split of EVAL_JSON by frame, every number
out of fold. Models, each fitted per target radius tau (cfg.certainty.targets_m) as P(error < tau):
  inlier_isotonic   the RANSAC inlier ratio alone, isotonic (the gate of 2026-09-25, made a probability)
  logistic          L2 logistic regression on the standardised statistics of eval_vigor.py (STAT_KEYS, the
                    --hyp spread when present) + the inlier ratio
  gbm               gradient-boosted trees on the same features, only when sklearn is importable
A frame without a pose is a failure (error = inf) with confidence 0. Metrics per model and target: AUROC, average
precision, 10-bin reliability + ECE, Brier; the abstention curve (frames kept in decreasing P(error <
cfg.certainty.rank_target_m)) at cfg.certainty.coverage_points; split conformal on the binary event "error < tau"
(sets {correct} / {wrong} / both = abstain, coverage >= 1 - alpha guaranteed; its calibration half is disjoint from
the half the model is fitted on) with the achieved test coverage; the guarantee is marginal and needs exchangeable
calibration and test frames, so it does not hold for a cross-area split (warned). Univariate AUROC of every raw
statistic at the rank target shows which statistics carry the information.
Writes <out>/certainty_<tag>.json (all numbers) and certainty_<tag>.png; prints one summary table.

Task 06 step 4 (the pose-correctness head, bevloc.model.certainty_head): --cache <test cache pkl> takes the test frames
from a certainty cache of scripts/certainty_cache_vigor.py instead of --eval-json (same errors and statistics, gated
against eval_vigor.py at caching), --fit-cache <calib cache pkl> fits the isotonic / logistic rows on the cache the
head trained on (the same frames: a like-for-like comparison), and --head <ckpt> [<ckpt> ...] adds one row per trained
head (scripts/train_certainty_head.py), named head:<checkpoint stem>, with its calibrated P(error < tau) on the test
cache's frames. The head's conformal half is its own validation frames (the last val_frames of --fit-cache, never
trained on; early stopping and the temperature were fitted there, so that half is not as clean as the logistic's) and
is skipped when --fit-cache is absent. With several cities in the test cache, per-city AUROC at the rank target is
reported for every row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bevloc import config as C
from bevloc.eval import calibration as K

# (row key, transform): log1p for counts and heavy-tailed lengths
FEATURES = (
    ("inliers_{row}", None), ("vote_entropy", None), ("vote_support_cells", "log"), ("vote_top1", None),
    ("vote_mass_2cells", None), ("n_modes", "log"), ("n_inlier_modes", "log"), ("peak_means_m", "log"),
    ("pose_centre_m", None), ("cert_mean", None), ("cert_median", None), ("n_valid_tokens", "log"),
    ("placed_depth_m", None), ("ego_entropy", None), ("ego_support_cells", "log"), ("ego_top1", None),
    ("ego_mass_2cells", None), ("ego_peak_pose_m", "log"), ("spread_hyp_m", "log"),
    ("vote_offtile_mass", None),      # since step 4 (2026-09-27): the logistic and the head's frame stream see the same 19
)
CONFORMAL_NOTE = ("split conformal: the coverage guarantee is MARGINAL (on average over test frames, not per frame or "
                  "per confidence level) and holds only when calibration and test frames are exchangeable; it does "
                  "NOT hold for a cross-area (or cross-city, cross-season) calibration/test split")
COLORS = {"inlier_isotonic": "#2a78d6", "logistic": "#eb6834", "gbm": "#1baf7a", "oracle": "#7a7a74"}
HEAD_COLORS = ("#8e44ad", "#c0392b", "#16a085", "#d4a017", "#2c3e50", "#e67e22")


def color_of(name, i=0):
    return COLORS.get(name, HEAD_COLORS[i % len(HEAD_COLORS)])


def load_rows(path):
    d = json.loads(Path(path).read_text())
    return d["frames"], d.get("meta", {})


def rows_from_cache(cache):
    """(rows, meta) in eval_vigor's row layout from a certainty cache: id, city, pose_peak_m, inliers_peak and the
    frame statistics (FRAME_KEYS of certainty_cache_vigor.py, which include every STAT_KEY)."""
    m = cache["meta"]
    rows = [dict(id=f["id"], city=f["city"], pose_peak_m=f["pose_peak_m"], inliers_peak=f["inliers_peak"],
                 **{k: v for k, v in f["stats"].items() if k != "inliers_peak"}) for f in cache["frames"]]
    meta = dict(ckpt=m.get("ckpt"), split=m.get("split"), cities=m.get("cities"), n=m.get("n"), draw=m.get("draw"),
                draw_info=m.get("draw_info"), solver=m.get("solver"), consensus=m.get("consensus"), gate=m.get("gate"),
                stats=m.get("stats"))
    return rows, meta


def head_predictions(paths, test_arrays, fit_arrays, err_test, cfg):
    """{head name: {tau: dict(prob, qhat, has_ok, has_bad, n_cal)}} of every head checkpoint on the test cache. The
    conformal half is the head's own validation frames looked up by id in the fit cache (None: no conformal numbers).
    A test frame without a pose gets probability 0, as for every other model."""
    import torch
    from bevloc.model.certainty_head import load_head, predict
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    pt = np.isfinite(err_test)
    for path in paths:
        head, ck = load_head(path, dev)
        taus = [float(t) for t in ck.get("targets_m", cfg.targets_m)]
        if taus != [float(t) for t in cfg.targets_m]:
            sys.exit(f"{path} was trained for targets {taus}, the config asks for {list(cfg.targets_m)}")
        if list(ck.get("frame_keys") or []) != list(test_arrays.frame_keys):
            sys.exit(f"{path}: frame statistics {ck.get('frame_keys')} differ from the test cache's {test_arrays.frame_keys}")
        name = "head:" + Path(path).stem.replace("certainty_head_", "", 1)
        P = predict(head, test_arrays, dev)
        cal = None
        if fit_arrays is not None:
            pos = {i: j for j, i in enumerate(fit_arrays.ids)}
            idx = [pos[i] for i in ck.get("val_ids") or [] if i in pos]
            if idx:
                sub = fit_arrays.subset(idx)
                cal = (predict(head, sub, dev), sub.err)
        out[name] = {}
        for j, tau in enumerate(taus):
            prob = np.zeros(len(err_test))
            prob[pt] = P[pt, j]
            if cal is None:
                out[name][tau] = dict(prob=prob, qhat=None, has_ok=None, has_bad=None, n_cal=0)
                continue
            p_cal = np.where(np.isfinite(cal[1]), cal[0][:, j], 0.0)
            q, _, _, _, _, (has_ok, has_bad) = K.conformal_sets(p_cal, cal[1] < tau, prob, float(cfg.conformal_alpha))
            out[name][tau] = dict(prob=prob, qhat=q, has_ok=has_ok, has_bad=has_bad, n_cal=int(len(p_cal)))
        print(f"{name}: streams {ck.get('streams')}, temperature {ck.get('temperature')}, best epoch "
              f"{ck.get('best_epoch')}, trained on {ck.get('n_train')} frames of {(ck.get('cache') or {}).get('path')}"
              f"{'' if cal is not None else ' (no --fit-cache validation frames: no conformal numbers)'}", flush=True)
    return out


def errors(rows, row):
    return np.array([np.inf if r.get(f"pose_{row}_m") is None else float(r[f"pose_{row}_m"]) for r in rows])


def feature_names(rows_sets, row):
    """Features present (key in every row, not all None) in every row set."""
    names = []
    for key, tr in FEATURES:
        k = key.format(row=row)
        if all(rows and all(k in r for r in rows) and any(r[k] is not None for r in rows) for rows in rows_sets):
            names.append((k, tr))
    return names


def matrix(rows, names):
    X = np.full((len(rows), len(names)), np.nan)
    for j, (k, tr) in enumerate(names):
        for i, r in enumerate(rows):
            v = r.get(k)
            if v is not None and np.isfinite(float(v)):
                X[i, j] = np.log1p(max(float(v), 0.0)) if tr == "log" else float(v)
    return X


def model_list():
    return ["inlier_isotonic", "logistic"] + (["gbm"] if K.gbm_available() else [])


def fit_predict(name, Xf, yf, Xt, inl_col, cfg, seed):
    """Fit one model on (Xf, yf) (frames with a pose only) and predict P(correct) on Xt."""
    if name == "inlier_isotonic":
        return K.Isotonic().fit(Xf[:, inl_col], yf).predict(Xt[:, inl_col])
    if name == "logistic":
        return K.Logistic(l2=float(cfg.l2)).fit(Xf, yf).predict(Xt)
    return K.GBM(seed=seed).fit(Xf, yf).predict(Xt)


def run_split(fit, test, names, row, cfg, seed):
    """Fit every model on the `fit` frames and predict the `test` frames; conformal: model refitted on one half of
    `fit`, q-hat on the other half. Returns {model: {tau: dict(prob, conformal indicators)}}."""
    Xf, ef = matrix(fit, names), errors(fit, row)
    Xt, et = matrix(test, names), errors(test, row)
    inl = [k for k, _ in names].index(f"inliers_{row}")
    pf, pt = np.isfinite(ef), np.isfinite(et)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(fit))
    half_a = np.zeros(len(fit), bool)
    half_a[perm[: len(fit) // 2]] = True
    out = {}
    for name in model_list():
        out[name] = {}
        for tau in cfg.targets_m:
            yf = ef < tau
            prob = np.zeros(len(test))
            prob[pt] = fit_predict(name, Xf[pf], yf[pf], Xt[pt], inl, cfg, seed)
            # split conformal: fit on half A, calibrate on half B (all of B, pose-less frames with probability 0)
            pa = pf & half_a
            p_b = np.zeros(int((~half_a).sum()))
            b_pose = pf[~half_a]
            p_b[b_pose] = fit_predict(name, Xf[pa], yf[pa], Xf[~half_a][b_pose], inl, cfg, seed)
            p_c = np.zeros(len(test))
            p_c[pt] = fit_predict(name, Xf[pa], yf[pa], Xt[pt], inl, cfg, seed)
            q, acc, rej, abst, empty, (has_ok, has_bad) = K.conformal_sets(p_b, yf[~half_a], p_c, float(cfg.conformal_alpha))
            out[name][tau] = dict(prob=prob, qhat=q, has_ok=has_ok, has_bad=has_bad, n_cal=int((~half_a).sum()))
    return out


def pool(parts, idx_list, n):
    """Merge per-fold predictions back into frame order."""
    merged = {}
    for part, idx in zip(parts, idx_list):
        for name, per_tau in part.items():
            for tau, d in per_tau.items():
                m = merged.setdefault(name, {}).setdefault(tau, dict(prob=np.zeros(n), has_ok=np.zeros(n, bool),
                                                                     has_bad=np.zeros(n, bool), qhat=[], n_cal=[]))
                m["prob"][idx], m["has_ok"][idx], m["has_bad"][idx] = d["prob"], d["has_ok"], d["has_bad"]
                m["qhat"].append(d["qhat"])
                m["n_cal"].append(d["n_cal"])
    return merged


def pct(c):
    return f"at{int(round(100 * float(c)))}"


def coverage_grid(cfg):
    lo, hi, step = (float(v) for v in cfg.coverage_grid)
    return np.round(np.arange(lo, hi + step / 100, step), 6)


def abstention_entry(curve, cfg):
    return dict(curve=curve, full=K.at_coverage(curve, 1.0), **{pct(c): K.at_coverage(curve, float(c))
                                                               for c in cfg.coverage_points})


def conformal_entry(d, y, cfg):
    """The conformal numbers of one (model, tau); None when the model has no conformal half (a head without
    --fit-cache)."""
    if d.get("has_ok") is None:
        return None
    covered = np.where(y, d["has_ok"], d["has_bad"])
    acc = d["has_ok"] & ~d["has_bad"]
    return dict(alpha=float(cfg.conformal_alpha), target_coverage=1 - float(cfg.conformal_alpha),
                qhat=d["qhat"], n_cal=d["n_cal"], coverage=float(covered.mean()),
                accept=float(acc.mean()), reject=float((d["has_bad"] & ~d["has_ok"]).mean()),
                abstain=float((d["has_ok"] & d["has_bad"]).mean()),
                empty=float((~d["has_ok"] & ~d["has_bad"]).mean()),
                accuracy_accepted=float(y[acc].mean()) if acc.any() else None)


def evaluate(pred, err, names, test_rows, row, cfg):
    res = dict(models={}, abstention={}, feature_auroc={})
    covs = coverage_grid(cfg)
    for name, per_tau in pred.items():
        mres = {}
        for tau, d in per_tau.items():
            y = err < tau
            bins, ece = K.reliability(d["prob"], y, int(cfg.n_bins))
            mres[str(tau)] = dict(
                base_rate=float(y.mean()), auroc=K.auroc(d["prob"], y), ap=K.average_precision(d["prob"], y),
                ece=ece, brier=K.brier(d["prob"], y), reliability=bins, conformal=conformal_entry(d, y, cfg))
        res["models"][name] = mres
        rank = per_tau[float(cfg.rank_target_m)]["prob"]
        curve = K.abstention_curve(rank, err, covs, float(cfg.gross_m))
        res["abstention"][name] = abstention_entry(curve, cfg)
    curve = K.abstention_curve(-err, err, covs, float(cfg.gross_m))
    res["abstention"]["oracle"] = abstention_entry(curve, cfg)
    # several cities: AUROC at the rank target per city (the head trains on four cities, the logistic may not)
    cities = sorted({r["city"] for r in test_rows})
    if len(cities) > 1:
        city = np.array([r["city"] for r in test_rows])
        y = err < float(cfg.rank_target_m)
        res["per_city"] = {}
        for c in cities:
            sel = city == c
            res["per_city"][c] = dict(n=int(sel.sum()), base_rate=float(y[sel].mean()),
                                      auroc={name: K.auroc(per_tau[float(cfg.rank_target_m)]["prob"][sel], y[sel])
                                             for name, per_tau in pred.items()})
    # univariate: AUROC of each raw statistic for "error < rank target" over the frames with a pose; > 0.5 = higher
    # is better, < 0.5 = lower is better
    X = matrix(test_rows, names)
    y = err < float(cfg.rank_target_m)
    pose = np.isfinite(err)
    for j, (k, _) in enumerate(names):
        ok = pose & np.isfinite(X[:, j])
        res["feature_auroc"][k] = dict(auroc=K.auroc(X[ok, j], y[ok]) if ok.sum() > 1 else None, n=int(ok.sum()))
    return res


def plot(res, err, cfg, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tau = str(float(cfg.rank_target_m))
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2), dpi=110)
    a = ax[0]
    a.plot([0, 1], [0, 1], color="#b0afa8", lw=1.5, ls="--", label="perfect calibration")
    for i, (name, m) in enumerate(res["models"].items()):
        b = [x for x in m[tau]["reliability"] if x["n"]]
        a.plot([x["conf"] for x in b], [x["acc"] for x in b], "-o", color=color_of(name, i), lw=2, ms=7,
               label=f"{name} (ECE {m[tau]['ece']:.3f}, AUROC {m[tau]['auroc']:.3f})")
    a.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"predicted P(error < {cfg.rank_target_m:g} m)",
          ylabel="observed fraction with error < %g m" % cfg.rank_target_m,
          title=f"Reliability, {int(cfg.n_bins)} bins (test frames)")
    a.legend(loc="upper left", fontsize=9, frameon=False)
    for i, (key, lab) in enumerate((("median_m", "median position error (m)"),
                                     ("gross_frac", f"fraction of errors > {cfg.gross_m:g} m"))):
        a = ax[i + 1]
        for j, (name, d) in enumerate(res["abstention"].items()):
            c = d["curve"]
            a.plot(np.array(c["coverage"]) * 100, c[key], color=color_of(name, j), lw=2,
                   ls="--" if name == "oracle" else "-", label=name + (" (sorted by true error)" if name == "oracle" else ""))
        for cv in cfg.coverage_points:
            a.axvline(100 * float(cv), color="#d9d8d2", lw=1, zorder=0)
        a.set(xlabel="coverage: frames kept, most confident first (%)", ylabel=lab,
              xlim=(100 * float(cfg.coverage_grid[0]), 100),
              title=f"Abstention: {lab}")
        a.set_ylim(bottom=0)
        a.legend(loc="upper left", fontsize=9, frameon=False)
        a.grid(alpha=0.25)
    ax[0].grid(alpha=0.25)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summary_table(res, cfg):
    taus = [str(float(t)) for t in cfg.targets_m]
    rt = str(float(cfg.rank_target_m))
    pts = [int(round(100 * float(c))) for c in cfg.coverage_points]
    head = (["model"] + [f"AUROC<{float(t):g}" for t in taus] + [f"AP<{float(rt):g}", f"ECE<{float(rt):g}", "med@100"]
            + [x for c in pts for x in (f"med@{c}", f">{cfg.gross_m:g}m@{c}")]
            + [f"conf.cov<{float(rt):g} (target {1 - cfg.conformal_alpha:.2f})", "accept"])
    lines = [head]
    for name, d in res["abstention"].items():
        m = res["models"].get(name)
        full = d["full"]
        row = [name]
        row += [f"{m[t]['auroc']:.3f}" for t in taus] if m else ["-"] * len(taus)
        row += [f"{m[rt]['ap']:.3f}", f"{m[rt]['ece']:.3f}"] if m else ["-", "-"]
        row += [f"{full['median_m']:.2f}"]
        row += [x for c in pts for x in (f"{d[f'at{c}']['median_m']:.2f}", f"{d[f'at{c}']['gross_frac'] * 100:.1f}%")]
        cf = m[rt]["conformal"] if m else None
        row += [f"{cf['coverage']:.3f}", f"{cf['accept']:.2f}"] if cf else ["-", "-"]
        lines.append(row)
    return _table(lines)


def _table(lines):
    w = [max(len(r[i]) for r in lines) for i in range(len(lines[0]))]
    return "\n".join("  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r)) for r in lines)


def per_city_table(res, cfg):
    pc = res.get("per_city") or {}
    models = list(res["models"])
    lines = [["city", "n", f"base<{float(cfg.rank_target_m):g}"] + models]
    for c, d in pc.items():
        lines.append([c, str(d["n"]), f"{d['base_rate']:.3f}"] + [f"{d['auroc'][m]:.3f}" for m in models])
    return _table(lines)


def main(argv=None):
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--eval-json", default="", help="test draw (eval_vigor.py json); or --cache")
    ap.add_argument("--calib-json", default="", help="calibration frames (eval_vigor.py --calib); empty = k-fold")
    ap.add_argument("--cache", default="", help="test draw as a certainty cache (certainty_cache_vigor.py --draw test)")
    ap.add_argument("--fit-cache", default="",
                    help="calibration frames as a certainty cache (--draw calib): the isotonic / logistic rows are "
                         "fitted on the same frames the heads trained on")
    ap.add_argument("--head", nargs="*", default=[],
                    help="pose-correctness head checkpoint(s) (train_certainty_head.py), one row each; needs --cache")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="", help="output folder (default: the folder of --eval-json / --cache)")
    ap.add_argument("--row", default="peak", help="which pose row is scored: pose_<row>_m / inliers_<row>")
    a = ap.parse_args(argv)
    cfg = C.load(a.config).certainty
    cfg.targets_m = [float(t) for t in cfg.targets_m]
    cfg.rank_target_m = float(cfg.rank_target_m)
    if cfg.rank_target_m not in cfg.targets_m:
        sys.exit(f"certainty.rank_target_m {cfg.rank_target_m} must be one of targets_m {cfg.targets_m}")
    if bool(a.eval_json) == bool(a.cache):
        sys.exit("give exactly one of --eval-json and --cache")
    if a.calib_json and a.fit_cache:
        sys.exit("give at most one of --calib-json and --fit-cache")
    if a.head and not a.cache:
        sys.exit("--head needs the test frames as a cache (--cache): the heads read the maps and token rows")
    test_arrays = fit_arrays = None
    if a.cache:
        from bevloc.model.certainty_head import CacheArrays, load_cache
        cache_t = load_cache(a.cache)
        if a.row != "peak":
            sys.exit("a certainty cache holds the peak row's statistics only (--row peak)")
        test, tmeta = rows_from_cache(cache_t)
        test_arrays = CacheArrays(cache_t)
        if tmeta.get("draw") != "test":
            print(f"WARNING: --cache {a.cache} is a {tmeta.get('draw')!r} draw, not the test draw", flush=True)
    else:
        test, tmeta = load_rows(a.eval_json)
    err = errors(test, a.row)
    calib_src = a.calib_json or a.fit_cache
    if calib_src:
        if a.fit_cache:
            from bevloc.model.certainty_head import CacheArrays, load_cache
            cache_c = load_cache(a.fit_cache)
            calib, cmeta = rows_from_cache(cache_c)
            fit_arrays = CacheArrays(cache_c)
            if cmeta.get("draw") != "calib":
                sys.exit(f"--fit-cache {a.fit_cache} is a {cmeta.get('draw')!r} draw: the calibrators are fitted on "
                         f"held-out training frames only")
        else:
            calib, cmeta = load_rows(a.calib_json)
        shared = {r["id"] for r in calib} & {r["id"] for r in test}
        if shared:
            sys.exit(f"{len(shared)} frames are in both the calibration and the test json: never fit and test on the "
                     f"same frames")
        names = feature_names([calib, test], a.row)
        pred = pool([run_split(calib, test, names, a.row, cfg, int(cfg.seed))], [np.arange(len(test))], len(test))
        protocol = dict(mode="calib->test", calib_json=calib_src, n_calib=len(calib), calib_meta=cmeta)
        if sorted(cmeta.get("cities") or []) != sorted(tmeta.get("cities") or []) or cmeta.get("split") != tmeta.get("split"):
            protocol["warning"] = (f"calibration cities/split {cmeta.get('cities')}/{cmeta.get('split')} differ from "
                                   f"the test's {tmeta.get('cities')}/{tmeta.get('split')}: the conformal guarantee "
                                   f"does not apply")
    else:
        names = feature_names([test], a.row)
        perm = np.random.default_rng(int(cfg.seed)).permutation(len(test))
        folds = np.array_split(perm, int(cfg.folds))
        parts, idx_list = [], []
        for k, idx in enumerate(folds):
            rest = np.setdiff1d(perm, idx)
            parts.append(run_split([test[i] for i in rest], [test[i] for i in idx], names, a.row, cfg,
                                   int(cfg.seed) + k))
            idx_list.append(idx)
        pred = pool(parts, idx_list, len(test))
        protocol = dict(mode=f"{int(cfg.folds)}-fold out-of-fold on the test json (by frame)")
    if a.head:
        pred.update(head_predictions(a.head, test_arrays, fit_arrays, err, cfg))
        protocol["heads"] = list(a.head)
    print(f"test {a.eval_json or a.cache}: {len(test)} frames, {int(np.isinf(err).sum())} without a pose; protocol: "
          f"{protocol['mode']}; features {[k for k, _ in names]}; models {list(pred)}", flush=True)
    res = evaluate(pred, err, names, test, a.row, cfg)
    out = Path(a.out) if a.out else Path(a.eval_json or a.cache).parent
    out.mkdir(parents=True, exist_ok=True)
    res["meta"] = dict(eval_json=a.eval_json, cache=a.cache, fit_cache=a.fit_cache, heads=list(a.head), eval_meta=tmeta,
                       n_test=len(test), row=a.row, protocol=protocol,
                       features=[dict(key=k, transform=t) for k, t in names], models=list(pred),
                       config=C._plain(cfg), command=" ".join(sys.argv))
    rt = str(float(cfg.rank_target_m))
    c0 = cfg.coverage_points[0]
    res["gate"] = {name: dict(auroc_min=float(cfg.gate_auroc), gross_frac_max=float(cfg.gate_gross_frac),
                              at_coverage=float(c0),
                              auroc_ok=bool(res["models"][name][rt]["auroc"] >= float(cfg.gate_auroc)),
                              gross_ok=bool(res["abstention"][name][pct(c0)]["gross_frac"] <= float(cfg.gate_gross_frac)))
                   for name in res["models"]}
    res["conformal_note"] = CONFORMAL_NOTE
    jp, pp = out / f"certainty_{a.tag}.json", out / f"certainty_{a.tag}.png"
    jp.write_text(json.dumps(res, indent=2, default=float))
    plot(res, err, cfg, pp, f"Per-frame confidence, {a.tag}: {len(test)} test frames, {protocol['mode']}")
    print(summary_table(res, cfg))
    if res.get("per_city"):
        print(f"per-city AUROC for error < {cfg.rank_target_m:g} m:")
        print(per_city_table(res, cfg))
    fa = sorted(res["feature_auroc"].items(), key=lambda kv: -abs((kv[1]["auroc"] or 0.5) - 0.5))
    print(f"univariate AUROC for error < {cfg.rank_target_m:g} m (frames with a pose; < 0.5 = lower is better): "
          + ", ".join(f"{k} {v['auroc']:.3f}" for k, v in fa if v["auroc"] is not None))
    print(f"gate (AUROC >= {cfg.gate_auroc:g} at {cfg.rank_target_m:g} m and <= {100 * cfg.gate_gross_frac:g} % "
          f"errors > {cfg.gross_m:g} m at {100 * float(c0):g} % coverage): "
          + ", ".join(f"{n} {'PASS' if g['auroc_ok'] and g['gross_ok'] else 'fail'}" for n, g in res["gate"].items()))
    print("note: " + CONFORMAL_NOTE)
    if protocol.get("warning"):
        print("WARNING: " + protocol["warning"])
    print(f"wrote {jp} and {pp}")
    return res


if __name__ == "__main__":
    main()
