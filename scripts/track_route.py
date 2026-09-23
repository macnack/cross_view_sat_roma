"""Track a Mapillary route with an SE(2) particle filter fed by Sat-RoMa's vote heatmap.

  make track-route CKPT=checkpoints/05_lift_splat_fixtor_seq_best.pt ROUTE=Fixtor/IcRzj0wTLZX874qitxVsQa TAG=seq

Per frame: (1) predict with the proxy relative pose plus Gaussian noise (stands in for odometry;
the noise level is the PF process noise), (2) render the reference crop around the CURRENT PF
estimate (the prior now comes from the filter, not from a random window), (3) query -> decoder ->
``vote_heatmap`` over the 56x56 reference cells, (4) weight every particle by the bilinear lookup
of that log-heatmap at its position, plus a compass observation on yaw. Logged next to it: the
per-frame RANSAC pose on the same crop and plain dead reckoning. Mapillary poses are a proxy,
not survey GT; errors are against that proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bevloc import config as C
from bevloc.baselines.common import pose_from_homography
from bevloc.data.mapillary import (
    MAP_ROOT, MapillaryPairs, PoznanOrtho, load_frames, poznan_tiles, src_in_query_se2,
)
from bevloc.data.ortho import Oriented
from bevloc.match.satroma import SatRoMa
from bevloc.model.coarse import FeatureQueryMatcher, vote_heatmap
from bevloc.model.query import build_query, load_query_state
from bevloc.track.pf import SE2ParticleFilter


def wrap_deg(d):
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def bilinear_log_heat(heat_kk, cells_xy):
    """heat (K, K) log-probs; cells_xy (N, 2) in cell units (col, row), centre convention. Outside -> min."""
    K = heat_kk.shape[0]
    x, y = cells_xy[:, 0], cells_xy[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    out = np.full(len(x), float(heat_kk.min()))
    ok = (x0 >= 0) & (x0 < K - 1) & (y0 >= 0) & (y0 < K - 1)
    xa, ya = x0[ok], y0[ok]
    out[ok] = ((1 - fx[ok]) * (1 - fy[ok]) * heat_kk[ya, xa] + fx[ok] * (1 - fy[ok]) * heat_kk[ya, xa + 1]
               + (1 - fx[ok]) * fy[ok] * heat_kk[ya + 1, xa] + fx[ok] * fy[ok] * heat_kk[ya + 1, xa + 1])
    return out


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--route", default="Fixtor/IcRzj0wTLZX874qitxVsQa")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--n", type=int, default=0, help="max frames after striding (0 = all)")
    ap.add_argument("--solver", default="")
    ap.add_argument("--seq-dists", default="")
    ap.add_argument("--out", default="experiments/07_track")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    cfg = C.load(a.config)
    L, P = cfg.lift, cfg.pf
    if a.seq_dists.strip():
        L.seq_dists_m = [float(x) for x in a.seq_dists.split(",") if x.strip()]
    if a.solver:
        cfg.matcher.solver = a.solver
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.matcher.seed)

    ortho = PoznanOrtho(poznan_tiles(a.year))
    frames = load_frames([MAP_ROOT / a.route], ortho, margin_m=L.margin_m)
    frames.sort(key=lambda fr: fr.get("captured_at", 0))
    frames = frames[:: max(1, a.stride)]
    if a.n:
        frames = frames[: a.n]
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = state.get("mode", "lift")
    L.query_mode = mode
    ds = MapillaryPairs(frames, {a.year: ortho}, cfg, train=False, seed=cfg.train.seed,
                        erp_size=tuple(L.erp_size), years=[a.year])
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    query = build_query(cfg, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    ransac = SatRoMa.from_config(cfg, use_means=False, min_valid_frac=L.min_patch_valid)
    ransac.m.model.decoder.load_state_dict(state["decoder"], strict=False)

    g = cfg.grid
    n, gsd, scale = int(g.n), float(g.cell_m), int(cfg.reference.scale)
    K, cell_px = 56, n * scale / 56.0
    pf = SE2ParticleFilter(n=int(P.n), seed=cfg.matcher.seed)
    en0, up0 = ds.frames[0]["_en"], ds._up_of(ds.frames[0])
    pf.init(en0, up0, sigma_xy=P.init_sigma_xy, sigma_yaw=P.init_sigma_yaw)
    dr_en, dr_yaw = np.asarray(en0, float) + rng.normal(0, P.init_sigma_xy, 2), up0 + rng.normal(0, P.init_sigma_yaw)
    prev = None
    rows = []
    for i, fr in enumerate(ds.frames):
        en_true, up_true = np.asarray(fr["_en"], float), ds._up_of(fr)
        if prev is not None:
            yaw_rel, tx, ty = src_in_query_se2(prev[0], prev[1], en_true, up_true)
            dyaw = float(wrap_deg(up_true - prev[1]))
            dx, dy, dyw = tx + rng.normal(0, P.sigma_xy), ty + rng.normal(0, P.sigma_xy), dyaw + rng.normal(0, P.sigma_yaw)
            pf.predict(dx, dy, dyw, sigma_xy=P.sigma_xy, sigma_yaw=P.sigma_yaw)
            b = np.radians(dr_yaw)
            dr_en = dr_en + dx * np.array([np.sin(b), np.cos(b)]) + dy * np.array([-np.cos(b), np.sin(b)])
            dr_yaw = (dr_yaw + dyw) % 360.0
        prev = (en_true, up_true)

        en_est, yaw_est = pf.estimate()
        ref_o = Oriented(tuple(en_est), float(yaw_est), n * scale, gsd)
        try:
            s = ds.sample_for(fr["id"], ref_o, a.year)
        except RuntimeError as e:                       # crop off the map: keep predicting
            rows.append(dict(frame_id=fr["id"], skipped=str(e)[:60]))
            continue
        batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = query(batch, matcher)
            f_s = matcher.reference_features(batch["ref"])
            out = matcher.model.decoder({16: f_q}, f_s, scale_factor=0.4)[16]
            gm, cert = out["gm_cls"], out.get("gm_certainty")
            matchable = frac >= L.min_patch_valid          # no GT here: validity only
            heat = vote_heatmap(gm, cert, matchable)[0].cpu().numpy()
        # particle log-likelihood from the heatmap at each particle's map position
        inv = np.linalg.inv(ref_o.px_to_world)
        pe = np.c_[pf.particles[:, :2], np.ones(pf.n)] @ inv.T
        cells = (pe[:, :2] + 0.5) / cell_px - 0.5
        pf.update(P.temperature * bilinear_log_heat(heat, cells))
        pf.update(-0.5 * (wrap_deg(pf.particles[:, 2] - up_true) / P.sigma_yaw_obs) ** 2)
        en_pf, yaw_pf = pf.estimate()
        # per-frame RANSAC on the same crop
        mask = torch.nn.functional.interpolate(frac[:, None].float(), size=(n, n), mode="nearest")[0, 0]
        m = ransac.match_encoded(f_q, f_s[16], scale_factor=0.4, mask=(mask >= L.min_patch_valid).cpu().numpy())
        r_err = None
        if m.H is not None:
            en_r, _ = pose_from_homography(m.H, n, ref_o)
            r_err = float(np.linalg.norm(np.asarray(en_r) - en_true))
        row = dict(frame_id=fr["id"], pf_err_m=float(np.linalg.norm(en_pf - en_true)),
                   pf_yaw_err_deg=float(abs(wrap_deg(yaw_pf - up_true))),
                   prior_err_m=float(np.linalg.norm(en_est - en_true)),
                   ransac_err_m=r_err, dr_err_m=float(np.linalg.norm(dr_en - en_true)),
                   ess=float(pf.ess()), en_true=en_true.tolist(), en_pf=en_pf.tolist(),
                   en_ransac=None if r_err is None else list(map(float, en_r)), en_dr=dr_en.tolist())
        rows.append(row)
        if i % 10 == 0 or i == len(ds.frames) - 1:
            print(f"  {i:4d} {fr['id']}  prior {row['prior_err_m']:.1f}  pf {row['pf_err_m']:.1f}  "
                  f"ransac {'-' if r_err is None else round(r_err, 1)}  dr {row['dr_err_m']:.1f}  ess {row['ess']:.0f}",
                  flush=True)

    ok = [r for r in rows if "pf_err_m" in r]
    def summ(key):
        v = np.asarray([np.inf if r[key] is None else r[key] for r in ok], float)
        return dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)),
                    frac_le_5m=float((v <= 5).mean()), frac_le_10m=float((v <= 10).mean()), n=int(len(v)))
    summary = {k: summ(k) for k in ("pf_err_m", "ransac_err_m", "dr_err_m", "prior_err_m")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(a.route).name}_{a.tag}_y{a.year}"
    (out / f"track_{tag}.json").write_text(json.dumps(
        {"meta": dict(ckpt=a.ckpt, route=a.route, year=a.year, stride=a.stride, mode=mode,
                      solver=cfg.matcher.solver, pf=vars(P) if not isinstance(P, dict) else P,
                      note="errors vs the Mapillary pose proxy; motion = proxy relative pose + noise"),
         "summary": summary, "frames": rows}, indent=1))
    for k, s_ in summary.items():
        print(f"{k:13s} median {s_['median']:.1f} m  p95 {s_['p95']:.1f} m  <=5 m {s_['frac_le_5m']:.2f}  "
              f"<=10 m {s_['frac_le_10m']:.2f}  n {s_['n']}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    t = np.array([r["en_true"] for r in ok]); p_ = np.array([r["en_pf"] for r in ok]); d = np.array([r["en_dr"] for r in ok])
    ax[0].plot(t[:, 0], t[:, 1], "-", color="green", lw=2, label="proxy pose")
    ax[0].plot(d[:, 0], d[:, 1], "-", color="grey", lw=1, label="dead reckoning")
    ax[0].plot(p_[:, 0], p_[:, 1], "-", color="red", lw=1.5, label="particle filter")
    rr = np.array([r["en_ransac"] for r in ok if r["en_ransac"] is not None])
    if len(rr):
        ax[0].plot(rr[:, 0], rr[:, 1], ".", color="blue", ms=3, label="per-frame RANSAC")
    ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_title(f"{Path(a.route).name} {a.year} {mode}")
    ax[0].plot([t[0, 0], t[0, 0] + 100], [t[:, 1].min() - 20] * 2, "k-", lw=3); ax[0].text(t[0, 0], t[:, 1].min() - 30, "100 m")
    for k, c in (("pf_err_m", "red"), ("ransac_err_m", "blue"), ("dr_err_m", "grey")):
        v = [np.nan if r[k] is None else r[k] for r in ok]
        ax[1].plot(v, color=c, lw=1, label=k)
    ax[1].set_ylim(0, 60); ax[1].set_xlabel("frame"); ax[1].set_ylabel("error vs proxy [m]"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / f"track_{tag}.jpg", dpi=110)
    print(f"wrote {out / f'track_{tag}.json'} and .jpg", flush=True)
    C.snapshot(cfg, out, dict(ckpt=a.ckpt, route=a.route, tag=a.tag, year=a.year))
    ortho.close()


if __name__ == "__main__":
    main()
