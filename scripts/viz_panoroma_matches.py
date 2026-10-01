"""PanoRoMa match figures: what the two-pass matcher does on frames where it works.

  make panoroma-viz CKPT=checkpoints/vigor_samearea_4city_erp_depth_cell0125_e100_last.pt \\
       FINE_CKPT=checkpoints/vigor_samearea_4city_fine00625_erp_depth_e100_last.pt \\
       EVAL_JSON=experiments/13_panoroma_long/eval_vigor_e100_full_chicago_twopass_se2_samearea.json \\
       N=10 PICK=quantiles [IDS="Chicago/<pano> ..."] OUT=experiments/13_panoroma_long/viz

Frames: PICK=quantiles takes the frames at the given percentiles (--quantiles, default 5 15 25 35 45 50) of the
FINAL (two-pass, gated) error of EVAL_JSON, an eval_vigor.py run of the same checkpoints on the same draw; PICK=ids
takes --ids (their percentile in EVAL_JSON is reported). Each frame is re-matched exactly as eval_vigor.py does
(se2 solver, decoder dtype, fine gate) and the recomputed errors are printed next to the json's.

Two stages (--stage all = both): `compute` (GPU) writes one npz bundle per frame to <out>/bundles/ (panorama,
depth, both reference canvases, every token's placement and the modes with their inlier flags), `render` (CPU)
draws one JPEG per frame from the bundles, a contact sheet of all of them and, with --diagram-id, one annotated
explanation figure. Figure layout:
  top            the panorama with the 56 x 28 token grid; dot per token placed by depth (filled = RANSAC inlier of
                 the coarse pass, hollow grey = placed but outlier, darkened = not placed: sky / >= max depth);
                 colour = azimuth (the band under it, the same colours on the maps)
  middle-left    UniK3D depth along each ray, metres
  middle-right   legend and the numbers
  bottom-left    the 112 m canvas (north up, metres from the tile centre): the tokens through the PREDICTED coarse
                 pose, a line from each inlier token to the centre of the 2 m reference cell it matched, GT (green
                 cross), coarse pose (orange), the 56 m second-pass window (dashed)
  bottom-right   the 56 m window: the same for the second pass (1 m cells), GT, coarse (orange), final (red)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

R_EARTH_NOTE = "known orientation: the panorama is north-aligned (centre column = north = ego forward)"
QUANTILES = (5, 15, 25, 35, 45, 50)


# ---------------------------------------------------------------------------------------------------------- picking

def final_err(r):
    v = r.get("pose_fine_gated_m")
    return np.inf if v is None else float(v)


def percentile_of(rows, rid):
    """Percentile (0-100) of frame rid's final error among rows (rank / (n - 1))."""
    errs = np.sort([final_err(r) for r in rows])
    e = final_err(next(r for r in rows if r["id"] == rid))
    return 100.0 * float(np.searchsorted(errs, e, side="left")) / max(1, len(errs) - 1)


def pick_rows(rows, pick, quantiles=QUANTILES, ids=None):
    """[(row, percentile)] for PICK=quantiles (the frame at rank round(q/100 * (n-1)) of the final error, distinct
    frames) or PICK=ids (in the given order)."""
    if pick == "ids":
        by = {r["id"]: r for r in rows}
        miss = [i for i in ids or [] if i not in by]
        if miss:
            raise SystemExit(f"ids not in the eval json: {miss}")
        return [(by[i], percentile_of(rows, i)) for i in ids]
    order = sorted(rows, key=final_err)
    out, seen = [], set()
    for q in quantiles:
        k = int(round(float(q) / 100.0 * (len(order) - 1)))
        while order[k]["id"] in seen and k + 1 < len(order):
            k += 1
        seen.add(order[k]["id"])
        out.append((order[k], 100.0 * k / max(1, len(order) - 1)))
    return out


def token_status(h, w, mode_tok, mode_inlier):
    """Per token (h, w): 1 = at least one of its modes is a RANSAC inlier, 0 = otherwise."""
    st = np.zeros((h, w), np.uint8)
    if len(mode_tok):
        t = np.asarray(mode_tok, int)[np.asarray(mode_inlier, bool)]
        st[t[:, 1], t[:, 0]] = 1
    return st


def safe_name(rid):
    city, pano = rid.split("/", 1)
    return f"{city}_{pano.split(',')[0]}"


# ---------------------------------------------------------------------------------------------------------- compute

def _apply_h(H, xy):
    p = np.c_[np.asarray(xy, np.float64).reshape(-1, 2), np.ones(int(np.prod(np.shape(xy)[:-1])))] @ np.asarray(H).T
    return (p[:, :2] / p[:, 2:3]).reshape(np.shape(xy))


def token_table(cons, gm, cert, query, batch, frac, n, m, prefix):
    """The placed tokens and the modes the consensus of Match m used, with their inlier flags (grid_errors <=
    reproj, the rule of the Match's own n_inliers). Arrays keyed with `prefix`."""
    import torch
    from bevloc.match.satroma import _pick_modes, grid_errors, query_tokens
    valid, xy = query_tokens(cons, query, batch, frac, n)
    gm2 = gm.clone()
    gm2[:, ~valid.bool()] = 0.0                                   # consensus_from_gm's masking
    md, idx, _ = _pick_modes(cons, gm2, cert, valid)
    err = grid_errors(cons, m)
    if err is None or len(err) != len(idx):
        raise RuntimeError(f"mode table does not match the Match ({None if err is None else len(err)} vs {len(idx)})")
    out_dim = int(round(gm.shape[0] ** 0.5))
    sb = float(cons.m.im_b_size) / out_dim                         # reference px per cell
    tgt = md["means" if cons.use_means else "peaks"][idx].astype(np.float64)
    xy_np = torch.as_tensor(xy).detach().float().cpu().numpy()
    return {f"{prefix}valid": valid.cpu().numpy().astype(bool), f"{prefix}xy": xy_np,
            f"{prefix}xy_ref": _apply_h(m.H, xy_np),
            f"{prefix}mode_tok": md["tok"][idx].astype(np.int16),
            f"{prefix}mode_ref": tgt * sb + sb / 2.0 - 0.5,
            f"{prefix}mode_height": md["height"][idx].astype(np.float32),
            f"{prefix}mode_err_cells": err.astype(np.float32),
            f"{prefix}mode_inlier": err <= float(cons.reproj),
            f"{prefix}cell_px": np.float64(sb), f"{prefix}reproj_cells": np.float64(cons.reproj)}


def compute(a, picks, bdir):
    import torch
    from bevloc import config as C
    from bevloc.data.vigor import VigorPairs, en_to_canvas, pose_en
    from bevloc.match.satroma import SatRoMa, consensus_for_query
    from bevloc.model.coarse import FeatureQueryMatcher
    from bevloc.model.query import apply_query_cfg, build_query, load_query_state
    from eval_vigor import _centre_of, apply_decoder_dtype, decode_refs, decoder_fine

    cfg = C.load(a.config)
    cfg.matcher.solver = a.solver
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = state.get("mode", "lift")
    if mode != "erp_depth":
        raise SystemExit(f"{a.ckpt} is a {mode} checkpoint; this figure is for the depth-placed (erp_depth) query")
    cfg.lift.query_mode = mode
    apply_query_cfg(cfg, state)
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    apply_decoder_dtype(matcher, a)
    query = build_query(cfg, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    cons = SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=False, min_valid_frac=0.05)
    ds = VigorPairs(a.root, cfg, cities=a.cities, split=a.split, train=False,
                    row_sign=a.row_sign, height_m=a.height)
    by_id = {f"{lab['city']}/{lab['pano']}": i for i, lab in enumerate(ds.labels)}
    ds.labels = [ds.labels[by_id[r["id"]]] for r, _ in picks]      # only the picked frames
    fa = SimpleNamespace(fine_config=a.fine_config, fine_ckpt=a.fine_ckpt, fine_gate=a.fine_gate,
                         decoder_dtype=a.decoder_dtype, consensus_json=None, ref_source=None, ref_sources=None,
                         train_split=False, calib=False, root=a.root, cities=a.cities, split=a.split)
    fine, _ = decoder_fine(fa, cfg, ds, dev)
    n, cell = int(cfg.grid.n), float(cfg.grid.cell_m)
    nf, cell_f = int(fine.cfg.grid.n), float(fine.cfg.grid.cell_m)
    o, of = (n - 1) / 2.0, (nf - 1) / 2.0
    by_id = {f"{lab['city']}/{lab['pano']}": i for i, lab in enumerate(ds.labels)}
    for r, pct in picks:
        i = by_id[r["id"]]
        s = ds[i]
        batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = query(batch, matcher)
        gms, certs = decode_refs(matcher, f_q, [batch["ref"]])
        m = consensus_for_query(cons, gms[0], query, batch, frac, n, min_frac=0.05, certainty=certs[0], stats=False)
        if m.H is None:
            print(f"  {r['id']}: no coarse pose, skipped", flush=True)
            continue
        S = int(s["ref"].shape[-1])
        H_gt = s["H"].numpy().astype(np.float64)
        en_gt = pose_en(H_gt, _centre_of(s), n, cell, S)
        en_c = pose_en(m.H, _centre_of(s), n, cell, S)
        gms_f, certs_f, fbatch, ffrac, sf = fine.decode_all(i, torch.from_numpy(np.asarray(en_c, np.float64)))
        mf = consensus_for_query(fine.cons, gms_f[0], fine.query, fbatch, ffrac, nf, min_frac=0.05,
                                 certainty=certs_f[0], stats=False)
        Sf = int(sf["ref"].shape[-1])
        centre_f = _centre_of(sf)
        en_f = pose_en(mf.H, centre_f, nf, cell_f, Sf) if mf.H is not None else None
        shift = None if en_f is None else float(np.linalg.norm(en_f - en_c))
        fallback = en_f is None or shift > float(a.fine_gate)
        en_final = en_c if fallback else en_f
        b = dict(id=r["id"], city=s["city"], percentile=pct, n_draw=a.n_draw,
                 json_coarse_m=r.get("pose_peak_m"), json_final_m=r.get("pose_fine_gated_m"),
                 coarse_m=float(np.linalg.norm(en_c - en_gt)), final_m=float(np.linalg.norm(en_final - en_gt)),
                 fallback=bool(fallback), cell_m=cell, cell_f_m=cell_f, n=n, nf=nf, S=S, Sf=Sf,
                 max_depth_m=float(query.max_depth), n_modes=int(m.n_modes), n_inl_modes=int(m.n_inliers),
                 n_modes_f=int(mf.n_modes), n_inl_modes_f=int(mf.n_inliers))
        pano = cv2.imread(str(Path(a.root) / s["city"] / "panorama" / r["id"].split("/", 1)[1]), cv2.IMREAD_COLOR)
        ok, pano_jpg = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
        arr = dict(pano_jpg=pano_jpg.reshape(-1), depth=s["depth"][0].numpy().astype(np.float16),
                   ref=(s["ref"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                   ref_f=(sf["ref"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                   en_gt=en_gt, en_c=en_c, en_final=en_final, en_f=np.full(2, np.nan) if en_f is None else en_f,
                   centre_f=centre_f, H=m.H, H_gt=H_gt, H_f=np.full((3, 3), np.nan) if mf.H is None else mf.H,
                   H_gt_f=sf["H"].numpy().astype(np.float64),
                   gt_px=_apply_h(H_gt, [o, o]), coarse_px=_apply_h(m.H, [o, o]),
                   gt_px_f=_apply_h(sf["H"].numpy(), [of, of]),
                   fine_px_f=np.full(2, np.nan) if mf.H is None else _apply_h(mf.H, [of, of]),
                   coarse_px_f=en_to_canvas(en_c, centre_f, cell_f, Sf))
        arr.update(token_table(cons, gms[0], certs[0], query, batch, frac, n, m, ""))
        if mf.H is not None:
            arr.update(token_table(fine.cons, gms_f[0], certs_f[0], fine.query, fbatch, ffrac, nf, mf, "f_"))
        np.savez_compressed(bdir / f"{safe_name(r['id'])}.npz", meta=json.dumps(b), **arr)
        print(f"  {r['id']}  p{pct:.0f}  coarse {b['coarse_m']:.2f} m (json {r.get('pose_peak_m'):.2f})  final "
              f"{b['final_m']:.2f} m (json {r.get('pose_fine_gated_m'):.2f})  fallback {fallback}", flush=True)


# ----------------------------------------------------------------------------------------------------------- render

def load_bundle(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    d["pano"] = cv2.cvtColor(cv2.imdecode(d.pop("pano_jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return d


def brighten(rgb):
    """Display only: stretch to the 1st..99.5th percentile; extra gamma when the image is dark."""
    x = rgb.astype(np.float32) / 255.0
    lo, hi = np.percentile(x, 1), np.percentile(x, 99.5)
    x = np.clip((x - lo) / max(1e-3, hi - lo), 0, 1)
    if x.mean() < 0.4:
        x = x ** 0.75
    return (x * 255).astype(np.uint8)


def az_colour(cols, w):
    """Colour per token column (azimuth): a cyclic HSV hue, centre column = north."""
    import matplotlib.colors as mc
    hue = (np.asarray(cols, float) + 0.5) / w
    return mc.hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.95), np.full_like(hue, 1.0)], -1))


def px_to_en(uv, centre_en, cell_m, size):
    from bevloc.data.vigor import canvas_to_en
    return canvas_to_en(uv, centre_en, cell_m, size)


def _scale_bar(ax, length_m, label=None):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    xs, ys = x0 + 0.05 * w, y0 + 0.05 * h
    for lw, c in ((9, "black"), (5, "white")):
        ax.plot([xs, xs + length_m], [ys, ys], color=c, lw=lw, solid_capstyle="butt", zorder=20)
    ax.text(xs + length_m / 2, ys + 0.025 * h, label or f"{length_m:g} m", ha="center", va="bottom", fontsize=20,
            color="white", fontweight="bold", zorder=21,
            path_effects=_halo())


def _north(ax, fx=0.06):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    ax.annotate("N", xy=(x0 + fx * w, y1 - 0.05 * h), xytext=(x0 + fx * w, y1 - 0.16 * h), ha="center",
                va="center", fontsize=22, color="white", fontweight="bold", zorder=21, path_effects=_halo(),
                arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.8", color="white", lw=3))


def _halo(w=3):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=w, foreground="black")]


def draw_matches(ax, b, prefix, centre_en, cell_m, size, cell_label, show_cells=True, line_alpha=0.55,
                 dot=34, highlight=None):
    """Tokens through the predicted pose (filled = inlier, hollow grey = outlier), lines inlier token -> matched cell
    centre, matched cells as squares. Everything in tile metres (east, north)."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    valid, xy_ref = b[prefix + "valid"], b[prefix + "xy_ref"]
    h, w = valid.shape
    tok, mref, inl = b[prefix + "mode_tok"].astype(int), b[prefix + "mode_ref"], b[prefix + "mode_inlier"]
    st = token_status(h, w, tok, inl)
    cols = np.tile(np.arange(w), (h, 1))
    en = px_to_en(xy_ref, centre_en, cell_m, size)
    rgb = az_colour(cols, w)
    out = valid & (st == 0)
    ax.scatter(en[out][:, 0], en[out][:, 1], s=dot, facecolors="none", edgecolors=(0.85, 0.85, 0.85, 0.85),
               linewidths=1.2, zorder=5)
    if inl.any():
        t = tok[inl]
        p0 = en[t[:, 1], t[:, 0]]
        p1 = px_to_en(mref[inl], centre_en, cell_m, size)
        c = az_colour(t[:, 0], w)
        ax.add_collection(LineCollection(np.stack([p0, p1], 1), colors=np.c_[c, np.full(len(c), line_alpha)],
                                         linewidths=1.3, zorder=6))
        if show_cells:
            half = float(b[prefix + "cell_px"]) * cell_m / 2.0
            for u in np.unique(np.round(p1, 3), axis=0):
                ax.add_patch(Rectangle((u[0] - half, u[1] - half), 2 * half, 2 * half, fill=False,
                                       edgecolor=(1, 1, 1, 0.5), lw=0.9, zorder=7))
    ins = valid & (st == 1)
    ax.scatter(en[ins][:, 0], en[ins][:, 1], s=dot, c=rgb[ins], edgecolors="black", linewidths=0.6, zorder=8)
    return en, st


def _pos(ax, en, kind, size=1.0):
    style = {"gt": dict(marker="P", color="#22dd22", s=700, label="ground truth"),
             "coarse": dict(marker="o", color="#ff9900", s=420, label="coarse prediction"),
             "final": dict(marker="X", color="#ff2020", s=620, label="final prediction")}[kind]
    if en is None or not np.isfinite(en).all():
        return
    ax.scatter([en[0]], [en[1]], marker=style["marker"], s=style["s"] * size, c=style["color"], edgecolors="black",
               linewidths=2.0, zorder=30)


def _map_axes(ax, img, centre_en, cell_m, size, title):
    half = size * cell_m / 2.0
    c = np.asarray(centre_en, float)
    ext = [c[0] - half - cell_m / 2, c[0] + half - cell_m / 2, c[1] - half + cell_m / 2, c[1] + half + cell_m / 2]
    ax.imshow(brighten(img), extent=ext, interpolation="lanczos", zorder=0)
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_title(title, fontsize=22, loc="left", pad=10)
    ax.set_xlabel("east of tile centre (m)", fontsize=16)
    ax.set_ylabel("north of tile centre (m)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_aspect("equal")


def _crop_to_tile(ax, ref, cell_m, size, margin_m=3.0):
    """Limit a whole-canvas axis to the satellite tile (the canvas is black outside it), square, plus a margin."""
    nz = np.argwhere(ref.max(-1) > 8)
    if not len(nz):
        return
    (v0, u0), (v1, u1) = nz.min(0), nz.max(0)
    e = px_to_en(np.array([[u0, v1], [u1, v0]], float), (0.0, 0.0), cell_m, size)
    c = e.mean(0)
    half = max(e[1, 0] - e[0, 0], e[1, 1] - e[0, 1]) / 2.0 + margin_m
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)


def render_frame(b, path, out_w=2400):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    mt = b["meta"]
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(24, 34.2), dpi=out_w / 24, facecolor="white")
    gs = fig.add_gridspec(5, 2, height_ratios=[12, 0.55, 6.2, 0.35, 12], hspace=0.1, wspace=0.14,
                          left=0.045, right=0.985, top=0.935, bottom=0.025)
    city, pano = mt["id"].split("/", 1)
    valid = b["valid"]
    h, w = valid.shape
    st = token_status(h, w, b["mode_tok"], b["mode_inlier"])
    n_inl_tok, n_placed = int(st.sum()), int(valid.sum())
    if "f_valid" in b:
        stf = token_status(h, w, b["f_mode_tok"], b["f_mode_inlier"])
        fine_txt = f"{int(stf.sum())} / {int(b['f_valid'].sum())}"
    else:
        stf, fine_txt = None, "no fine pose"
    fig.suptitle(f"{city}  ·  {pano.split(',')[0]}  ·  coarse error {mt['coarse_m']:.2f} m  →  final error "
                 f"{mt['final_m']:.2f} m\n{mt['percentile']:.0f}th percentile of the final error over "
                 f"{mt['n_draw']} Chicago test panoramas  ·  inlier tokens {n_inl_tok} / {n_placed} placed (coarse), "
                 f"{fine_txt} (fine)", fontsize=22, fontweight="bold", y=0.997)

    # --- panorama -------------------------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    P = brighten(b["pano"])
    Hp, Wp = P.shape[:2]
    ax.imshow(P, extent=[0, w, h, 0], interpolation="lanczos")
    shade = np.zeros((h, w, 4))
    shade[~valid] = (0.05, 0.05, 0.05, 0.55)
    ax.imshow(shade, extent=[0, w, h, 0], interpolation="nearest")
    for x in range(w + 1):
        ax.axvline(x, color="white", lw=0.35, alpha=0.35)
    for y in range(h + 1):
        ax.axhline(y, color="white", lw=0.35, alpha=0.35)
    cols = np.tile(np.arange(w), (h, 1))
    rows = np.tile(np.arange(h)[:, None], (1, w))
    rgb = az_colour(cols, w)
    out = valid & (st == 0)
    ax.scatter(cols[out] + 0.5, rows[out] + 0.5, s=60, facecolors="none", edgecolors=(0.9, 0.9, 0.9, 0.9),
               linewidths=1.3)
    ins = valid & (st == 1)
    ax.scatter(cols[ins] + 0.5, rows[ins] + 0.5, s=75, c=rgb[ins], edgecolors="black", linewidths=0.8)
    ax.axvline(w / 2, color="yellow", lw=2.5, ls="--")
    ax.text(w / 2 + 0.3, 1.2, "forward = north\n(centre column)", color="yellow", fontsize=20, fontweight="bold",
            va="top", path_effects=_halo())
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"1  Panorama → DINOv3 sat493m at 896×448 → {w}×{h} tokens (grid).  Filled dot = token that is a "
                 "RANSAC inlier of the coarse pass,\n    hollow = placed by depth but outlier, "
                 f"dark = not placed (sky / depth ≥ {mt['max_depth_m']:.0f} m)", fontsize=19, loc="left", pad=8)
    # azimuth band
    axb = fig.add_subplot(gs[1, :])
    axb.imshow(az_colour(np.arange(w), w)[None], extent=[0, w, 0, 1], aspect="auto")
    axb.set_xlim(0, w)
    axb.set_yticks([])
    axb.set_xticks([0, w / 4, w / 2, 3 * w / 4, w])
    axb.set_xticklabels(["S (−180°)", "W (−90°)", "N (0°, forward)", "E (+90°)", "S (+180°)"], fontsize=17)
    axb.set_title("colour = azimuth of the token (the same colours on the maps below)", fontsize=16, loc="left",
                  pad=4)

    # --- depth ----------------------------------------------------------------------------------------------------
    axd = fig.add_subplot(gs[2, 0])
    d = b["depth"].astype(np.float32)
    dm = np.ma.masked_where(~(np.isfinite(d) & (d > 0) & (d < mt["max_depth_m"])), d)
    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_bad((0.55, 0.55, 0.55))
    im = axd.imshow(dm, cmap=cmap, vmin=0, vmax=mt["max_depth_m"], extent=[0, w, h, 0], interpolation="nearest")
    axd.axvline(w / 2, color="white", lw=2, ls="--")
    axd.set_xticks([])
    axd.set_yticks([])
    axd.set_title("2  UniK3D metric depth along each ray (grey = not placed;\n    the row stripes are in the stored "
                  "depth files)", fontsize=19, loc="left", pad=8)
    cb = fig.colorbar(im, ax=axd, fraction=0.035, pad=0.015)
    cb.set_label("depth (m)", fontsize=17)
    cb.ax.tick_params(labelsize=14)

    # --- legend / numbers -----------------------------------------------------------------------------------------
    axl = fig.add_subplot(gs[2, 1])
    axl.axis("off")
    rc = float(b["reproj_cells"])
    cm2 = float(b["cell_px"]) * mt["cell_m"]
    lines = [
        "How to read the maps",
        f"• each placed token is moved to the satellite map by the PREDICTED pose;",
        f"• the decoder gave each token a probability over the reference cells",
        f"  ({cm2:.0f} m cells coarse, {float(b['f_cell_px']) * mt['cell_f_m'] if 'f_cell_px' in b else 1:.0f} m fine); "
        "the line goes to the cell it matched (top mode);",
        f"• RANSAC inlier (filled) = matched cell within {rc:.0f} cells of where the pose",
        f"  puts the token ({rc * cm2:.0f} m coarse); outliers are hollow grey and ignored.",
        "",
        f"coarse modes: {mt['n_inl_modes']} inliers of {mt['n_modes']};  fine: {mt['n_inl_modes_f']} of {mt['n_modes_f']}",
        f"eval json: coarse {mt['json_coarse_m']:.2f} m, final {mt['json_final_m']:.2f} m"
        + ("  (fine pose rejected by the 6 m gate)" if mt["fallback"] else ""),
    ]
    axl.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=18, transform=axl.transAxes, linespacing=1.45)
    handles = [Line2D([], [], marker="o", ls="", ms=13, mfc="#3a7bd5", mec="black", label="token, RANSAC inlier"),
               Line2D([], [], marker="o", ls="", ms=13, mfc="none", mec="0.5", mew=2, label="token, outlier"),
               Line2D([], [], color="#3a7bd5", lw=2.5, label="token → its matched cell"),
               Rectangle((0, 0), 1, 1, fill=False, ec="0.3", lw=2, label="matched reference cell"),
               Line2D([], [], marker="P", ls="", ms=22, mfc="#22dd22", mec="black", label="ground truth"),
               Line2D([], [], marker="o", ls="", ms=19, mfc="#ff9900", mec="black", label="coarse prediction"),
               Line2D([], [], marker="X", ls="", ms=22, mfc="#ff2020", mec="black", label="final prediction")]
    axl.legend(handles=handles, loc="lower left", fontsize=17, ncol=2, frameon=True, borderpad=0.8,
               handletextpad=0.6, columnspacing=1.6)

    # --- coarse canvas --------------------------------------------------------------------------------------------
    axc = fig.add_subplot(gs[4, 0])
    _map_axes(axc, b["ref"], (0.0, 0.0), mt["cell_m"], mt["S"],
              f"3  Coarse: {mt['S'] * mt['cell_m']:.0f} m canvas (shown: the tile), {cm2:.0f} m cells — error "
              f"{mt['coarse_m']:.2f} m")
    draw_matches(axc, b, "", (0.0, 0.0), mt["cell_m"], mt["S"], "2 m", dot=26)
    _crop_to_tile(axc, b["ref"], mt["cell_m"], mt["S"])
    half = mt["Sf"] * mt["cell_f_m"] / 2.0
    cf = b["centre_f"]
    axc.add_patch(Rectangle((cf[0] - half, cf[1] - half), 2 * half, 2 * half, fill=False, ec="#ff9900", lw=2.5,
                            ls="--", zorder=25))
    axc.text(cf[0] - half + 1, cf[1] + half - 1, f"second-pass window ({2 * half:.0f} m)", color="#ff9900",
             fontsize=16, fontweight="bold", va="top", zorder=26, path_effects=_halo())
    _pos(axc, b["en_gt"], "gt")
    _pos(axc, b["en_c"], "coarse")
    _scale_bar(axc, 10)
    _north(axc)

    # --- fine window ----------------------------------------------------------------------------------------------
    axf = fig.add_subplot(gs[4, 1])
    _map_axes(axf, b["ref_f"], cf, mt["cell_f_m"], mt["Sf"],
              f"4  Fine: {mt['Sf'] * mt['cell_f_m']:.0f} m window, 1 m cells — final error {mt['final_m']:.2f} m")
    if "f_valid" in b:
        draw_matches(axf, b, "f_", cf, mt["cell_f_m"], mt["Sf"], "1 m", dot=30)
    _pos(axf, b["en_gt"], "gt")
    _pos(axf, b["en_c"], "coarse", 0.8)
    _pos(axf, b["en_final"], "final")
    _scale_bar(axf, 5)
    _north(axf)
    fig.savefig(path, dpi=out_w / 24, pil_kwargs=dict(quality=80, optimize=True))
    plt.close(fig)


def contact_sheet(paths, captions, out, width=2400, cols=5):
    tiles = []
    tw = width // cols
    for p, cap in zip(paths, captions):
        im = cv2.imread(str(p))
        im = cv2.resize(im, (tw - 12, int(im.shape[0] * (tw - 12) / im.shape[1])), interpolation=cv2.INTER_AREA)
        bar = np.full((56, im.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, cap, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(np.pad(np.vstack([bar, im]), ((6, 6), (6, 6), (0, 0)), constant_values=255))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def render_diagram(b, path, out_w=2400):
    """One frame, the parts named: a token on the panorama, its depth, its placement around the camera, the cell it
    matched, inlier vs outlier, the coarse and final poses against the ground truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    mt = b["meta"]
    valid = b["valid"]
    h, w = valid.shape
    tok, inl = b["mode_tok"].astype(int), b["mode_inlier"]
    st = token_status(h, w, tok, inl)
    d_tok = sample_depth(b["depth"].astype(np.float32), h, w)
    cell = mt["cell_m"]
    n = mt["n"]
    xy_m = np.stack([((n / 2 - 0.5) - b["xy"][..., 0]) * cell, ((n / 2 - 0.5) - b["xy"][..., 1]) * cell], -1)
    # ego metres: y left (west), x forward (north) -> plot east = -y_left, north = x
    ego_e, ego_n = -xy_m[..., 0], xy_m[..., 1]
    en_ref = px_to_en(b["xy_ref"], (0.0, 0.0), cell, mt["S"])
    gt = b["en_gt"]
    # the highlighted inlier token
    # rank: an agreeing mode (< 1.5 cells), then the farthest such token below the horizon (a long, visible ray)
    cand = [(-(float(e) < 1.5), -float(d_tok[t[1], t[0]]), int(t[0]), int(t[1]), k)
            for k, (t, e, ok) in enumerate(zip(tok, b["mode_err_cells"], inl))
            if ok and 4 < d_tok[t[1], t[0]] < 15 and t[1] > 0.55 * h and 0.1 * w < t[0] < 0.9 * w]
    if not cand:
        cand = [(0, float(e), int(t[0]), int(t[1]), k) for k, (t, e, ok) in enumerate(zip(tok, b["mode_err_cells"], inl))
                if ok]
    _, _, hc, hr, hk = min(cand)
    # an outlier token whose matched cell lies inside the zoom, far from where the pose puts it
    zoom = 18.0
    centre = gt
    best_out = None
    for k, (t, e, ok) in enumerate(zip(tok, b["mode_err_cells"], inl)):
        if ok or st[t[1], t[0]]:
            continue
        p0 = en_ref[t[1], t[0]]
        p1 = px_to_en(b["mode_ref"][k], (0.0, 0.0), cell, mt["S"])
        if np.all(np.abs(p0 - centre) < zoom * 0.8) and np.all(np.abs(p1 - centre) < zoom * 0.8) and e > 5:
            if best_out is None or abs(t[0] - hc) > abs(best_out[1] - hc):
                best_out = (k, int(t[0]), int(t[1]))
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(24, 23), dpi=out_w / 24, facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[11.5, 7.6], hspace=0.13, wspace=0.16, left=0.035, right=0.99,
                          top=0.935, bottom=0.04)
    fig.suptitle("How PanoRoMa localises a panorama (one example, Chicago same-area test; "
                 f"coarse {mt['coarse_m']:.2f} m → final {mt['final_m']:.2f} m)", fontsize=27, fontweight="bold")
    col = az_colour([hc], w)[0]
    # (a) panorama
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(brighten(b["pano"]), extent=[0, w, h, 0], interpolation="lanczos")
    shade = np.zeros((h, w, 4))
    shade[~valid] = (0.05, 0.05, 0.05, 0.5)
    ax.imshow(shade, extent=[0, w, h, 0], interpolation="nearest")
    for x in range(w + 1):
        ax.axvline(x, color="white", lw=0.3, alpha=0.3)
    for y in range(h + 1):
        ax.axhline(y, color="white", lw=0.3, alpha=0.3)
    ax.add_patch(Rectangle((hc, hr), 1, 1, fill=False, ec=col, lw=4, zorder=10))
    ax.annotate(f"one token = a 16×16 px patch → one DINOv3 feature\n"
                f"UniK3D depth along its ray: {d_tok[hr, hc]:.1f} m",
                xy=(hc + 0.5, hr + 1), xytext=(0.50, 0.10), textcoords="axes fraction", fontsize=20, color="white", fontweight="bold",
                path_effects=_halo(4), arrowprops=dict(arrowstyle="-|>", color="white", lw=3), zorder=11)
    if best_out is not None:
        _, oc, orow = best_out
        ax.add_patch(Rectangle((oc, orow), 1, 1, fill=False, ec="white", lw=3, ls="--", zorder=10))
        ax.annotate("an outlier token", xy=(oc + 0.5, orow), xytext=(max(oc + 1, 15), 1.5), fontsize=19, color="white",
                    path_effects=_halo(4), arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5), zorder=11)
    ax.text(0.3, 1.0, "dark = sky / too far: not placed", fontsize=18, color="white", path_effects=_halo(4), va="top")
    ax.axvline(w / 2, color="yellow", lw=2, ls="--")
    ax.text(w / 2 + 0.3, h - 0.6, "forward = north", fontsize=18, color="yellow", path_effects=_halo(4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title("a  The panorama is cut into 56 × 28 tokens; each token gets a feature and a depth",
                 fontsize=22, loc="left")
    # (b) ego placement
    axe = fig.add_subplot(gs[1, 0])
    axe.set_facecolor("#1e1e1e")
    for r in (10, 20, 30):
        axe.add_patch(Circle((0, 0), r, fill=False, ec="0.6", lw=1, ls=":"))
        axe.text(r * 0.71, r * 0.71, f"{r} m", color="0.8", fontsize=14)
    cols = np.tile(np.arange(w), (h, 1))
    axe.scatter(ego_e[valid], ego_n[valid], s=14, c=az_colour(cols[valid], w), zorder=3)
    axe.plot([0, ego_e[hr, hc]], [0, ego_n[hr, hc]], color=col, lw=3, zorder=4)
    axe.scatter([ego_e[hr, hc]], [ego_n[hr, hc]], s=260, c=[col], edgecolors="white", linewidths=2.5, zorder=5)
    axe.annotate("token placed by depth\n(along its ray, height dropped)", xy=(ego_e[hr, hc], ego_n[hr, hc]),
                 xytext=(-30, -31), fontsize=18, color="white", fontweight="bold", path_effects=_halo(3),
                 arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5), zorder=6)
    axe.scatter([0], [0], marker="^", s=400, c="white", edgecolors="black", zorder=6)
    axe.text(1.5, -3.5, "camera", color="white", fontsize=17)
    axe.set_xlim(-36, 36)
    axe.set_ylim(-36, 36)
    axe.set_aspect("equal")
    axe.set_xlabel("east of camera (m)", fontsize=15)
    axe.set_ylabel("north of camera (m)", fontsize=15)
    axe.tick_params(labelsize=13)
    axe.set_title("b  Tokens placed on the ground around the\ncamera (the query; no satellite yet)\n",
                  fontsize=18, loc="left")
    # (c) coarse zoom
    axc = fig.add_subplot(gs[1, 1])
    _map_axes(axc, b["ref"], (0.0, 0.0), cell, mt["S"], "")
    draw_matches(axc, b, "", (0.0, 0.0), cell, mt["S"], "2 m", dot=40, line_alpha=0.35)
    p_h = en_ref[hr, hc]
    c_h = px_to_en(b["mode_ref"][hk], (0.0, 0.0), cell, mt["S"])
    half = float(b["cell_px"]) * cell / 2.0
    axc.plot([p_h[0], c_h[0]], [p_h[1], c_h[1]], color=col, lw=4, zorder=12)
    axc.add_patch(Rectangle((c_h[0] - half, c_h[1] - half), 2 * half, 2 * half, fill=False, ec="yellow", lw=3.5,
                            zorder=12))
    axc.scatter([p_h[0]], [p_h[1]], s=240, c=[col], edgecolors="white", linewidths=2.5, zorder=13)
    xl = (centre[0] - zoom, centre[0] + zoom)
    yl = (centre[1] - zoom, centre[1] + zoom)
    ann = dict(fontsize=17, color="white", fontweight="bold", path_effects=_halo(4), zorder=40)
    axc.annotate("RANSAC inlier: the cell it matched\nagrees with the pose", xy=p_h,
                 xytext=(0.03, 0.97), textcoords="axes fraction", va="top",
                 arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5), **ann)
    axc.annotate(f"matched cell\n({2 * half:.0f} m × {2 * half:.0f} m)", xy=(c_h[0] + half, c_h[1]),
                 xytext=(0.62, 0.80), textcoords="axes fraction", va="top",
                 arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2.5),
                 **{**ann, "color": "yellow"})
    if best_out is not None:
        k, oc, orow = best_out
        p0 = en_ref[orow, oc]
        p1 = px_to_en(b["mode_ref"][k], (0.0, 0.0), cell, mt["S"])
        axc.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=2.5, ls="--", zorder=12)
        axc.add_patch(Rectangle((p1[0] - half, p1[1] - half), 2 * half, 2 * half, fill=False, ec="white", lw=2.5,
                                ls="--", zorder=12))
        axc.annotate("outlier: matched a cell far from\nwhere the pose puts it → ignored", xy=p0,
                     xytext=(0.03, 0.24), textcoords="axes fraction", va="top",
                     arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5), **ann)
    _pos(axc, gt, "gt", 0.8)
    _pos(axc, b["en_c"], "coarse", 0.8)
    axc.annotate("ground truth", xy=gt, xytext=(0.55, 0.30), textcoords="axes fraction", arrowprops=dict(arrowstyle="-|>",
                 color="#22dd22", lw=2.5), **{**ann, "color": "#22dd22"})
    axc.annotate(f"coarse prediction\n({mt['coarse_m']:.2f} m off)", xy=b["en_c"],
                 xytext=(0.62, 0.58), textcoords="axes fraction", va="top",
                 arrowprops=dict(arrowstyle="-|>", color="#ff9900", lw=2.5), **{**ann, "color": "#ff9900"})
    axc.set_xlim(*xl)
    axc.set_ylim(*yl)
    _scale_bar(axc, 10)
    _north(axc, 0.93)
    axc.set_title("c  Coarse pass (zoom of the 112 m map): each\ntoken votes for a 2 m cell; RANSAC keeps the\n"
                  "pose that most votes agree with", fontsize=18, loc="left")
    # (d) fine
    axf = fig.add_subplot(gs[1, 2])
    cf = b["centre_f"]
    _map_axes(axf, b["ref_f"], cf, mt["cell_f_m"], mt["Sf"], "")
    if "f_valid" in b:
        draw_matches(axf, b, "f_", cf, mt["cell_f_m"], mt["Sf"], "1 m", dot=40, line_alpha=0.35)
    _pos(axf, gt, "gt", 0.8)
    _pos(axf, b["en_c"], "coarse", 0.7)
    _pos(axf, b["en_final"], "final", 0.8)
    zf = 10.0
    axf.set_xlim(gt[0] - zf, gt[0] + zf)
    axf.set_ylim(gt[1] - zf, gt[1] + zf)
    axf.annotate(f"final prediction\n({mt['final_m']:.2f} m off)", xy=b["en_final"],
                 xytext=(0.03, 0.97), textcoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color="#ff2020", lw=2.5),
                 **{**ann, "color": "#ff2020", "fontsize": 17}, va="top")
    axf.annotate("ground truth", xy=gt, xytext=(0.55, 0.12), textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#22dd22", lw=2.5), **{**ann, "color": "#22dd22"})
    _scale_bar(axf, 5)
    _north(axf, 0.93)
    axf.set_title("d  Fine pass (zoom): 56 m window at 6 cm/px\naround the coarse pose, 1 m cells, the same\n"
                  "voting → final pose", fontsize=18, loc="left")
    fig.savefig(path, dpi=out_w / 24, pil_kwargs=dict(quality=88, optimize=True))
    plt.close(fig)


def sample_depth(depth, h, w):
    """Token-centre depth (nearest pixel), the rule of bevloc.model.depth_query.sample_token_depth."""
    H, W = depth.shape
    r = np.clip(np.floor((np.arange(h) + 0.5) * H / h).astype(int), 0, H - 1)
    c = np.clip(np.floor((np.arange(w) + 0.5) * W / w).astype(int), 0, W - 1)
    return depth[r][:, c]


# ------------------------------------------------------------------------------------------------------------- main

def main():
    from bevloc import config as C
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--ckpt", default="checkpoints/vigor_samearea_4city_erp_depth_cell0125_e100_last.pt")
    ap.add_argument("--fine-config", default="configs/vigor_cell00625_fine.yaml")
    ap.add_argument("--fine-ckpt", default="checkpoints/vigor_samearea_4city_fine00625_erp_depth_e100_last.pt")
    ap.add_argument("--fine-gate", type=float, default=6.0)
    ap.add_argument("--solver", default="se2")
    ap.add_argument("--decoder-dtype", default="bfloat16")
    ap.add_argument("--eval-json", required=True, help="eval_vigor.py json of the same checkpoints (frame choice)")
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--pick", default="quantiles", choices=("quantiles", "ids"))
    ap.add_argument("--quantiles", type=float, nargs="+", default=list(QUANTILES))
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=0, help="keep the first N picks (0 = all)")
    ap.add_argument("--stage", default="all", choices=("all", "compute", "render"))
    ap.add_argument("--diagram-id", default=None, help="frame id for the annotated explanation figure")
    ap.add_argument("--out", default="experiments/13_panoroma_long/viz")
    a = ap.parse_args()
    ev = json.loads(Path(a.eval_json).read_text())
    rows, meta = ev["frames"], ev["meta"]
    a.split, a.cities = meta["split"], meta["cities"]
    a.row_sign, a.height, a.n_draw = meta.get("row_sign"), meta.get("height_m"), len(rows)
    picks = pick_rows(rows, a.pick, a.quantiles, a.ids)
    if a.n:
        picks = picks[:a.n]
    out = Path(a.out)
    bdir = out / "bundles"
    bdir.mkdir(parents=True, exist_ok=True)
    (out / "picks.json").write_text(json.dumps(dict(eval_json=a.eval_json, ckpt=a.ckpt, fine_ckpt=a.fine_ckpt,
                                                    pick=a.pick, quantiles=a.quantiles,
                                                    frames=[dict(id=r["id"], percentile=p,
                                                                 coarse_m=r.get("pose_peak_m"),
                                                                 final_m=r.get("pose_fine_gated_m"))
                                                            for r, p in picks]), indent=2))
    if a.stage in ("all", "compute"):
        compute(a, picks, bdir)
    if a.stage in ("all", "render"):
        paths, caps = [], []
        for k, (r, p) in enumerate(picks, 1):
            f = bdir / f"{safe_name(r['id'])}.npz"
            if not f.exists():
                print(f"  no bundle for {r['id']}", flush=True)
                continue
            b = load_bundle(f)
            path = out / f"{k:02d}_p{int(round(p)):02d}_{safe_name(r['id'])}.jpg"
            render_frame(b, path)
            paths.append(path)
            caps.append(f"{k}. p{p:.0f}: {b['meta']['coarse_m']:.2f} -> {b['meta']['final_m']:.2f} m")
            print(f"wrote {path}", flush=True)
        if paths:
            contact_sheet(paths, caps, out / "contact_sheet.jpg")
            print(f"wrote {out / 'contact_sheet.jpg'}", flush=True)
        if a.diagram_id:
            f = bdir / f"{safe_name(a.diagram_id)}.npz"
            render_diagram(load_bundle(f), out / "diagram_explained.jpg")
            print(f"wrote {out / 'diagram_explained.jpg'}", flush=True)


if __name__ == "__main__":
    main()
