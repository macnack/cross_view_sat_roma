"""Failure figures: frames of the full Chicago same-area test split where FG² and/or PanoRoMa D fail.

  make failure-viz IDS_JSON=experiments/16_full_test/failures/fail_cases.json OUT=experiments/16_full_test/failures

IDS_JSON lists panorama files per category ({"fg2_fails_pano_ok" | "pano_fails_fg2_ok" | "both_fail": [[pano, pano_m,
fg2_m], ...]}); --extra-both adds that many "both fail" frames with the largest common error min(PanoRoMa, FG²).
Errors come from the two full-split runs (--pano-json: eval_vigor.py of PanoRoMa D two-pass; --fg2-csv:
eval_fg2_vigor.py of the released same-area FG² checkpoint) and are re-computed here, on the same frames:

  compute-pano (GPU)  the compute stage of scripts/viz_panoroma_matches.py (coarse + fine checkpoint, se2, bf16 decoder,
                      6 m gate) -> <out>/bundles/<frame>.npz; the recomputed final error must equal the json's.
  compute-fg2  (GPU)  FG²'s model + RANSAC solver exactly as scripts/eval_fg2_vigor.py ran them (seed 0, batch 24,
                      `--solver both`), predicted position kept -> <out>/bundles/<frame>_fg2.json. FG²'s RANSAC draws
                      from the CUDA generator, and a frame's draws depend on its batch mates and on the generator offset
                      at its batch, so the frame's whole batch of the full run is re-run with the CUDA Philox offset set
                      to (batch index x the offset one batch consumes, measured on batch 0, every batch being the same
                      shape); every frame of each such batch must match the csv within --tol-fg2.
  render       (CPU)  one figure per frame (<out>/<category>/NN_<frame>.jpg) and a contact sheet per category.

Figure: left the panorama (brightened, a sample of PanoRoMa's placed tokens as dots, colour = azimuth); right the VIGOR
tile, north up, metres from the tile centre: ground truth (green +), PanoRoMa final (red x), FG² (blue triangle), lines
GT -> prediction with the error, the same token sample placed through PanoRoMa's final pass (filled = RANSAC inlier).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CATS = {"fg2_fails_pano_ok": ("fg2_fails", "FG² fails, PanoRoMa works"),
        "pano_fails_fg2_ok": ("pano_fails", "PanoRoMa fails, FG² works"),
        "both_fail": ("both_fail", "both fail")}
TOKENS_SHOWN = 160          # placed tokens drawn per figure (evenly subsampled)


# --------------------------------------------------------------------------------------------------------- selection

def load_runs(a):
    pano = {r["id"]: r for r in json.loads(Path(a.pano_json).read_text())["frames"]}
    with open(a.fg2_csv, newline="") as f:
        fg2 = list(csv.DictReader(f))
    fg2_pos = {r["pano"]: i for i, r in enumerate(fg2)}
    return pano, fg2, fg2_pos


def select(a, pano, fg2, fg2_pos):
    """[{id, pano, cat, pano_m, fg2_m, fg2_index}] in IDS_JSON order, then the --extra-both frames."""
    spec = json.loads(Path(a.ids_json).read_text())
    sel, seen = [], set()
    for key, (cat, _) in CATS.items():
        for p, *_ in spec.get(key, []):
            sel.append(dict(id=f"{a.city}/{p}", pano=p, cat=cat, extra=False))
            seen.add(p)
    common = sorted(((min(np.inf if pano[f"{a.city}/{r['pano']}"]["pose_fine_gated_m"] is None
                          else pano[f"{a.city}/{r['pano']}"]["pose_fine_gated_m"], float(r["ransac_m"])), r["pano"])
                     for r in fg2), reverse=True)
    extra = [p for _, p in common if p not in seen][:a.extra_both]
    sel += [dict(id=f"{a.city}/{p}", pano=p, cat="both_fail", extra=True) for p in extra]
    for s in sel:
        r = pano[s["id"]]
        s["pano_m"] = r["pose_fine_gated_m"]
        s["fg2_index"] = fg2_pos[s["pano"]]
        s["fg2_m"] = float(fg2[s["fg2_index"]]["ransac_m"])
    return sel


def safe_name(rid):
    city, p = rid.split("/", 1)
    return f"{city}_{p.split(',')[0]}"


# ----------------------------------------------------------------------------------------------------- compute: pano

def compute_pano(a, sel, pano, bdir):
    import viz_panoroma_matches as V
    meta = json.loads(Path(a.pano_json).read_text())["meta"]
    na = SimpleNamespace(config=a.config, solver=meta.get("solver", "se2"), ckpt=a.ckpt, fine_config=a.fine_config,
                         fine_ckpt=a.fine_ckpt, fine_gate=a.fine_gate, decoder_dtype=a.decoder_dtype, root=a.root,
                         cities=meta["cities"], split=meta["split"], row_sign=meta.get("row_sign"),
                         height=meta.get("height_m"), n_draw=len(pano))
    if meta["ckpt"] != a.ckpt or meta["fine"]["ckpt"] != a.fine_ckpt:
        raise SystemExit(f"--ckpt/--fine-ckpt differ from the json's ({meta['ckpt']}, {meta['fine']['ckpt']})")
    rows = list(pano.values())
    picks = [(pano[s["id"]], V.percentile_of(rows, s["id"])) for s in sel]
    V.compute(na, picks, bdir)
    bad = []
    for s in sel:
        f = bdir / f"{safe_name(s['id'])}.npz"
        if not f.exists():
            bad.append(f"{s['id']}: no bundle")
            continue
        mt = json.loads(str(np.load(f)["meta"]))
        d = abs(mt["final_m"] - s["pano_m"])
        print(f"  PanoRoMa {s['id'][:40]}  recomputed {mt['final_m']:.4f} m  json {s['pano_m']:.4f} m  |d| {d:.2e}",
              flush=True)
        if d > a.tol_pano:
            bad.append(f"{s['id']}: recomputed {mt['final_m']:.4f} vs json {s['pano_m']:.4f}")
    if bad:
        raise SystemExit("PanoRoMa recomputation does not match the json:\n  " + "\n  ".join(bad))


# ------------------------------------------------------------------------------------------------------ compute: FG²

def _cuda_offset(state):
    """CUDA generator state = seed (uint64) + Philox offset (int64), 16 bytes."""
    if state.numel() != 16:
        raise RuntimeError(f"unexpected CUDA RNG state size {state.numel()} (expected 16 = seed + offset)")
    return int(np.frombuffer(state.numpy().tobytes()[8:16], np.int64)[0])


def _with_offset(state, off):
    import torch
    b = bytearray(state.numpy().tobytes())
    b[8:16] = np.int64(off).tobytes()
    return torch.frombuffer(b, dtype=torch.uint8).clone()


def compute_fg2(a, sel, fg2, bdir):
    """FG² on the batches of the full run that hold the selected frames; see the module docstring."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import random

    import torch
    from torch.utils.data import DataLoader, Subset

    from bevloc import config as C
    from bevloc.baselines import fg2 as fg2_wrap
    from bevloc.data.vigor import CITY_RES, VigorPairs, find_label_root
    from eval_fg2_vigor import fg2_modules

    fmeta = json.loads(Path(a.fg2_csv).with_suffix(".json").read_text())["meta"]
    if fmeta.get("limit") or fmeta.get("seed") or fmeta["orientation"] != "known_ori":
        raise SystemExit(f"FG² run is not limit 0 / seed 0 / known_ori: {fmeta}")
    cfg = C.load(a.config)
    B = int(a.fg2_batch or getattr(cfg.vigor, "fg2_batch", 24))
    dev = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    VIGORDataset, create_metric_grid, Solver, procrustes, ini = fg2_modules()

    class CityDataset(VIGORDataset):
        def __init__(self, cities, **kw):
            self._cities = list(cities)
            super().__init__(**kw)

        def _get_city_list(self):
            return self._cities

    cities, split = fmeta["cities"], fmeta["split"]
    root = Path(a.root)
    ds = CityDataset(cities, root=str(root), label_root=str(find_label_root(root)), split=split, train=False,
                     random_orientation=False, first_run=False)
    ours = VigorPairs(a.root, cfg, cities=cities, split=split, train=False, limit=0, seed=0)
    names = [lab["pano"] for lab in ours.labels]
    if names != [r["pano"] for r in fg2]:
        raise SystemExit("VigorPairs order differs from the FG² csv order")
    lab_by = {lab["pano"]: lab for lab in ours.labels}
    by_name = {os.path.basename(p): i for i, p in enumerate(ds.grd_list)}
    idx = [by_name[p] for p in names]
    N = fg2_wrap.NATIVE
    model, _ = fg2_wrap.load_cvm(dev, area=split, orientation="known_ori")
    dino = fg2_wrap.load_dino(dev)
    Vi = ini["VIGOR"]
    grid_h, sat_px = float(Vi["grid_size_h"]), N["satellite_image_size"][0]
    n_samples = int(ini["Model"]["num_samples_matches"])
    sat_grid = create_metric_grid(grid_h, N["sat_bev_res"], B).to(dev)
    grd_grid = create_metric_grid(grid_h, N["grd_bev_res"], B).to(dev)
    ransac_args = (int(Vi["it_RANSAC_procrustes"]), int(Vi["it_matches"]), int(Vi["num_samples_matches_ransac"]),
                   int(Vi["num_corr_2d_2d"]), int(Vi["num_ref_steps"]), float(Vi["th_inlier"]), float(Vi["th_soft_inlier"]))

    def run_batch(k):
        """Batch k of the full run exactly as eval_fg2_vigor.py --solver both: (t_px (B, 2), tgt (B, 2), err_m)."""
        sub = idx[k * B:(k + 1) * B]
        grd, sat, tgt, _R, city = next(iter(DataLoader(Subset(ds, sub), batch_size=len(sub), shuffle=False,
                                                     num_workers=0)))
        b = grd.shape[0]
        grd, sat, tgt = grd.to(dev), sat.to(dev), tgt.to(dev)
        with torch.no_grad():
            score, _, _ = model(dino(grd), dino(sat))
            _, _n_sat, n_grd = score.shape
            flat = score.flatten(1)
            bidx = torch.arange(b, device=dev)[:, None].expand(b, n_samples)
            s = torch.multinomial(flat, n_samples)                      # the procrustes solver's draw (RNG order)
            procrustes(sat_grid[:b][bidx, torch.div(s, n_grd, rounding_mode="trunc")], grd_grid[:b][bidx, s % n_grd],
                       use_weights=True, use_mask=True, w=flat[bidx, s])
            _R, t, _, _ = Solver(*ransac_args, sat_grid[:b], grd_grid[:b]).estimate_pose(score, return_inliers=False)
        t_px = (t / grid_h * sat_px)[:, 0].double().cpu().numpy()
        tg = tgt[:, 0].double().cpu().numpy()
        px = np.linalg.norm(t_px - tg, axis=1)
        err = np.array([p * CITY_RES[c] * 640 / sat_px for p, c in zip(px, city)])
        return sub, t_px, tg, err, list(city)

    def check(k, err):
        ref = np.array([float(r["ransac_m"]) for r in fg2[k * B:k * B + len(err)]])
        d = np.abs(err - ref)
        return int((d <= a.tol_fg2).sum()), float(d.max())

    state0 = torch.cuda.get_rng_state()
    off0 = _cuda_offset(state0)
    _, _, _, err0, _ = run_batch(0)
    delta = _cuda_offset(torch.cuda.get_rng_state()) - off0
    ok0, d0 = check(0, err0)
    print(f"FG² batch 0: {ok0}/{len(err0)} frames match the csv (max |d| {d0:.2e} m); CUDA offset per batch {delta}",
          flush=True)
    if ok0 != len(err0):
        raise SystemExit("FG² batch 0 does not reproduce the csv; the RNG replay cannot work")
    need = sorted({s["fg2_index"] // B for s in sel})
    bad = []
    for k in need:
        torch.cuda.set_rng_state(_with_offset(state0, off0 + k * delta))
        sub, t_px, tg, err, city = run_batch(k)
        ok, dmax = check(k, err)
        print(f"FG² batch {k}: {ok}/{len(err)} frames match the csv (max |d| {dmax:.2e} m)", flush=True)
        for s in sel:
            if s["fg2_index"] // B != k:
                continue
            j = s["fg2_index"] - k * B
            res = CITY_RES[city[j]] * 640.0 / sat_px                 # metres per px of the 630 px tile
            # FG²'s target is (-row_offset, col_offset) in 630 px with (row, col) offset = the label's (dy, dx);
            # our tile frame (VigorPairs): east = col_sign * dx * res640, north = -row_sign * dy * res640, so
            # east = col_sign * p1 * res630 and north = row_sign * p0 * res630 for p = (-dy, dx) in 630 px.
            to_en = np.array([[0.0, ours.col_sign], [ours.row_sign, 0.0]]) * res   # (p0, p1) -> (east, north)
            en_pred, en_gt_fg2 = to_en @ t_px[j], to_en @ tg[j]
            lab = lab_by[s["pano"]]
            en_gt = np.array([ours.col_sign * lab["dx"] * CITY_RES[lab["city"]],
                              -ours.row_sign * lab["dy"] * CITY_RES[lab["city"]]])
            out = dict(id=s["id"], batch=k, batch_pos=j, batch_matched=ok, batch_n=len(err),
                       err_m=float(err[j]), csv_m=s["fg2_m"], en_pred=en_pred.tolist(), en_gt_fg2=en_gt_fg2.tolist(),
                       en_gt_ours=en_gt.tolist(), err_vs_ours_gt_m=float(np.linalg.norm(en_pred - en_gt)))
            (bdir / f"{safe_name(s['id'])}_fg2.json").write_text(json.dumps(out, indent=2))
            print(f"  FG² {s['id'][:40]}  recomputed {err[j]:.4f} m  csv {s['fg2_m']:.4f} m  pred EN "
                  f"({en_pred[0]:.2f}, {en_pred[1]:.2f})  GT FG² ({en_gt_fg2[0]:.2f}, {en_gt_fg2[1]:.2f}) "
                  f"ours ({en_gt[0]:.2f}, {en_gt[1]:.2f})", flush=True)
            if abs(err[j] - s["fg2_m"]) > a.tol_fg2:
                bad.append(f"{s['id']}: recomputed {err[j]:.4f} vs csv {s['fg2_m']:.4f}")
            if np.linalg.norm(en_gt_fg2 - en_gt) > 0.05:
                bad.append(f"{s['id']}: FG² GT {en_gt_fg2} vs ours {en_gt} (frame convention)")
    if bad:
        raise SystemExit("FG² recomputation does not match the csv:\n  " + "\n  ".join(bad))


# ----------------------------------------------------------------------------------------------------------- render

def render_frame(b, f, s, title_cat, path, out_w=2400):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import viz_panoroma_matches as V
    mt = b["meta"]
    pre = "f_" if ("f_valid" in b and not mt["fallback"]) else ""
    valid = b[pre + "valid"]
    h, w = valid.shape
    st = V.token_status(h, w, b[pre + "mode_tok"].astype(int), b[pre + "mode_inlier"])
    if pre:
        en_tok = V.px_to_en(b["f_xy_ref"], b["centre_f"], mt["cell_f_m"], mt["Sf"])
    else:
        en_tok = V.px_to_en(b["xy_ref"], (0.0, 0.0), mt["cell_m"], mt["S"])
    rr, cc = np.nonzero(valid)
    keep = np.linspace(0, len(rr) - 1, min(TOKENS_SHOWN, len(rr))).round().astype(int) if len(rr) else []
    rr, cc = rr[keep], cc[keep]
    col = V.az_colour(cc, w)
    inl = st[rr, cc] == 1

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(24, 10.6), dpi=out_w / 24, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.07, left=0.012, right=0.985, top=0.80, bottom=0.17)
    city, pano = s["id"].split("/", 1)
    fig.suptitle(f"{city} · {pano.split(',')[0]}   [{title_cat}]\nPanoRoMa {mt['final_m']:.2f} m  ·  "
                 f"FG² {f['err_m']:.2f} m", fontsize=26, fontweight="bold", y=0.985)

    # --- panorama
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(V.brighten(b["pano"]), extent=[0, w, h, 0], interpolation="lanczos")
    ax.scatter(cc[~inl] + 0.5, rr[~inl] + 0.5, s=55, facecolors="none", edgecolors=col[~inl], linewidths=2.0)
    ax.scatter(cc[inl] + 0.5, rr[inl] + 0.5, s=55, c=col[inl], edgecolors="black", linewidths=0.8)
    for x, lab in ((0, "S"), (w / 4, "W"), (w / 2, "N (forward)"), (3 * w / 4, "E")):
        ax.axvline(x, color="yellow", lw=1.4, ls="--", alpha=0.8)
        ax.text(x + 0.3, 1.0, lab, color="yellow", fontsize=19, fontweight="bold", va="top", path_effects=V._halo())
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"panorama (brightened). Dots: {len(rr)} of PanoRoMa's {int(valid.sum())} depth-placed tokens "
                 f"({'second' if pre else 'first'} pass; filled = RANSAC inlier,\nhollow = outlier, colour = azimuth); "
                 "the same dots are drawn on the tile where PanoRoMa's final pose puts them", fontsize=16.5, loc="left")

    # --- tile
    axm = fig.add_subplot(gs[0, 1])
    ref, cell, S = b["ref"], mt["cell_m"], mt["S"]
    V._map_axes(axm, ref, (0.0, 0.0), cell, S, "")
    V._crop_to_tile(axm, ref, cell, S, margin_m=1.0)
    gt = np.asarray(b["en_gt"], float)
    pts = {"pano": np.asarray(b["en_final"], float), "fg2": np.asarray(f["en_pred"], float)}
    (x0, x1), (y0, y1) = axm.get_xlim(), axm.get_ylim()
    allp = np.vstack([gt, *pts.values()])
    lo, hi = np.minimum(allp.min(0) - 3, [x0, y0]), np.maximum(allp.max(0) + 3, [x1, y1])
    half = max(hi - lo) / 2
    c = (lo + hi) / 2
    axm.set_xlim(c[0] - half, c[0] + half)
    axm.set_ylim(c[1] - half, c[1] + half)
    axm.scatter(en_tok[rr[~inl], cc[~inl], 0], en_tok[rr[~inl], cc[~inl], 1], s=26, facecolors="none",
                edgecolors=col[~inl], linewidths=1.4, zorder=5)
    axm.scatter(en_tok[rr[inl], cc[inl], 0], en_tok[rr[inl], cc[inl], 1], s=26, c=col[inl], edgecolors="black",
                linewidths=0.5, zorder=6)
    style = {"pano": ("#ff2020", "X", 520, mt["final_m"], "PanoRoMa"),
             "fg2": ("#2f6dff", "^", 460, f["err_m"], "FG²")}
    for k, (colr, mk, sz, err, name) in style.items():
        p = pts[k]
        axm.plot([gt[0], p[0]], [gt[1], p[1]], color=colr, lw=2.2, zorder=24, path_effects=V._halo(4))
        axm.scatter([p[0]], [p[1]], marker=mk, s=sz, c=colr, edgecolors="white", linewidths=2.0, zorder=31)
        # label next to the prediction, PanoRoMa above-right, FG² below-right (they often land close together)
        up = k == "pano"
        right = p[0] > c[0] + 0.1 * half                    # right part of the view: label to the left of it
        axm.text(p[0] + (-2.0 if right else 2.0), p[1] + (2.5 if up else -2.5), f"{name} {err:.1f} m", color=colr,
                 fontsize=18, fontweight="bold", ha="right" if right else "left", va="bottom" if up else "top", zorder=32, path_effects=V._halo(4),
                 clip_on=True)
    axm.scatter([gt[0]], [gt[1]], marker="P", s=640, c="#22dd22", edgecolors="black", linewidths=2.0, zorder=30)
    V._scale_bar(axm, 10)
    V._north(axm, 0.07)
    axm.set_title("satellite tile (north up)", fontsize=18, loc="left")
    axm.set_xlabel("east of tile centre (m)", fontsize=14)
    axm.set_ylabel("north of tile centre (m)", fontsize=14)
    hd = [Line2D([], [], marker="P", ls="", ms=20, mfc="#22dd22", mec="black", label="ground truth"),
          Line2D([], [], marker="X", ls="", ms=19, mfc="#ff2020", mec="white", label="PanoRoMa final"),
          Line2D([], [], marker="^", ls="", ms=18, mfc="#2f6dff", mec="white", label="FG²"),
          Line2D([], [], marker="o", ls="", ms=10, mfc="0.6", mec="black", label="token, inlier"),
          Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec="0.6", mew=2, label="token, outlier")]
    axm.legend(handles=hd, loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=14, ncol=3, frameon=False,
               columnspacing=1.0, handletextpad=0.3)
    fig.savefig(path, dpi=out_w / 24, pil_kwargs=dict(quality=85, optimize=True))
    plt.close(fig)


def render(a, sel, bdir, out):
    import viz_panoroma_matches as V
    by_cat = {}
    for s in sel:
        f = bdir / f"{safe_name(s['id'])}.npz"
        g = bdir / f"{safe_name(s['id'])}_fg2.json"
        if not (f.exists() and g.exists()):
            print(f"  missing bundle(s) for {s['id']}", flush=True)
            continue
        cat = s["cat"]
        title = {c: t for c, t in CATS.values()}[cat] + (" (extra: largest common error)" if s["extra"] else "")
        d = out / cat
        d.mkdir(parents=True, exist_ok=True)
        k = len(by_cat.setdefault(cat, [])) + 1
        path = d / f"{k:02d}_{safe_name(s['id'])}.jpg"
        b, fj = V.load_bundle(f), json.loads(g.read_text())
        render_frame(b, fj, s, title, path)
        by_cat[cat].append((path, f"{k}. {s['pano'].split(',')[0][:12]}  PanoRoMa {b['meta']['final_m']:.1f} m  "
                                  f"FG2 {fj['err_m']:.1f} m"))
        print(f"wrote {path}", flush=True)
    for cat, items in by_cat.items():
        V.contact_sheet([p for p, _ in items], [c for _, c in items], out / cat / "contact_sheet.jpg", cols=2)
        print(f"wrote {out / cat / 'contact_sheet.jpg'}", flush=True)


# ------------------------------------------------------------------------------------------------------------- main

def main():
    from bevloc import config as C
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--ids-json", required=True)
    ap.add_argument("--extra-both", type=int, default=2)
    ap.add_argument("--city", default="Chicago")
    ap.add_argument("--pano-json", default="experiments/16_full_test/eval_vigor_full_chicago_pano_D_samearea.json")
    ap.add_argument("--fg2-csv", default="experiments/16_full_test/eval_fg2_full_chicago_fg2_samearea.csv")
    ap.add_argument("--ckpt", default="checkpoints/vigor_v2_reg_D_e100_last.pt")
    ap.add_argument("--fine-config", default="configs/vigor_cell00625_fine.yaml")
    ap.add_argument("--fine-ckpt", default="checkpoints/vigor_v2_reg_D_fine_e100_last.pt")
    ap.add_argument("--fine-gate", type=float, default=6.0)
    ap.add_argument("--decoder-dtype", default="bfloat16")
    ap.add_argument("--fg2-batch", type=int, default=0, help="batch of the FG² run (0 = cfg.vigor.fg2_batch)")
    ap.add_argument("--tol-pano", type=float, default=1e-3)
    ap.add_argument("--tol-fg2", type=float, default=1e-3)
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--stage", default="all", choices=("all", "compute", "compute-pano", "compute-fg2", "render"))
    ap.add_argument("--out", default="experiments/16_full_test/failures")
    a = ap.parse_args()
    pano, fg2, fg2_pos = load_runs(a)
    sel = select(a, pano, fg2, fg2_pos)
    out = Path(a.out)
    bdir = out / "bundles"
    bdir.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(dict(ids_json=a.ids_json, extra_both=a.extra_both,
                                                        pano_json=a.pano_json, fg2_csv=a.fg2_csv, frames=sel), indent=2))
    for s in sel:
        print(f"  {s['cat']:10s} {s['pano'][:40]:40s} PanoRoMa {s['pano_m']:6.2f} m  FG² {s['fg2_m']:6.2f} m", flush=True)
    if a.stage in ("all", "compute", "compute-pano"):
        compute_pano(a, sel, pano, bdir)
    if a.stage in ("all", "compute", "compute-fg2"):
        compute_fg2(a, sel, fg2, bdir)
    if a.stage in ("all", "render"):
        render(a, sel, bdir, out)


if __name__ == "__main__":
    main()
