"""Pose evaluation of a query checkpoint on an immutable manifest (bootstrap CIs, chance row).

  make eval-pose CKPT=checkpoints/05_lift_splat_fixtor_ipm_best.pt \
       MANIFEST=experiments/06_fg2_bevsplat/manifest_test.json TAG=ipm [EVAL_ARGS="--seq-dists 0,2,5 --solver se2"]

The reference crop of every entry is fixed by the manifest (centre, bearing, size), so every
checkpoint, query mode and solver is scored on exactly the same pairs. Misses count as inf.
One encoder, one decoder pass per entry; the "peak" (published one-peak-per-patch RANSAC) and
"means" (distinct GMM means) rows are two consensus runs on the same logits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bevloc import config as C
from bevloc.baselines.common import load_manifest
from bevloc.data.mapillary import MAP_ROOT, MapillaryPairs, PoznanOrtho, load_frames, poznan_tiles
from bevloc.data.ortho import Oriented
from bevloc.eval.metrics import pose_errors
from bevloc.eval.report import summarise_pose
from bevloc.match.satroma import SatRoMa
from bevloc.model.coarse import FeatureQueryMatcher
from bevloc.model.query import build_query, load_query_state

CELL_M = 4.0


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="experiments/06_fg2_bevsplat/manifest.json")
    ap.add_argument("--years", default="2025,2024")
    ap.add_argument("--n", type=int, default=0, help="0 = every entry; small values for smoke tests")
    ap.add_argument("--query", default="", help="lift | ipm | hybrid | erp; default = the mode stored in the checkpoint")
    ap.add_argument("--solver", default="", help="srt | sim | se2; default = cfg.matcher.solver")
    ap.add_argument("--seq-dists", default="", help="multi-frame distances, e.g. 0,2,5 (seq / IPM-mosaic queries)")
    ap.add_argument("--out", default="experiments/05_lift_splat/eval")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    cfg = C.load(a.config)
    L = cfg.lift
    if a.seq_dists.strip():
        L.seq_dists_m = [float(x) for x in a.seq_dists.split(",") if x.strip()]
    if a.solver:
        cfg.matcher.solver = a.solver
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    man = load_manifest(a.manifest)
    years = [int(y) for y in a.years.split(",") if y.strip()]
    entries = [e for e in man["frames"] if int(e["year"]) in years]
    if a.n:
        entries = entries[: a.n]
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ckpt_mode = state.get("mode", "lift")
    mode = a.query or ckpt_mode
    L.query_mode = mode                                    # the dataset builds the picture this mode needs
    orthos = {y: PoznanOrtho(poznan_tiles(y)) for y in years}
    seq_dir = MAP_ROOT / "Fixtor" / man["meta"]["held_out_seq"]
    frames = load_frames([seq_dir], orthos[years[0]], margin_m=L.margin_m)
    ds = MapillaryPairs(frames, orthos, cfg, train=False, seed=cfg.train.seed,
                        erp_size=tuple(L.erp_size), years=years)

    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    query = build_query(cfg, mode).to(dev)
    n_loaded, n_own = load_query_state(query, state)
    if mode != ckpt_mode:
        print(f"WARNING: --query {mode} differs from the checkpoint's mode {ckpt_mode}: "
              f"{n_loaded}/{n_own} query tensors loaded, the rest are initialised", flush=True)
    query.eval()
    cons = {tag: SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=means, min_valid_frac=L.min_patch_valid)
            for tag, means in (("peak", False), ("means", True))}

    rows = []
    n = int(cfg.grid.n)
    for e in entries:
        ref_o = Oriented(tuple(e["crop_centre_en"]), float(e["crop_up_bearing_deg"]),
                         int(e["ref_size"]), float(e["gsd_m"]))
        s = ds.sample_for(e["frame_id"], ref_o, int(e["year"]))
        batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = query(batch, matcher)
            f_s = matcher.reference_features(batch["ref"])
            sf = float(((f_q.shape[-2] * 16) * (f_q.shape[-1] * 16)) ** 0.5 / 560.0)
            with matcher.model.exposed_intermediates():
                gm = matcher.model.decoder({16: f_q}, f_s, scale_factor=sf)[16]["gm_cls"][0]
        H = np.asarray(e["H_gt"], float)
        placed = query.placement(batch) if hasattr(query, "placement") else None
        row = dict(frame_id=e["frame_id"], year=int(e["year"]),
                   centre_guess_m=float(np.hypot(*e["crop_offset_m"])), argmax_m=None)
        for tag, c in cons.items():
            if placed is not None:
                m = SatRoMa.consensus_from_gm(c, gm, placed[0][0], placed[1][0], H_gt=H)
            else:
                mask = torch.nn.functional.interpolate(frac[:, None].float(), size=(n, n), mode="nearest")[0, 0]
                mask = (mask >= L.min_patch_valid)
                gmm = gm.clone()
                gmm[:, ~c.query_patches(mask)] = 0.0
                cell_err = c._argmax_cells(gmm, mask, H)
                m = c._ransac(gmm, cell_err)
            err = pose_errors(m.H, H, n, cfg.grid.cell_m) if m.H is not None else None
            row[f"pose_{tag}_m"] = None if err is None else err["position_m"]
            row[f"yaw_{tag}_deg"] = None if err is None else err["yaw_deg"]
            row[f"inliers_{tag}"] = m.inlier_ratio
            row[f"modes_{tag}"] = m.n_modes
            if m.argmax_cells is not None:
                row["argmax_m"] = m.argmax_cells * CELL_M       # None for placed (ERP) queries: no patch grid
        rows.append(row)
        pk, mn = row["pose_peak_m"], row["pose_means_m"]
        print(f"  {row['frame_id']} y{row['year']}  peak {None if pk is None else round(pk, 1)}  "
              f"means {None if mn is None else round(mn, 1)}  centre {row['centre_guess_m']:.1f}", flush=True)

    summary = {}
    for y in years:
        sub = [r for r in rows if r["year"] == y]
        if not sub:
            continue
        summary[str(y)] = {
            "peak": summarise_pose([r["pose_peak_m"] for r in sub]),
            "means": summarise_pose([r["pose_means_m"] for r in sub]),
            "centre_guess": summarise_pose([r["centre_guess_m"] for r in sub]),
        }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{a.tag}_{Path(a.manifest).stem}.json"
    path.write_text(json.dumps({"meta": dict(ckpt=a.ckpt, manifest=a.manifest, mode=mode, ckpt_mode=ckpt_mode,
                                              solver=cfg.matcher.solver, seq_dists=list(L.seq_dists_m),
                                              held_out_seq=man["meta"]["held_out_seq"], n=len(rows)),
                                "frames": rows, "summary": summary}, indent=2))
    for y, s in summary.items():
        p, c = s["peak"], s["centre_guess"]
        print(f"{y}: peak median {p['median_m']:.1f} m {tuple(round(v, 1) for v in p['median_ci'])}  "
              f"R@5 {p['recall@5m']:.2f} {tuple(round(v, 2) for v in p['recall@5m_ci'])}  "
              f"R@10 {p['recall@10m']:.2f}  >30m {p['frac_gt_30m']:.2f}  matched {p['matched']}/{p['n']}"
              f"  | centre-guess median {c['median_m']:.1f} m  R@5 {c['recall@5m']:.2f}  R@10 {c['recall@10m']:.2f}",
              flush=True)
    print(f"wrote {path}", flush=True)
    C.snapshot(cfg, out, dict(ckpt=a.ckpt, manifest=a.manifest, tag=a.tag, mode=mode))
    for o in orthos.values():
        o.close()


if __name__ == "__main__":
    main()
