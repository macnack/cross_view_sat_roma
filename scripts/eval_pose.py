"""Pose evaluation of a query checkpoint on an immutable manifest (bootstrap CIs, chance row).

  make eval-pose CKPT=checkpoints/05_lift_splat_fixtor_seq_best.pt \
       MANIFEST=experiments/06_fg2_bevsplat/manifest.json TAG=seq EVAL_ARGS="--seq-dists 0,2,5"

The reference crop of every entry is fixed by the manifest (centre, bearing, size), so every
checkpoint, query mode and solver is scored on exactly the same pairs. Misses count as inf.
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
from bevloc.model.coarse import FeatureQueryMatcher, coarse_targets, ref_cell_validity
from bevloc.model.query import build_query, load_query_state

CELL_M = 4.0


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="experiments/06_fg2_bevsplat/manifest.json")
    ap.add_argument("--years", default="2025,2024")
    ap.add_argument("--n", type=int, default=0, help="0 = every entry; small values for smoke tests")
    ap.add_argument("--query", default="", help="lift | ipm | hybrid; default = the mode stored in the checkpoint")
    ap.add_argument("--solver", default="", help="srt | se2; default = cfg.matcher.solver")
    ap.add_argument("--seq-dists", default="", help="multi-frame splat distances, e.g. 0,2,5 (seq checkpoints)")
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
    orthos = {y: PoznanOrtho(poznan_tiles(y)) for y in years}
    seq_dir = MAP_ROOT / "Fixtor" / man["meta"]["held_out_seq"]
    frames = load_frames([seq_dir], orthos[years[0]], margin_m=L.margin_m)
    if getattr(L, "query_mode", "lift") != (a.query or "lift"):
        L.query_mode = a.query or L.query_mode
    ds = MapillaryPairs(frames, orthos, cfg, train=False, seed=cfg.train.seed,
                        erp_size=tuple(L.erp_size), years=years)

    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mode = a.query or state.get("mode", "lift")
    L.query_mode = mode
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    query = build_query(cfg, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    wraps = {}
    for tag, means in (("peak", False), ("means", True)):
        w = SatRoMa.from_config(cfg, use_means=means, min_valid_frac=L.min_patch_valid)
        w.m.model.decoder.load_state_dict(state["decoder"], strict=False)
        wraps[tag] = w

    rows = []
    for e in entries:
        ref_o = Oriented(tuple(e["crop_centre_en"]), float(e["crop_up_bearing_deg"]),
                         int(e["ref_size"]), float(e["gsd_m"]))
        s = ds.sample_for(e["frame_id"], ref_o, int(e["year"]))
        batch = {k: (v[None].to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = query(batch, matcher)
            f_s = matcher.reference_features(batch["ref"])
        H = np.asarray(e["H_gt"], float)
        rv = ref_cell_validity(batch["ref"])
        idx, use = coarse_targets(batch["H"], frac >= L.min_patch_valid, ref_valid=rv)
        mask = torch.nn.functional.interpolate(frac[:, None].float(), size=(cfg.grid.n, cfg.grid.n),
                                               mode="nearest")[0, 0]
        mask = (mask >= L.min_patch_valid).cpu().numpy()
        row = dict(frame_id=e["frame_id"], year=int(e["year"]),
                   centre_guess_m=float(np.hypot(*e["crop_offset_m"])), argmax_m=None)
        for tag, w in wraps.items():
            m = w.match_encoded(f_q, f_s[16], scale_factor=0.4, mask=mask, H_gt=H)
            err = pose_errors(m.H, H, cfg.grid.n, cfg.grid.cell_m) if m.H is not None else None
            row[f"pose_{tag}_m"] = None if err is None else err["position_m"]
            row[f"yaw_{tag}_deg"] = None if err is None else err["yaw_deg"]
            row[f"inliers_{tag}"] = m.inlier_ratio
            row[f"modes_{tag}"] = m.n_modes
            if m.argmax_cells is not None:
                row["argmax_m"] = m.argmax_cells * CELL_M
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
    path.write_text(json.dumps({"meta": dict(ckpt=a.ckpt, manifest=a.manifest, mode=mode,
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
