"""PanoRoMa (panorama tokens placed by UniK3D metric depth + Sat-RoMa decoder + multi-hypothesis RANSAC + second pass
at 1 m cells; decision 2026-09-29) zero-shot on the Fixtor × Poznań manifest, scored like the FG² / Loc² rows.

  make panoroma-poznan                                       # cfg.poznan: heading prior, north-up reference, 400 entries
  make panoroma-poznan PZ_ARGS="--heading gt --limit 20"     # smoke / the reported baselines' orientation
  make panoroma-poznan PZ_ARGS="--sanity 0"                  # the unit sanity image only (no matcher)

Per entry (bevloc.data.poznan_pano.ManifestPanoPairs; geometry in its docstring): the reference is the Poznań
orthophoto of the entry's year on the coarse canvas of configs/vigor_cell0125.yaml (896 px at 0.125 m/px = 112 m, 56 x
56 cells of 2 m) centred on the manifest's crop centre (the prior every baseline gets; the +-22 m offset stays inside
the 56 m half-width), north-up (--ref-up north, VIGOR's geometry) or crop-up (--ref-up crop, the baselines' crop);
the panorama is rolled so that the prior's heading puts the reference's up at the ERP centre (--heading prior: the
residual rotation -crop_rot, U(-10, 10) deg, is left to the se2 solver; gt: none, as in the reported FG² / Loc² rows);
tokens are placed with UniK3D depth (make poznan-depth), car body masked. Coarse pass -> second pass with the fine
checkpoint on a 56 m window at 0.0625 m/px around the coarse pose, gate 6 m (scripts/eval_vigor.py `score` +
`DecoderFine`, unchanged) -> poses mapped back to the map frame -> bevloc.eval.manifest_eval rows:
panoroma_coarse (the peak row), panoroma_fine (ungated), panoroma_two_pass (gated: the row of the VIGOR paper table).
Heading errors are the solver's (pose_errors on the query footprint), reported next to |crop_rot| (no rotation).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C
from bevloc.baselines.common import HEADINGS, load_manifest, select_entries
from bevloc.data.poznan_pano import ManifestPanoPairs, local_to_map, residual_rotation_deg
from bevloc.eval import manifest_eval as ME

ROWS = (("panoroma_coarse", "en_coarse", "pose_peak_m", "yaw_peak_deg"),
        ("panoroma_fine", "en_fine", "pose_fine_m", "yaw_fine_deg"),
        ("panoroma_two_pass", None, "pose_fine_gated_m", None))


def _eval_vigor():
    """scripts/eval_vigor.py under a unique module name (third_party/Loc2 ships an eval_vigor.py too), as
    scripts/sweep_consensus_vigor.py loads it."""
    import importlib.util
    import sys
    name = "bevloc_scripts_eval_vigor"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "eval_vigor.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def build_datasets(a, P, cfg, entries, orthos, cfg_f=None):
    kw = dict(heading=a.heading, ref_up=a.ref_up, ego_mask_deg=a.ego_mask_deg, base_size=tuple(P.panoroma.base_size))
    ds = ManifestPanoPairs(entries, orthos, cfg, extent_m=a.extent_m, **kw)
    ds_f = ManifestPanoPairs(entries, orthos, cfg_f, **kw) if cfg_f is not None else None
    return ds, ds_f


def sanity_image(ds, ds_f, i, cfg, out_path):
    """Unit sanity check (CLAUDE.md): the entry's tokens placed from UniK3D depth, projected with the GROUND-TRUTH pose
    onto the coarse canvas and onto a fine window centred at the true position. Tokens must land on the road /
    pavement / building fronts they see. Colour = placed range (blue near, red far); white cross = true camera."""
    from bevloc.model.depth_query import ErpDepthQuery
    s = ds[i]
    batch = {k: (v[None] if torch.is_tensor(v) else v) for k, v in s.items()}
    panels = []
    for dset, sample, label in ((ds, s, "coarse 112 m @0.125"), (ds_f, None, "fine 56 m @0.0625")):
        if dset is None:
            continue
        if sample is None:
            sample = dset.item(i, ref_centre_en=s["en"])
            batch = {k: (v[None] if torch.is_tensor(v) else v) for k, v in sample.items()}
        q = ErpDepthQuery(dset.cfg)
        xy, valid = q.placement(batch)
        xy, valid = xy[0].reshape(-1, 2).numpy(), valid[0].reshape(-1).numpy()
        H = sample["H"].numpy().astype(float)
        p = np.c_[xy, np.ones(len(xy))] @ H.T
        uv = p[:, :2] / p[:, 2:3]
        n, cell = int(dset.cfg.grid.n), float(dset.cfg.grid.cell_m)
        rng = np.hypot(*(((xy - (n - 1) / 2.0) * cell).T))
        img = (sample["ref"].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        img = np.clip(img.astype(np.float32) * 1.3 + 10, 0, 255).astype(np.uint8)               # brighten
        cmap = cv2.applyColorMap(np.clip(rng / 35.0 * 255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0]
        for (u, v), ok, c in zip(uv, valid, cmap):
            if ok:
                cv2.circle(img, (int(round(u)), int(round(v))), 4, tuple(int(x) for x in c[::-1]), -1)
        cam = (H @ np.array([(n - 1) / 2.0, (n - 1) / 2.0, 1.0]))
        cam = cam[:2] / cam[2]
        fwd = H @ np.array([(n - 1) / 2.0, (n - 1) / 2.0 - 40.0, 1.0])
        fwd = fwd[:2] / fwd[2]
        cv2.drawMarker(img, (int(cam[0]), int(cam[1])), (255, 255, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.arrowedLine(img, (int(cam[0]), int(cam[1])), (int(fwd[0]), int(fwd[1])), (255, 255, 255), 3)
        cv2.putText(img, f"{label}: {int(valid.sum())} tokens, GT pose", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 3)
        panels.append(img)
    top = np.concatenate(panels, 1) if len(panels) > 1 else panels[0]
    erp = (s["erp"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    d = s["depth"][0].numpy() if "depth" in s else None
    if d is not None:
        dv = np.clip(d / 35.0 * 255, 0, 255).astype(np.uint8)
        dcol = cv2.applyColorMap(dv, cv2.COLORMAP_JET)[..., ::-1]
        dcol[d <= 0] = 0
        erp = np.concatenate([erp, dcol], 0)
    erp = cv2.resize(erp, (top.shape[1], int(erp.shape[0] * top.shape[1] / erp.shape[1])))
    sheet = np.concatenate([top, erp], 0)
    cv2.putText(sheet, f"{s['id']} y{s['year']} heading={ds.heading} ref_up={ds.ref_up} rho={s['rho_deg']:.1f} "
                       f"beta={s['beta_deg']:.1f}", (10, top.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet[..., ::-1])
    return out_path


def load_matcher_query(ckpt, cfg, dev, decoder_dtype="float16"):
    from bevloc.model.coarse import FeatureQueryMatcher
    from bevloc.model.query import apply_query_cfg, build_query, load_query_state
    from bevloc.model.speed import DECODER_DTYPE_DEFAULT, set_decoder_dtype
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    mode = state.get("mode", "lift")
    if mode != "erp_depth":
        raise SystemExit(f"{ckpt}: mode {mode!r}, PanoRoMa needs an erp_depth checkpoint")
    cfg.lift.query_mode = mode
    apply_query_cfg(cfg, state)
    matcher = FeatureQueryMatcher(cfg.matcher.checkpoint, dev, train_decoder=False)
    matcher.model.decoder.load_state_dict(state["decoder"], strict=False)
    if decoder_dtype != DECODER_DTYPE_DEFAULT:
        set_decoder_dtype(matcher.model.decoder, decoder_dtype)
    query = build_query(cfg, mode).to(dev)
    load_query_state(query, state)
    query.eval()
    return state, matcher, query


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--manifest", default=None, help="default cfg.poznan.manifest")
    ap.add_argument("--years", default=None, help="comma list (default cfg.poznan.years)")
    ap.add_argument("--heading", default=None, choices=HEADINGS, help="default cfg.poznan.heading")
    ap.add_argument("--ref-up", default=None, choices=("north", "crop"), help="default cfg.poznan.panoroma.ref_up")
    ap.add_argument("--extent-m", type=float, default=None, help="coarse imagery extent (default cfg: whole canvas)")
    ap.add_argument("--ego-mask-deg", type=float, default=None)
    ap.add_argument("--coarse-config", default=None)
    ap.add_argument("--coarse-ckpt", default=None)
    ap.add_argument("--fine-config", default=None)
    ap.add_argument("--fine-ckpt", default=None)
    ap.add_argument("--solver", default=None)
    ap.add_argument("--decoder-dtype", default="float16", choices=("bfloat16", "float16", "float32"),
                    help="autocast dtype of both Sat-RoMa decoders on CUDA; default float16 = the package's, every "
                         "published row (bevloc.model.speed.set_decoder_dtype)")
    ap.add_argument("--fine-gate", type=float, default=None)
    ap.add_argument("--limit", "--n", dest="limit", type=int, default=0,
                    help="first N unique frame ids (0 = all); --n as in eval_fg2.py / eval_loc2.py")
    ap.add_argument("--sanity", type=int, default=None, help="write the sanity image of entry INDEX and stop")
    ap.add_argument("--out", default=None, help="default experiments/12_poznan_three_way/panoroma_<heading>_<ref_up>")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    base = C.load(a.config)
    P = base.poznan
    Q = P.panoroma
    a.heading = a.heading or P.heading
    a.ref_up = a.ref_up or Q.ref_up
    a.extent_m = Q.coarse_extent_m if a.extent_m is None else a.extent_m
    a.ego_mask_deg = float(P.ego_mask_deg if a.ego_mask_deg is None else a.ego_mask_deg)
    manifest = a.manifest or P.manifest
    years = [int(y) for y in (a.years.split(",") if a.years else P.years)]
    out = Path(a.out or f"experiments/12_poznan_three_way/panoroma_{a.heading}_{a.ref_up}{a.tag}")
    cfg = C.load(a.coarse_config or Q.coarse_config)
    cfg.matcher.solver = a.solver or Q.solver
    cfg.lift.query_mode = "erp_depth"
    cfg_f = C.load(a.fine_config or Q.fine_config)
    cfg_f.matcher = copy.deepcopy(cfg.matcher)
    cfg_f.lift.query_mode = "erp_depth"
    gate = float(Q.fine_gate_m if a.fine_gate is None else a.fine_gate)
    entries = select_entries(load_manifest(manifest), years, a.limit)
    orthos = ME.open_orthos(years)
    ds, ds_f = build_datasets(a, P, cfg, entries, orthos, cfg_f)
    n_drop = ds.keep_with_depth()
    ds_f.labels = ds.labels
    if n_drop:
        raise SystemExit(f"{n_drop} of {len(entries)} entries have no UniK3D depth: make poznan-depth first")
    if a.sanity is not None:
        p = sanity_image(ds, ds_f, a.sanity, cfg, out / f"sanity_{ds.labels[a.sanity]['frame_id']}.jpg")
        print(f"wrote {p}")
        return

    ev = _eval_vigor()
    DecoderFine, check_fine_grid, score = ev.DecoderFine, ev.check_fine_grid, ev.score
    from bevloc.match.satroma import SatRoMa
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    coarse_ckpt, fine_ckpt = a.coarse_ckpt or Q.coarse_ckpt, a.fine_ckpt or Q.fine_ckpt
    state, matcher, query = load_matcher_query(coarse_ckpt, cfg, dev, a.decoder_dtype)
    state_f, matcher_f, query_f = load_matcher_query(fine_ckpt, cfg_f, dev, a.decoder_dtype)
    check_fine_grid(state_f, cfg_f, fine_ckpt, a.fine_config or Q.fine_config)
    cons = {tag: SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=m, min_valid_frac=0.05)
            for tag, m in (("peak", False), ("means", True))}
    cons_f = SatRoMa.from_wrapper(matcher_f.wrapper, cfg_f, use_means=False, min_valid_frac=0.05)
    fine = DecoderFine(ds_f, query_f, matcher_f, cons_f, cfg_f, dev)
    print(f"PanoRoMa on {manifest}: {len(ds)} entries, years {years}, heading {a.heading}, ref_up {a.ref_up}, extent "
          f"{a.extent_m or 'canvas'}, solver {cons['peak'].solver}, coarse {coarse_ckpt} (step {state.get('step')}), "
          f"fine {fine_ckpt} (step {state_f.get('step')}), gate {gate} m, device {dev}", flush=True)
    t0 = time.time()
    rows = score(ds, query, matcher, cons, cfg, dev, verbose=True, fine=fine, fine_gate=gate)
    sec = (time.time() - t0) / max(1, len(rows))
    orthos_close = [o.close() for o in orthos.values()]  # noqa: F841

    index = {(str(e["frame_id"]), str(int(e["year"]))): k for k, e in enumerate(ds.labels)}
    out_rows, max_check = [], 0.0
    scored = {(r["id"], r["city"]) for r in rows}
    for key, k in index.items():            # `score` skips an entry whose sample raised: a miss here, as for FG² / Loc²
        if key not in scored:
            for name, *_ in ROWS:
                out_rows.append(ME.failed_row(ds.labels[k], name, "sample failed (skipped by score)",
                                              heading=a.heading, ref_up=a.ref_up))
    for r in rows:
        e = ds.labels[index[(r["id"], r["city"])]]
        beta = ds.geometry(index[(r["id"], r["city"])])[0]
        extra = dict(heading=a.heading, ref_up=a.ref_up, residual_rot_deg=residual_rotation_deg(e, a.heading),
                     inliers_peak=r.get("inliers_peak"), inliers_fine=r.get("inliers_fine"),
                     fallback_fine=r.get("fallback_fine"), fine_shift_m=r.get("fine_shift_m"))
        for name, en_key, err_key, yaw_key in ROWS:
            if name == "panoroma_two_pass":
                fb = bool(r.get("fallback_fine"))
                en_key, yaw_key = ("en_coarse", "yaw_peak_deg") if fb else ("en_fine", "yaw_fine_deg")
            en_l = r.get(en_key)
            if en_l is None or r.get(err_key) is None:
                out_rows.append(ME.failed_row(e, name, r.get("fine_error") or "no pose", **extra))
                continue
            en = local_to_map(np.asarray(en_l, float), e["crop_centre_en"], beta).reshape(2)
            row = ME.pose_row(e, name, en, None, **extra)
            row["err_deg"] = r.get(yaw_key)
            max_check = max(max_check, abs(row["err_m"] - float(r[err_key])))
            out_rows.append(row)
    ME.write_rows(out, out_rows)
    summary = {name: ME.summarise_by_year([x for x in out_rows if x["method"] == name]) for name, *_ in ROWS}
    no_rot = [abs(x["residual_rot_deg"]) for x in out_rows if x["method"] == "panoroma_coarse"]
    (out / "metrics.json").write_text(json.dumps(dict(
        method="PanoRoMa", manifest=manifest, years=years, heading=a.heading, ref_up=a.ref_up, extent_m=a.extent_m,
        ego_mask_deg=a.ego_mask_deg, coarse=dict(config=a.coarse_config or Q.coarse_config, ckpt=coarse_ckpt,
                                                  step=state.get("step")),
        fine=dict(config=a.fine_config or Q.fine_config, ckpt=fine_ckpt, step=state_f.get("step"), gate_m=gate),
        solver=cons["peak"].solver, decoder_dtype=a.decoder_dtype, n=len(rows), n_entries=len(ds), sec_per_entry=sec,
        map_vs_tile_error_max_diff_m=max_check, median_abs_residual_rot_deg=float(np.median(no_rot)) if no_rot else None,
        summary=summary), indent=2))
    (out / "score_rows.json").write_text(json.dumps(rows, indent=1, default=float))
    C.snapshot(cfg, out, dict(manifest=manifest, heading=a.heading, ref_up=a.ref_up, coarse_ckpt=coarse_ckpt,
                              fine_ckpt=fine_ckpt, fine_config=a.fine_config or Q.fine_config,
                              decoder_dtype=a.decoder_dtype))
    for name, s in summary.items():
        ME.print_summary(name, s)
    print(f"map-frame vs tile-frame error, max difference {max_check:.2e} m; median |residual rotation| "
          f"{np.median(no_rot) if no_rot else float('nan'):.2f} deg; {sec:.2f} s/entry", flush=True)
    print(f"wrote {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
