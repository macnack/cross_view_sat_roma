"""H1: unchanged Sat-RoMa on BEV queries against the EA tiles that cover the drive.

Also the cosine probe: mean-removed cosine of scale-16 tokens of the BEV against
the reference token at the ground-truth cell, for sat493m and for the frozen
ConvNeXt-Tiny LVD, on the EA crop and on the NLP intensity crop of the same pose.

No training. One reference encode is shared across query variants.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bevloc import config as C
from bevloc.bev.variants import build_variants
from bevloc.data.ortho import OrthoMap, gt_homography, sample_reference
from bevloc.data.oxts import read_track
from bevloc.data.pairs import fully_covered_names
from bevloc.eval.metrics import mean_removed_cosine, pose_errors, recall
from bevloc.match import satroma
from bevloc.model.coarse import coarse_targets
from bevloc.run import context

VARIANTS = ("ipm_cl", "oracle_a", "oracle_b")
NLP = "data/ortho/durham/2021_nlp/durham_2021_nlp_intensity_1m_uint8.tif"
CELL_M = 4.0          # one coarse cell on the 896 px reference at 0.25 m/px


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--out", default="experiments/01_kickoff/h1_ea")
    ap.add_argument("--limit", type=int, default=0, help="evenly spaced subset; 0 = every fully covered frame")
    ap.add_argument("--overlays", type=int, default=8)
    a = ap.parse_args()
    cfg = C.load(a.config)
    out = Path(a.out)
    C.snapshot(cfg, out)

    ds, calib, erp_valid = context(cfg)
    ortho_path = C.REPO / cfg.data.ortho
    names = ds.names(("image", "scan", "oxts"))
    track = read_track(ds.root, names)
    edge_m = cfg.grid.n * cfg.grid.cell_m * cfg.reference.scale
    covered = set(fully_covered_names(ortho_path, names, track.en, edge_m, min_frac=0.999))
    names = [n for n in names if n in covered]
    n_covered = len(names)
    if a.limit and a.limit < len(names):
        pick = np.linspace(0, len(names) - 1, a.limit).astype(int)
        names = [names[i] for i in dict.fromkeys(pick.tolist())]
    if not names:
        raise SystemExit(f"no frame is fully covered by {ortho_path}")
    index = {n: i for i, n in enumerate(track.names)}
    print(f"H1 frames: {len(names)} of {n_covered} fully covered by {ortho_path.name} (min_frac 0.999)", flush=True)

    nlp_path = C.REPO / NLP
    nlp = OrthoMap(nlp_path) if nlp_path.exists() else None
    ea = OrthoMap(ortho_path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = satroma.SatRoMa.from_config(cfg)
    cnx = convnext(dev)
    sf = float(matcher.m.im_a_size / 560.0)     # square query; matches bevloc.match.satroma
    r = cfg.reference

    rows = []
    for k, name in enumerate(names):
        i = index[name]
        qpose = sample_query(cfg, track, i)
        rng = np.random.default_rng([cfg.matcher.seed, int(name)])
        ref_o = sample_reference(qpose, rng, r.scale, r.max_offset_frac, r.max_rot_deg)
        ref_ea, inside = ea.render(ref_o)
        if inside.mean() < 0.999 or (ref_ea.sum(-1) > 0).mean() < 0.95:
            print(f"  skip {name}: rendered EA crop is not full", flush=True)
            continue
        H = gt_homography(qpose, ref_o)
        bev = build_variants(ds.erp(name), ds.points(name), erp_valid, calib, cfg, name=name, variants=VARIANTS)
        f_ea = matcher.encode(ref_ea, matcher.m.im_b_size)
        f_nlp = None
        ref_nlp = None
        if nlp is not None:
            ref_nlp, _ = nlp.render(ref_o)
            f_nlp = matcher.encode(ref_nlp, matcher.m.im_b_size)
        cnx_ea = cnx_encode(cnx, ref_ea, matcher.m.im_b_size, dev)
        cnx_nlp = cnx_encode(cnx, ref_nlp, matcher.m.im_b_size, dev) if ref_nlp is not None else None

        for variant in VARIANTS:
            img, valid = bev.images[variant], bev.valid[variant]
            f_q = matcher.encode(img, matcher.m.im_a_size)
            hit = matcher.match_encoded(f_q, f_ea, sf, mask=valid, H_gt=H)
            err = pose_errors(hit.H, H, cfg.grid.n, cfg.grid.cell_m) if hit.H is not None else None
            pv = patch_valid(matcher, valid)
            row = dict(name=name, variant=variant,
                       position_m=None if err is None else err["position_m"],
                       yaw_deg=None if err is None else err["yaw_deg"],
                       corner_m=None if err is None else err["corner_m"],
                       inlier_ratio=hit.inlier_ratio, n_modes=hit.n_modes,
                       argmax_m=None if hit.argmax_cells is None else hit.argmax_cells * CELL_M,
                       cos_ea=paired_cosine(f_q, f_ea, H, pv),
                       cos_nlp=paired_cosine(f_q, f_nlp, H, pv) if f_nlp is not None else None,
                       cos_cnx_ea=paired_cosine(cnx_encode(cnx, img, matcher.m.im_a_size, dev), cnx_ea, H, pv),
                       cos_cnx_nlp=paired_cosine(cnx_encode(cnx, img, matcher.m.im_a_size, dev), cnx_nlp, H, pv)
                       if cnx_nlp is not None else None)
            rows.append(row)
            if k < a.overlays and variant in ("ipm_cl", "oracle_b"):
                write_overlay(out / f"{name}_{variant}.jpg", img, ref_ea, H, hit.H, row)
        ipm, ora = rows[-len(VARIANTS)], rows[-1]
        print(f"  {k + 1}/{len(names)} {name}  ipm_cl { _pose(ipm) }  oracle_b { _pose(ora) }  "
              f"cos sat/cnx {ipm['cos_ea']:.3f}/{ipm['cos_cnx_ea']:.3f}", flush=True)

    summary = summarize(rows)
    json.dump(dict(n=len({r["name"] for r in rows}), summary=summary), open(out / "metrics.json", "w"), indent=1)
    write_csv(out / "frames.csv", rows)
    print(json.dumps(summary, indent=1))
    print("wrote", out)


def sample_query(cfg, track, i):
    from bevloc.data.ortho import Oriented
    bearing = float(track.bearing(cfg.oxts.convention, grid=True)[i])
    return Oriented(tuple(track.en[i]), bearing, cfg.grid.n, cfg.grid.cell_m)


def patch_valid(matcher, valid):
    return matcher.query_patches(valid)[None]


def paired_cosine(f_q, f_s, H, pv):
    """Mean-removed cosine of each valid query token with the reference token at its GT cell."""
    if f_q.shape[-2:] != pv.shape[-2:]:
        raise RuntimeError(f"query tokens {tuple(f_q.shape)} do not match the {tuple(pv.shape[-2:])} patch grid")
    dev = f_q.device
    idx, use = coarse_targets(torch.as_tensor(H, dtype=torch.float32, device=dev)[None], pv.to(dev))
    q = f_q[0].detach().float().permute(1, 2, 0)[use[0]]
    ref = f_s[0].detach().float().permute(1, 2, 0).reshape(-1, f_s.shape[1])
    r = ref[idx[0][use[0]]]
    if q.shape[0] < 2:
        return float("nan")
    return mean_removed_cosine(q.cpu().numpy(), r.cpu().numpy())


def _pose(row):
    return "fail" if row["position_m"] is None else f"{row['position_m']:.1f} m"


_CNX_MEAN = (0.485, 0.456, 0.406)
_CNX_STD = (0.229, 0.224, 0.225)


def convnext(device):
    import timm
    net = timm.create_model("convnext_tiny.dinov3_lvd1689m", pretrained=True, features_only=True,
                            out_indices=(2,)).to(device).eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def cnx_encode(net, image, size, device):
    """Stride-16 ConvNeXt-Tiny tokens. ImageNet norm, matching the LVD weights."""
    t = satroma.SatRoMaMatcher.load_image(image, size).unsqueeze(0).to(device)
    mean = t.new_tensor(_CNX_MEAN).view(1, 3, 1, 1)
    std = t.new_tensor(_CNX_STD).view(1, 3, 1, 1)
    with torch.no_grad():
        return net((t - mean) / std)[0]


def summarize(rows):
    out = {}
    for variant in VARIANTS:
        sub = [r for r in rows if r["variant"] == variant]
        pos = [r["position_m"] for r in sub]
        yaw = [r["yaw_deg"] for r in sub if r["yaw_deg"] is not None]
        arg = [r["argmax_m"] for r in sub if r["argmax_m"] is not None]
        block = dict(n=len(sub), n_matched=sum(p is not None for p in pos), **recall(pos))
        block["median_position_m"] = _median([p for p in pos if p is not None])
        block["median_yaw_deg"] = _median(yaw)
        block["median_argmax_m"] = _median(arg)
        for key in ("cos_ea", "cos_nlp", "cos_cnx_ea", "cos_cnx_nlp"):
            block[key] = _median([r[key] for r in sub if r[key] is not None])
        out[variant] = block
    return out


def _median(xs):
    return None if not xs else float(np.median(xs))


def write_csv(path, rows):
    import csv
    keys = list(rows[0]) if rows else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def write_overlay(path, query, ref, H_gt, H_est, row):
    o = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)

    def box(H, color, thickness):
        if H is None:
            return
        c = np.array([[0, 0, 1], [223, 0, 1], [223, 223, 1], [0, 223, 1]], float) @ np.asarray(H, float).T
        c = c[:, :2] / c[:, 2:3]
        cv2.polylines(o, [np.round(c).astype(np.int32)], True, color, thickness)

    box(H_gt, (0, 255, 0), 2)
    box(H_est, (40, 40, 255), 2)
    q = cv2.resize(cv2.cvtColor(query, cv2.COLOR_RGB2BGR), (ref.shape[0], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
    body = np.hstack([q, o])
    head = np.zeros((64, body.shape[1], 3), np.uint8)
    pose = "RANSAC failed" if row["position_m"] is None else f"pose {row['position_m']:.1f} m  yaw {row['yaw_deg']:.1f} deg"
    text = f"{row['name']}  {row['variant']}  {pose}  argmax {row['argmax_m']:.1f} m  cos {row['cos_ea']:.3f}"
    cv2.putText(head, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([head, body]), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
