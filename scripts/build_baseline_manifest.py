"""Build the immutable ≥200-frame Fixtor held-out manifest for baseline comparison.

Uses the same route split and local prior as ``05_lift_splat``:
  held-out ``IcRzj0wTLZX874qitxVsQa``, ±10% of the 224 m reference edge / ±10°,
  primary year 2025 + cross-year 2024 entries for the same frame IDs.

  make baselines-manifest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bevloc import config as C
from bevloc.baselines.common import ManifestFrame, frame_pose_proxy, write_manifest
from bevloc.data.mapillary import MAP_ROOT, TRAIN_SEQS, PoznanOrtho, load_frames, poznan_tiles
from bevloc.data.ortho import Oriented, gt_homography, sample_reference



def open_year(year: int) -> PoznanOrtho:
    return PoznanOrtho(poznan_tiles(year))


def uniform_along_route(frames, n, seed=0):
    """Pick ``n`` frames spread uniformly by path length (not by list index)."""
    if len(frames) <= n:
        return list(frames)
    ens = np.asarray([fr["_en"] for fr in frames], float)
    # sort by capture time when present so path order matches driving
    order = sorted(range(len(frames)),
                   key=lambda i: frames[i].get("captured_at", i))
    cum = np.zeros(len(order), float)
    for k in range(1, len(order)):
        cum[k] = cum[k - 1] + float(np.linalg.norm(ens[order[k]] - ens[order[k - 1]]))
    targets = np.linspace(0.0, cum[-1], n)
    chosen, used = [], set()
    for t in targets:
        k = int(np.searchsorted(cum, t))
        k = min(max(k, 0), len(order) - 1)
        # walk to nearest unused
        for d in range(len(order)):
            for cand in (k - d, k + d):
                if 0 <= cand < len(order) and order[cand] not in used:
                    used.add(order[cand])
                    chosen.append(frames[order[cand]])
                    break
            else:
                continue
            break
    return chosen


def build_entries(frames, years, cfg, seed=0):
    g = cfg.grid
    scale = int(cfg.reference.scale)
    # Task protocol: local prior ±10% of 224 m edge / ±10° (lift val defaults).
    max_off = float(getattr(cfg.lift, "max_offset_frac", 0.10))
    max_rot = float(getattr(cfg.lift, "max_rot_deg", 10.0))
    out = []
    for fr in frames:
        en, up = frame_pose_proxy(fr)
        query = Oriented(en, up, g.n, g.cell_m)
        # Per-frame RNG from matcher seed + id so years share the same crop
        # geometry (fair cross-year comparison).
        rng = np.random.default_rng([int(cfg.matcher.seed) + int(seed),
                                     int(fr["id"]) % (2**32)])
        ref = sample_reference(query, rng, scale=scale,
                               max_offset_frac=max_off, max_rot_deg=max_rot)
        # Recover the (right, up) offset that sample_reference applied.
        b = np.radians(ref.up_bearing_deg)
        right = np.array([np.cos(b), -np.sin(b)])
        up_v = np.array([np.sin(b), np.cos(b)])
        d = np.asarray(query.centre_en) - np.asarray(ref.centre_en)
        off = (float(d @ right), float(d @ up_v))
        rot = float(ref.up_bearing_deg - query.up_bearing_deg)
        H = gt_homography(query, ref)
        lon, lat = fr["computed_geometry"]["coordinates"]
        # repo-relative so the manifest is portable (Eagle, worktrees)
        panorama = str((Path(fr["_seq"]) / "images" / f"{fr['id']}.jpg").relative_to(C.REPO))
        for year in years:
            out.append(ManifestFrame(
                frame_id=str(fr["id"]),
                seq=Path(fr["_seq"]).name,
                panorama=panorama,
                lon=float(lon),
                lat=float(lat),
                en=(float(en[0]), float(en[1])),
                up_bearing_deg=float(up),
                compass_angle=float(fr["computed_compass_angle"]),
                year=int(year),
                crop_centre_en=(float(ref.centre_en[0]), float(ref.centre_en[1])),
                crop_up_bearing_deg=float(ref.up_bearing_deg),
                crop_offset_m=off,
                crop_rot_deg=rot,
                query_size=int(g.n),
                ref_size=int(g.n * scale),
                gsd_m=float(g.cell_m),
                H_gt=H.tolist(),
            ))
    return out


def overview_plot(entries, out_path: Path):
    """Map overview of selected held-out frames (2025 entries only)."""
    rows = [e for e in entries if e.year == 2025]
    e = np.array([r.en[0] for r in rows])
    n = np.array([r.en[1] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(e, n, s=8, c="C0", alpha=0.8, label=f"n={len(rows)}")
    ax.set_aspect("equal")
    ax.set_xlabel("E (EPSG:2180)")
    ax.set_ylabel("N (EPSG:2180)")
    ax.set_title(f"Fixtor held-out manifest ({rows[0].seq if rows else ''})")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--out", default="experiments/06_fg2_bevsplat/manifest.json")
    ap.add_argument("--n", type=int, default=200, help="held-out frames (≥200)")
    ap.add_argument("--years", default="2025,2024")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq", default="Fixtor/IcRzj0wTLZX874qitxVsQa",
                    help="route under data/mapillary used for this manifest (never a training route)")
    a = ap.parse_args()
    cfg = C.load(a.config)
    years = [int(y) for y in a.years.split(",") if y.strip()]
    seq_dir = MAP_ROOT / a.seq
    if seq_dir in TRAIN_SEQS:
        raise SystemExit(f"{a.seq} is a training route")
    if not (seq_dir / "images.json").exists():
        raise SystemExit(f"{seq_dir} has no images.json (download it: make mapillary-seq SEQ=<id>)")
    ortho = open_year(years[0])
    margin = float(getattr(cfg.lift, "margin_m", 120.0))
    all_frames = load_frames([seq_dir], ortho, margin_m=margin)
    # Also require coverage on every other year.
    for y in years[1:]:
        o = open_year(y)
        keep = []
        for fr in all_frames:
            if o.tile_for(fr["_en"], margin=margin) is not None:
                keep.append(fr)
        print(f"  year {y}: {len(keep)}/{len(all_frames)} still covered", flush=True)
        all_frames = keep
        o.close()
    ortho.close()
    if len(all_frames) < a.n:
        raise SystemExit(f"only {len(all_frames)} covered frames, need ≥{a.n}")
    picked = uniform_along_route(all_frames, a.n, seed=a.seed)
    entries = build_entries(picked, years, cfg, seed=a.seed)
    meta = {
        "protocol": "docs/tasks/02_fg2_bevsplat.md",
        "held_out_seq": seq_dir.name,
        "train_seqs": [p.name for p in TRAIN_SEQS],
        "n_frames": a.n,
        "years": years,
        "seed": a.seed,
        "max_offset_frac": float(cfg.lift.max_offset_frac),
        "max_rot_deg": float(cfg.lift.max_rot_deg),
        "query_size": int(cfg.grid.n),
        "ref_scale": int(cfg.reference.scale),
        "gsd_m": float(cfg.grid.cell_m),
        "pose_proxy": "Mapillary computed_geometry + computed_compass_angle",
        "note": "Footprint overlay is approximate; do not call it survey GT.",
    }
    out = Path(a.out)
    write_manifest(entries, out, meta)
    overview_plot(entries, out.parent / f"{out.stem}_overview.jpg")
    # Snapshot config
    snap = out.parent / "config.yaml"
    if not snap.exists():
        snap.write_text(Path(a.config).read_text())
    print(json.dumps({
        "manifest": str(out),
        "n_entries": len(entries),
        "n_unique_frames": a.n,
        "years": years,
        "overview": str(out.parent / f"{out.stem}_overview.jpg"),
    }, indent=2))


if __name__ == "__main__":
    main()
