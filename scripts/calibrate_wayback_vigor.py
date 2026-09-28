"""Georeferencing calibration VIGOR tile -> Esri Wayback window (task 05, deliverable 2).

  make wayback-calib CITY=Chicago                     # 150 tiles, the release closest to 1 July wayback.calib_year
  make wayback-calib CITY=Chicago WAYBACK_ARGS="--year 2021 --tiles 150 --split samearea"

Google's tile centres and Esri's tiles disagree by a constant few pixels per city. On `wayback.calib_tiles` distinct
tiles of the city (a seeded choice among the unique tiles of a 4x larger VigorPairs test-list draw, seed 0; several
panoramas share a tile) the window of the release closest to `wayback.calib_year`
(the VIGOR capture years, 2020-2021) is fetched UNCALIBRATED into <root>/<City>/wayback_calib_<year>/ (offset 0,
sidecars as for any fetch; scripts/fetch_wayback_vigor.fetch_tile) and phase-correlated with the VIGOR tile resampled
to the same raster (`bevloc.data.wayback.phase_correlation`, Hann window, 20x upsampled DFT, i.e. 0.05 px;
`vigor_at_out_gsd` = VigorPairs' resampling). Per tile that gives (dx, dy): the content at VIGOR pixel (u, v) sits at
Wayback pixel (u + dx, v + dy). Aggregate: tiles whose correlation peak is below wayback.calib_min_response are
dropped; the median offset over the rest; tiles farther than wayback.calib_max_dev_px from it are outliers (dropped,
counted, the median recomputed). Reported: the median offset in px and metres, the residual = median over the kept
tiles of |offset_i - median| in metres (the misalignment a constant offset leaves), the robust sigma (1.4826 MAD per
axis) and the standard error of the median. Gate (docs/tasks/05_reference_years.md): residual < wayback.calib_gate_m
-> PASS, else FAIL (printed and stored; the fetcher applies the offset either way and records it).
Output: <root>/<City>/wayback_calibration.json (offset_px, offset_m, gsd_m, residual_m, robust_sigma_m, se_m, pass,
release(s), per-tile offsets and responses); every later `make wayback-fetch` of that city applies it.
No design decision of the 2026-09-24 label calibration is reused here: that one regressed the labels on the
lat/lon of the file names (scripts/vigor_check_labels.py); the phase correlation is new code (bevloc.data.wayback).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from bevloc import config as C
from bevloc.data.vigor import CITY_RES, VigorPairs
from bevloc.data import wayback as W


def _fetch_module():
    spec = importlib.util.spec_from_file_location("bevloc_scripts_fetch_wayback_vigor",
                                                  Path(__file__).resolve().parent / "fetch_wayback_vigor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def aggregate(offsets, responses, gsd, min_response, max_dev_px, gate_m):
    """The calibration numbers from per-tile (dx, dy) and correlation responses (module doc). Returns a dict."""
    off = np.asarray(offsets, np.float64).reshape(-1, 2)
    resp = np.asarray(responses, np.float64).reshape(-1)
    ok = resp >= float(min_response)
    n_low = int((~ok).sum())
    if not ok.any():
        return dict(n=int(len(off)), n_used=0, n_low_response=n_low, n_outliers=0, offset_px=None, offset_m=None,
                    residual_m=None, robust_sigma_m=None, se_m=None, gsd_m=float(gsd), gate_m=float(gate_m), **{"pass": False})
    med = np.median(off[ok], 0)
    dev = np.linalg.norm(off - med, axis=1)
    keep = ok & (dev <= float(max_dev_px))
    n_out = int((ok & ~keep).sum())
    med = np.median(off[keep], 0)
    dev = np.linalg.norm(off[keep] - med, axis=1)
    mad = np.median(np.abs(off[keep] - med), 0)
    sigma = 1.4826 * mad * gsd
    n_used = int(keep.sum())
    residual = float(np.median(dev) * gsd)
    se = float(np.linalg.norm(sigma) * 1.2533 / np.sqrt(max(n_used, 1)))
    return dict(n=int(len(off)), n_used=n_used, n_low_response=n_low, n_outliers=n_out,
                offset_px=[float(med[0]), float(med[1])], offset_m=[float(med[0] * gsd), float(med[1] * gsd)],
                residual_m=residual, robust_sigma_m=[float(sigma[0]), float(sigma[1])], se_m=se,
                gsd_m=float(gsd), gate_m=float(gate_m), **{"pass": bool(residual < float(gate_m))})


def calibrate(a, cfg, opener=None):
    F = _fetch_module()
    wcfg = F.wayback_cfg(cfg)
    year = int(a.year if a.year is not None else wcfg.calib_year)
    n_tiles = int(a.tiles if a.tiles is not None else wcfg.calib_tiles)
    root = Path(a.root)
    city = a.city
    # n_tiles distinct tiles: several panoramas share a tile, so a draw of n labels has fewer; draw 4 n labels and
    # take a seeded choice of n of their unique tiles (file order kept)
    ds = VigorPairs(root, cfg, cities=[city], split=a.split, train=False, limit=4 * n_tiles, seed=a.seed)
    tiles = F.unique_tiles(ds.labels)
    pick = np.sort(np.random.default_rng(a.seed).permutation(len(tiles))[:n_tiles])
    tiles = [tiles[j] for j in pick]
    print(f"{city}: {len(tiles)} unique tiles of a {len(ds.labels)}-label draw (split {a.split}, seed {a.seed}); "
          f"release closest to 1 July {year}; out GSD {wcfg.out_gsd_m} m", flush=True)
    client = F.make_client(a, wcfg, opener)
    releases = client.releases(root / "wayback" / "waybackconfig.json", refresh=a.refresh_releases)
    walks = F.Walks(root / "wayback_tiles" / f"tilemap_z{int(wcfg.walk_zoom)}.json")
    meta = None if a.no_metadata else F.Metadata(root / "wayback_tiles" / "metadata.json")
    dest = f"wayback_calib_{year}"
    per_tile, rel_hist = [], collections.Counter()
    t0 = time.time()
    try:
        for k, tile in enumerate(tiles):
            rel, _ = F.choose_release(walks, client, releases, tile, year, int(wcfg.walk_zoom))
            if rel is None:
                per_tile.append(dict(sat=tile["sat"], status="no_data"))
                continue
            r = F.fetch_tile(client, rel, tile, year, wcfg, root, (0.0, 0.0), dest_name=dest, metadata=meta,
                             attribution=str(wcfg.attribution), calibration=None)
            if r["status"] == "missing":
                per_tile.append(dict(sat=tile["sat"], status="missing", release=rel.num))
                continue
            wb = cv2.imread(r["path"], cv2.IMREAD_COLOR)
            vg = cv2.imread(str(root / city / "satellite" / tile["sat"]), cv2.IMREAD_COLOR)
            if wb is None or vg is None:
                per_tile.append(dict(sat=tile["sat"], status="unreadable", release=rel.num))
                continue
            vg = W.vigor_at_out_gsd(vg, CITY_RES[city], float(wcfg.out_gsd_m))
            if vg.shape != wb.shape:
                wb = cv2.resize(wb, (vg.shape[1], vg.shape[0]), interpolation=cv2.INTER_LINEAR)
            dx, dy, resp = W.phase_correlation(vg, wb, upsample=int(a.upsample))
            rel_hist[(rel.date.isoformat(), rel.num)] += 1
            per_tile.append(dict(sat=tile["sat"], status="ok", release=rel.num, release_date=rel.date.isoformat(),
                                 dx_px=dx, dy_px=dy, response=resp))
            if (k + 1) % 25 == 0 or k + 1 == len(tiles):
                ok = [t for t in per_tile if t["status"] == "ok"]
                if ok:
                    m = np.median([[t["dx_px"], t["dy_px"]] for t in ok], 0)
                    print(f"  {k + 1}/{len(tiles)}  running median offset ({m[0]:+.2f}, {m[1]:+.2f}) px over {len(ok)}  "
                          f"requests {client.n_requests}  {time.time() - t0:.0f} s", flush=True)
    finally:
        walks.save()
        if meta is not None:
            meta.save()
    ok = [t for t in per_tile if t["status"] == "ok"]
    gsd = W.window_geometry(tiles[0]["lat"], tiles[0]["lon"], CITY_RES[city], float(wcfg.out_gsd_m))["gsd"] if tiles else float(wcfg.out_gsd_m)
    agg = aggregate([[t["dx_px"], t["dy_px"]] for t in ok], [t["response"] for t in ok], gsd,
                    float(wcfg.calib_min_response), float(wcfg.calib_max_dev_px), float(a.gate if a.gate is not None else wcfg.calib_gate_m))
    out = dict(city=city, year=year, split=a.split, seed=a.seed, draw_tiles=len(tiles), out_gsd_m=float(wcfg.out_gsd_m),
               releases={f"{d} (#{n})": c for (d, n), c in sorted(rel_hist.items(), reverse=True)},
               method="phase correlation, Hann window, upsampled DFT (1/%d px); median over tiles; residual = median "
                      "|offset_i - median| in metres" % int(a.upsample),
               convention="content at VIGOR px (u, v) is at Wayback px (u + dx, v + dy); the fetcher samples the "
                          "release at (u + dx, v + dy) for output px (u, v)",
               calib_dir=str(root / city / dest), **agg, per_tile=per_tile,
               written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    path = W.calibration_path(root, city)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    if agg["offset_px"] is None:
        print(f"{city}: no usable tile (responses below {wcfg.calib_min_response}); wrote {path}; FAIL", flush=True)
        return out
    print(f"{city}: offset ({agg['offset_px'][0]:+.2f}, {agg['offset_px'][1]:+.2f}) px = "
          f"({agg['offset_m'][0]:+.3f} m east, {agg['offset_m'][1]:+.3f} m south) over {agg['n_used']} tiles "
          f"({agg['n_low_response']} low response, {agg['n_outliers']} outliers of {agg['n']}); "
          f"residual {agg['residual_m']:.3f} m, robust sigma ({agg['robust_sigma_m'][0]:.3f}, {agg['robust_sigma_m'][1]:.3f}) m, "
          f"SE of the median {agg['se_m']:.3f} m; releases {out['releases']}", flush=True)
    print(f"{city}: gate residual < {agg['gate_m']} m: {'PASS' if agg['pass'] else 'FAIL'}   wrote {path}", flush=True)
    return out


def build_parser():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--city", required=True)
    ap.add_argument("--split", default="samearea", choices=("crossarea", "samearea"))
    ap.add_argument("--year", type=int, default=None, help="default cfg.wayback.calib_year")
    ap.add_argument("--tiles", type=int, default=None, help="distinct tiles used (default cfg.wayback.calib_tiles)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", type=float, default=None, help="metres (default cfg.wayback.calib_gate_m)")
    ap.add_argument("--upsample", type=int, default=20, help="sub-pixel factor of the phase correlation")
    ap.add_argument("--no-metadata", action="store_true")
    ap.add_argument("--refresh-releases", action="store_true")
    return ap


def main(argv=None, opener=None):
    a = build_parser().parse_args(argv)
    return calibrate(a, C.load(a.config), opener)


if __name__ == "__main__":
    main()
