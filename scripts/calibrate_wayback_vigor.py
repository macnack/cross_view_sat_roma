"""Georeferencing calibration VIGOR tile -> Esri Wayback window (task 05, deliverable 2).

  make wayback-calib CITY=Chicago                     # 150 tiles, the release closest to 1 July wayback.calib_year
  make wayback-calib CITY=Chicago WAYBACK_ARGS="--year 2021 --tiles 150 --split samearea"
  make wayback-calib CITY=Chicago WAYBACK_ARGS="--pairs-dir data/vigor"   # offline: re-measure the windows on disk

Google's tile centres and Esri's tiles disagree by a constant few decimetres per city. On `wayback.calib_tiles`
distinct tiles of the city (a seeded choice among the unique tiles of a 4x larger VigorPairs test-list draw, seed 0;
several panoramas share a tile) the window of the version whose capture date is closest to 1 July
`wayback.calib_year` (the VIGOR capture years, 2020-2021; `wayback.select_by`, the fetcher's rule) is fetched
UNCALIBRATED into <root>/<City>/wayback_calib_<year>/ (offset 0, sidecars as for any fetch;
scripts/fetch_wayback_vigor.fetch_tile; `wayback.workers` tiles at a time under the global rate limit) and measured
against the VIGOR tile resampled to the same raster (`vigor_at_out_gsd` = VigorPairs' resampling) by
`bevloc.data.wayback.calib_match` (module doc "Cross-source calibration": VIGOR blurred to the Wayback source
resolution, gradient domain, band-limited phase correlation constrained to wayback.calib_max_shift_m, 20x upsampled
DFT). Per tile: (dx, dy) — the content at VIGOR pixel (u, v) sits at Wayback pixel (u + dx, v + dy) — the response
and the peak-to-sidelobe ratio (PSR). Aggregate (`aggregate`): tiles with PSR < wayback.calib_min_psr are dropped;
the median offset over the rest; tiles farther than wayback.calib_max_dev_px from it are outliers (dropped, counted,
the median recomputed). Reported: the median offset in px and metres, the residual = median over the kept tiles of
|offset_i - median| in metres (the misalignment a constant offset leaves), the per-axis IQR, the fraction of the
PSR-accepted tiles within 0.5 m of the median, the robust sigma (1.4826 MAD per axis), the standard error of the
median and the PSR percentiles over all tiles. Gate: residual < wayback.calib_gate_m AND n_used >=
wayback.calib_min_tiles (30) -> PASS, else FAIL (printed and stored). The fetcher applies a PASS offset; a FAIL is
not applied unless asked (scripts/fetch_wayback_vigor.py --apply-failed-calibration).
--pairs-dir DIR: offline — every window under DIR/<City>/wayback_calib_<year>/ (sorted, first --tiles) against
DIR/<City>/satellite/, no network and no draw; writes DIR/<City>/wayback_calibration.json.
Output: <root>/<City>/wayback_calibration.json (offset_px, offset_m, gsd_m, residual_m, iqr_m, frac_within_0_5m,
psr_percentiles, pass, release(s), per-tile offsets / responses / PSRs / blur sigma / source resolution).
Real pairs, 2026-09-28 (docs/decisions.md): New York PASS (+0.50 m E, +0.21 m S, residual 0.16 m, 39 tiles);
Chicago, San Francisco, Seattle FAIL (4-6 tiles of 150 above PSR 7; relief displacement of the off-nadir satellite
scenes, not a translation, is what separates the sources there).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
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


def source_res_of(side):
    """The Wayback source resolution (m) of a window sidecar: the sensor's SRC_RES from the metadata layer when it
    is known and coarser than the tile GSD (WorldView-2 = 0.5 m served on 0.22 m zoom-19 tiles), else the tile GSD."""
    gsd = float(side.get("source_gsd_m") or 0.0)
    try:
        res = float(((side.get("capture") or {}).get("SRC_RES")))
    except (TypeError, ValueError):
        res = 0.0
    return max(gsd, res if math.isfinite(res) else 0.0)


def measure_pair(vg_bgr, wb_bgr, side, city, wcfg, upsample=20):
    """Per-tile calibration measurement (module doc): the VIGOR tile at the output GSD, blurred to the Wayback source
    resolution, both through `calib_preprocess(wayback.calib_domain)`, phase correlation constrained to
    wayback.calib_max_shift_m with the peak-to-sidelobe ratio. Returns dict(dx_px, dy_px, response, psr,
    blur_sigma_px, source_res_m)."""
    out_gsd = float(wcfg.out_gsd_m)
    vg = W.vigor_at_out_gsd(vg_bgr, CITY_RES[city], out_gsd)
    wb = wb_bgr
    if vg.shape != wb.shape:
        wb = cv2.resize(wb, (vg.shape[1], vg.shape[0]), interpolation=cv2.INTER_LINEAR)
    src = source_res_of(side or {})
    sigma = W.calib_blur_sigma(src, CITY_RES[city], out_gsd, float(getattr(wcfg, "calib_blur_k", 0.5)))
    max_m = getattr(wcfg, "calib_max_shift_m", None)
    r = W.calib_match(vg, wb, str(getattr(wcfg, "calib_domain", "gradient")), sigma,
                      None if max_m is None else float(max_m) / out_gsd, int(upsample),
                      float(getattr(wcfg, "calib_psr_exclude_px", 5)), bool(getattr(wcfg, "calib_band", True)))
    return dict(dx_px=r["dx"], dy_px=r["dy"], response=r["response"], psr=r["psr"], blur_sigma_px=sigma,
                source_res_m=src)


def aggregate(offsets, psrs, gsd, min_psr, max_dev_px, gate_m, min_tiles=30):
    """The calibration numbers from per-tile (dx, dy) and peak-to-sidelobe ratios (module doc). Tiles with PSR below
    `min_psr` are dropped (their peak is indistinguishable from the correlation noise); the median over the rest; tiles
    farther than `max_dev_px` from it are outliers (dropped, counted, the median recomputed). Reported: the median
    offset, the residual = median |offset_i - median| (m), the robust sigma (1.4826 MAD per axis, m), the standard
    error of the median, the per-axis inter-quartile range (m) and the fraction of the PSR-accepted tiles within
    0.5 m of the median. PASS = residual < gate_m AND n_used >= min_tiles."""
    off = np.asarray(offsets, np.float64).reshape(-1, 2)
    psr = np.asarray(psrs, np.float64).reshape(-1)
    ok = psr >= float(min_psr)
    n_low = int((~ok).sum())
    base = dict(n=int(len(off)), n_low_psr=n_low, gsd_m=float(gsd), gate_m=float(gate_m), min_tiles=int(min_tiles),
                min_psr=float(min_psr), psr_percentiles={f"p{q}": float(np.percentile(psr, q)) for q in (10, 50, 90, 99)}
                if len(psr) else {})
    if not ok.any():
        return dict(base, n_used=0, n_outliers=0, offset_px=None, offset_m=None, residual_m=None, robust_sigma_m=None,
                    se_m=None, iqr_m=None, frac_within_0_5m=None, **{"pass": False})
    med = np.median(off[ok], 0)
    dev = np.linalg.norm(off - med, axis=1)
    keep = ok & (dev <= float(max_dev_px))
    n_out = int((ok & ~keep).sum())
    if keep.any():
        med = np.median(off[keep], 0)
    else:                                                          # no two accepted tiles agree: keep them all
        keep, n_out = ok, 0
    dev_k = np.linalg.norm(off[keep] - med, axis=1)
    mad = np.median(np.abs(off[keep] - med), 0)
    sigma = 1.4826 * mad * gsd
    n_used = int(keep.sum())
    residual = float(np.median(dev_k) * gsd)
    se = float(np.linalg.norm(sigma) * 1.2533 / np.sqrt(max(n_used, 1)))
    q75, q25 = np.percentile(off[keep], [75, 25], axis=0)
    within = float(np.mean(np.linalg.norm(off[ok] - med, axis=1) * gsd <= 0.5))
    return dict(base, n_used=n_used, n_outliers=n_out,
                offset_px=[float(med[0]), float(med[1])], offset_m=[float(med[0] * gsd), float(med[1] * gsd)],
                residual_m=residual, robust_sigma_m=[float(sigma[0]), float(sigma[1])], se_m=se,
                iqr_m=[float((q75[0] - q25[0]) * gsd), float((q75[1] - q25[1]) * gsd)], frac_within_0_5m=within,
                **{"pass": bool(residual < float(gate_m) and n_used >= int(min_tiles))})


def _method(wcfg, upsample):
    return (f"{getattr(wcfg, 'calib_domain', 'gradient')} domain (VIGOR blurred to the Wayback source resolution, "
            f"k {getattr(wcfg, 'calib_blur_k', 0.5)}), Hann window, phase correlation "
            f"{'band-limited to the same sigma, ' if getattr(wcfg, 'calib_band', True) else ''}constrained to "
            f"{getattr(wcfg, 'calib_max_shift_m', None)} m, upsampled DFT (1/{int(upsample)} px); tiles with PSR < "
            f"{wcfg.calib_min_psr} dropped; median over tiles; residual = median |offset_i - median| in metres")


def finish(city, root, year, per_tile, gsd, wcfg, a, extra):
    """Aggregate, print and write <root>/<City>/wayback_calibration.json."""
    ok = [t for t in per_tile if t["status"] == "ok"]
    min_psr = float(a.min_psr if a.min_psr is not None else wcfg.calib_min_psr)
    min_tiles = int(a.min_tiles if a.min_tiles is not None else getattr(wcfg, "calib_min_tiles", 30))
    agg = aggregate([[t["dx_px"], t["dy_px"]] for t in ok], [t["psr"] for t in ok], gsd, min_psr,
                    float(wcfg.calib_max_dev_px), float(a.gate if a.gate is not None else wcfg.calib_gate_m), min_tiles)
    for t in ok:
        t["accepted"] = bool(t["psr"] >= min_psr)
    out = dict(city=city, year=year, out_gsd_m=float(wcfg.out_gsd_m), **extra,
               domain=str(getattr(wcfg, "calib_domain", "gradient")), blur_k=float(getattr(wcfg, "calib_blur_k", 0.5)),
               max_shift_m=getattr(wcfg, "calib_max_shift_m", None), method=_method(wcfg, a.upsample),
               convention="content at VIGOR px (u, v) is at Wayback px (u + dx, v + dy); the fetcher samples the "
                          "release at (u + dx, v + dy) for output px (u, v)",
               calib_dir=str(root / city / f"wayback_calib_{year}"), **agg, per_tile=per_tile,
               written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    path = W.calibration_path(root, city)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    pp = agg["psr_percentiles"]
    if pp:
        print(f"{city}: PSR over {agg['n']} tiles p10 {pp['p10']:.1f} / p50 {pp['p50']:.1f} / p90 {pp['p90']:.1f} / "
              f"p99 {pp['p99']:.1f}; accepted (PSR >= {min_psr}) {agg['n'] - agg['n_low_psr']}", flush=True)
    if agg["offset_px"] is None:
        print(f"{city}: no usable tile (every PSR below {min_psr}); wrote {path}; FAIL", flush=True)
        return out
    print(f"{city}: offset ({agg['offset_px'][0]:+.2f}, {agg['offset_px'][1]:+.2f}) px = "
          f"({agg['offset_m'][0]:+.3f} m east, {agg['offset_m'][1]:+.3f} m south) over {agg['n_used']} tiles "
          f"({agg['n_low_psr']} low PSR, {agg['n_outliers']} outliers of {agg['n']}); residual {agg['residual_m']:.3f} m, "
          f"IQR ({agg['iqr_m'][0]:.3f}, {agg['iqr_m'][1]:.3f}) m, {100 * agg['frac_within_0_5m']:.0f}% of the accepted "
          f"tiles within 0.5 m, robust sigma ({agg['robust_sigma_m'][0]:.3f}, {agg['robust_sigma_m'][1]:.3f}) m, "
          f"SE of the median {agg['se_m']:.3f} m", flush=True)
    print(f"{city}: gate residual < {agg['gate_m']} m and >= {min_tiles} usable tiles: "
          f"{'PASS' if agg['pass'] else 'FAIL'}   wrote {path}", flush=True)
    return out


def calibrate_offline(a, cfg):
    """--pairs-dir: the windows already written under <pairs-dir>/<City>/wayback_calib_<year>/ (with sidecars) against
    <pairs-dir>/<City>/satellite/; no network, no draw (every window there, sorted by name, first --tiles)."""
    F = _fetch_module()
    wcfg = F.wayback_cfg(cfg)
    year = int(a.year if a.year is not None else wcfg.calib_year)
    root, city = Path(a.pairs_dir), a.city
    d = root / city / f"wayback_calib_{year}"
    pngs = sorted(p for p in d.glob("*.png") if not p.name.endswith(".part.png"))
    if a.tiles:
        pngs = pngs[: int(a.tiles)]
    print(f"{city}: offline calibration on {len(pngs)} windows in {d}", flush=True)
    per_tile, gsd = [], float(wcfg.out_gsd_m)
    for p in pngs:
        side_p = W.sidecar_path(p)
        side = json.loads(side_p.read_text()) if side_p.exists() else {}
        wb = cv2.imread(str(p), cv2.IMREAD_COLOR)
        vg = cv2.imread(str(root / city / "satellite" / p.name), cv2.IMREAD_COLOR)
        if wb is None or vg is None:
            per_tile.append(dict(sat=p.name, status="unreadable"))
            continue
        lat, lon = W.latlon_of_sat(p.name)
        gsd = W.window_geometry(lat, lon, CITY_RES[city], float(wcfg.out_gsd_m))["gsd"]
        m = measure_pair(vg, wb, side, city, wcfg, a.upsample)
        per_tile.append(dict(sat=p.name, status="ok", release=side.get("release"), capture_date=side.get("capture_date"),
                             **m))
    return finish(city, root, year, per_tile, gsd, wcfg, a, dict(split=None, seed=None, draw_tiles=len(pngs),
                                                                  source="offline --pairs-dir " + str(root)))


def calibrate(a, cfg, opener=None):
    if a.pairs_dir:
        return calibrate_offline(a, cfg)
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
    walks = F.Walks(root / "wayback_tiles" / f"tilemap_z{int(wcfg.walk_zoom)}.json", newest=releases[-1].num)
    select_by = F.select_by_of(a, wcfg)
    meta = None if (a.no_metadata and select_by != "capture") else F.Metadata(root / "wayback_tiles" / "metadata.json")
    print(f"release selected by {select_by} date (closest to 1 July {year}; ties -> newer publication)", flush=True)
    dest = f"wayback_calib_{year}"
    workers = max(1, int(a.workers if a.workers is not None else getattr(wcfg, "workers", 1) or 1))

    def one(tile):
        rel, _, sel = F.choose_release(walks, client, releases, tile, year, int(wcfg.walk_zoom), meta, select_by)
        if rel is None:
            return dict(sat=tile["sat"], status="no_data"), None, None
        r = F.fetch_tile(client, rel, tile, year, wcfg, root, (0.0, 0.0), dest_name=dest, metadata=meta,
                         attribution=str(wcfg.attribution), calibration=None, selection=sel)
        if r["status"] == "missing":
            return dict(sat=tile["sat"], status="missing", release=rel.num), None, None
        wb = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        vg = cv2.imread(str(root / city / "satellite" / tile["sat"]), cv2.IMREAD_COLOR)
        if wb is None or vg is None:
            return dict(sat=tile["sat"], status="unreadable", release=rel.num), None, None
        side_p = W.sidecar_path(r["path"])
        side = json.loads(side_p.read_text()) if side_p.exists() else {}
        m = measure_pair(vg, wb, side, city, wcfg, a.upsample)
        return (dict(sat=tile["sat"], status="ok", release=rel.num, release_date=rel.date.isoformat(),
                     capture_date=sel["capture_date"], **m), rel, sel)

    per_tile, rel_hist, cap_hist, n_fallback = [], collections.Counter(), collections.Counter(), 0
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for k, (t, rel, sel) in enumerate(pool.map(one, tiles)):
                per_tile.append(t)
                if rel is not None:
                    rel_hist[(rel.date.isoformat(), rel.num)] += 1
                    cap_hist[sel["capture_date"][:4] if sel["capture_date"] else "unknown"] += 1
                    n_fallback += int(bool(sel["fallback"]))
                if (k + 1) % 25 == 0 or k + 1 == len(tiles):
                    ok = [q for q in per_tile if q["status"] == "ok"]
                    if ok:
                        print(f"  {k + 1}/{len(tiles)}  median PSR {np.median([q['psr'] for q in ok]):.1f} over "
                              f"{len(ok)}  requests {client.n_requests}  {time.time() - t0:.0f} s", flush=True)
    finally:
        walks.save()
        if meta is not None:
            meta.save()
    gsd = W.window_geometry(tiles[0]["lat"], tiles[0]["lon"], CITY_RES[city], float(wcfg.out_gsd_m))["gsd"] if tiles else float(wcfg.out_gsd_m)
    extra = dict(split=a.split, seed=a.seed, draw_tiles=len(tiles),
                 releases={f"{d} (#{n})": c for (d, n), c in sorted(rel_hist.items(), reverse=True)},
                 select_by=select_by, capture_years=dict(sorted(cap_hist.items(), reverse=True)),
                 n_fallback_publication=int(n_fallback))
    out = finish(city, root, year, per_tile, gsd, wcfg, a, extra)
    print(f"{city}: releases {out['releases']}; capture years {out['capture_years']}"
          + (f"; {n_fallback} tile(s) chosen on the publication date (no capture date)" if n_fallback else ""), flush=True)
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
    ap.add_argument("--select-by", default=None, choices=W.SELECT_BY,
                    help="the release by capture (metadata identify per version) or publication date "
                         "(default cfg.wayback.select_by); the fetcher uses the same rule")
    ap.add_argument("--no-metadata", action="store_true", help="publication selection only: skip the identify")
    ap.add_argument("--refresh-releases", action="store_true")
    ap.add_argument("--pairs-dir", default=None,
                    help="offline: calibrate the windows already under <DIR>/<City>/wayback_calib_<year>/ against "
                         "<DIR>/<City>/satellite/ (no network, no draw; --tiles = the first N by name)")
    ap.add_argument("--min-psr", type=float, default=None, help="default cfg.wayback.calib_min_psr")
    ap.add_argument("--min-tiles", type=int, default=None,
                    help="PASS needs this many usable tiles (default cfg.wayback.calib_min_tiles)")
    ap.add_argument("--workers", type=int, default=None, help="default cfg.wayback.workers")
    return ap


def main(argv=None, opener=None):
    a = build_parser().parse_args(argv)
    return calibrate(a, C.load(a.config), opener)


if __name__ == "__main__":
    main()
