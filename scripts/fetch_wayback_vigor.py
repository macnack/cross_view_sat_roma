"""Esri World Imagery Wayback windows of the VIGOR tiles of an evaluation draw (task 05, deliverable 1).

  make wayback-fetch SPLIT=samearea CITIES=Chicago LIMIT=3000 YEARS="2025 2021 2019"
  make wayback-fetch ... WAYBACK_ARGS="--dry-run"                        # releases per tile + counts, no imagery
  make wayback-fetch ... WAYBACK_ARGS="--draw calib --val-samples 400"   # eval_vigor.py --calib draw (train lists)
  make wayback-fetch CITIES=Chicago WAYBACK_ARGS="--all"                 # every tile of the city's satellite_list

The draw is exactly eval_vigor.py's: VigorPairs(root, cfg, cities, split, train, limit, stride) with seed 0 on the
test lists (--draw test), or eval_vigor.calib_split on the train lists (--draw calib: --val-frac / --val-samples as
the checkpoint was trained, or --ckpt to read them), reduced to its unique tiles (several panoramas share a tile).
Per tile and year: the distinct imagery versions at the tile centre (`WaybackClient.versions_at`, cached in
<root>/wayback_tiles/tilemap_z<walk_zoom>.json), the release closest to 1 July of the year among them
(`pick_release`; ties -> newer), its tiles at the finest zoom that has the centre tile (`fetch_window`; the raw tiles
are cached under <root>/wayback_tiles/<release>/<z>/), the mosaic rendered into the tile's frame at
wayback.out_gsd_m (bevloc.data.wayback module doc) with the city's calibration offset (make wayback-calib,
<root>/<City>/wayback_calibration.json; none = zero offset, recorded as such) and written as
<root>/<City>/wayback_<year>/<sat_name> plus the JSON sidecar <stem>.json (release number and date, capture date /
sensor / resolution from the metadata layer, zoom, source GSD, tiles and the sha256 of their bytes, offset applied,
attribution). Resumable: a tile whose sidecar records the same release, offset and width is skipped; a different
offset (a new calibration) re-renders from the tile cache without downloads; --force re-renders everything.
--dry-run: the version walks (small JSON requests) and the choice per year, counts and volume estimates, nothing
written. The release list is pinned at <root>/wayback/waybackconfig.json on first use (--refresh-releases updates it,
which can change which release is "closest" for recent years).
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
from bevloc.data.vigor import CITY_RES, VigorPairs, split_cities
from bevloc.data import wayback as W

TILE_BYTES_EST = 15_000          # observed zoom-19 JPEG tiles over the four cities: 7-22 kB
PNG_BYTES_EST = 500_000          # a ~570 px RGB PNG of imagery


def wayback_cfg(cfg):
    """The `wayback:` block: the run's config, else configs/default.yaml (the VIGOR configs predate the block)."""
    return getattr(cfg, "wayback", None) or C.load().wayback


def _eval_vigor():
    spec = importlib.util.spec_from_file_location("bevloc_scripts_eval_vigor_for_wayback",
                                                  Path(__file__).resolve().parent / "eval_vigor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draw_labels(a, cfg):
    """The labels of the requested draw (eval_vigor.py's rules) or, with --all, one pseudo-label per tile of
    satellite_list.txt. Returns (labels, description)."""
    cities = a.cities or split_cities(a.split, a.draw == "calib" or a.train_split)
    if a.all:
        from bevloc.data.vigor import find_label_root
        lr = find_label_root(a.root)
        labs = []
        for city in cities:
            for name in (lr / city / "satellite_list.txt").read_text().split():
                labs.append(dict(city=city, pano="", sat=name, dy=0.0, dx=0.0))
        return labs, f"all tiles of {cities}"
    if a.draw == "calib":
        ev = _eval_vigor()
        ds = VigorPairs(a.root, cfg, cities=cities, split=a.split, train=True)
        if a.ckpt:
            import torch
            train_meta = torch.load(a.ckpt, map_location="cpu", weights_only=False).get("train")
            assume = None
        else:
            if a.val_samples is None:
                raise SystemExit("--draw calib needs --ckpt or --val-samples (with --val-frac) as the checkpoint was trained")
            train_meta, assume = None, dict(val_frac=a.val_frac, val_samples=a.val_samples, cities=list(cities))
        held, info = ev.calib_split(len(ds.labels), train_meta, a.val_frac, cities, a.limit, assume)
        return [ds.labels[i] for i in held], f"calib draw {info['n']} of {cities} ({info['source']})"
    ds = VigorPairs(a.root, cfg, cities=cities, split=a.split, train=a.train_split, limit=a.limit, stride=a.stride,
                    seed=a.seed)
    return ds.labels, f"{'train' if a.train_split else 'test'} draw {len(ds.labels)} of {cities} (split {a.split}, limit {a.limit}, seed {a.seed})"


def unique_tiles(labels):
    """One dict(city, sat, lat, lon) per distinct tile, in first-seen order."""
    seen, out = set(), []
    for lab in labels:
        key = (lab["city"], lab["sat"])
        if key in seen:
            continue
        seen.add(key)
        lat, lon = W.latlon_of_sat(lab["sat"])
        out.append(dict(city=lab["city"], sat=lab["sat"], lat=lat, lon=lon))
    return out


class Walks:
    """Version walks per (z, x, y), persisted as JSON: {"z/x/y": [owner release numbers, newest first]}."""

    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.dirty = 0

    def get(self, client, releases, z, x, y):
        key = f"{z}/{x}/{y}"
        if key not in self.d:
            self.d[key] = [r.num for r in client.versions_at(releases, z, x, y)]
            self.dirty += 1
            if self.dirty % 50 == 0:
                self.save()
        by_num = {r.num: r for r in releases}
        return [by_num[n] for n in self.d[key] if n in by_num]

    def save(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d))
            tmp.replace(self.path)
            self.dirty = 0


class Metadata:
    """Capture metadata per (release, walk tile), persisted as JSON."""

    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.dirty = 0

    def get(self, client, release, lat, lon, z):
        x, y = W.tile_xy(*W.latlon_to_merc_px(lat, lon, z))
        key = f"{release.num}/{z}/{x}/{y}"
        if key not in self.d:
            self.d[key] = client.metadata(release, lat, lon)
            self.dirty += 1
            if self.dirty % 50 == 0:
                self.save()
        return self.d[key]

    def save(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d))
            tmp.replace(self.path)
            self.dirty = 0


def choose_release(walks, client, releases, tile, year, walk_zoom):
    """(release closest to `year` among the distinct versions at the tile centre, the versions) — (None, []) when the
    service has nothing there."""
    x, y = W.tile_xy(*W.latlon_to_merc_px(tile["lat"], tile["lon"], walk_zoom))
    versions = walks.get(client, releases, walk_zoom, x, y)
    return W.pick_release(versions, year), versions


def fetch_tile(client, release, tile, year, wcfg, root, offset_px, dest_name=None, metadata=None, force=False,
               attribution=W.ATTRIBUTION, calibration=None):
    """Fetch, render and write one tile for one year. Returns dict(status = written | skipped | missing, path, ...).
    dest_name: the folder under <root>/<City> (default wayback_<year>)."""
    city = tile["city"]
    out_dir = Path(root) / city / (dest_name or f"wayback_{year}")
    png = out_dir / tile["sat"]
    side = W.sidecar_path(png)
    out_gsd = float(wcfg.out_gsd_m)
    geom = W.window_geometry(tile["lat"], tile["lon"], CITY_RES[city], out_gsd)
    off_m = [float(offset_px[0]) * geom["gsd"], float(offset_px[1]) * geom["gsd"]]
    if png.exists() and side.exists() and not force:
        try:
            old = json.loads(side.read_text())
        except json.JSONDecodeError:
            old = {}
        if (old.get("release") == release.num and old.get("width") == geom["width"]
                and np.allclose(old.get("offset_m", [np.nan, np.nan]), off_m, atol=1e-6)):
            return dict(status="skipped", path=str(png), release=release.num)
    img, info = W.fetch_window(client, release, tile["lat"], tile["lon"], CITY_RES[city], out_gsd,
                               zoom_prefs=tuple(int(z) for z in wcfg.zoom_pref), offset_px=offset_px)
    if img is None:
        return dict(status="missing", path=str(png), release=release.num)
    meta = None
    if metadata is not None:
        try:
            meta = metadata.get(client, release, tile["lat"], tile["lon"], int(wcfg.walk_zoom))
        except RuntimeError as e:                                  # the metadata service is not essential
            meta = dict(error=str(e)[:200])
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = png.with_suffix(".part.png")
    cv2.imwrite(str(tmp), img)
    tmp.replace(png)
    side.write_text(json.dumps(dict(
        sat=tile["sat"], city=city, lat=tile["lat"], lon=tile["lon"], year_requested=int(year),
        **info, capture=meta, calibration=calibration, source="Esri World Imagery Wayback",
        attribution=attribution, fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        frame="Web Mercator zoom-20 px x CITY_RES (the VIGOR tile's frame), north up, east right; output pixel "
              "(u, v) samples the release at (u + dx, v + dy) of the uncalibrated window (offset_px)"), indent=1))
    return dict(status="written", path=str(png), release=release.num, zoom=info["zoom"], missing=info["missing_tiles"])


def build_parser():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--root", default=os.environ.get("VIGOR_DIR", "data/vigor"))
    ap.add_argument("--split", default="samearea", choices=("crossarea", "samearea"))
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="eval_vigor.py --limit (0 = all)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="the draw's seed (eval_vigor.py: 0)")
    ap.add_argument("--draw", default="test", choices=("test", "calib"))
    ap.add_argument("--train-split", action="store_true", help="eval_vigor.py --train-split")
    ap.add_argument("--val-frac", type=float, default=0.2, help="--draw calib: as eval_vigor.py")
    ap.add_argument("--val-samples", type=int, default=None, help="--draw calib without --ckpt: the training run's")
    ap.add_argument("--ckpt", default=None, help="--draw calib: read val_frac / val_samples from the checkpoint")
    ap.add_argument("--years", nargs="+", type=int, default=None, help="default cfg.wayback.years")
    ap.add_argument("--all", action="store_true", help="every tile of the cities' satellite_list.txt instead of a draw")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-render every tile (the tile cache still avoids downloads)")
    ap.add_argument("--no-metadata", action="store_true")
    ap.add_argument("--no-calibration", action="store_true", help="zero offset even when wayback_calibration.json exists")
    ap.add_argument("--dest-name", default=None, help="folder under <root>/<City> (default wayback_<year>)")
    ap.add_argument("--refresh-releases", action="store_true")
    ap.add_argument("--max-tiles", type=int, default=0, help="stop after this many tiles (smoke tests)")
    return ap


def make_client(a, wcfg, opener=None):
    root = Path(a.root)
    return W.WaybackClient(rate_hz=float(wcfg.rate_hz), user_agent=str(wcfg.user_agent), retries=int(wcfg.retries),
                           cache_dir=root / "wayback_tiles", opener=opener)


def main(argv=None, opener=None):
    a = build_parser().parse_args(argv)
    cfg = C.load(a.config)
    wcfg = wayback_cfg(cfg)
    years = a.years or [int(y) for y in wcfg.years]
    root = Path(a.root)
    labels, desc = draw_labels(a, cfg)
    tiles = unique_tiles(labels)
    if a.max_tiles:
        tiles = tiles[: a.max_tiles]
    print(f"{desc}: {len(labels)} labels, {len(tiles)} unique tiles; years {years}; root {root}", flush=True)
    client = make_client(a, wcfg, opener)
    releases = client.releases(root / "wayback" / "waybackconfig.json", refresh=a.refresh_releases)
    print(f"{len(releases)} releases {releases[0].date} .. {releases[-1].date}", flush=True)
    walk_zoom = int(wcfg.walk_zoom)
    walks = Walks(root / "wayback_tiles" / f"tilemap_z{walk_zoom}.json")
    meta = None if (a.no_metadata or not bool(wcfg.metadata)) else Metadata(root / "wayback_tiles" / "metadata.json")
    calib = {}
    for city in sorted({t["city"] for t in tiles}):
        c = None if a.no_calibration else W.load_calibration(root, city)
        calib[city] = c
        print(f"{city}: calibration " + ("none (offset 0)" if c is None else
                                          f"offset {c['offset_m'][0]:+.3f} m E, {c['offset_m'][1]:+.3f} m S "
                                          f"(residual {c['residual_m']:.3f} m, {'PASS' if c.get('pass') else 'FAIL'})"),
              flush=True)
    counts = collections.Counter()
    chosen = {y: collections.Counter() for y in years}
    n_versions = []
    t0 = time.time()
    try:
        for k, tile in enumerate(tiles):
            city = tile["city"]
            off = W.offset_px_for(calib[city], float(wcfg.out_gsd_m))
            for year in years:
                rel, versions = choose_release(walks, client, releases, tile, year, walk_zoom)
                if year == years[0]:
                    n_versions.append(len(versions))
                if rel is None:
                    counts["no_data"] += 1
                    continue
                chosen[year][(rel.date.isoformat(), rel.num)] += 1
                if a.dry_run:
                    counts["dry"] += 1
                    continue
                r = fetch_tile(client, rel, tile, year, wcfg, root, off, dest_name=a.dest_name, metadata=meta,
                               force=a.force, attribution=str(wcfg.attribution),
                               calibration=None if calib[city] is None else str(W.calibration_path(root, city)))
                counts[r["status"]] += 1
                if r["status"] == "written" and r.get("missing"):
                    counts["with_missing_tiles"] += 1
            if (k + 1) % 50 == 0 or k + 1 == len(tiles):
                print(f"  {k + 1}/{len(tiles)} tiles  {dict(counts)}  requests {client.n_requests}  "
                      f"{client.n_bytes / 1e6:.1f} MB  {time.time() - t0:.0f} s", flush=True)
    finally:
        walks.save()
        if meta is not None:
            meta.save()
    print(f"versions per tile centre: median {np.median(n_versions) if n_versions else 0:.0f}, "
          f"min {min(n_versions) if n_versions else 0}, max {max(n_versions) if n_versions else 0}", flush=True)
    for year in years:
        print(f"year {year}: releases chosen " + ", ".join(f"{d} (#{n}) x{c}" for (d, n), c in
                                                           sorted(chosen[year].items(), reverse=True)), flush=True)
    if a.dry_run:
        z = int(wcfg.zoom_pref[-1])
        n_tiles = 0
        for tile in tiles:
            g = W.window_geometry(tile["lat"], tile["lon"], CITY_RES[tile["city"]], float(wcfg.out_gsd_m))
            x0, x1, y0, y1 = W.tile_range(g, z)
            n_tiles += (x1 - x0 + 1) * (y1 - y0 + 1)
        per_year = n_tiles
        print(f"dry run: {len(tiles)} tiles x {len(years)} years -> ~{per_year * len(years)} XYZ tiles at zoom {z} "
              f"(~{per_year * len(years) * TILE_BYTES_EST / 1e6:.0f} MB; x4 where zoom {z + 1} exists), "
              f"~{len(tiles) * len(years) * PNG_BYTES_EST / 1e9:.1f} GB of PNGs, "
              f"~{len(tiles) * len(years)} metadata requests; walks made this run: {client.n_requests}", flush=True)
    print(f"done: {dict(counts)}; {client.n_requests} requests, {client.n_bytes / 1e6:.1f} MB", flush=True)
    return counts


if __name__ == "__main__":
    main()
