"""Esri World Imagery Wayback as a second reference source for the VIGOR tiles (task 05).

Service facts, verified with real requests on 2026-09-28 (scripts/fetch_wayback_vigor.py, `make wayback-fetch`):

* Release list: https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json — a JSON object keyed
  by release number (196 releases, 2014-02-20 .. 2026-08-05), each with `itemTitle` "World Imagery (Wayback
  YYYY-MM-DD)", `itemURL` = the tile template
  https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/{release}/{level}/{row}/{col}
  (Web Mercator, 256 px tiles, XYZ numbering: level = zoom, row = y, col = x), `metadataLayerUrl` (a MapServer with one
  layer per source resolution) and `layerIdentifier` (WB_YYYY_RNN). The release NUMBERS are not chronological; the
  date in the title is.
* Finest zoom: 19 over Chicago and Seattle (zoom 20 -> HTTP 404), 20 over New York and San Francisco (newest release).
  Zoom 19 is 0.30 m/px at the equator, 0.22 m/px at Chicago's latitude; the VIGOR footprint (640 px at zoom 20 =
  71 m) is ~320 px of zoom-19 imagery, ~640 px of zoom-20. Older releases may stop at a coarser zoom: the fetcher walks
  down `zoom_pref` until the centre tile exists.
* Which release actually changed a tile: releases share tile data — most releases point at an older release's tile
  for a given area. The undocumented but public `tilemap` endpoint
  https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tilemap/{release}/{z}/{y}/{x}
  returns {"data": [1], "select": [<owner release>], ...}: `select[0]` is the release whose imagery this release serves
  at that tile (absent = the release is its own owner; "data": [0] / "valid": false = nothing there). Walking newest ->
  owner -> the release just older than the owner -> ... lists the DISTINCT versions at one tile in as many requests as
  there are versions (`WaybackClient.versions_at`; this is what Esri's wayback-core does). Chicago 41.88 N 87.63 W:
  7 versions (2026-03-26, 2022-11-02, 2021-02-24, 2019-12-12, 2018-04-11, 2018-01-08, 2014-02-20); New York 11,
  Seattle 11, San Francisco 8 (2025-12-18, 2023-06-29, 2023-06-13, 2022-06-08, 2020-10-14, 2018-04-11, 2018-01-08,
  2014-02-20). Comparing tile bytes gives the same answer (identical `size` for shared tiles) at 20x the traffic.
* Capture date: the release date is the PUBLICATION date. The metadata MapServer's `identify` at a point
  (`WaybackClient.metadata`) returns SRC_DATE (capture, YYYYMMDD), SRC_RES (m), SRC_ACC (CE90, m), SRC_DESC (sensor)
  and NICE_DESC (vendor): Chicago's 2026-03-26 release carries WorldView-3 imagery captured 2025-04-24 at 0.31 m
  (Vantor/Maxar, CE90 8.5 m). Both dates go into the sidecar. Publication year != capture year, tile by tile: the
  release published 2025-03-27 carries imagery captured 2022-06-20 at the Chicago Water Tower (review, 2026-09-28).
  The year requested on the command line therefore selects by CAPTURE date (`wayback.select_by: capture`, the
  default; `pick_version`): over the distinct versions at the tile centre, one identify per version (cached), the
  version whose capture date is closest to 1 July of the year, ties -> the newer publication; a version without a
  usable SRC_DATE falls back to its publication date (counted). `select_by: publication` is the old rule
  (`pick_release`). Probe of the city centres (identify per version, 2026-09-28): the two rules agree for 2025 and
  2021 in Chicago and Seattle and for 2025 in San Francisco; they differ for 2021 in New York (publication ->
  2021-05-19 = 2020-03-21 imagery, capture -> 2025-04-24 = 2022-03-16) and San Francisco (2020-10-14 = 2019-09-24
  vs 2022-06-08 = 2021-09-05), and for "2019" everywhere but Seattle (the 2019-12-12 release shows 2018 imagery; the
  capture-closest 2019 version is usually a 2020-2021 release). Several distinct tile versions can share one capture
  date (re-processings: six releases 2022-11 .. 2026-02 at the Water Tower all show 2022-06-20 imagery).
* Terms (https://www.esri.com/en-us/legal/terms/full-master-agreement, Living Atlas): free for research with
  attribution "Esri, Maxar, Earthstar Geographics" (`ATTRIBUTION`); no redistribution of the tiles — they stay under
  data/ (never committed); the paper cites the source and the release dates. Requests carry a descriptive User-Agent
  (`wayback.user_agent`) and a rate limit (`wayback.rate_hz`).

Geometry. A VIGOR tile is a 640 px Google Static Maps image at zoom 20 centred on the (lat, lon) of its file name,
i.e. a Web Mercator raster; VIGOR's per-city resolution (CITY_RES, m per tile px) is the zoom-20 ground resolution at
the city's latitude, and `bevloc.data.vigor.VigorPairs` treats the tile as a metric, north-up, east-right raster of
640 * CITY_RES metres. The "local metric frame" of the tile is therefore Web Mercator pixels at zoom 20 scaled by
CITY_RES, and the Wayback window is built in exactly that frame: W = round(640 * CITY_RES / out_gsd) output pixels
covering the same 640 zoom-20 pixels around the same centre (`window_geometry`), sampled from the zoom-z mosaic by one
affine map (`render_window`, cv2.warpAffine, pixel centres at integer + 0.5 in Mercator pixel space). No datum or
projection resampling is involved: both sources are the same projection, so the map is a pure scale + translation and
the only free parameter is the constant offset between Google's and Esri's georeferencing, measured by
`scripts/calibrate_wayback_vigor.py` (`calib_match`, below) and applied through `offset_px`. The
file written for a tile has the SAME footprint (640 * CITY_RES m) as the VIGOR tile at a different pixel count, so
`VigorPairs` reads it unchanged (`ref_source: wayback_<year>`): its resampling scale is footprint / width / cell.

Sign conventions. `phase_correlation(a, b)` returns (dx, dy) such that the content at a's pixel (u, v) is found at
b's pixel (u + dx, v + dy). The calibration stores the offset of the Wayback window relative to the VIGOR tile in that
sense (a = VIGOR, b = Wayback, both at out_gsd), and the fetcher samples the Wayback mosaic at (u + dx, v + dy) for
output pixel (u, v), which puts the Wayback content where the VIGOR content is.

Cross-source calibration (`calib_match`; 2026-09-28). Whole-image intensity phase correlation fails on these pairs:
the Esri source is often much coarser than the Google tile (WorldView-2 0.5 m in Chicago / SF, GeoEye-1 0.46 m in
Seattle, served on 0.22 m zoom-19 tiles; New York is the 0.15 m NYS aerial ortho), cars, shadows and seasons differ,
and the whitened spectrum above the coarse source's cut-off is pure noise. Per pair: (1) the VIGOR tile at the output
GSD is blurred to the Wayback source resolution (`calib_blur_sigma`: sigma = k sqrt(src^2 - vigor^2) / out_gsd,
src = the sidecar's SRC_RES, else the tile GSD); (2) both go to the gradient-magnitude domain (`calib_preprocess`,
`wayback.calib_domain: gradient | intensity`); (3) Hann window after removing the window-weighted mean (a plain mean
leaves a spurious zero-shift peak); (4) phase correlation whose whitened cross-power spectrum is weighted by the same
Gaussian (`band_sigma_px`: only the common band votes; 0.25 -> 0.08 px on the synthetic cross-source test); (5) the
integer peak is searched only within `wayback.calib_max_shift_m` of zero (+2 px, so that a peak just outside is
found and flagged `edge` -> rejected, instead of one of its ringing lobes inside), refined by the upsampled DFT; (6) the
peak-to-sidelobe ratio PSR = (peak - mean) / std of the surface inside that disc, 5 px around the peak excluded, is
the acceptance statistic (`wayback.calib_min_psr`), not the absolute response (which is 0.01-0.1 on real pairs even
when right). Real pairs (150 per city, the release closest to 2021): unrelated-content PSRs are 3-6 (median 3.6-4.2
in every city); New York is bimodal with a matched mode at 7-35 -> default 7.0. New York PASSes (+0.50 m E, +0.21 m
S, residual 0.16 m over 39 tiles); Chicago / SF / Seattle do not: 4-6 tiles of 150 reach PSR 7, and even at PSR 5 the
accepted offsets scatter by 0.8-1.1 m (median |offset - median|). The imagery there is not related to the Google tile
by a translation: an off-nadir satellite scene orthorectified on the terrain displaces roofs and trees by
height x tan(off-nadir), so the averaged correlation surface over 150 tiles is a streak, not a peak (from ~+1 m to
~+3 m east in Chicago, south-east in SF): the ground and the roofs want different offsets.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
TILEMAP_URL = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tilemap/{release}/{z}/{y}/{x}"
ATTRIBUTION = "Esri, Maxar, Earthstar Geographics"
USER_AGENT = "cross_view_sat_roma/0.1 (academic research; maciej.krupka@gmail.com)"
VIGOR_ZOOM = 20              # VIGOR tiles: Google Static Maps, zoom 20
VIGOR_TILE_PX = 640          # ... 640 px
TILE_PX = 256                # Web Mercator XYZ tiles
EQUATOR_M_PER_PX_Z0 = 156543.03392  # Web Mercator ground resolution at zoom 0 (256 px tiles)
SOURCE_RE = re.compile(r"^(vigor|wayback_\d{4})$")


# ---- Web Mercator ----------------------------------------------------------------------------------------------

def latlon_to_merc_px(lat, lon, zoom):
    """Continuous Web Mercator pixel coordinates (x, y) at `zoom` (256 px tiles; pixel i covers [i, i + 1))."""
    n = TILE_PX * (2.0 ** zoom)
    phi = math.radians(float(lat))
    x = (float(lon) + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(phi) + 1.0 / math.cos(phi)) / math.pi) / 2.0 * n
    return x, y


def merc_px_to_latlon(x, y, zoom):
    """Inverse of `latlon_to_merc_px`."""
    n = TILE_PX * (2.0 ** zoom)
    lon = float(x) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * float(y) / n))))
    return lat, lon


def tile_xy(px, py):
    """XYZ tile indices (x, y) of a Mercator pixel coordinate."""
    return int(math.floor(px / TILE_PX)), int(math.floor(py / TILE_PX))


def ground_res_m(lat, zoom):
    """Web Mercator ground resolution (m per px) at `lat` and `zoom`."""
    return EQUATOR_M_PER_PX_Z0 * math.cos(math.radians(float(lat))) / (2.0 ** zoom)


def latlon_of_sat(name):
    """(lat, lon) of a VIGOR tile file name satellite_<lat>_<lon>.png."""
    m = re.match(r"satellite_(-?\d+\.\d+)_(-?\d+\.\d+)\.png$", Path(name).name)
    if m is None:
        raise ValueError(f"not a VIGOR tile name: {name}")
    return float(m.group(1)), float(m.group(2))


# ---- releases --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Release:
    num: int                 # release number (the {release} of the URLs; not chronological)
    date: _dt.date           # publication date from the title
    tile_url: str            # template with {level}/{row}/{col}
    metadata_url: str        # metadata MapServer (may be "")
    layer_id: str            # WB_YYYY_RNN

    def tile(self, z, x, y):
        return self.tile_url.format(level=int(z), row=int(y), col=int(x))


def parse_releases(config):
    """Releases of the waybackconfig.json dict, oldest first by date (ties by number)."""
    out = []
    for k, v in config.items():
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", v.get("itemTitle", ""))
        if m is None:
            continue
        out.append(Release(int(k), _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))),
                           v["itemURL"], v.get("metadataLayerUrl", "") or "", v.get("layerIdentifier", "") or ""))
    return sorted(out, key=lambda r: (r.date, r.num))


def pick_release(releases, year, month=7, day=1):
    """The release closest (in days) to `year`-month-day; ties -> the newer one. None when `releases` is empty."""
    if not releases:
        return None
    target = _dt.date(int(year), month, day)
    return min(releases, key=lambda r: (abs((r.date - target).days), -r.date.toordinal()))


SELECT_BY = ("capture", "publication")


def capture_date(meta):
    """The capture date of a `WaybackClient.metadata` dict (SRC_DATE, YYYYMMDD as int or str) as a date; None when
    absent, "Null" or malformed."""
    sd = (meta or {}).get("SRC_DATE")
    if sd is None:
        return None
    try:
        return _dt.datetime.strptime(str(int(str(sd).strip())), "%Y%m%d").date()
    except (TypeError, ValueError):
        return None


def pick_version(versions, year, captures=None, select_by="capture", month=7, day=1):
    """The version of `versions` (Release objects) for `year` under `select_by` (module doc). captures: {release
    number: capture date or None} for select_by "capture" (a version missing from it, or with None, falls back to
    its publication date). Returns (release or None, info dict: select_by, date used, capture_date, publication_date,
    fallback (the chosen version used its publication date), n_fallback (versions without a capture date))."""
    if select_by not in SELECT_BY:
        raise ValueError(f"select_by must be one of {SELECT_BY}, got {select_by!r}")
    if not versions:
        return None, dict(select_by=select_by, date=None, capture_date=None, publication_date=None, fallback=False,
                          n_fallback=0)
    target = _dt.date(int(year), month, day)
    caps = dict(captures or {})

    def used(r):
        c = caps.get(r.num) if select_by == "capture" else None
        return (c, False) if c is not None else (r.date, select_by == "capture")
    best = min(versions, key=lambda r: (abs((used(r)[0] - target).days), -r.date.toordinal()))
    d, fb = used(best)
    return best, dict(select_by=select_by, date=d.isoformat(), capture_date=None if caps.get(best.num) is None
                      else caps[best.num].isoformat(), publication_date=best.date.isoformat(), fallback=fb,
                      n_fallback=sum(used(r)[1] for r in versions))


# ---- HTTP client -----------------------------------------------------------------------------------------------

class WaybackClient:
    """Rate-limited, retrying HTTP access to the Wayback services with an on-disk tile cache.

    opener(url, headers) -> bytes, or raises urllib.error.HTTPError; tests inject a fake. cache_dir: tiles are kept
    at <cache_dir>/<release>/<z>/<x>_<y>.jpg (a 404 leaves an empty <x>_<y>.missing marker), so a re-render of the
    windows (new calibration) costs no requests.

    Thread-safe (the fetcher's worker pool shares one client): request STARTS are spaced by at least 1 / rate_hz
    across all threads (the wait happens under a lock, the request itself outside it, so up to `workers` requests are
    in flight but never more than rate_hz start per second); the counters are updated under the lock; a tile is
    written to a per-thread `.part` file and renamed, so two threads fetching the same tile both leave a complete
    file (the rename is atomic, the bytes identical)."""

    def __init__(self, rate_hz=10.0, user_agent=USER_AGENT, retries=3, timeout=30.0, cache_dir=None, opener=None):
        self.min_interval = 0.0 if not rate_hz else 1.0 / float(rate_hz)
        self.headers = {"User-Agent": user_agent, "Accept": "*/*"}
        self.retries = int(retries)
        self.timeout = float(timeout)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.opener = opener or self._urlopen
        self._last = 0.0
        self._lock = threading.Lock()
        self.n_requests = 0
        self.n_bytes = 0

    def _urlopen(self, url, headers):
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as f:
            return f.read()

    def get(self, url):
        """Bytes of `url`; None on 404. Retries (1, 2, 4 s) on other errors, then raises."""
        err = None
        for attempt in range(self.retries + 1):
            with self._lock:                                          # global rate limit over all threads
                wait = self._last + self.min_interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
                self.n_requests += 1
            try:
                b = self.opener(url, self.headers)
                with self._lock:
                    self.n_bytes += len(b)
                return b
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                err = e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                err = e
            if attempt < self.retries:
                time.sleep(2.0 ** attempt)
        raise RuntimeError(f"GET {url} failed after {self.retries + 1} attempts: {err}")

    def json(self, url):
        b = self.get(url)
        return None if b is None else json.loads(b.decode("utf-8"))

    def releases(self, cache_file=None, refresh=False):
        """`parse_releases` of the config, pinned to `cache_file` when given (re-downloaded only with refresh)."""
        if cache_file is not None and Path(cache_file).exists() and not refresh:
            return parse_releases(json.loads(Path(cache_file).read_text()))
        cfg = self.json(CONFIG_URL)
        if cfg is None:
            raise RuntimeError(f"{CONFIG_URL} returned 404")
        if cache_file is not None:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_file).write_text(json.dumps(cfg))
        return parse_releases(cfg)

    def tile(self, release, z, x, y):
        """Tile bytes (JPEG/PNG as served), None when the service has no tile there. Cached on disk."""
        if self.cache_dir is not None:
            d = self.cache_dir / str(release.num) / str(int(z))
            f, miss = d / f"{int(x)}_{int(y)}.jpg", d / f"{int(x)}_{int(y)}.missing"
            if f.exists():
                return f.read_bytes()
            if miss.exists():
                return None
        b = self.get(release.tile(z, x, y))
        if self.cache_dir is not None:
            d.mkdir(parents=True, exist_ok=True)
            if b is None:
                miss.touch()
            else:
                tmp = f.with_name(f"{f.stem}.{os.getpid()}.{threading.get_ident()}.part")
                tmp.write_bytes(b)
                tmp.replace(f)
        return b

    def tilemap(self, release, z, x, y):
        """Owner release number of (z, x, y) under `release`, or None when the release has no data there."""
        d = self.json(TILEMAP_URL.format(release=release.num, z=int(z), y=int(y), x=int(x)))
        if not d or not d.get("valid", True) or not (d.get("data") or [0])[0]:
            return None
        sel = d.get("select")
        return int(sel[0]) if sel else int(release.num)

    def versions_at(self, releases, z, x, y):
        """Distinct imagery versions at tile (z, x, y): the owner releases, newest first (module doc). One tilemap
        request per version."""
        order = list(releases)                                        # oldest -> newest
        by_num = {r.num: i for i, r in enumerate(order)}
        out = []
        i = len(order) - 1
        while i >= 0:
            owner = self.tilemap(order[i], z, x, y)
            if owner is None:
                break
            j = by_num.get(owner)
            if j is None or (out and j >= by_num[out[-1].num]):        # unknown owner / no progress: stop cleanly
                out.append(order[i])
                break
            out.append(order[j])
            i = j - 1
        return out

    def metadata(self, release, lat, lon):
        """Capture metadata of the finest source at (lat, lon) in `release` (identify on its metadata MapServer):
        dict(SRC_DATE, SRC_RES, SRC_ACC, SRC_DESC, NICE_DESC, layer) or None."""
        if not release.metadata_url:
            return None
        e = 0.005
        url = (f"{release.metadata_url}/identify?geometry={lon},{lat}&geometryType=esriGeometryPoint&sr=4326"
               f"&layers=all&tolerance=1&mapExtent={lon - e},{lat - e},{lon + e},{lat + e}"
               f"&imageDisplay=400,400,96&returnGeometry=false&f=json")
        d = self.json(url)
        res = (d or {}).get("results") or []
        if not res:
            return None
        r = min(res, key=lambda r: int(r.get("layerId", 99)))         # the finest layer with a feature
        a = r.get("attributes", {})
        return {k: a.get(k) for k in ("SRC_DATE", "SRC_RES", "SRC_ACC", "SRC_DESC", "NICE_DESC")} | {
            "layer": r.get("layerName")}

    def finest_zoom(self, release, lat, lon, prefs=(20, 19)):
        """The first zoom of `prefs` whose tile at (lat, lon) exists in `release`; None when none does."""
        for z in prefs:
            x, y = tile_xy(*latlon_to_merc_px(lat, lon, z))
            if self.tile(release, z, x, y) is not None:
                return int(z)
        return None


# ---- window geometry, mosaic, rendering ------------------------------------------------------------------------

def window_geometry(lat, lon, city_res, out_gsd):
    """Output raster of a VIGOR footprint at ~out_gsd m/px: dict(width, k (zoom-20 px per output px), cx, cy (the tile
    centre in zoom-20 Mercator px), gsd (exact m per output px), footprint_m)."""
    footprint = VIGOR_TILE_PX * float(city_res)
    width = max(1, int(round(footprint / float(out_gsd))))
    cx, cy = latlon_to_merc_px(lat, lon, VIGOR_ZOOM)
    return dict(width=width, k=VIGOR_TILE_PX / float(width), cx=cx, cy=cy, gsd=footprint / width, footprint_m=footprint)


def output_to_merc(u, v, geom, zoom, offset_px=(0.0, 0.0)):
    """Output pixel (u, v) (centre convention) -> continuous Mercator pixel coordinates at `zoom`, with the calibration
    offset (content sampled at output (u + dx, v + dy))."""
    c = (geom["width"] - 1) / 2.0
    s = 2.0 ** (zoom - VIGOR_ZOOM)
    x = (geom["cx"] + (u + offset_px[0] - c) * geom["k"]) * s
    y = (geom["cy"] + (v + offset_px[1] - c) * geom["k"]) * s
    return x, y


def merc_to_output(x, y, geom, zoom, offset_px=(0.0, 0.0)):
    """Inverse of `output_to_merc`."""
    c = (geom["width"] - 1) / 2.0
    s = 2.0 ** (zoom - VIGOR_ZOOM)
    u = (x / s - geom["cx"]) / geom["k"] + c - offset_px[0]
    v = (y / s - geom["cy"]) / geom["k"] + c - offset_px[1]
    return u, v


def tile_range(geom, zoom, offset_px=(0.0, 0.0), margin_px=2.0):
    """XYZ tiles (x0, x1, y0, y1 inclusive) covering the output raster at `zoom`."""
    w = geom["width"]
    xs, ys = [], []
    for u, v in ((-margin_px, -margin_px), (w - 1 + margin_px, w - 1 + margin_px)):
        x, y = output_to_merc(u, v, geom, zoom, offset_px)
        xs.append(x)
        ys.append(y)
    x0, y0 = tile_xy(min(xs), min(ys))
    x1, y1 = tile_xy(max(xs), max(ys))
    return x0, x1, y0, y1


def assemble_mosaic(get_tile, x0, x1, y0, y1):
    """Mosaic (BGR uint8, rows = (y1 - y0 + 1) * 256) from get_tile(x, y) -> bytes or None (black). Returns
    (mosaic, missing tile count, sha256 over the tile bytes in (y, x) order)."""
    ny, nx = y1 - y0 + 1, x1 - x0 + 1
    out = np.zeros((ny * TILE_PX, nx * TILE_PX, 3), np.uint8)
    h = hashlib.sha256()
    missing = 0
    for j, y in enumerate(range(y0, y1 + 1)):
        for i, x in enumerate(range(x0, x1 + 1)):
            b = get_tile(x, y)
            if b is None:
                missing += 1
                h.update(b"missing")
                continue
            h.update(b)
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (TILE_PX, TILE_PX):
                missing += 1
                continue
            out[j * TILE_PX:(j + 1) * TILE_PX, i * TILE_PX:(i + 1) * TILE_PX] = img
    return out, missing, h.hexdigest()


def render_window(mosaic, x0, y0, geom, zoom, offset_px=(0.0, 0.0), interpolation=cv2.INTER_LINEAR):
    """The output raster (width x width, BGR) sampled from a mosaic whose top-left tile is (x0, y0) at `zoom`:
    output pixel (u, v) at Mercator (x, y) of `output_to_merc` is mosaic index (x - 256 x0 - 0.5, y - 256 y0 - 0.5)."""
    w = geom["width"]
    s = 2.0 ** (zoom - VIGOR_ZOOM)
    a = geom["k"] * s                                                 # mosaic px per output px
    bx, by = output_to_merc(0.0, 0.0, geom, zoom, offset_px)
    M = np.array([[a, 0.0, bx - TILE_PX * x0 - 0.5], [0.0, a, by - TILE_PX * y0 - 0.5]], np.float64)
    return cv2.warpAffine(mosaic, M, (w, w), flags=interpolation | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def fetch_window(client, release, lat, lon, city_res, out_gsd, zoom_prefs=(20, 19), offset_px=(0.0, 0.0)):
    """One VIGOR footprint from `release`: the tiles at the finest zoom of `zoom_prefs` whose centre tile exists,
    mosaicked and rendered (`render_window`). Returns (image BGR (W, W, 3) or None when no zoom has the centre tile,
    info dict for the sidecar)."""
    geom = window_geometry(lat, lon, city_res, out_gsd)
    zoom = client.finest_zoom(release, lat, lon, zoom_prefs)
    info = dict(release=release.num, release_date=release.date.isoformat(), layer=release.layer_id, zoom=zoom,
                width=geom["width"], out_gsd_m=geom["gsd"], footprint_m=geom["footprint_m"],
                offset_px=[float(offset_px[0]), float(offset_px[1])],
                offset_m=[float(offset_px[0]) * geom["gsd"], float(offset_px[1]) * geom["gsd"]],
                tile_url=release.tile_url)
    if zoom is None:
        info.update(source_gsd_m=None, tiles=None, tiles_sha256=None, missing_tiles=None)
        return None, info
    x0, x1, y0, y1 = tile_range(geom, zoom, offset_px)
    mosaic, missing, digest = assemble_mosaic(lambda x, y: client.tile(release, zoom, x, y), x0, x1, y0, y1)
    img = render_window(mosaic, x0, y0, geom, zoom, offset_px)
    info.update(source_gsd_m=ground_res_m(lat, zoom), tiles=dict(z=zoom, x0=x0, y0=y0, nx=x1 - x0 + 1, ny=y1 - y0 + 1),
                tiles_sha256=digest, missing_tiles=missing)
    return img, info


# ---- phase correlation (calibration) ---------------------------------------------------------------------------

def _gray(img):
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.shape[2] == 3 else a[..., 0]
    return a.astype(np.float64)


def _upsampled_dft(R, up, region, cy, cx):
    """Inverse DFT of the spectrum R on a (region x region) grid of 1/up px steps centred on (cy, cx) (matrix
    multiply, Guizar-Sicairos et al. 2008). Returns (values complex (region, region), ys, xs)."""
    h, w = R.shape
    fy, fx = np.fft.fftfreq(h), np.fft.fftfreq(w)
    ys = cy + (np.arange(region) - region // 2) / float(up)
    xs = cx + (np.arange(region) - region // 2) / float(up)
    ky = np.exp(2j * np.pi * np.outer(ys, fy))                       # (region, h)
    kx = np.exp(2j * np.pi * np.outer(fx, xs))                       # (w, region)
    return ky @ R @ kx, ys, xs


def phase_correlation(a, b, upsample=20, window=True):
    """Sub-pixel translation between two same-size images by phase correlation with an upsampled-DFT refinement.
    Returns (dx, dy, response): the content at a's pixel (u, v) is at b's pixel (u + dx, v + dy); response = the
    normalised correlation peak in [0, 1] (1 = identical up to a shift)."""
    A, B = _gray(a), _gray(b)
    if A.shape != B.shape:
        raise ValueError(f"shapes differ: {A.shape} vs {B.shape}")
    A, B = A - A.mean(), B - B.mean()
    if window:
        h, w = A.shape
        win = np.outer(np.hanning(h), np.hanning(w))
        A, B = A * win, B * win
    FA, FB = np.fft.fft2(A), np.fft.fft2(B)
    R = np.conj(FA) * FB
    mag = np.abs(R)
    R = R / np.where(mag > 1e-12, mag, 1.0)
    r = np.fft.ifft2(R).real
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)
    h, w = r.shape
    dy0 = py - h if py > h // 2 else py                               # wrap to signed shifts
    dx0 = px - w if px > w // 2 else px
    if upsample and upsample > 1:
        region = int(math.ceil(1.5 * upsample)) * 2 + 1
        vals, ys, xs = _upsampled_dft(R, upsample, region, dy0, dx0)
        v = vals.real
        iy, ix = np.unravel_index(int(np.argmax(v)), v.shape)
        dy, dx = float(ys[iy]), float(xs[ix])
        peak = float(v[iy, ix]) / (h * w)
    else:
        dy, dx, peak = float(dy0), float(dx0), float(r[py, px])
    return dx, dy, float(np.clip(peak, 0.0, 1.0))


CALIB_DOMAINS = ("gradient", "intensity")


def calib_blur_sigma(source_res_m, vigor_res_m, out_gsd, k=0.5):
    """Gaussian sigma (output px at `out_gsd`) that degrades the VIGOR tile (vigor_res_m m per px) to the Wayback
    source resolution (source_res_m: the sensor's SRC_RES when the metadata layer gives it, else the tile GSD):
    k * sqrt(max(source^2 - vigor^2, 0)) / out_gsd (k ~ 0.5: a pixel of pitch p ~ a Gaussian PSF of sigma p / 2).
    0 when the source is not coarser."""
    if not source_res_m or not k:
        return 0.0
    d2 = float(source_res_m) ** 2 - float(vigor_res_m) ** 2
    return float(k) * math.sqrt(d2) / float(out_gsd) if d2 > 0 else 0.0


def calib_preprocess(img, domain="gradient", blur_sigma=0.0):
    """The image the calibration correlates: grey, Gaussian blur of `blur_sigma` px (0 = none), then for domain
    'gradient' the Sobel gradient magnitude (edges survive a change of source, sensor and season; flat albedo and
    illumination differences do not), for 'intensity' the grey values."""
    if domain not in CALIB_DOMAINS:
        raise ValueError(f"calib domain must be one of {CALIB_DOMAINS}, got {domain!r}")
    g = _gray(img)
    if blur_sigma and blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), float(blur_sigma))
    if domain == "gradient":
        g = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3))
    return g


def phase_correlation_psr(A, B, upsample=20, max_shift_px=None, exclude_px=5, band_sigma_px=0.0):
    """Phase correlation of two preprocessed same-size float images with a constrained peak search and a
    peak-to-sidelobe ratio. Hann window after removing the WINDOW-WEIGHTED mean (a plain mean leaves the window's own
    spectrum common to both images: a spurious peak at zero shift). The integer peak is searched only within
    `max_shift_px` of zero (None = everywhere), refined by the upsampled DFT (1 / upsample px). PSR = (peak - mean) /
    std of the correlation surface inside the search disc excluding `exclude_px` around the peak (the whole surface
    when unconstrained). band_sigma_px > 0: the whitened cross-power spectrum is weighted by exp(-2 pi^2 f^2 s^2),
    f in cycles / px (the transfer function of a Gaussian of sigma s px, i.e. the blur applied to the finer image),
    so only the band both sources carry votes: above the coarser source's cut-off the whitened spectrum is pure
    noise with unit weight, which costs the sub-pixel peak ~0.2 px on blurred pairs with clutter (tests).
    Rim guard (constrained search): the integer peak is searched over a disc 2 px wider than the radius and refined
    there; when the refined peak lies outside the radius, `edge` is True and the caller rejects the tile (searching
    only inside the radius would return a ringing lobe of that outside peak, 1-2 px inside the rim, with a PSR of
    15-20 on synthetic pairs). Returns dict(dx, dy, response, psr, edge) with `phase_correlation`'s sign
    convention."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    if A.shape != B.shape:
        raise ValueError(f"shapes differ: {A.shape} vs {B.shape}")
    h, w = A.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    A = (A - (A * win).sum() / win.sum()) * win
    B = (B - (B * win).sum() / win.sum()) * win
    R = np.conj(np.fft.fft2(A)) * np.fft.fft2(B)
    mag = np.abs(R)
    R = R / np.where(mag > 1e-12, mag, 1.0)
    if band_sigma_px and band_sigma_px > 0:
        fy, fx = np.fft.fftfreq(h), np.fft.fftfreq(w)
        R = R * np.exp(-2.0 * np.pi ** 2 * float(band_sigma_px) ** 2 * (fy[:, None] ** 2 + fx[None, :] ** 2))
    r = np.fft.ifft2(R).real
    sy = np.rint(np.fft.fftfreq(h) * h)                               # signed integer shift of each surface row
    sx = np.rint(np.fft.fftfreq(w) * w)                               # (rint: fftfreq(320) * 320 has 24.000000000000004)
    d2 = sy[:, None] ** 2 + sx[None, :] ** 2
    inside = np.ones_like(r, bool) if max_shift_px is None else d2 <= float(max_shift_px) ** 2
    # the integer peak is searched 2 px beyond the radius: a peak just outside it would otherwise hand the search one
    # of its own ringing lobes inside the disc, with a high PSR; such a peak is found, refined and flagged `edge`
    wide = np.ones_like(r, bool) if max_shift_px is None else d2 <= (float(max_shift_px) + 2.0) ** 2
    py, px = np.unravel_index(int(np.argmax(np.where(wide, r, -np.inf))), r.shape)
    dy0, dx0 = float(sy[py]), float(sx[px])
    side = inside & ((sy[:, None] - dy0) ** 2 + (sx[None, :] - dx0) ** 2 > float(exclude_px) ** 2)
    vals = r[side]
    psr = float((r[py, px] - vals.mean()) / vals.std()) if vals.size > 1 and vals.std() > 0 else 0.0
    dx, dy, peak = dx0, dy0, float(r[py, px])
    if upsample and upsample > 1:
        region = int(math.ceil(1.5 * upsample)) * 2 + 1
        v, ys, xs = _upsampled_dft(R, upsample, region, dy0, dx0)
        v = v.real
        iy, ix = np.unravel_index(int(np.argmax(v)), v.shape)
        dy, dx, peak = float(ys[iy]), float(xs[ix]), float(v[iy, ix]) / (h * w)
    edge = max_shift_px is not None and bool(math.hypot(dx, dy) > float(max_shift_px))
    return dict(dx=dx, dy=dy, response=float(np.clip(peak, 0.0, 1.0)), psr=psr, edge=edge)


def calib_match(vigor_img, wayback_img, domain="gradient", blur_sigma=0.0, max_shift_px=None, upsample=20,
                exclude_px=5, band=True):
    """One calibration pair (both at the output GSD): the VIGOR tile blurred by `blur_sigma` to the Wayback source
    resolution, both images through `calib_preprocess(domain)`, then `phase_correlation_psr` (band-limited to the
    same sigma when `band`). Returns dict(dx, dy, response, psr, edge): the content at VIGOR px (u, v) is at Wayback
    px (u + dx, v + dy); edge = the peak is not inside the search disc (the tile is rejected)."""
    A = calib_preprocess(vigor_img, domain, blur_sigma)
    B = calib_preprocess(wayback_img, domain, 0.0)
    return phase_correlation_psr(A, B, upsample, max_shift_px, exclude_px, blur_sigma if band else 0.0)


def fourier_shift(img, dx, dy):
    """`img` (2-D float) translated by (dx, dy) px with the Fourier shift theorem (periodic; for tests): the content
    at (u, v) moves to (u + dx, v + dy)."""
    a = np.asarray(img, np.float64)
    h, w = a.shape
    fy, fx = np.fft.fftfreq(h), np.fft.fftfreq(w)
    ph = np.exp(-2j * np.pi * (np.outer(fy, np.ones(w)) * dy + np.outer(np.ones(h), fx) * dx))
    return np.fft.ifft2(np.fft.fft2(a) * ph).real


# ---- calibration file and layout -------------------------------------------------------------------------------

def calibration_path(root, city):
    return Path(root) / city / "wayback_calibration.json"


def load_calibration(root, city):
    """dict of scripts/calibrate_wayback_vigor.py (offset_px, offset_m, gsd_m, ...) or None when absent."""
    p = calibration_path(root, city)
    return json.loads(p.read_text()) if p.exists() else None


def offset_px_for(calib, out_gsd):
    """The calibration's offset in output px at `out_gsd` (it is stored in metres): (dx, dy), zeros without one."""
    if not calib or calib.get("offset_m") is None:                # none, or a calibration without a usable tile
        return (0.0, 0.0)
    m = calib["offset_m"]
    return (float(m[0]) / float(out_gsd), float(m[1]) / float(out_gsd))


def source_dir(root, city, source):
    """Directory of a reference source: `vigor` -> <root>/<City>/satellite, `wayback_<year>` -> <root>/<City>/<source>."""
    if not SOURCE_RE.match(str(source)):
        raise ValueError(f"ref_source must be 'vigor' or 'wayback_<year>', got {source!r}")
    return Path(root) / city / ("satellite" if source == "vigor" else str(source))


def sidecar_path(png_path):
    return Path(png_path).with_suffix(".json")


def vigor_at_out_gsd(tile_bgr, city_res, out_gsd):
    """The VIGOR tile resampled exactly as `VigorPairs.reference` does (the width of `window_geometry`, INTER_AREA
    when shrinking): the image the calibration compares the Wayback window with."""
    h0, w0 = tile_bgr.shape[:2]
    s = float(city_res) * VIGOR_TILE_PX / w0 / float(out_gsd)
    new = (max(1, int(round(w0 * s))), max(1, int(round(h0 * s))))
    return cv2.resize(tile_bgr, new, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
