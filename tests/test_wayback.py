"""Esri Wayback reference years (task 05): Mercator <-> tile-frame geometry, release selection, the version walk and
the mosaic on a fake tile server, the phase-correlation calibration on planted offsets, VigorPairs' ref_source and the
cross-source union of modes in scripts/eval_vigor.py. CPU, no network (the HTTP layer is a fake opener), no data."""
from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import time
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from bevloc import config as C
from bevloc.data import wayback as W
from bevloc.data.vigor import CITY_RES, VigorPairs
from bevloc.match.satroma import SatRoMa, consensus_for_query, consensus_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_vigor_window import DX, DY, SIGMA, _camera_px, _make, _marker_centroid  # noqa: E402
from tiny_satroma import StubPictureQuery, plant_translation, tiny_decoder, tiny_matcher  # noqa: E402
from vigor_labels import corrected  # noqa: E402

LAT, LON = 41.88, -87.63
SAT = f"satellite_{LAT}_{LON}.png"
RES = CITY_RES["Chicago"]
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bevloc_scripts_{name}_wayback_test", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- fake Wayback service -----------------------------------------------------------------------------------------

RELEASES = {  # number: (date, owner at every tile); owner None = itself
    "10": ("2014-02-20", None), "4230": ("2014-03-26", 10), "9812": ("2021-02-24", None),
    "58924": ("2025-09-25", None), "22869": ("2026-03-26", None),
}
# capture date (SRC_DATE) the metadata layer reports per release: the 2025 release re-publishes 2020 imagery, the
# 2026 one carries 2022 imagery (publication year != capture year, as on the real service), 2014 has no date
CAPTURE = {"10": "Null", "4230": "20140101", "9812": "20200904", "58924": "20200501", "22869": "20220620"}
WORLD_ZOOM = 19


def world(x, y):
    """Smooth synthetic imagery as a function of continuous zoom-19 Mercator px (several sinusoids)."""
    v = 128.0
    for amp, per, th, ph in ((40, 23.0, 0.3, 0.0), (30, 41.0, 1.9, 1.0), (25, 13.0, 2.6, 2.0), (20, 61.0, 0.9, 0.5)):
        v = v + amp * np.sin(2 * np.pi * (x * np.cos(th) + y * np.sin(th)) / per + ph)
    return np.clip(v, 0, 255)


def world_tile(x, y, z=WORLD_ZOOM):
    ii = np.arange(256) + 0.5
    X = 256 * x + ii[None, :]
    Y = 256 * y + ii[:, None]
    g = world(X, Y).astype(np.uint8)
    return np.stack([g, g, g], -1)


class FakeService:
    """opener(url, headers) for WaybackClient: config, tiles (zoom 19 only), tilemap, metadata. Counts requests."""

    def __init__(self):
        self.calls = []

    def config(self):
        return {n: dict(itemTitle=f"World Imagery (Wayback {d})",
                        itemURL=f"https://wayback.example/tile/{n}/{{level}}/{{row}}/{{col}}",
                        metadataLayerUrl=f"https://meta.example/{n}/MapServer", layerIdentifier=f"WB_{d[:4]}")
                for n, (d, _) in RELEASES.items()}

    def __call__(self, url, headers):
        assert "cross_view_sat_roma" in headers["User-Agent"]
        self.calls.append(url)
        if url == W.CONFIG_URL:
            return json.dumps(self.config()).encode()
        m = re.match(r"https://wayback\.example/tile/(\d+)/(\d+)/(\d+)/(\d+)$", url)
        if m:
            rel, z, y, x = (int(v) for v in m.groups())
            if z != WORLD_ZOOM or str(rel) not in RELEASES:
                raise urllib.error.HTTPError(url, 404, "no tile", None, None)
            ok, buf = cv2.imencode(".png", world_tile(x, y, z))
            return buf.tobytes()
        m = re.match(r".*/tilemap/(\d+)/(\d+)/(\d+)/(\d+)$", url)
        if m:
            rel = m.group(1)
            owner = RELEASES[rel][1]
            d = {"data": [1], "valid": True, "size": [1000]}
            if owner is not None:
                d["select"] = [owner]
            return json.dumps(d).encode()
        if "/identify?" in url:
            rel = re.search(r"meta\.example/(\d+)/", url).group(1)
            return json.dumps({"results": [
                {"layerId": 5, "layerName": "60cm", "attributes": {"SRC_DATE": "20250101", "SRC_RES": "0.6"}},
                {"layerId": 4, "layerName": "30cm", "attributes": {"SRC_DATE": CAPTURE[rel],
                                                                     "SRC_RES": "0.31", "SRC_ACC": "8.5",
                                                                     "SRC_DESC": "WV03", "NICE_DESC": "Vantor"}}]}).encode()
        raise urllib.error.HTTPError(url, 404, "unknown", None, None)


def _client(service, cache_dir=None):
    return W.WaybackClient(rate_hz=0, cache_dir=cache_dir, opener=service, retries=0)


def _releases(service):
    return W.parse_releases(service.config())


def _layout(tmp_path, offset_px=(0.0, 0.0), out_gsd=0.125):
    """A VIGOR layout with one Chicago tile whose imagery is the fake world rendered at 640 px, shifted by `offset_px`
    (in out_gsd output px, the calibration's units): Google's georeferencing error, planted."""
    root = _make(tmp_path)
    (root / "Chicago" / "satellite" / "s1.png").unlink()
    g570 = W.window_geometry(LAT, LON, RES, out_gsd)
    g640 = W.window_geometry(LAT, LON, RES, RES)                      # 640 px, k = 1
    assert g640["width"] == 640
    off640 = (offset_px[0] * g570["k"], offset_px[1] * g570["k"])   # the same shift in zoom-20 px = 640-px units
    x0, x1, y0, y1 = W.tile_range(g640, WORLD_ZOOM, off640, margin_px=8)
    mosaic, _, _ = W.assemble_mosaic(lambda x, y: cv2.imencode(".png", world_tile(x, y))[1].tobytes(), x0, x1, y0, y1)
    tile = W.render_window(mosaic, x0, y0, g640, WORLD_ZOOM, off640, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(root / "Chicago" / "satellite" / SAT), tile)
    for f in ("pano_label_balanced.txt", "same_area_balanced_test.txt", "same_area_balanced_train.txt", "satellite_list.txt"):
        p = root / "splits" / "VIGOR" / "Chicago" / f
        p.write_text(p.read_text().replace("s1.png", SAT))
    corrected(p.parent)
    return root


def _cfg(cell_m=0.125, solver="se2"):
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.lift.query_mode = "ipm"
    cfg.grid.cell_m = cell_m
    cfg.vigor.ref_window_m = None
    cfg.vigor.ref_jitter_m = 0.0
    cfg.matcher.solver = solver
    return cfg


# ---- geometry ----------------------------------------------------------------------------------------------------

def test_mercator_round_trip_and_known_tile():
    for lat, lon in ((LAT, LON), (40.75, -73.98), (-33.9, 151.2), (0.0, 0.0)):
        for z in (0, 19, 20):
            x, y = W.latlon_to_merc_px(lat, lon, z)
            assert np.allclose(W.merc_px_to_latlon(x, y, z), (lat, lon), atol=1e-9)
    assert W.latlon_to_merc_px(0.0, 0.0, 0) == (128.0, 128.0)
    assert W.tile_xy(*W.latlon_to_merc_px(LAT, LON, 19)) == (134523, 194859)      # the probed Chicago tile
    assert abs(W.ground_res_m(41.80, 20) - RES) < 2e-4                              # CITY_RES = zoom-20 GSD


def test_output_pixel_round_trip_is_sub_pixel_and_centre_is_the_tile_centre():
    g = W.window_geometry(LAT, LON, RES, 0.125)
    assert g["width"] == 570 and abs(g["gsd"] - 0.125) < 2e-4 and abs(g["footprint_m"] - 640 * RES) < 1e-9
    for off in ((0.0, 0.0), (1.7, -2.3)):
        for z in (19, 20):
            u, v = 123.4, 456.7
            x, y = W.output_to_merc(u, v, g, z, off)
            assert np.allclose(W.merc_to_output(x, y, g, z, off), (u, v), atol=1e-6)
    c = (g["width"] - 1) / 2.0
    x, y = W.output_to_merc(c, c, g, 20)
    assert np.allclose(W.merc_px_to_latlon(x, y, 20), (LAT, LON), atol=1e-9)
    # one output px east = k zoom-20 px east; one px down = south
    x1, y1 = W.output_to_merc(c + 1, c + 1, g, 20)
    assert np.isclose(x1 - x, g["k"]) and np.isclose(y1 - y, g["k"])
    lat1, lon1 = W.merc_px_to_latlon(x1, y1, 20)
    assert lon1 > LON and lat1 < LAT
    # the window's outer pixel edges are the VIGOR tile's: 320 zoom-20 px either side of the centre (a 640 px Google
    # Static Maps image centred on the label), whatever the output pixel count; half of that at zoom 19
    w = g["width"]
    assert np.allclose(W.output_to_merc(-0.5, -0.5, g, 20), (g["cx"] - 320.0, g["cy"] - 320.0))
    assert np.allclose(W.output_to_merc(w - 0.5, w - 0.5, g, 20), (g["cx"] + 320.0, g["cy"] + 320.0))
    assert np.allclose(W.output_to_merc(-0.5, -0.5, g, 19), ((g["cx"] - 320.0) / 2, (g["cy"] - 320.0) / 2))
    # a Mercator point on the seam of tiles x0 / x0 + 1 is mosaic index 255.5 (pixel i of tile x covers [256 x + i, +1))
    mosaic = np.zeros((256, 512, 3), np.uint8)
    mosaic[:, :256] = 200
    x0, y0 = W.tile_xy(*W.output_to_merc(0.0, 0.0, g, 20))
    g1 = dict(g, width=1, k=1.0, cx=256.0 * (x0 + 1), cy=256.0 * y0 + 0.5)   # one output px centred on the seam
    assert W.render_window(mosaic, x0, y0, g1, 20)[0, 0, 0] == 100          # half of each tile


def test_pick_release_closest_to_mid_year_ties_go_to_the_newer():
    rel = W.parse_releases(FakeService().config())
    assert [r.date.isoformat() for r in rel] == sorted(r.date.isoformat() for r in rel)
    assert W.pick_release(rel, 2021).num == 9812
    assert W.pick_release(rel, 2025).num == 58924
    assert W.pick_release(rel, 2014).num == 4230                     # 03-26 is nearer 07-01 than 02-20
    import datetime as dt
    a = W.Release(1, dt.date(2020, 6, 1), "", "", "")
    b = W.Release(2, dt.date(2020, 8, 1), "", "", "")                # both 30/31 days from 07-01? no: 30 vs 31
    c = W.Release(3, dt.date(2020, 7, 31), "", "", "")               # exactly 30 days, like a
    assert W.pick_release([a, b, c], 2020) is c                       # tie a / c -> the newer c
    assert W.pick_release([], 2020) is None


def test_client_retries_with_backoff_and_rate_limits_every_request(monkeypatch):
    """Transient errors: retries 1 / 2 / 4 s (no sleep after the last attempt), then RuntimeError; 404 = None at once;
    every request (tile, tilemap json, config) waits for the rate interval."""
    sleeps = []
    monkeypatch.setattr(W.time, "sleep", lambda s: sleeps.append(round(s, 3)))
    calls = []

    def flaky(url, headers):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.URLError("down")
        return b"ok"
    cl = W.WaybackClient(rate_hz=0, retries=3, opener=flaky)
    assert cl.get("u") == b"ok" and len(calls) == 3 and sleeps == [1.0, 2.0]
    calls.clear(); sleeps.clear()
    cl = W.WaybackClient(rate_hz=0, retries=2, opener=lambda u, h: (_ for _ in ()).throw(urllib.error.URLError("x")))
    with pytest.raises(RuntimeError, match="3 attempts"):
        cl.get("u")
    assert sleeps == [1.0, 2.0]
    sleeps.clear()
    monkeypatch.setattr(W.time, "monotonic", lambda: 100.0)           # time stands still: every wait is the interval
    svc = FakeService()
    cl = W.WaybackClient(rate_hz=4.0, retries=0, opener=svc)
    assert cl.get("https://wayback.example/tile/22869/20/1/1") is None and not sleeps   # 404: no retry, first: no wait
    cl.releases()
    cl.tilemap(_releases(svc)[0], 19, 1, 1)
    cl.tile(_releases(svc)[-1], 19, 1, 1)
    assert sleeps == [0.25, 0.25, 0.25] and cl.n_requests == 4        # config, tilemap, tile: each waits 1 / rate_hz


# ---- fake service: walk, mosaic, fetch ---------------------------------------------------------------------------

def test_version_walk_lists_distinct_owners_in_one_request_each():
    svc = FakeService()
    cl = _client(svc)
    rel = _releases(svc)
    x, y = W.tile_xy(*W.latlon_to_merc_px(LAT, LON, 19))
    v = cl.versions_at(rel, 19, x, y)
    assert [r.num for r in v] == [22869, 58924, 9812, 10]
    assert sum("/tilemap/" in u for u in svc.calls) == 4
    assert W.pick_release(v, 2025).num == 58924 and W.pick_release(v, 2021).num == 9812 and W.pick_release(v, 2019).num == 9812


def test_walk_cache_is_reused_and_dropped_when_the_release_list_changes(tmp_path):
    """Walks: one tilemap request per version on the first call, none on the second (persisted), and a newer
    release list (--refresh-releases) drops the cached walks (a walk starts at the newest release)."""
    F = _load("fetch_wayback_vigor")
    svc = FakeService()
    cl = _client(svc)
    rel = _releases(svc)
    x, y = W.tile_xy(*W.latlon_to_merc_px(LAT, LON, 19))
    p = tmp_path / "tilemap_z19.json"
    w = F.Walks(p, newest=rel[-1].num)
    assert [r.num for r in w.get(cl, rel, 19, x, y)] == [22869, 58924, 9812, 10] and sum("/tilemap/" in u for u in svc.calls) == 4
    w.save()
    w2 = F.Walks(p, newest=rel[-1].num)
    assert [r.num for r in w2.get(cl, rel, 19, x, y)] == [22869, 58924, 9812, 10] and sum("/tilemap/" in u for u in svc.calls) == 4
    w3 = F.Walks(p, newest=rel[-1].num + 1)                            # a newer release appeared
    assert "19/%d/%d" % (x, y) not in w3.d and w3.dirty
    w3.get(cl, rel, 19, x, y)
    assert sum("/tilemap/" in u for u in svc.calls) == 8


def test_mosaic_render_matches_the_world_and_the_zoom_falls_back(tmp_path):
    svc = FakeService()
    cl = _client(svc, cache_dir=tmp_path / "tiles")
    rel = {r.num: r for r in _releases(svc)}[22869]
    img, info = W.fetch_window(cl, rel, LAT, LON, RES, 0.125, zoom_prefs=(20, 19))
    assert info["zoom"] == 19 and img.shape == (570, 570, 3) and info["missing_tiles"] == 0
    assert info["tiles"]["nx"] == 3 and info["tiles"]["ny"] == 3 and len(info["tiles_sha256"]) == 64
    g = W.window_geometry(LAT, LON, RES, 0.125)
    uu, vv = np.meshgrid(np.arange(570, dtype=float), np.arange(570, dtype=float))
    X, Y = W.output_to_merc(uu, vv, g, 19)
    expect = world(X, Y)
    err = np.abs(img[..., 0].astype(float) - expect)
    assert np.median(err) < 1.0 and err.mean() < 2.0                 # bilinear on a smooth pattern
    # the cache serves a second render with no tile requests; a 404 zoom leaves a marker
    n = len(svc.calls)
    img2, _ = W.fetch_window(cl, rel, LAT, LON, RES, 0.125, zoom_prefs=(20, 19))
    assert np.array_equal(img, img2) and len(svc.calls) == n
    assert list((tmp_path / "tiles" / "22869" / "20").glob("*.missing"))


def test_planted_offset_moves_the_content_the_documented_way():
    """render_window with offset (dx, dy) shows at output (u, v) what the offset-free render shows at (u + dx, v + dy)."""
    g = W.window_geometry(LAT, LON, RES, 0.125)
    x0, x1, y0, y1 = W.tile_range(g, 19, margin_px=8)
    mosaic, _, _ = W.assemble_mosaic(lambda x, y: cv2.imencode(".png", world_tile(x, y))[1].tobytes(), x0, x1, y0, y1)
    a = W.render_window(mosaic, x0, y0, g, 19)
    b = W.render_window(mosaic, x0, y0, g, 19, offset_px=(5.0, -3.0))
    d = b[10:-10, 10:-10].astype(float) - a[7:-13, 15:-5]           # b(u, v) = a(u + 5, v - 3) ...
    assert np.abs(d).mean() < 1.5 and np.abs(d).max() < 12           # ... up to bilinear resampling at another phase
    dx, dy, resp = W.phase_correlation(a, b)
    # a's content at (u, v) sits at b's (u - 5, v + 3): the documented sign of phase_correlation
    assert abs(dx + 5.0) < 0.05 and abs(dy - 3.0) < 0.05 and resp > 0.3


# ---- phase correlation and calibration -----------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [(3.27, -1.63), (0.0, 0.0), (-7.4, 12.9), (0.5, 0.5)])
def test_phase_correlation_recovers_a_planted_sub_pixel_shift(shift):
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur(rng.random((256, 300)).astype(np.float32), (0, 0), 2.0)
    b = W.fourier_shift(a, *shift)
    dx, dy, resp = W.phase_correlation(a, b, upsample=20)
    assert abs(dx - shift[0]) < 0.05 and abs(dy - shift[1]) < 0.05
    assert 0.3 < resp <= 1.0
    with pytest.raises(ValueError):
        W.phase_correlation(a, a[:-1])


def test_calibration_recovers_the_planted_google_offset_and_the_fetch_applies_it(tmp_path):
    planted = (2.6, -1.4)                                             # output px at 0.125 m: 0.33 m east, 0.18 m north
    root = _layout(tmp_path, offset_px=planted)
    svc = FakeService()
    cal = _load("calibrate_wayback_vigor")
    args = ["--root", str(root), "--city", "Chicago", "--split", "samearea", "--tiles", "5", "--upsample", "50"]
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    out = cal.calibrate(cal.build_parser().parse_args(args), cfg, opener=svc)
    assert out["n_used"] == 1 and out["pass"] is False                 # one tile < calib_min_tiles (30): FAIL
    out = cal.calibrate(cal.build_parser().parse_args(args + ["--min-tiles", "1"]), cfg, opener=svc)
    assert out["n_used"] == 1 and out["releases"] == {"2021-02-24 (#9812)": 1}
    t = out["per_tile"][0]
    assert t["psr"] >= cfg.wayback.calib_min_psr and t["accepted"] and t["source_res_m"] == pytest.approx(0.31)
    assert t["blur_sigma_px"] == pytest.approx(W.calib_blur_sigma(0.31, RES, 0.125, cfg.wayback.calib_blur_k))
    assert abs(out["offset_px"][0] - planted[0]) < 0.05 and abs(out["offset_px"][1] - planted[1]) < 0.05
    assert out["pass"] is True and out["residual_m"] == 0.0            # one tile: no spread
    assert np.allclose(out["offset_m"], np.multiply(out["offset_px"], out["gsd_m"]))
    assert W.calibration_path(root, "Chicago").exists()
    assert (root / "Chicago" / "wayback_calib_2021" / SAT).exists()
    side = json.loads((root / "Chicago" / "wayback_calib_2021" / SAT.replace(".png", ".json")).read_text())
    assert side["release"] == 9812 and side["offset_px"] == [0.0, 0.0] and side["capture"]["SRC_DATE"] == "20200904"
    assert side["capture_date"] == "2020-09-04" and side["selection"]["select_by"] == "capture"
    assert side["selection"]["date"] == "2020-09-04" and side["selection"]["publication_date"] == "2021-02-24"
    assert out["select_by"] == "capture" and out["capture_years"] == {"2020": 1} and out["n_fallback_publication"] == 0
    assert side["attribution"] == W.ATTRIBUTION and side["zoom"] == 19
    # the fetcher applies the calibration: the 2021 window now coincides with the VIGOR tile at the output GSD
    F = _load("fetch_wayback_vigor")
    counts = F.main(["--root", str(root), "--split", "samearea", "--cities", "Chicago", "--years", "2021", "2025"],
                    opener=svc)
    assert counts["written"] == 2
    wb = cv2.imread(str(root / "Chicago" / "wayback_2021" / SAT))
    vg = W.vigor_at_out_gsd(cv2.imread(str(root / "Chicago" / "satellite" / SAT)), RES, 0.125)
    dx, dy, _ = W.phase_correlation(vg, wb, upsample=50)
    assert abs(dx) < 0.1 and abs(dy) < 0.1                           # 1 cm: two resamplings (area vs bilinear)
    assert np.abs(wb[20:-20, 20:-20].astype(float) - vg[20:-20, 20:-20]).mean() < 3.0
    side = json.loads((root / "Chicago" / "wayback_2021" / SAT.replace(".png", ".json")).read_text())
    assert np.allclose(side["offset_px"], planted, atol=0.05) and side["calibration"].endswith("wayback_calibration.json")
    side25 = json.loads((root / "Chicago" / "wayback_2025" / SAT.replace(".png", ".json")).read_text())
    assert side25["release"] == 22869 and side25["release_date"] == "2026-03-26"
    # resumable: the second run skips both and makes no tile request
    n = len(svc.calls)
    counts = F.main(["--root", str(root), "--split", "samearea", "--cities", "Chicago", "--years", "2021", "2025"],
                    opener=svc)
    assert counts["skipped"] == 2 and counts.get("written", 0) == 0
    assert not any("/tile/" in u for u in svc.calls[n:])
    # dry run writes nothing and lists the choice per year
    counts = F.main(["--root", str(root), "--split", "samearea", "--cities", "Chicago", "--years", "2019", "--dry-run"],
                    opener=svc)
    assert counts["dry"] == 1 and not (root / "Chicago" / "wayback_2019").exists()


def test_aggregate_is_robust_to_outliers_and_low_psr_and_needs_30_tiles():
    cal = _load("calibrate_wayback_vigor")
    rng = np.random.default_rng(1)
    off = np.array([[2.0, -1.0]]) + rng.normal(0, 0.3, (40, 2))
    off = np.vstack([off, [[30.0, 30.0], [2.0, -1.0]]])              # a gross outlier, a low-PSR tile
    psr = np.r_[np.full(41, 12.0), 3.5]
    agg = cal.aggregate(off, psr, 0.125, 7.0, 4.0, 0.3, 30)
    assert agg["n"] == 42 and agg["n_low_psr"] == 1 and agg["n_outliers"] == 1 and agg["n_used"] == 40
    assert abs(agg["offset_px"][0] - 2.0) < 0.15 and abs(agg["offset_px"][1] + 1.0) < 0.15
    assert 0.0 < agg["residual_m"] < 0.1 and agg["pass"]
    assert agg["frac_within_0_5m"] == pytest.approx(40 / 41) and all(0 < q < 0.15 for q in agg["iqr_m"])
    assert agg["psr_percentiles"]["p50"] == 12.0
    none = cal.aggregate(off, np.full(42, 3.0), 0.125, 7.0, 4.0, 0.3, 30)
    assert none["offset_px"] is None and none["n_used"] == 0 and none["pass"] is False
    # the same tight cluster on 20 tiles: residual far below the gate, but n_used < 30 -> FAIL
    few = cal.aggregate(off[:20], np.full(20, 12.0), 0.125, 7.0, 4.0, 0.3, 30)
    assert few["n_used"] == 20 and few["residual_m"] < 0.1 and few["pass"] is False
    assert cal.aggregate(off[:20], np.full(20, 12.0), 0.125, 7.0, 4.0, 0.3, 20)["pass"] is True


# ---- VigorPairs.ref_source -----------------------------------------------------------------------------------------

def _wayback_tile(root, source="wayback_2025", colour=(10, 200, 30), size=570):
    d = root / "Chicago" / source
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "s1.png"), np.full((size, size, 3), colour, np.uint8))


def test_ref_source_picks_the_file_and_keeps_the_label_geometry(tmp_path):
    root = _make(tmp_path)
    _wayback_tile(root)
    cfg = _cfg()
    v = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea")[0]
    w = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea", ref_source="wayback_2025")[0]
    assert torch.equal(v["H"], w["H"]) and torch.equal(v["en"], w["en"]) and torch.equal(v["bev"], w["bev"])
    assert w["ref"].shape == v["ref"].shape == (3, 896, 896)
    c = 447
    assert np.allclose(w["ref"][:, c, c].numpy() * 255, (30, 200, 10), atol=1)   # RGB of the BGR colour
    assert np.allclose(v["ref"][:, c, c].numpy() * 255, (60, 60, 60), atol=60)   # the vigor background (+ marker)
    # the 570 px file covers the same footprint: its edge lands where the 640 px tile's edge lands (+-1 px)
    hw = int(round(640 * RES / 0.125 / 2))
    assert w["ref"][:, c, c + hw - 2].max() > 0 and w["ref"][:, c, c + hw + 2].max() == 0
    assert v["ref"][:, c, c + hw - 2].max() > 0 and v["ref"][:, c, c + hw + 2].max() == 0
    # config default, ref_canvas, missing_refs
    cfg.vigor.ref_source = "wayback_2025"
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea")
    assert ds.ref_source == "wayback_2025" and torch.equal(ds[0]["ref"], w["ref"])
    assert torch.equal(ds.ref_canvas(0, "vigor"), v["ref"]) and torch.equal(ds.ref_canvas(0), w["ref"])
    assert ds.missing_refs() == [] and ds.missing_refs("wayback_2019") == ds.labels
    with pytest.raises(ValueError):
        VigorPairs(root, cfg, cities=["Chicago"], split="crossarea", ref_source="esri")


def test_missing_wayback_file_is_a_clear_error(tmp_path):
    root = _make(tmp_path)
    ds = VigorPairs(root, _cfg(), cities=["Chicago"], split="crossarea", ref_source="wayback_2019")
    with pytest.raises(RuntimeError, match="wayback_2019.*wayback-fetch"):
        ds[0]


def test_window_reference_of_another_source_shares_the_geometry(tmp_path):
    root = _make(tmp_path)
    _wayback_tile(root)
    cfg = _cfg(cell_m=0.0625)
    cfg.vigor.ref_window_m = 56.0
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea")
    centre = torch.tensor([3.0, -2.0], dtype=torch.float64)
    s = ds.item(0, ref_centre_en=centre)
    w = ds.ref_canvas(0, "wayback_2025", ref_centre_en=s["ref_centre_en"])
    assert w.shape == s["ref"].shape
    nz_v, nz_w = (s["ref"].sum(0) > 0), (w.sum(0) > 0)
    assert (nz_v ^ nz_w).float().mean() < 0.01                       # the same footprint on the window canvas
    assert np.allclose(w[:, 447, 447].numpy() * 255, (30, 200, 10), atol=1)


def test_default_ref_source_is_bit_identical(tmp_path):
    root = _make(tmp_path)
    cfg = _cfg()
    a = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea")
    b = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea", ref_source="vigor")
    sa, sb = a[0], b[0]
    for k in ("erp", "R_w2c", "se2", "ref", "H", "en", "ref_centre_en", "bev", "bev_valid"):
        assert torch.equal(sa[k], sb[k]), k
    assert a.ref_source == "vigor" and str(a.ref_path("Chicago", "s1.png")).endswith("Chicago/satellite/s1.png")


def _marker_layout(tmp_path, width=570):
    """`_make`'s VIGOR tile (a Gaussian marker at the label, 640 px) plus a `width` px Wayback file of the same
    footprint with the same marker: tile px t -> file px (t + 0.5) * width / 640 - 0.5 (both cover 640 * RES m)."""
    root = _make(tmp_path)
    k = width / 640.0
    xc, yc = ((639 / 2.0 - DX) + 0.5) * k - 0.5, ((639 / 2.0 + DY) + 0.5) * k - 0.5
    yy, xx = np.mgrid[0:width, 0:width].astype(np.float64)
    g = np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * (SIGMA * k) ** 2))
    img = np.full((width, width, 3), 60.0)
    img[..., 2] += 195.0 * g
    d = root / "Chicago" / "wayback_2025"
    d.mkdir()
    cv2.imwrite(str(d / "s1.png"), np.round(img).astype(np.uint8))
    return root


@pytest.mark.parametrize("cell_m,window,centre", [(0.125, None, None), (0.25, None, None), (0.0625, None, None),
                                                  (0.0625, 56.0, (3.0, -2.0)), (0.125, 40.0, (-2.5, 4.0))])
def test_vigorpairs_places_the_wayback_file_on_the_tile_footprint(tmp_path, cell_m, window, centre):
    """A 570 px Wayback file and the 640 px VIGOR tile of the same footprint put the same marker on the same canvas
    pixel (whole tile at three cell sizes, two windows): the resampling rule footprint / width / cell of `reference`
    and `window_reference` is source-independent to a few hundredths of a px (the label's camera px is the check)."""
    root = _marker_layout(tmp_path)
    cfg = _cfg(cell_m)
    cfg.vigor.ref_window_m = window
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea")
    c = None if centre is None else torch.tensor(centre, dtype=torch.float64)
    s = ds.item(0, ref_centre_en=c)
    near = _camera_px(s)
    cv = _marker_centroid(ds.ref_canvas(0, "vigor", ref_centre_en=c), near)
    cw = _marker_centroid(ds.ref_canvas(0, "wayback_2025", ref_centre_en=c), near)
    assert np.abs(cv - cw).max() < 0.05, (cv, cw)                     # the two sources agree to 5 mm at 0.125 m
    assert np.abs(cv - near).max() < 0.6                              # both sit on the label (whole tile: +-0.5 px)


def test_other_ref_builds_the_canvas_by_the_samples_rule(tmp_path):
    """eval_vigor.other_ref: the other source's canvas of a coarse sample is built by the rule that built the
    sample's (whole tile without ref_window_m, the sample's own window otherwise, also under a fresh training
    jitter); a window at the whole-tile sample's zeros is NOT that canvas (half a px at an odd size gap)."""
    ev = _load("eval_vigor")
    root = _make(tmp_path)
    d = root / "Chicago" / "wayback_2025"
    d.mkdir()
    (d / "s1.png").write_bytes((root / "Chicago" / "satellite" / "s1.png").read_bytes())   # the same file, other source
    for cell, window, train in ((0.25, None, False), (0.125, None, False), (0.0625, 56.0, False), (0.0625, 56.0, True)):
        cfg = _cfg(cell)
        cfg.vigor.ref_window_m = window
        cfg.vigor.ref_jitter_m = 6.0 if window else 0.0
        ds = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea", train=train)
        s = ds[0]
        assert torch.equal(ev.other_ref(ds, 0, s, "vigor"), s["ref"]), (cell, window, train)
        assert torch.equal(ev.other_ref(ds, 0, s, "wayback_2025"), s["ref"]), (cell, window, train)
    ds = VigorPairs(root, _cfg(0.25), cities=["Chicago"], split="crossarea")   # 896 canvas, 285 px tile: odd gap
    s = ds[0]
    assert torch.equal(s["ref_centre_en"], torch.zeros(2, dtype=torch.float64))
    assert not torch.equal(ds.ref_canvas(0, "vigor", ref_centre_en=s["ref_centre_en"]), s["ref"])


# ---- union of modes ---------------------------------------------------------------------------------------------------

T = (336.0, 352.0)                                                    # planted translation (cell-aligned)


def _toy(tmp_path, solver="se2"):
    root = _make(tmp_path)
    _wayback_tile(root)
    cfg = _cfg(solver=solver)
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea")
    matcher = tiny_matcher(plant_translation(tiny_decoder(), T))
    cons = {tag: SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=means, min_valid_frac=0.05)
            for tag, means in (("peak", False), ("means", True))}
    return root, cfg, ds, matcher, cons


@pytest.mark.parametrize("solver", ["se2", "srt"])
def test_consensus_union_pools_the_modes_and_reports_the_share_per_source(tmp_path, solver):
    ev = _load("eval_vigor")
    root, cfg, ds, matcher, cons = _toy(tmp_path, solver)
    q = StubPictureQuery()
    s = ds[0]
    batch = {k: (v[None] if torch.is_tensor(v) else v) for k, v in s.items()}
    f_q, frac = q(batch, matcher)
    refs = [batch["ref"], ds.ref_canvas(0, "wayback_2025")[None]]
    gms, certs = ev.decode_refs(matcher, f_q, refs)
    assert len(gms) == 2 and gms[0].shape == gms[1].shape
    m1 = consensus_for_query(cons["peak"], gms[0], q, batch, frac, 224, min_frac=0.05, certainty=certs[0], stats=False)
    m = consensus_union(cons["peak"], gms, q, batch, frac, 224, certainties=certs, sources=["vigor", "wayback_2025"])
    assert m.H is not None and np.allclose(m.H[:2, 2], T, atol=0.05) and np.allclose(m.H[:2, :2], np.eye(2), atol=1e-3)
    assert m.n_modes == 2 * m1.n_modes and m.n_inliers == 2 * m1.n_inliers
    assert set(m.sources) == {"vigor", "wayback_2025"}
    assert sum(v["n_inliers"] for v in m.sources.values()) == m.n_inliers
    assert sum(v["n_modes"] for v in m.sources.values()) == m.n_modes
    assert m.sources["vigor"] == m.sources["wayback_2025"] == dict(n_modes=m1.n_modes, n_inliers=m1.n_inliers)
    assert m.stats is None
    # one source = the single-source consensus
    m_one = consensus_union(cons["peak"], gms[:1], q, batch, frac, 224, certainties=certs[:1], sources=["vigor"])
    assert np.allclose(m_one.H, m1.H) and m_one.n_inliers == m1.n_inliers


def test_score_union_rows_and_the_single_source_rows_on_identical_frames(tmp_path):
    ev = _load("eval_vigor")
    root, cfg, ds, matcher, cons = _toy(tmp_path)
    rows = ev.score(ds, StubPictureQuery(), matcher, cons, cfg, "cpu", sources=["vigor", "wayback_2025"])
    r = rows[0]
    assert r["union_inlier_share_vigor"] == 0.5 == r["union_inlier_share_wayback_2025"]
    assert r["union_modes_vigor"] == r["union_modes_wayback_2025"] > 0
    assert r["pose_peak_m"] is not None and abs(r["pose_peak_m"] - r["pose_peak_vigor_m"]) < 1e-6
    assert abs(r["pose_peak_wayback_2025_m"] - r["pose_peak_vigor_m"]) < 1e-6
    assert r["vote_entropy"] is None                                  # statistics are per decode: None in union mode
    single = ev.score(ds, StubPictureQuery(), matcher, cons, cfg, "cpu")
    assert abs(single[0]["pose_peak_m"] - r["pose_peak_m"]) < 1e-6
    with pytest.raises(ValueError, match="refine"):
        ev.score(ds, StubPictureQuery(), matcher, cons, cfg, "cpu", sources=["vigor", "wayback_2025"], refine=16)


def _run(ev, tmp_path, root, extra, monkeypatch, out):
    built = []

    def fake_matcher(ckpt, dev, train_decoder=False):
        m = tiny_matcher(plant_translation(tiny_decoder(), T if not built else (352.0, 320.0)))
        built.append(m)
        return m

    class Q(torch.nn.Module):
        def __init__(self, *a):
            super().__init__()
            self.stub = StubPictureQuery()

        def forward(self, batch, matcher):
            return self.stub(batch, matcher)
    monkeypatch.setattr(ev, "FeatureQueryMatcher", fake_matcher)
    monkeypatch.setattr(ev, "build_query", lambda cfg, mode: Q())
    monkeypatch.setattr(ev.torch.cuda, "is_available", lambda: False)
    ck = tmp_path / "ck.pt"
    torch.save({"mode": "ipm", "decoder": {}, "query": {}, "step": 1}, ck)
    a = ev.build_parser().parse_args([
        "--config", str(C.REPO / "configs/vigor_cell0125.yaml"), "--ckpt", str(ck), "--root", str(root),
        "--cities", "Chicago", "--solver", "se2", "--fine-config", str(C.REPO / "configs/vigor_cell00625_fine.yaml"),
        "--fine-ckpt", str(ck), "--out", str(out), "--tag", "t"] + extra)
    ev.run(a, ev.decoder_fine)
    return json.loads((out / "eval_vigor_t_crossarea.json").read_text())


def test_run_with_ref_sources_writes_union_rows_and_the_fine_pass_unions_too(tmp_path, monkeypatch):
    ev = _load("eval_vigor")
    root = _make(tmp_path / "vigor")
    _wayback_tile(root)
    d = _run(ev, tmp_path, root, ["--ref-sources", "vigor", "wayback_2025"], monkeypatch, tmp_path / "u")
    assert d["meta"]["ref_sources"] == ["vigor", "wayback_2025"] and d["meta"]["ref_source"] == "vigor"
    assert d["meta"]["fine"]["ref_sources"] == ["vigor", "wayback_2025"]
    r = d["frames"][0]
    assert r["union_inlier_share_vigor"] == 0.5 and r["pose_fine_m"] is not None and r["fallback_fine"] is False
    s = d["summary"]["all"]
    assert s["peak_wayback_2025"]["n"] == 1 and s["union_inlier_share_wayback_2025"] == 0.5
    assert abs(s["peak_vigor"]["median_m"] - s["peak"]["median_m"]) < 1e-9


def test_run_default_and_ref_source_vigor_are_identical_and_missing_source_stops(tmp_path, monkeypatch):
    ev = _load("eval_vigor")
    root = _make(tmp_path / "vigor")
    d0 = _run(ev, tmp_path, root, [], monkeypatch, tmp_path / "a")
    d1 = _run(ev, tmp_path, root, ["--ref-source", "vigor"], monkeypatch, tmp_path / "b")
    assert d0["frames"] == d1["frames"] and d0["summary"] == d1["summary"]
    assert d1["meta"]["ref_source"] == "vigor" and d1["meta"]["ref_sources"] is None
    assert "union_inlier_share_vigor" not in d0["frames"][0] and "peak_vigor" not in d0["summary"]["all"]
    with pytest.raises(SystemExit, match="wayback_2019"):
        _run(ev, tmp_path, root, ["--ref-source", "wayback_2019"], monkeypatch, tmp_path / "c")
    with pytest.raises(SystemExit, match="first of --ref-sources"):
        _run(ev, tmp_path, root, ["--ref-source", "wayback_2025", "--ref-sources", "vigor", "wayback_2025"],
             monkeypatch, tmp_path / "d")


# ---- selection by capture date ------------------------------------------------------------------------------------

def test_pick_version_by_capture_differs_from_publication_and_falls_back():
    import datetime as dt
    svc = FakeService()
    by_num = {r.num: r for r in _releases(svc)}
    versions = [by_num[n] for n in (22869, 58924, 9812, 10)]                # the walk, newest first
    caps = {n: W.capture_date({"SRC_DATE": CAPTURE[str(n)]}) for n in (22869, 58924, 9812, 10)}
    assert caps[10] is None and caps[22869] == dt.date(2022, 6, 20)
    # 2025: the publication rule takes the 2025-09-25 release (2020 imagery); the capture rule the 2022 imagery
    pub, ip = W.pick_version(versions, 2025, caps, "publication")
    cap, ic = W.pick_version(versions, 2025, caps, "capture")
    assert pub.num == 58924 and cap.num == 22869
    assert ip == dict(select_by="publication", date="2025-09-25", capture_date="2020-05-01",
                      publication_date="2025-09-25", fallback=False, n_fallback=0)
    assert ic == dict(select_by="capture", date="2022-06-20", capture_date="2022-06-20",
                      publication_date="2026-03-26", fallback=False, n_fallback=1)
    # 2021: both rules agree; 2019: capture picks the 2020-05-01 imagery published in 2025
    assert W.pick_version(versions, 2021, caps, "capture")[0].num == 9812 == W.pick_version(versions, 2021, caps, "publication")[0].num
    assert W.pick_version(versions, 2019, caps, "capture")[0].num == 58924
    # a version without a capture date falls back to its publication date (and is the choice when nearest)
    r, info = W.pick_version(versions, 2014, caps, "capture")
    assert r.num == 10 and info["fallback"] is True and info["date"] == "2014-02-20" and info["capture_date"] is None
    assert W.pick_version(versions, 2014, None, "capture")[1]["n_fallback"] == 4     # no captures at all
    # ties on the capture date -> the newer publication
    same = {22869: dt.date(2021, 1, 1), 9812: dt.date(2021, 1, 1)}
    assert W.pick_version([by_num[9812], by_num[22869]], 2021, same, "capture")[0].num == 22869
    assert W.pick_version([], 2021, caps, "capture")[0] is None
    with pytest.raises(ValueError):
        W.pick_version(versions, 2021, caps, "release")
    for bad in (None, "Null", "", "2025-04-24", 2025):
        assert W.capture_date({"SRC_DATE": bad}) is None
    assert W.capture_date({"SRC_DATE": 20250424}) == dt.date(2025, 4, 24) == W.capture_date({"SRC_DATE": " 20250424 "})
    assert W.capture_date(None) is None


def test_choose_release_by_capture_identifies_each_version_once(tmp_path):
    F = _load("fetch_wayback_vigor")
    svc = FakeService()
    cl = _client(svc, cache_dir=tmp_path / "tiles")
    rel = _releases(svc)
    walks = F.Walks(tmp_path / "walks.json", newest=rel[-1].num)
    meta = F.Metadata(tmp_path / "metadata.json")
    tile = dict(city="Chicago", sat=SAT, lat=LAT, lon=LON)
    r, versions, info = F.choose_release(walks, cl, rel, tile, 2025, 19, meta, "capture")
    assert r.num == 22869 and len(versions) == 4 and info["capture_date"] == "2022-06-20" and info["n_fallback"] == 1
    assert meta.n_fetched == 4 and sum("/identify?" in u for u in svc.calls) == 4
    r2, _, _ = F.choose_release(walks, cl, rel, tile, 2019, 19, meta, "capture")
    assert r2.num == 58924 and meta.n_fetched == 4                                 # cached: no new identify
    rp, _, ip = F.choose_release(walks, cl, rel, tile, 2025, 19, None, "publication")
    assert rp.num == 58924 and ip["select_by"] == "publication" and sum("/identify?" in u for u in svc.calls) == 4
    with pytest.raises(ValueError, match="metadata"):
        F.choose_release(walks, cl, rel, tile, 2025, 19, None, "capture")
    meta.save()
    meta2 = F.Metadata(tmp_path / "metadata.json")
    assert len(meta2.d) == 4 and meta2.n_fetched == 0


def test_dry_run_prints_the_capture_year_histogram_and_the_request_count(tmp_path, capsys):
    root = _make(tmp_path)
    for f in ("pano_label_balanced.txt", "same_area_balanced_test.txt", "same_area_balanced_train.txt", "satellite_list.txt"):
        p = root / "splits" / "VIGOR" / "Chicago" / f
        p.write_text(p.read_text().replace("s1.png", SAT))
    corrected(p.parent)
    F = _load("fetch_wayback_vigor")
    svc = FakeService()
    counts = F.main(["--root", str(root), "--split", "samearea", "--cities", "Chicago", "--years", "2025", "2019",
                     "--dry-run"], opener=svc)
    out = capsys.readouterr().out
    assert counts["dry"] == 2 and not (root / "Chicago" / "wayback_2025").exists()
    assert "release per year selected by capture date" in out
    assert "year 2025: releases chosen 2026-03-26 (#22869) x1" in out
    assert "year 2025: capture years of the chosen versions 2022 x1 (100%)" in out
    assert "year 2019: releases chosen 2025-09-25 (#58924) x1" in out
    assert "year 2019: capture years of the chosen versions 2020 x1 (100%)" in out
    assert "metadata identify requests this run: 4 (cached: 0)" in out
    assert "capture date missing: 2 version(s) fell back" in out           # release 10, once per year
    assert sum("/identify?" in u for u in svc.calls) == 4 and not any("/tile/" in u for u in svc.calls)
    # the old rule on request: no identify, the 2025-09-25 release for 2025
    svc2 = FakeService()
    F.main(["--root", str(root), "--split", "samearea", "--cities", "Chicago", "--years", "2025", "--dry-run",
            "--select-by", "publication"], opener=svc2)
    out = capsys.readouterr().out
    assert "selected by publication date" in out and "year 2025: releases chosen 2025-09-25 (#58924) x1" in out
    assert "capture years of the chosen versions 2020 x1" in out            # still reported from the sidecar metadata
    assert (root / "wayback_tiles" / "metadata.json").exists()


# ---- cross-source calibration: gradient domain, constrained search, PSR ----------------------------------------------

def _city_block(seed=0, n=320):
    """Synthetic imagery with sharp structure: 60 flat rectangles (roofs, roads), band-limited (sigma 1 px)."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n), 110.0)
    for _ in range(60):
        x0, y0 = rng.integers(0, n - 20, 2)
        w, h = rng.integers(8, 70, 2)
        img[y0:y0 + h, x0:x0 + w] = rng.uniform(40, 220)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def _cars(img, seed, k=60):
    """k random elongated +-blobs ("cars", 60-120 grey levels): differ between the two sources."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    for _ in range(k):
        cx, cy = rng.uniform(0, img.shape[1], 2)
        out += rng.choice([-1, 1]) * rng.uniform(60, 120) * np.exp(-((xx - cx) ** 2 / 6 + (yy - cy) ** 2 / 2))
    return out


def _cross_source_pair(shift, blur=2.0, k=60, noise=3.0):
    """(google, esri): the same block; Esri blurred by `blur` px, its own cars, another gain / offset, sensor noise,
    and its content shifted by `shift` (google (u, v) -> esri (u + dx, v + dy))."""
    base = _city_block()
    a = _cars(base, 1, k) + np.random.default_rng(4).normal(0, noise, base.shape)
    b = _cars(cv2.GaussianBlur(base, (0, 0), blur), 2, k) * 0.6 + 40
    b = W.fourier_shift(b, *shift) + np.random.default_rng(3).normal(0, noise, base.shape)
    return a, b


@pytest.mark.parametrize("shift", [(3.27, -1.63), (-7.4, 12.9), (0.5, 0.5), (10.2, 4.4)])
def test_gradient_constrained_psr_recovers_a_cross_source_shift_the_intensity_path_rejects(shift):
    a, b = _cross_source_pair(shift)
    g = W.calib_match(a, b, "gradient", blur_sigma=2.0, max_shift_px=24, upsample=20)
    assert abs(g["dx"] - shift[0]) < 0.1 and abs(g["dy"] - shift[1]) < 0.1
    assert g["psr"] >= 7.0                                           # above the default acceptance
    # the previous calibration (whole-image intensity phase correlation, absolute response floor 0.05) rejects it
    dx, dy, resp = W.phase_correlation(a, b, upsample=20)
    assert resp < 0.05
    # and the intensity domain of the new path, without the blur, is off by more than the gradient path's tolerance
    i = W.calib_match(a, b, "intensity", blur_sigma=0.0, max_shift_px=24, upsample=20, band=False)
    assert max(abs(i["dx"] - shift[0]), abs(i["dy"] - shift[1])) > 0.1


def test_constrained_search_ignores_a_stronger_peak_outside_the_radius():
    """Two copies of the content: a strong one far away (a repeated facade / the wrong block) and a weaker one at a
    small shift: unconstrained -> the far peak; constrained to 24 px -> the near one, to 0.1 px, with a PSR."""
    base = _city_block(seed=5)
    near, far = (2.4, -1.3), (41.0, -33.0)
    b = W.fourier_shift(base, *far) + 0.6 * W.fourier_shift(base, *near)
    A, B = W.calib_preprocess(base, "gradient"), W.calib_preprocess(b, "gradient")
    free = W.phase_correlation_psr(A, B, upsample=20)
    assert abs(free["dx"] - far[0]) < 0.5 and abs(free["dy"] - far[1]) < 0.5
    con = W.phase_correlation_psr(A, B, upsample=20, max_shift_px=24)
    assert abs(con["dx"] - near[0]) < 0.1 and abs(con["dy"] - near[1]) < 0.1 and con["psr"] > 7.0


def test_psr_is_noise_level_for_unrelated_images_and_blur_sigma_rule():
    psrs = []
    for s in range(5):
        a = _city_block(seed=10 + s)
        b = _city_block(seed=20 + s)
        psrs.append(W.calib_match(a, b, "gradient", 2.0, 24)["psr"])
    assert max(psrs) < 7.0                                           # unrelated content stays below the default
    assert W.calib_blur_sigma(0.5, 0.111, 0.125, 0.5) == pytest.approx(0.5 * math.sqrt(0.25 - 0.111 ** 2) / 0.125)
    assert W.calib_blur_sigma(0.1, 0.111, 0.125, 0.5) == 0.0 and W.calib_blur_sigma(None, 0.111, 0.125) == 0.0
    with pytest.raises(ValueError):
        W.calib_preprocess(np.zeros((8, 8)), "sobel")


def test_offline_pairs_dir_calibrates_without_network(tmp_path):
    """--pairs-dir: windows + sidecars under <dir>/<City>/wayback_calib_<year>/ and the VIGOR tiles under
    <dir>/<City>/satellite/: the planted offset is recovered, nothing is requested."""
    planted = (2.6, -1.4)
    root = _layout(tmp_path, offset_px=planted)
    svc = FakeService()
    cal = _load("calibrate_wayback_vigor")
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cal.calibrate(cal.build_parser().parse_args(["--root", str(root), "--city", "Chicago", "--tiles", "5"]), cfg,
                  opener=svc)                                         # writes the uncalibrated window once
    W.calibration_path(root, "Chicago").unlink()
    def boom(url, headers):
        raise AssertionError(f"network in offline mode: {url}")
    out = cal.calibrate(cal.build_parser().parse_args(["--pairs-dir", str(root), "--city", "Chicago", "--min-tiles",
                                                       "1", "--upsample", "50"]), cfg, opener=boom)
    assert out["n"] == 1 and out["n_used"] == 1 and out["pass"] is True
    assert abs(out["offset_px"][0] - planted[0]) < 0.05 and abs(out["offset_px"][1] - planted[1]) < 0.05
    assert out["per_tile"][0]["release"] == 9812 and out["source"].startswith("offline")
    assert W.calibration_path(root, "Chicago").exists()


# ---- concurrency: worker pool under a global rate limit -------------------------------------------------------------

def _multi_tile_root(tmp_path, n=8):
    """A layout with only satellite_list.txt (enough for `--all`): n tiles around LAT, LON, ~100 m apart."""
    names = [f"satellite_{LAT + 0.001 * (i // 3):.6f}_{LON + 0.0012 * (i % 3):.6f}.png" for i in range(n)]
    d = tmp_path / "splits" / "VIGOR" / "Chicago"
    d.mkdir(parents=True)
    (d / "satellite_list.txt").write_text("\n".join(names) + "\n")
    (d / "pano_label_balanced__corrected.txt").write_text("")
    return tmp_path, names


def test_worker_pool_writes_the_same_files_as_sequential(tmp_path):
    F = _load("fetch_wayback_vigor")
    outs = {}
    for workers in (1, 4):
        root, names = _multi_tile_root(tmp_path / f"w{workers}")
        svc = FakeService()
        counts = F.main(["--root", str(root), "--cities", "Chicago", "--all", "--years", "2021", "2025",
                         "--workers", str(workers)], opener=svc)
        assert counts["written"] == 2 * len(names) and not counts.get("missing")
        files = {}
        for year in (2021, 2025):
            for nm in names:
                png = root / "Chicago" / f"wayback_{year}" / nm
                side = json.loads(png.with_suffix(".json").read_text())
                side.pop("fetched_at")
                files[(year, nm)] = (png.read_bytes(), side)
        assert not list((root / "wayback_tiles").rglob("*.part"))    # every cache write completed and renamed
        walks = json.loads((root / "wayback_tiles" / "tilemap_z19.json").read_text())
        outs[workers] = (files, walks, json.loads((root / "wayback_tiles" / "metadata.json").read_text()))
    f1, w1, m1 = outs[1]
    f4, w4, m4 = outs[4]
    assert f1.keys() == f4.keys()
    for k in f1:
        assert f1[k][0] == f4[k][0], f"PNG differs: {k}"             # byte-identical imagery
        assert f1[k][1] == f4[k][1], f"sidecar differs: {k}"
    assert w1 == w4 and m1 == m4


def test_rate_limit_is_global_over_threads():
    """4 threads x 15 requests at rate_hz 100: every request start is >= 1 / rate after the previous one, over all
    threads (so <= rate_hz per second), while requests themselves overlap (they take 30 ms each)."""
    import threading
    starts, lock = [], threading.Lock()
    inflight, peak = [0], [0]

    def slow(url, headers):
        with lock:
            starts.append(time.monotonic())
            inflight[0] += 1
            peak[0] = max(peak[0], inflight[0])
        time.sleep(0.03)
        with lock:
            inflight[0] -= 1
        return b"x"
    cl = W.WaybackClient(rate_hz=100.0, retries=0, opener=slow)
    ths = [threading.Thread(target=lambda: [cl.get(f"u{i}") for i in range(15)]) for _ in range(4)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    st = np.sort(starts)
    assert len(st) == 60 and cl.n_requests == 60 and cl.n_bytes == 60
    assert np.diff(st).min() >= 0.01 - 2e-3                          # spacing 1 / rate_hz (timer slack)
    assert (st[-1] - st[0]) >= 59 * 0.01 - 0.01                      # 60 starts need >= 59 intervals
    assert peak[0] >= 2                                              # ... but the requests overlapped


def test_fetch_applies_only_a_passing_calibration_and_survives_an_empty_one(tmp_path, capsys):
    """A FAIL calibration (or one without any usable tile: offset_m None, which crashed the Eagle dry run 8813461) is
    not applied: offset 0, the status and reason recorded in the sidecar; --apply-failed-calibration applies a FAIL
    offset that exists. Resumable WITHOUT --force: a change of the applied offset re-renders the window, the same
    offset skips it (and only refreshes the recorded status)."""
    root = _layout(tmp_path)
    F = _load("fetch_wayback_vigor")
    p = W.calibration_path(root, "Chicago")
    args = ["--root", str(root), "--split", "samearea", "--cities", "Chicago", "--years", "2021"]
    side = root / "Chicago" / "wayback_2021" / SAT.replace(".png", ".json")
    rd = lambda: json.loads(side.read_text())                        # noqa: E731
    p.write_text(json.dumps({"offset_m": None, "offset_px": None, "residual_m": None, "n_used": 0, "pass": False}))
    assert F.main(args + ["--dry-run"], opener=FakeService())["dry"] == 1          # the 8813461 crash path
    assert "no usable tile" in capsys.readouterr().out
    assert F.main(args, opener=FakeService())["written"] == 1
    s = rd()
    assert s["offset_px"] == [0.0, 0.0] and s["calibration"] is None
    assert s["calibration_status"]["status"] == "not_applied" and "no usable tile" in s["calibration_status"]["reason"]
    p.write_text(json.dumps({"offset_m": [0.25, -0.125], "residual_m": 0.9, "n_used": 5, "pass": False}))
    assert F.main(args, opener=FakeService())["skipped"] == 1                       # same offset 0: not re-rendered
    s = rd()
    assert s["offset_px"] == [0.0, 0.0] and s["calibration_status"]["status"] == "not_applied"
    assert "FAIL (residual 0.9 m over 5 tiles)" in s["calibration_status"]["reason"]   # the record follows the file
    assert s["calibration_status"]["offset_m"] == [0.25, -0.125]
    assert F.main(args + ["--apply-failed-calibration"], opener=FakeService())["written"] == 1   # new offset: re-render
    s = rd()
    assert np.allclose(s["offset_px"], [0.25 / 0.125, -0.125 / 0.125]) and s["calibration_status"]["status"] == "applied_fail"
    assert F.main(args, opener=FakeService())["written"] == 1              # back to the default policy: offset 0 again
    assert rd()["offset_px"] == [0.0, 0.0]
    p.write_text(json.dumps({"offset_m": [0.25, -0.125], "residual_m": 0.1, "n_used": 40, "pass": True}))
    assert F.main(args, opener=FakeService())["written"] == 1
    s = rd()
    assert np.allclose(s["offset_px"], [2.0, -1.0]) and s["calibration_status"]["status"] == "applied"
    assert s["calibration"].endswith("wayback_calibration.json")
    assert F.main(args + ["--no-calibration"], opener=FakeService())["written"] == 1
    assert rd()["calibration_status"]["status"] == "disabled" and rd()["offset_px"] == [0.0, 0.0]
    assert W.offset_px_for({"offset_m": None}, 0.125) == (0.0, 0.0)


def test_rim_guard_rejects_a_peak_outside_the_search_radius():
    """A true shift just outside the radius: the constrained search must not return one of its ringing lobes inside
    the disc (which has a PSR far above 7); it returns the outside peak flagged `edge`, and the aggregate drops it.
    Shifts just inside the radius are measured to 0.15 px and not flagged."""
    base = _city_block(seed=5)
    A = W.calib_preprocess(base, "gradient")
    for shift in ((24.3, 0.0), (24.6, 0.0), (17.5, 17.8), (0.0, -25.5)):            # |shift| > 24 px
        r = W.phase_correlation_psr(A, W.calib_preprocess(W.fourier_shift(base, *shift), "gradient"), 20, 24)
        assert r["edge"] and math.hypot(r["dx"], r["dy"]) > 24
    for shift in ((23.2, 0.0), (-16.6, 16.6), (-20.1, -8.7), (0.3, -0.4)):          # |shift| < 24 px
        r = W.phase_correlation_psr(A, W.calib_preprocess(W.fourier_shift(base, *shift), "gradient"), 20, 24)
        assert not r["edge"] and abs(r["dx"] - shift[0]) < 0.15 and abs(r["dy"] - shift[1]) < 0.15
    cal = _load("calibrate_wayback_vigor")
    off = np.array([[2.0, -1.0]] * 3 + [[23.9, 0.0]])
    agg = cal.aggregate(off, np.full(4, 20.0), 0.125, 7.0, 40.0, 0.3, 3, edges=[False, False, False, True])
    assert agg["n_edge"] == 1 and agg["n_used"] == 3 and agg["offset_px"] == [2.0, -1.0]


def test_a_failing_tile_is_counted_reported_and_the_others_are_written(tmp_path, monkeypatch, capsys):
    """A worker exception (e.g. a request failing after its retries) must not vanish in the pool nor stop the other
    tiles: it is counted as `error`, printed, and the CLI exits 1."""
    F = _load("fetch_wayback_vigor")
    root, names = _multi_tile_root(tmp_path, n=6)
    real = F.fetch_tile

    def flaky(client, rel, tile, *a, **k):
        if tile["sat"] == names[2]:
            raise RuntimeError("GET ... failed after 4 attempts")
        return real(client, rel, tile, *a, **k)
    monkeypatch.setattr(F, "fetch_tile", flaky)
    counts = F.main(["--root", str(root), "--cities", "Chicago", "--all", "--years", "2021", "--workers", "3"],
                    opener=FakeService())
    assert counts["error"] == 1 and counts["written"] == len(names) - 1
    out = capsys.readouterr().out
    assert f"ERROR {names[2]}: RuntimeError" in out and "1 tile(s) FAILED" in out
    assert not (root / "Chicago" / "wayback_2021" / names[2]).exists()
