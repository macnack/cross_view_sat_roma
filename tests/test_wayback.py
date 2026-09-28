"""Esri Wayback reference years (task 05): Mercator <-> tile-frame geometry, release selection, the version walk and
the mosaic on a fake tile server, the phase-correlation calibration on planted offsets, VigorPairs' ref_source and the
cross-source union of modes in scripts/eval_vigor.py. CPU, no network (the HTTP layer is a fake opener), no data."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
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
from test_vigor_window import _make  # noqa: E402
from tiny_satroma import StubPictureQuery, plant_translation, tiny_decoder, tiny_matcher  # noqa: E402

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
    "58924": ("2025-09-25", 9812), "22869": ("2026-03-26", None),
}
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
                {"layerId": 4, "layerName": "30cm", "attributes": {"SRC_DATE": f"{RELEASES[rel][0][:4]}0424",
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


# ---- fake service: walk, mosaic, fetch ---------------------------------------------------------------------------

def test_version_walk_lists_distinct_owners_in_one_request_each():
    svc = FakeService()
    cl = _client(svc)
    rel = _releases(svc)
    x, y = W.tile_xy(*W.latlon_to_merc_px(LAT, LON, 19))
    v = cl.versions_at(rel, 19, x, y)
    assert [r.num for r in v] == [22869, 9812, 10]
    assert sum("/tilemap/" in u for u in svc.calls) == 3
    assert W.pick_release(v, 2025).num == 22869 and W.pick_release(v, 2021).num == 9812 and W.pick_release(v, 2019).num == 9812


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
    a = cal.build_parser().parse_args(["--root", str(root), "--city", "Chicago", "--split", "samearea", "--tiles", "5",
                                       "--upsample", "50"])
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    out = cal.calibrate(a, cfg, opener=svc)
    assert out["n_used"] == 1 and out["releases"] == {"2021-02-24 (#9812)": 1}
    assert abs(out["offset_px"][0] - planted[0]) < 0.05 and abs(out["offset_px"][1] - planted[1]) < 0.05
    assert out["pass"] is True and out["residual_m"] == 0.0            # one tile: no spread
    assert np.allclose(out["offset_m"], np.multiply(out["offset_px"], out["gsd_m"]))
    assert W.calibration_path(root, "Chicago").exists()
    assert (root / "Chicago" / "wayback_calib_2021" / SAT).exists()
    side = json.loads((root / "Chicago" / "wayback_calib_2021" / SAT.replace(".png", ".json")).read_text())
    assert side["release"] == 9812 and side["offset_px"] == [0.0, 0.0] and side["capture"]["SRC_DATE"] == "20210424"
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


def test_aggregate_is_robust_to_outliers_and_low_response():
    cal = _load("calibrate_wayback_vigor")
    rng = np.random.default_rng(1)
    off = np.array([[2.0, -1.0]]) + rng.normal(0, 0.3, (40, 2))
    off = np.vstack([off, [[30.0, 30.0], [2.0, -1.0]]])              # a gross outlier, a low-response tile
    resp = np.r_[np.full(41, 0.5), 0.01]
    agg = cal.aggregate(off, resp, 0.125, 0.05, 4.0, 0.3)
    assert agg["n"] == 42 and agg["n_low_response"] == 1 and agg["n_outliers"] == 1 and agg["n_used"] == 40
    assert abs(agg["offset_px"][0] - 2.0) < 0.15 and abs(agg["offset_px"][1] + 1.0) < 0.15
    assert 0.0 < agg["residual_m"] < 0.1 and agg["pass"]
    assert cal.aggregate(off, np.zeros(42), 0.125, 0.05, 4.0, 0.3)["offset_px"] is None


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
