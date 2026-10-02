"""Consensus sweep (scripts/sweep_consensus_vigor.py, bevloc.match.consensus): the cache reproduces the online
consensus exactly, the swept factors do what they claim, selection refuses the test draw, and the chosen settings
round-trip through eval_vigor.py --consensus-json. CPU, synthetic categoricals, no weights, no data."""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bevloc.match.consensus import (
    ConsensusCfg, apply_settings, extract_modes, load_chosen, select_modes, settings_of,
)
from bevloc.match.satroma import SatRoMa, consensus_for_query, find_gaussians
from bevloc.match.se2 import se2_ransac

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_satroma import cfg_stub, tiny_decoder, tiny_matcher  # noqa: E402

K = 56
ROOT = Path(__file__).resolve().parents[1]


def _load(name, file):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SW = _load("bevloc_scripts_sweep_consensus_vigor", "sweep_consensus_vigor.py")
EV = _load("bevloc_scripts_eval_vigor", "eval_vigor.py")


def _logits(seed, h=14, w=14, t=None, outlier=0.3, bimodal=0.3):
    """Planted translation t (cells) + outlier tokens + second modes, on N(0, 1) noise."""
    g = np.random.default_rng(seed)
    gm = g.normal(0, 1.0, (K * K, h, w)).astype(np.float32)
    t = g.uniform(10, 30, 2) if t is None else np.asarray(t, float)
    for r in range(h):
        for c in range(w):
            if g.random() < outlier:
                x, y = g.integers(0, K, 2)
            else:
                x, y = int(round(c + t[0] + g.normal(0, 0.7))), int(round(r + t[1] + g.normal(0, 0.7)))
            x, y = np.clip([x, y], 0, K - 1)
            gm[y * K + x, r, c] += 9.0
            if g.random() < bimodal:
                x2, y2 = g.integers(0, K, 2)
                gm[y2 * K + x2, r, c] += 8.5
    return torch.from_numpy(gm), t


class _Picture:
    """A picture query: no placement (the package path, `_ransac`)."""


class _Placed:
    def __init__(self, xy, valid):
        self.xy, self.valid = xy, valid

    def placement(self, batch):
        return self.xy[None], self.valid[None]


def _cons(solver="se2"):
    return SatRoMa.from_wrapper(tiny_matcher(tiny_decoder()).wrapper, cfg_stub(solver), use_means=False,
                                min_valid_frac=0.05)


def _sample(kind, seed):
    """(query, batch, frac, gm, cert) of one synthetic decoded sample."""
    if kind == "picture":
        gm, _ = _logits(seed)
        frac = torch.ones(1, 14, 14)
        frac[0, :3, :5] = 0.0                                        # masked tokens vote nothing
        query = _Picture()
    else:
        gm, _ = _logits(seed, h=14, w=28)
        g = torch.Generator().manual_seed(seed)
        xy = torch.rand(14, 28, 2, generator=g) * 223.0
        valid = torch.rand(14, 28, generator=g) > 0.2
        query, frac = _Placed(xy, valid), None
    cert = torch.from_numpy(np.random.default_rng(seed + 100).normal(0, 2, gm.shape[-2:]).astype(np.float32))
    return query, {}, frac, gm, cert


META = dict(im_a_size=224, im_b_size=896, seed=0, n=224, cell_m=0.125)


def _same(m1, m2):
    assert (m1.H is None) == (m2.H is None)
    if m1.H is not None:
        assert np.array_equal(m1.H, m2.H)
    assert m1.inlier_ratio == m2.inlier_ratio and m1.n_modes == m2.n_modes and m1.n_inliers == m2.n_inliers


# ---- the cache reproduces the consensus -----------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["picture", "placed"])
@pytest.mark.parametrize("solver", ["srt", "sim", "se2"])
@pytest.mark.parametrize("means", [False, True])
def test_cache_reproduces_the_default_consensus_exactly(kind, solver, means):
    cons = _cons(solver)
    cons.use_means = means
    for seed in range(2):
        query, batch, frac, gm, cert = _sample(kind, seed)
        online = consensus_for_query(cons, gm, query, batch, frac, 224, certainty=cert, stats=False)
        rec = SW.frame_record(cons, gm, query, batch, frac, 224, cert, 0.004)
        cached = SW.solve(rec, settings_of(cons), SW.consensus_stub(META))
        _same(online, cached)
        assert online.H is not None and online.n_modes > 0


@pytest.mark.parametrize("kind", ["picture", "placed"])
@pytest.mark.parametrize("c", [
    ConsensusCfg(reproj_cells=1.5, max_modes=2, mode_thr=0.016, solver="se2", ransac="4x"),
    ConsensusCfg(reproj_cells=1.0, target="means", max_modes=1, cert="filter", solver="srt"),
    ConsensusCfg(reproj_cells=2.0, cert="weight", solver="sim", mode_thr=0.03),
    ConsensusCfg(reproj_cells=0.75, cert="weight", solver="srt"),
    ConsensusCfg(reproj_cells=0.5, cert="weight", solver="se2", ransac="4x"),
])
def test_cache_reproduces_any_swept_configuration(kind, c):
    """The evaluator with the settings applied (the --consensus-json path) == the sweep on the cache."""
    cons = apply_settings(_cons(), c)
    query, batch, frac, gm, cert = _sample(kind, 3)
    online = consensus_for_query(cons, gm, query, batch, frac, 224, certainty=cert, stats=False)
    rec = SW.frame_record(_cons(), gm, query, batch, frac, 224, cert, 0.004)
    _same(online, SW.solve(rec, c, SW.consensus_stub(META)))


@pytest.mark.parametrize("thr", [0.004, 0.008, 0.016, 0.03, 0.06])
def test_mode_threshold_is_an_exact_filter_of_the_loosest_extraction(thr):
    gm, _ = _logits(5, bimodal=0.6)
    loose = extract_modes(gm, 0.004)
    idx = select_modes(loose, ConsensusCfg(mode_thr=thr))
    pts_A, means_B, peaks_B, _ = find_gaussians(gm, adaptive_gauss_fit=False, log_missing_gaussians=False,
                                                fixed_threshold=thr, fixed_window_size=4)
    assert len(idx) == len(pts_A) > 0
    assert np.array_equal(loose["tok"][idx], pts_A)
    assert np.array_equal(loose["means"][idx], means_B)
    assert np.array_equal(loose["peaks"][idx], peaks_B)


# ---- the factors ----------------------------------------------------------------------------------------------------

def _record(tok, peaks, heights=None):
    n = len(tok)
    return dict(placed=False, valid=np.ones((14, 14), bool), xy=None, cert=None, in_dim=14, out_dim=K,
                modes=dict(tok=np.asarray(tok, np.float32), peaks=np.asarray(peaks, np.float32),
                           means=np.asarray(peaks, np.float32), mass=np.ones(n, np.float32),
                           height=np.full(n, 0.1, np.float32) if heights is None else np.asarray(heights, np.float32)))


def _shift(m):
    """Translation (cells) of a Match: query centre -> reference, 16 px per patch and per cell."""
    p = m.H @ np.array([111.5, 111.5, 1.0])
    return (p[:2] / p[2] - 111.5) / 16.0


@pytest.mark.parametrize("solver", ["se2", "srt", "sim"])
def test_tighter_threshold_picks_the_tighter_cluster(solver):
    """70 tokens agree on tA within 0.2 cells; 126 tokens agree on tB but scattered over +-2.5 cells. At 3 cells the
    bigger (loose) cluster wins, at 0.5 cells the tight one."""
    g = np.random.default_rng(0)
    rc = np.stack(np.meshgrid(np.arange(14), np.arange(14)), -1).reshape(-1, 2).astype(float)   # (col, row)
    perm = g.permutation(len(rc))
    a, b = perm[:70], perm[70:]
    tA, tB = np.array([20.0, 15.0]), np.array([30.0, 26.0])
    tgt = np.empty_like(rc)
    tgt[a] = rc[a] + tA + g.uniform(-0.2, 0.2, (len(a), 2))
    tgt[b] = rc[b] + tB + g.uniform(-2.5, 2.5, (len(b), 2))
    rec = _record(rc, tgt)
    stub = SW.consensus_stub(META)
    wide = SW.solve(rec, ConsensusCfg(reproj_cells=3.0, solver=solver), stub)
    tight = SW.solve(rec, ConsensusCfg(reproj_cells=0.5, solver=solver), stub)
    assert np.linalg.norm(_shift(wide) - tB) < 1.0
    assert np.linalg.norm(_shift(tight) - tA) < 0.25


def test_max_modes_one_equals_the_peak_row_without_duplicates():
    """The peak row repeats a token's argmax once per mode; max_modes=1 keeps one correspondence per token (its
    highest mode), i.e. exactly the peak row's distinct (token, peak) pairs, and on a unimodal set it is the peak row."""
    gm, _ = _logits(7, bimodal=0.7)
    md = extract_modes(gm, 0.008)
    tok = md["tok"]
    assert len(np.unique(tok, axis=0)) < len(tok)                    # the synthetic set is multimodal
    idx1 = select_modes(md, ConsensusCfg(max_modes=1))
    assert len(idx1) == len(np.unique(tok, axis=0))
    _, first = np.unique(tok, axis=0, return_index=True)             # peak targets are identical within a token
    assert np.array_equal(np.sort(idx1), idx1)
    assert np.array_equal(md["peaks"][idx1], md["peaks"][np.sort(first)])
    for i in idx1:                                                   # the kept mode is its token's highest
        same = np.all(tok == tok[i], axis=1)
        assert md["height"][i] == md["height"][same].max()
    stub = SW.consensus_stub(META)
    rec = dict(placed=False, valid=np.ones((14, 14), bool), xy=None, cert=None, in_dim=14, out_dim=K, modes=md)
    dedup = {k: v[np.sort(first)] for k, v in md.items()}
    _same(SW.solve(rec, ConsensusCfg(max_modes=1), stub),
          SW.solve(dict(rec, modes=dedup), ConsensusCfg(), stub))
    gm1, _ = _logits(8, bimodal=0.0)                                 # unimodal: the cap changes nothing
    md1 = extract_modes(gm1, 0.008)
    assert len(np.unique(md1["tok"], axis=0)) == len(md1["tok"])
    rec1 = dict(rec, modes=md1)
    _same(SW.solve(rec1, ConsensusCfg(max_modes=1), stub), SW.solve(rec1, ConsensusCfg(), stub))


def test_certainty_filter_drops_the_lowest_quartile_of_valid_tokens():
    gm, _ = _logits(9)
    md = extract_modes(gm, 0.008)
    cert = np.arange(196, dtype=float).reshape(14, 14)
    idx = select_modes(md, ConsensusCfg(cert="filter"), cert, np.ones((14, 14), bool))
    t = md["tok"].astype(int)
    kept = cert[t[idx, 1], t[idx, 0]]
    assert kept.min() >= np.quantile(cert, 0.25)
    assert len(idx) < len(t)


def test_weighted_se2_samples_by_weight_and_uniform_is_unchanged():
    g = np.random.default_rng(1)
    a = g.uniform(0, 14, (60, 2))
    b = a + np.array([5.0, 3.0])
    b[:40] = g.uniform(0, 56, (40, 2))                               # 40 outliers, 20 inliers
    H0, m0 = se2_ransac(a, b, thresh=0.5, n_iter=3, seed=0)
    H1, m1 = se2_ransac(a, b, thresh=0.5, n_iter=3, seed=0, weights=None)
    assert (H0 is None and H1 is None) or np.array_equal(H0, H1)
    w = np.r_[np.full(40, 1e-6), np.ones(20)]                        # weights on the inliers: 3 trials suffice
    Hw, mw = se2_ransac(a, b, thresh=0.5, n_iter=3, seed=0, weights=w)
    assert Hw is not None and mw[40:].all() and np.allclose(Hw[:2, 2], [5.0, 3.0], atol=1e-6)


# ---- protocol and round trip ----------------------------------------------------------------------------------------

def _cache(tmp_path, draw, ids, seed0, name=None):
    cons = _cons("se2")
    frames = []
    for j, fid in enumerate(ids):
        query, batch, frac, gm, cert = _sample("picture", seed0 + j)
        rec = SW.frame_record(cons, gm, query, batch, frac, 224, cert, 0.004)
        H = np.eye(3)
        H[:2, 2] = 16.0 * _logits(seed0 + j)[1]                       # the planted translation, in px
        rec.update(id=fid, city="Chicago", centre_guess_m=0.0, H_gt=H)
        m0 = consensus_for_query(cons, gm, query, batch, frac, 224, certainty=cert, stats=False)
        rec["pose_online_m"] = SW.coarse_error(m0.H, rec, META)
        rec["online"] = SW.match_summary(m0)
        frames.append(rec)
    meta = dict(META, draw=draw, draw_info=dict(draw=draw), ids=list(ids), cache_thr=0.004, ckpt="x.pt",
                config="c.yaml", consensus=settings_of(cons).to_dict())
    p = tmp_path / f"{name or draw}.pkl"
    with open(p, "wb") as f:
        pickle.dump(dict(meta=meta, frames=frames), f)
    return p


def _args(tmp_path, *extra):
    return SW.build_parser().parse_args(["--tag", "t", "--out", str(tmp_path), "--workers", "1", "--n-boot", "50",
                                         "--reproj", "1", "3", "--targets", "peak", "--max-modes", "1", "all",
                                         "--mode-thr", "0.008", "0.03", "--cert", "off", "--solvers", "se2",
                                         "--ransac", "default", *extra])


def test_selection_refuses_the_test_draw(tmp_path):
    test = _cache(tmp_path, "test", ["Chicago/a", "Chicago/b"], 0)
    calib = _cache(tmp_path, "calib", ["Chicago/b", "Chicago/c"], 10)
    with pytest.raises(SystemExit, match="refusing to select"):
        SW.sweep(_args(tmp_path), test, None)
    with pytest.raises(SystemExit, match="share 1 frames"):
        SW.sweep(_args(tmp_path), calib, test)


def test_calibration_split_is_deterministic_by_id_and_disjoint():
    recs = [dict(id=f"Chicago/p{i}") for i in range(11)]
    sel, conf = SW.split_calib(recs)
    sel2, conf2 = SW.split_calib(list(reversed(recs)))

    def ids(rs):
        return sorted(r["id"] for r in rs)
    assert ids(sel) == ids(sel2) and ids(conf) == ids(conf2)
    assert not set(ids(sel)) & set(ids(conf)) and len(sel) + len(conf) == 11 and abs(len(sel) - len(conf)) <= 1


@pytest.mark.parametrize("draw", ["calib", "test"])
def test_a_cache_that_does_not_reproduce_the_online_match_stops_the_sweep(tmp_path, draw):
    calib = _cache(tmp_path, "calib", [f"Chicago/c{i}" for i in range(4)], 60)
    test = _cache(tmp_path, "test", [f"Chicago/t{i}" for i in range(3)], 70)
    bad = calib if draw == "calib" else test
    with open(bad, "rb") as f:
        c = pickle.load(f)
    c["frames"][1]["online"]["H"] = c["frames"][1]["online"]["H"] + 1e-4    # off by more than 1e-6
    with open(bad, "wb") as f:
        pickle.dump(c, f)
    fid = c["frames"][1]["id"]
    with pytest.raises(SystemExit, match=fid):
        SW.sweep(_args(tmp_path), calib, test)
    res = SW.sweep(_args(tmp_path, "--allow-mismatch"), calib, test)
    assert res["mismatch"][f"{draw}_coarse"]["mismatches"] == 1 and res["mismatch"][f"{draw}_coarse"]["ids"] == [fid]
    d = json.loads((tmp_path / "sweep_consensus_t.json").read_text())
    assert d["mismatch"][f"{draw}_coarse"]["mismatches"] == 1


def test_uniform_certainty_is_the_weighted_code_path_with_equal_weights():
    query, batch, frac, gm, cert = _sample("picture", 4)
    rec = SW.frame_record(_cons(), gm, query, batch, frac, 224, cert, 0.004)
    stub = SW.consensus_stub(META)
    md = rec["modes"]
    for solver in ("srt", "sim", "se2"):
        c = ConsensusCfg(solver=solver, cert="uniform")
        idx = select_modes(md, c)
        apply_settings(stub, c)
        ref = SatRoMa.package_fit(stub, md["tok"][idx], md["peaks"][idx], md["means"][idx], 14, K, None,
                                  weights=np.ones(len(idx)))
        _same(SW.solve(rec, c, stub), ref)
        _same(SW.solve(dict(rec, cert=np.zeros_like(rec["cert"])), c, stub), ref)   # the certainty plays no role


def test_chosen_config_round_trips_through_consensus_json(tmp_path):
    calib = _cache(tmp_path, "calib", [f"Chicago/c{i}" for i in range(6)], 20)
    test = _cache(tmp_path, "test", [f"Chicago/t{i}" for i in range(3)], 40)
    res = SW.sweep(_args(tmp_path), calib, test)
    path = tmp_path / "sweep_consensus_t.json"
    d = json.loads(path.read_text())
    assert d["mismatch"]["calib_coarse"]["mismatches"] == 0 and d["mismatch"]["test_coarse"]["mismatches"] == 0
    assert d["coarse"]["split"]["n_select"] == 3 and d["coarse"]["split"]["n_confirm"] == 3
    assert d["chosen"]["test"] is not None and d["chosen"]["baseline"]["test"] is not None
    assert d["chosen"]["paired_delta_test"]["n"] == 3
    chosen = ConsensusCfg.from_dict(d["chosen"]["config"])
    assert chosen == ConsensusCfg.from_dict(res["chosen"]["config"])
    if d["chosen"]["rule"].startswith("baseline kept"):
        assert d["chosen"]["config"] == d["chosen"]["baseline"]["config"]
    best = ConsensusCfg.from_dict(
        SW.sweep(_args(tmp_path, "--no-guard", "--tag", "t2"), calib, test)["coarse"]["best"]["config"])
    assert load_chosen(tmp_path / "sweep_consensus_t2.json")[0] == best  # --no-guard: always the best
    a = EV.build_parser().parse_args(["--ckpt", "x", "--tag", "y", "--consensus-json", str(path)])
    coarse, fine = EV.chosen_consensus(a)
    assert coarse == chosen and fine is None
    assert EV.chosen_consensus(EV.build_parser().parse_args(["--ckpt", "x", "--tag", "y"])) == (None, None)
    cons = apply_settings(_cons(), coarse)
    assert settings_of(cons) == chosen
    # the evaluator with the applied settings gives the sweep's pose on the same frame
    with open(test, "rb") as f:
        rec = pickle.load(f)["frames"][0]
    query, batch, frac, gm, cert = _sample("picture", 40)
    _same(consensus_for_query(cons, gm, query, batch, frac, 224, certainty=cert, stats=False),
          SW.solve(rec, chosen, SW.consensus_stub(META)))
    # a fine block is read from `chosen_fine`
    d["chosen_fine"] = dict(config=ConsensusCfg(reproj_cells=1.0, solver="se2").to_dict())
    p2 = tmp_path / "with_fine.json"
    p2.write_text(json.dumps(d))
    assert load_chosen([p2])[1] == ConsensusCfg(reproj_cells=1.0, solver="se2")
    md = (tmp_path / "sweep_consensus_t.md").read_text()
    assert md.startswith("# Consensus sweep") and "Headline (test draw" in md


def test_thresholds_below_the_cache_are_refused(tmp_path):
    calib = _cache(tmp_path, "calib", ["Chicago/c0", "Chicago/c1"], 50)
    with pytest.raises(SystemExit, match="below the cache threshold"):
        SW.sweep(_args(tmp_path, "--mode-thr", "0.002"), calib, None)


def _vigor_setup(tmp_path, monkeypatch, n_labels=10):
    """Synthetic VIGOR root with n_labels copies of one panorama (planted toy decoders of test_two_pass), the fine
    decoder injected in place of eval_vigor.decoder_fine."""
    import shutil
    from test_two_pass import _setup
    root = tmp_path / "vigor"
    _, ds, query, matcher, cons, cfg, fine = _setup(root, (352.0, 320.0))
    lab = root / "splits" / "VIGOR" / "Chicago" / "pano_label_balanced.txt"
    line = lab.read_text()
    for k in range(2, n_labels + 1):
        shutil.copy(root / "Chicago" / "panorama" / "p1,1.0,.jpg", root / "Chicago" / "panorama" / f"p{k},1.0,.jpg")
        lab.write_text(lab.read_text() + line.replace("p1,", f"p{k},"))
    tm = dict(val_frac=0.2, val_samples=0, cities=["Chicago"])

    def fake_decoder_fine(fa, cfg_, ds_, dev):
        fine.ds.labels = ds_.labels
        return fine, {}
    monkeypatch.setattr(SW._ev(), "decoder_fine", fake_decoder_fine)
    a = SW.build_parser().parse_args(["--tag", "s", "--out", str(tmp_path), "--root", str(root),
                                      "--cities", "Chicago", "--fine-config", "f.yaml", "--fine-ckpt", "f.pt",
                                      "--workers", "1", "--n-boot", "20", "--reproj", "1", "3", "--targets", "peak",
                                      "--max-modes", "all", "--mode-thr", "0.008", "--cert", "off",
                                      "--solvers", "se2", "--ransac", "default"])
    M = dict(cfg=cfg, dev="cpu", mode="ipm", train_meta=tm, matcher=matcher, query=query, cons=cons["peak"],
             fine_train=tm, fine_args=a)
    return a, M


def test_cache_stage_on_synthetic_vigor_with_a_fine_pass_reproduces_both_passes(tmp_path, monkeypatch):
    """The caching pass (calibration draw: held-out training frames of both checkpoints) through the eval_vigor
    machinery with planted toy decoders, then the sweep: the cached defaults give the online coarse and fine Match."""
    a, M = _vigor_setup(tmp_path, monkeypatch)
    path = SW.cache_draw(a, "calib", M)
    with open(path, "rb") as f:
        c = pickle.load(f)
    assert c["meta"]["draw"] == "calib" and len(c["meta"]["ids"]) == 2    # 10 labels, the last 20 % held out
    assert c["meta"]["draw_info"]["n_heldout_both"] == 2
    r, rf = c["frames"][0], c["fine"]["frames"][0]
    assert r["pose_online_m"] is not None and rf["pose_online_m"] is not None and rf["online"]["H"] is not None
    assert r["topk_idx"].shape[1] == 8 and r["modes"]["tok"].shape[1] == 2
    assert c["fine"]["meta"]["centred_on"] == c["meta"]["consensus"]
    res = SW.sweep(a, path, None, M=M)
    assert res["mismatch"]["calib_coarse"]["mismatches"] == 0 and res["mismatch"]["calib_fine"]["mismatches"] == 0
    assert res["chosen_fine"]["config"]["solver"] == "se2"
    base = [x for x in res["fine"]["ofat"] if x["factor"] == "baseline"][0]
    assert base["select"]["median_m"] == pytest.approx(rf["pose_online_m"])
    assert not res["meta"]["fine_centring"]["recached"]           # the planted decoder: every coarse config ties
    a.passes = ["fine"]                                           # --passes fine: printed and recorded
    res = SW.sweep(a, path, None)
    assert "coarse" not in res and "centred on the caching coarse consensus" in res["meta"]["fine_centring"]["note"]


def test_fine_pass_is_recached_around_the_chosen_coarse_pose(tmp_path, monkeypatch):
    a, M = _vigor_setup(tmp_path, monkeypatch)
    chosen = ConsensusCfg(reproj_cells=1.0, solver="se2", mode_thr=0.016)
    cal, test = SW.recache_fine(a, M, chosen, want_test=False)
    assert test is None and (tmp_path / "sweep_cache" / "s_finec_calib.pkl").exists()
    assert ConsensusCfg.from_dict(cal["fine"]["meta"]["centred_on"]) == chosen
    assert settings_of(M["cons"]) != chosen                        # the models' own consensus is untouched
    orig = SW.pass_block

    def forced(kind, cal_, test_, grid, a_):                       # pretend the coarse pass chose `chosen`
        rep, block, c = orig(kind, cal_, test_, grid, a_)
        return (rep, block, chosen) if kind == "coarse" else (rep, block, c)
    monkeypatch.setattr(SW, "pass_block", forced)
    SW.cache_draw(a, "calib", M)
    res = SW.sweep(a, SW.cache_path(a, "calib"), None, M=M)
    assert res["meta"]["fine_centring"]["recached"]
    assert ConsensusCfg.from_dict(res["meta"]["fine_centring"]["centred_on"]) == chosen
    assert res["mismatch"]["calib_fine_recached"]["mismatches"] == 0
    res = SW.sweep(a, SW.cache_path(a, "calib"), None)             # no models: recorded, not re-cached
    assert not res["meta"]["fine_centring"]["recached"] and "no models" in res["meta"]["fine_centring"]["note"]


# ---- pinned numbers --------------------------------------------------------------------------------------------------

# consensus_for_query on the fixed synthetic set (_sample), computed with the code of this commit and checked
# bit-identical against the pre-refactor satroma.py (6858803) when pinned: (kind, seed, solver, means, n_modes,
# n_inliers, H row-major to 10 significant digits). A refactor that changes any of these changes the default pipeline.
PINNED = [
    ('picture', 0, 'srt', False, 235, 141, [1.010378955, 0.03499621401, 344.8522789, -0.006523518968, 1.029698661, 376.2581405, -5.186581563e-06, 4.173101371e-05, 1.0]),
    ('picture', 0, 'srt', True, 235, 132, [1.034798573, 0.00861785831, 344.9582661, 0.01816341589, 1.021689062, 373.2654168, 2.337275869e-05, 2.584241941e-06, 1.0]),
    ('picture', 0, 'sim', False, 235, 141, [1.006150425, 0.008754314923, 346.4859043, -0.008754314923, 1.006150425, 377.010979, 0.0, 0.0, 1.0]),
    ('picture', 0, 'sim', True, 235, 132, [1.014477855, 0.0007380748162, 346.5808491, -0.0007380748162, 1.014477855, 375.2008517, 0.0, 0.0, 1.0]),
    ('picture', 0, 'se2', False, 235, 141, [0.9999631794, 0.008581368822, 346.3817929, -0.008581368822, 0.9999631794, 377.8412312, 0.0, 0.0, 1.0]),
    ('picture', 0, 'se2', True, 235, 132, [0.9999997353, 0.0007275413825, 348.2771459, -0.0007275413825, 0.9999997353, 376.9315087, 0.0, 0.0, 1.0]),
    ('picture', 1, 'srt', False, 234, 139, [0.7628921057, -0.02075931768, 247.5118663, -0.2092306603, 0.9396707176, 359.5411513, -0.0004376701126, 3.340345651e-05, 1.0]),
    ('picture', 1, 'srt', True, 234, 128, [0.9909336163, -0.09927363334, 245.3527544, 0.008436048942, 0.9106325799, 354.767704, 7.015810564e-05, -0.0001684074718, 1.0]),
    ('picture', 1, 'sim', False, 234, 139, [0.9970307684, -0.00546973317, 236.7970499, 0.00546973317, 0.9970307684, 350.412704, 0.0, 0.0, 1.0]),
    ('picture', 1, 'sim', True, 234, 128, [0.9948764054, -0.01310718648, 237.9044697, 0.01310718648, 0.9948764054, 350.1232927, 0.0, 0.0, 1.0]),
    ('picture', 1, 'se2', False, 234, 139, [0.9998351391, -0.01815749724, 237.7700589, 0.01815749724, 0.9998351391, 347.9332206, 0.0, 0.0, 1.0]),
    ('picture', 1, 'se2', True, 234, 128, [0.9999858261, -0.005324251839, 236.6843317, 0.005324251839, 0.9999858261, 350.9867183, 0.0, 0.0, 1.0]),
    ('placed', 0, 'srt', False, 383, 35, [-3.582528721, -4.528798289, 849.4200801, -2.301277302, -2.853293642, 539.9586813, -0.004229156683, -0.005266594972, 1.0]),
    ('placed', 0, 'srt', True, 383, 28, [-2.942404858, -0.1663105741, 680.5833344, -2.218342385, -0.6263557761, 596.2321, -0.003755688991, -0.000911597183, 1.0]),
    ('placed', 0, 'sim', False, 383, 28, [-0.761377798, -0.8160119291, 813.7990188, 0.8160119291, -0.761377798, 486.7048408, 0.0, 0.0, 1.0]),
    ('placed', 0, 'sim', True, 383, 24, [-0.2445895582, -0.1051436386, 694.8244097, 0.1051436386, -0.2445895582, 462.2343043, 0.0, 0.0, 1.0]),
    ('placed', 0, 'se2', False, 383, 29, [-0.7634263154, -0.6458949303, 813.6129952, 0.6458949303, -0.7634263154, 517.1118531, 0.0, 0.0, 1.0]),
    ('placed', 0, 'se2', True, 383, 24, [-0.8908529834, -0.4542917146, 792.952121, 0.4542917146, -0.8908529834, 547.9058514, 0.0, 0.0, 1.0]),
    ('placed', 1, 'srt', False, 382, 36, [-3.448918639, 1.146402608, 709.1674289, -2.651798813, 2.218254235, 447.572817, -0.005772226419, 0.004977791033, 1.0]),
    ('placed', 1, 'srt', True, 382, 28, [-3.400044957, 1.083320778, 699.979701, -2.60200561, 2.063561297, 445.0978787, -0.00571390006, 0.004692871879, 1.0]),
    ('placed', 1, 'sim', False, 382, 30, [-0.9069735525, -0.06634984942, 680.1959247, 0.06634984942, -0.9069735525, 487.2652587, 0.0, 0.0, 1.0]),
    ('placed', 1, 'sim', True, 382, 26, [0.03524211715, -0.8167682074, 610.6700601, 0.8167682074, 0.03524211715, 316.5683532, 0.0, 0.0, 1.0]),
    ('placed', 1, 'se2', False, 382, 33, [-0.969240973, -0.2461136652, 696.5526366, 0.2461136652, -0.969240973, 476.3406204, 0.0, 0.0, 1.0]),
    ('placed', 1, 'se2', True, 382, 24, [-0.9833358734, -0.1817981298, 695.1970201, 0.1817981298, -0.9833358734, 481.0161131, 0.0, 0.0, 1.0]),
]


@pytest.mark.parametrize("row", PINNED, ids=lambda r: "-".join(map(str, r[:4])))
def test_default_consensus_matches_the_pinned_numbers(row):
    kind, seed, solver, means, n_modes, n_inliers, H = row
    query, batch, frac, gm, cert = _sample(kind, seed)
    cons = _cons(solver)
    cons.use_means = means
    m = consensus_for_query(cons, gm, query, batch, frac, 224, certainty=cert, stats=False)
    assert m.n_modes == n_modes and m.n_inliers == n_inliers
    assert np.allclose(m.H.ravel(), H, rtol=1e-8, atol=1e-6)


def test_tail_veto_rejects_a_median_gain_bought_with_a_worse_mean():
    base = dict(mean_m=2.8, gross=0.067)
    assert SW.tail_not_worse(dict(mean_m=2.7, gross=0.060), base)
    assert SW.tail_not_worse(dict(mean_m=2.8, gross=0.067), base)
    assert not SW.tail_not_worse(dict(mean_m=17.0, gross=0.077), base)   # the fine-pass case of 2026-09-27
    assert not SW.tail_not_worse(dict(mean_m=2.7, gross=0.080), base)
    assert not SW.tail_not_worse(dict(mean_m=float("nan"), gross=0.0), base)
