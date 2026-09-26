"""Task 06: vote-map statistics, the calibrator script's metrics / conformal / abstention, and --hyp. CPU, no data."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from bevloc.eval import calibration as K
from bevloc.match import vote_stats as V
from bevloc.match.satroma import SatRoMa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_satroma import StubPictureQuery, cfg_stub, plant_translation, tiny_decoder, tiny_matcher  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(f"bevloc_scripts_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- vote-map statistics ---------------------------------------------------------------------------------------

def test_entropy_of_one_hot_is_zero_and_of_uniform_is_one_with_matching_support():
    k = 56
    one = np.zeros((k, k))
    one[10, 20] = 1.0
    s = V.vote_stats(one)
    assert s["vote_entropy"] == 0.0 and s["vote_support_cells"] == pytest.approx(1.0) and s["vote_top1"] == 1.0
    u = np.full((k, k), 1.0 / k ** 2)
    s = V.vote_stats(u)
    assert s["vote_entropy"] == pytest.approx(1.0) and s["vote_support_cells"] == pytest.approx(k * k)
    # restricted to the valid cells (the black canvas outside a VIGOR tile): uniform over those is 1 again
    rv = np.zeros((k, k), bool)
    rv[18:38, 18:38] = True
    s = V.vote_stats(np.where(rv, 1.0, 0.0) / rv.sum(), ref_valid=rv)
    assert s["vote_entropy"] == pytest.approx(1.0) and s["vote_support_cells"] == pytest.approx(400)
    # m equally likely cells: support m
    four = np.zeros((k, k))
    four[0, :4] = 0.25
    assert V.vote_stats(four)["vote_support_cells"] == pytest.approx(4.0)


def test_mass_within_2_cells_is_one_when_the_map_sits_at_the_pose():
    k, ref = 56, 896
    p = np.zeros((k, k))
    p[30, 12], p[31, 13], p[29, 12], p[30, 14] = 0.4, 0.3, 0.2, 0.1       # all within 2 cells of (col 12, row 30)
    pose_px = np.array([12 * 16 + 7.5, 30 * 16 + 7.5])                    # centre of cell (12, 30)
    cell = V.px_to_cell(pose_px, ref, k)
    assert np.allclose(cell, [12, 30])
    assert V.vote_stats(p, pose_cell=cell)["vote_mass_2cells"] == pytest.approx(1.0)
    assert V.vote_stats(p, pose_cell=(40, 40))["vote_mass_2cells"] == 0.0
    assert V.vote_stats(p)["vote_mass_2cells"] is None


def test_vote_map_ignores_masked_tokens_and_ego_map_concentrates_on_a_consistent_translation():
    k, h, w = 56, 14, 14
    gm = torch.zeros(k * k, h, w)
    t_cells = (9, 5)                                                      # every token lands 9 cols, 5 rows further
    for i in range(h):
        for j in range(w):
            gm[(i + 5 + 7) * k + (j + 9 + 7), i, j] = 30.0              # token (i, j) at cell (j + 16, i + 12)
    gm[:, :3] = 0.0                                                       # masked rows: zero logits
    p, n = V.vote_map(gm)
    assert n == (h - 3) * w and p.sum() == pytest.approx(1.0)
    assert p[12:12 + 3].sum() < 1e-6                                      # the masked rows' cells get no vote
    # token centres at 16 j + 7.5 query px, the ego at 111.5: token (i, j) sits (j - 6.5, i - 6.5) cells from the
    # ego, so every token puts the ego at ((j + 16) - (j - 6.5), (i + 12) - (i - 6.5)) = (22.5, 18.5): one half-cell
    # point, bilinearly split over 2 x 2 cells
    pe = V.ego_vote_map(gm, V.token_centres(h, w, 224), query_size=224, ref_px=896)
    ego = np.array([22.5, 18.5])
    assert V.vote_stats(pe, pose_cell=ego)["vote_mass_2cells"] == pytest.approx(1.0)
    assert V.vote_stats(pe)["vote_support_cells"] <= 4.0 + 1e-6


# ---- calibrator: metrics, conformal, abstention ----------------------------------------------------------------

def test_auroc_and_ap_recover_a_known_monotone_relation_and_chance():
    rng = np.random.default_rng(0)
    err = rng.exponential(4.0, 4000)
    y = err < 5.0
    assert K.auroc(-err, y) == pytest.approx(1.0)                         # the score is the error itself
    assert K.average_precision(-err, y) == pytest.approx(1.0)
    assert abs(K.auroc(rng.random(4000), y) - 0.5) < 0.03                 # random score
    assert abs(K.average_precision(rng.random(4000), y) - y.mean()) < 0.03
    assert K.auroc(err, y) == pytest.approx(0.0)


def test_isotonic_and_logistic_are_calibrated_on_a_known_probability():
    rng = np.random.default_rng(1)
    x = rng.random(20000)
    y = rng.random(20000) < x                                             # P(y | x) = x
    iso = K.Isotonic().fit(x[:10000], y[:10000])
    p = iso.predict(x[10000:])
    assert np.all(np.diff(iso.y_) >= -1e-12)                              # monotone
    assert K.reliability(p, y[10000:])[1] < 0.03
    lx = np.log(x / (1 - x))[:, None]                                    # logit(P) = feature: exact for logistic
    lg = K.Logistic(l2=1e-3).fit(lx[:10000], y[:10000])
    assert K.reliability(lg.predict(lx[10000:]), y[10000:])[1] < 0.03
    assert np.allclose(lg.predict(np.array([[0.0]])), 0.5, atol=0.03)


def test_split_conformal_gives_the_requested_coverage():
    rng = np.random.default_rng(2)
    for alpha in (0.1, 0.2):
        x = rng.random(8000)
        y = rng.random(8000) < x
        r = K.conformal_report(x[:4000], y[:4000], x[4000:], y[4000:], alpha)
        assert abs(r["coverage"] - (1 - alpha)) < 0.02 and r["coverage"] >= 1 - alpha - 0.02
        assert r["accept"] + r["reject"] + r["abstain"] + r["empty"] == pytest.approx(1.0)
    assert K.conformal_quantile(np.arange(10.0), 0.01) == float("inf")   # too few calibration frames: no guarantee


def test_abstention_curve_of_a_perfect_score_is_monotone():
    rng = np.random.default_rng(3)
    err = rng.exponential(5.0, 3000)
    c = K.abstention_curve(-err, err)
    assert np.all(np.diff(c["median_m"]) >= 0) and np.all(np.diff(c["gross_frac"]) >= 0)
    assert c["coverage"][-1] == 1.0 and c["median_m"][-1] == pytest.approx(np.median(err))
    assert K.at_coverage(c, 0.9)["n"] == 2700


def _rows(n, seed, informative=True):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        good = rng.random() < 0.75
        e = rng.exponential(1.5) if good else 10 + rng.exponential(20)
        inl = np.clip((0.8 if good else 0.55) + 0.2 * rng.standard_normal(), 0, 1)
        ent = np.clip((0.3 if good else 0.7) + 0.1 * rng.standard_normal(), 0, 1) if informative else rng.random()
        rows.append(dict(id=f"f{seed}_{i}", city="Chicago", pose_peak_m=float(e), inliers_peak=float(inl),
                         vote_entropy=float(ent), ego_mass_2cells=float(1 - ent), n_modes=200, peak_means_m=None))
    return rows


def test_certainty_script_end_to_end_calib_to_test_and_kfold(tmp_path):
    mod = _script("certainty_vigor")
    calib, test = tmp_path / "calib.json", tmp_path / "test.json"
    calib.write_text(json.dumps(dict(meta={}, frames=_rows(1500, 0))))
    test.write_text(json.dumps(dict(meta={}, frames=_rows(1500, 1))))
    res = mod.main(["--eval-json", str(test), "--calib-json", str(calib), "--tag", "t", "--out", str(tmp_path)])
    lg, iso = res["models"]["logistic"]["5.0"], res["models"]["inlier_isotonic"]["5.0"]
    assert lg["auroc"] > iso["auroc"] + 0.05                              # the entropy adds information
    assert abs(lg["conformal"]["coverage"] - 0.9) < 0.04
    assert (tmp_path / "certainty_t.json").is_file() and (tmp_path / "certainty_t.png").is_file()
    assert [f["key"] for f in res["meta"]["features"]] == ["inliers_peak", "vote_entropy", "n_modes",
                                                           "ego_mass_2cells"]
    a = res["abstention"]
    assert a["logistic"]["at90"]["median_m"] <= a["logistic"]["full"]["median_m"]
    res = mod.main(["--eval-json", str(test), "--tag", "k", "--out", str(tmp_path)])       # 5-fold, out of fold
    assert res["meta"]["protocol"]["mode"].startswith("5-fold")
    with pytest.raises(SystemExit):                                       # same frames in both jsons: refused
        mod.main(["--eval-json", str(test), "--calib-json", str(test), "--tag", "x", "--out", str(tmp_path)])


# ---- --hyp: bootstrap hypotheses ---------------------------------------------------------------------------------

def _consistent_gm(k=56, h=14, w=14, dx=9, dy=5):
    gm = torch.zeros(k * k, h, w)
    for i in range(h):
        for j in range(w):
            gm[(i + dy) * k + (j + dx), i, j] = 30.0
    return gm


@pytest.mark.parametrize("solver", ["se2", "sim", "srt"])
def test_hyp_spread_is_zero_on_consistent_modes(solver):
    cons = NS(m=NS(im_a_size=224, im_b_size=896), use_means=False, reproj=3.0, seed=0, solver=solver)
    xy = torch.from_numpy(V.token_centres(14, 14, 224)).float()
    m = SatRoMa.consensus_from_gm(cons, _consistent_gm(), xy, torch.ones(14, 14, dtype=torch.bool))
    assert m.H is not None and m.n_inliers == m.n_modes == 196
    hs = SatRoMa.hypothesis_spread(cons, m, 8)
    assert hs["n_ok"] == 8 and hs["spread_px"] < 1e-6
    assert np.allclose(V.pose_px(hs["H"], 224), V.pose_px(m.H, 224), atol=1e-6)
    assert SatRoMa.hypothesis_spread(cons, m, 0) is None
    st = m.stats
    assert st["n_valid_tokens"] == 196 and st["ego_mass_2cells"] == pytest.approx(1.0)
    assert st["placed_range_px"] > 0


class _DS:
    def __init__(self, t, n=2):
        g = torch.Generator().manual_seed(7)
        self.items = [dict(id=f"s{i}", city="Chicago", H=torch.tensor([[1.0, 0, t[0]], [0, 1.0, t[1]], [0, 0, 1.0]],
                                                                          dtype=torch.float64),
                           bev=torch.rand(3, 224, 224, generator=g), bev_valid=torch.ones(224, 224),
                           ref=torch.rand(3, 896, 896, generator=g)) for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def centre_guess_m(self, i):
        return 0.0


def test_eval_hyp_zero_is_identical_and_hyp_k_adds_only_the_hyp_keys():
    ev = _script("eval_vigor")
    t = (300.0, 280.0)
    cfg = cfg_stub("se2")
    matcher = tiny_matcher(plant_translation(tiny_decoder(), t))
    cons = {tag: SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=means, min_valid_frac=0.05)
            for tag, means in (("peak", False), ("means", True))}
    ds, q = _DS(t), StubPictureQuery()
    base = ev.score(ds, q, matcher, cons, cfg, "cpu")
    assert ev.score(ds, q, matcher, cons, cfg, "cpu", hyp=0) == base
    assert all(set(V.STAT_KEYS) <= set(r) for r in base)
    k4 = ev.score(ds, q, matcher, cons, cfg, "cpu", hyp=4)
    for r, b in zip(k4, base):
        assert {k: r[k] for k in b} == b and set(r) - set(b) == set(V.HYP_KEYS)
        assert r["n_hyp_ok"] == 4 and r["spread_hyp_m"] < 0.25 and r["pose_hyp_med_m"] < 2.0


# ---- review fixes: --calib split, spread definition, off-tile mass, peak-only statistics -------------------------

def test_calib_split_skips_the_selection_frames_and_matches_train_vigor():
    calib_split = _script("eval_vigor").calib_split
    n, vf, vs = 1000, 0.2, 50
    tm = dict(val_frac=vf, val_samples=vs, cities=["Chicago"], split="samearea")
    idx, info = calib_split(n, tm, vf, ["Chicago"], 100)
    perm = np.random.default_rng(0).permutation(n)                    # train_vigor.py's rule, restated
    n_tr = int(n * (1 - vf))
    assert np.array_equal(idx, perm[n_tr + vs:n_tr + vs + 100])
    assert not set(idx) & set(perm[:n_tr]) and not set(idx) & set(perm[n_tr:n_tr + vs])   # neither train nor val
    assert info["heldout_slice"] == [vs, vs + 100] and info["val_samples_skipped"] == vs and info["source"] == "checkpoint"
    idx_all, info = calib_split(n, tm, vf, ["Chicago"], 0)            # limit 0 = the rest of the held-out part
    assert np.array_equal(idx_all, perm[n_tr + vs:]) and info["n"] == n - n_tr - vs
    idx_big, _ = calib_split(n, tm, vf, ["Chicago"], 10 ** 6)         # a limit beyond the held-out part is clipped
    assert np.array_equal(idx_big, perm[n_tr + vs:])


@pytest.mark.parametrize("tm,vf,cities,msg", [
    (dict(steps=30000), 0.2, ["Chicago"], "no val_frac"),                          # old checkpoint, no split stored
    (dict(val_frac=0.2, val_samples=200, cities=["Chicago"]), 0.1, ["Chicago"], "differs from the checkpoint"),
    (dict(val_frac=0.2, val_samples=200, cities=["Chicago"]), 0.2, ["Seattle"], "differ from the training"),
    (dict(val_frac=0.2, val_samples=200, cities=["Chicago", "Seattle"]), 0.2, ["Chicago"], "differ from the training"),
    (dict(val_frac=0.0, val_samples=200, cities=["Chicago"]), 0.0, ["Chicago"], "no held-out"),
    (dict(val_frac=0.2, val_samples=200, cities=["Chicago"]), 0.2, ["Chicago"], "nothing left"),    # n = 1000: 200 held out
])
def test_calib_split_refuses(tm, vf, cities, msg):
    calib_split = _script("eval_vigor").calib_split
    with pytest.raises(ValueError, match=msg):
        calib_split(1000, tm, vf, cities, 100)


def test_calib_split_assumed_split_for_an_old_checkpoint_is_recorded():
    calib_split = _script("eval_vigor").calib_split
    idx, info = calib_split(1000, dict(steps=1), 0.2, ["Chicago"], 10,
                            assume=dict(val_frac=0.2, val_samples=20, cities=["Chicago"]))
    assert info["source"].startswith("asserted") and info["heldout_slice"] == [20, 30]


def test_spread_excludes_the_medoid_and_is_none_below_two_hypotheses():
    i, sp = V.medoid([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    assert i == 1 and sp == pytest.approx(1.5)                        # others at 1 and 2 (the medoid's 0 not counted)
    assert V.medoid([[5.0, 5.0]]) == (0, None)
    # one successful hypothesis (K = 1) -> no spread
    cons = NS(m=NS(im_a_size=224, im_b_size=896), use_means=False, reproj=3.0, seed=0, solver="se2")
    xy = torch.from_numpy(V.token_centres(14, 14, 224)).float()
    m = SatRoMa.consensus_from_gm(cons, _consistent_gm(), xy, torch.ones(14, 14, dtype=torch.bool))
    hs = SatRoMa.hypothesis_spread(cons, m, 1)
    assert hs["n_ok"] == 1 and hs["H"] is not None and hs["spread_px"] is None


def test_offtile_mass_is_the_mass_on_invalid_cells_before_renormalising():
    k = 8
    rv = np.zeros((k, k), bool)
    rv[:, :4] = True
    p = np.zeros((k, k))
    p[0, 0], p[0, 6], p[3, 7] = 0.5, 0.3, 0.2
    s = V.vote_stats(p, ref_valid=rv)
    assert s["vote_offtile_mass"] == pytest.approx(0.5) and s["vote_top1"] == pytest.approx(1.0)
    assert V.vote_stats(p)["vote_offtile_mass"] == 0.0


def test_statistics_are_computed_for_the_peak_row_only_and_do_not_change_the_pose():
    cons = NS(m=NS(im_a_size=224, im_b_size=896), use_means=False, reproj=3.0, seed=0, solver="se2")
    xy = torch.from_numpy(V.token_centres(14, 14, 224)).float()
    args = (cons, _consistent_gm(), xy, torch.ones(14, 14, dtype=torch.bool))
    a, b = SatRoMa.consensus_from_gm(*args), SatRoMa.consensus_from_gm(*args, stats=False)
    assert a.stats is not None and b.stats is None and np.array_equal(a.H, b.H)
    assert (a.n_modes, a.n_inliers, a.inlier_ratio) == (b.n_modes, b.n_inliers, b.inlier_ratio)
