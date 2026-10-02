"""Coarse-to-fine second pass (task 04): scripts/eval_vigor.py with a fine pass on synthetic VIGOR samples, with
planted toy decoders (tests/tiny_satroma.py) whose poses are known exactly. CPU, no weights, no data."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from bevloc import config as C
from bevloc.data.vigor import CITY_RES, R_NORTH, VigorPairs
from bevloc.match.satroma import Match, SatRoMa
from bevloc.model.depth_query import ErpDepthQuery

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_vigor_window import DX, DY, _make  # noqa: E402
from tiny_satroma import StubPictureQuery, plant_translation, tiny_decoder, tiny_matcher  # noqa: E402

O = (224 - 1) / 2.0            # BEV centre px
C896 = (896 - 1) / 2.0         # canvas centre px
T_COARSE = (336.0, 352.0)      # cell-aligned: camera at canvas (447.5, 463.5) -> en (0, -2) m at 0.125 m/px
EN_COARSE = np.array([(O + T_COARSE[0] - C896) * 0.125, -(O + T_COARSE[1] - C896) * 0.125])


def _eval_vigor():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_vigor.py"
    spec = importlib.util.spec_from_file_location("bevloc_scripts_eval_vigor_two_pass", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(cell_m, window=None, solver="se2"):
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.lift.query_mode = "ipm"
    cfg.grid.cell_m = cell_m
    cfg.vigor.ref_window_m = window
    cfg.vigor.ref_jitter_m = 6.0 if window else 0.0
    cfg.matcher.solver = solver
    return cfg


def _setup(tmp_path, t_fine, solver="se2"):
    ev = _eval_vigor()
    root = _make(tmp_path)
    cfg, cfg_f = _cfg(0.125, solver=solver), _cfg(0.0625, window=56.0, solver=solver)
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="crossarea")
    ds_f = VigorPairs(root, cfg_f, cities=["Chicago"], split="crossarea")
    ds_f.labels = ds.labels
    matcher = tiny_matcher(plant_translation(tiny_decoder(), T_COARSE))
    cons = {tag: SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=means, min_valid_frac=0.05)
            for tag, means in (("peak", False), ("means", True))}
    matcher_f = tiny_matcher(plant_translation(tiny_decoder(), t_fine))
    cons_f = SatRoMa.from_wrapper(matcher_f.wrapper, cfg_f, use_means=False, min_valid_frac=0.05)
    fine = ev.DecoderFine(ds_f, StubPictureQuery(), matcher_f, cons_f, cfg_f, "cpu")
    return ev, ds, StubPictureQuery(), matcher, cons, cfg, fine


def _en_fine(t_fine, centre):
    """Tile-frame position of a fine pose that is the pure translation t_fine on a window centred at `centre`."""
    return centre + np.array([(O + t_fine[0] - C896) * 0.0625, -(O + t_fine[1] - C896) * 0.0625])


@pytest.mark.parametrize("solver", ["se2", "srt"])
def test_two_pass_returns_the_injected_fine_pose_in_the_tile_frame(tmp_path, solver):
    t_fine = (352.0, 320.0)                                   # +1 m east, +1 m north of the window centre
    ev, ds, query, matcher, cons, cfg, fine = _setup(tmp_path, t_fine, solver)
    plain = ev.score(ds, query, matcher, cons, cfg, "cpu")
    rows = ev.score(ds, query, matcher, cons, cfg, "cpu", fine=fine, fine_gate=6.0)
    r = rows[0]
    assert {k: r[k] for k in plain[0]} == plain[0]            # the coarse rows are untouched by the second pass
    res = CITY_RES["Chicago"]
    en_gt = np.array([-DX * res, -DY * res])
    assert np.allclose(r["en_gt"], en_gt, atol=1e-5)               # H is float32 in the sample
    en_gt = np.array(r["en_gt"])
    assert np.allclose(r["en_coarse"], EN_COARSE, atol=1e-6)
    assert abs(r["pose_peak_m"] - np.linalg.norm(EN_COARSE - en_gt)) < 1e-4    # coarse rows: the same metric
    assert np.allclose(r["fine_centre_en"], EN_COARSE)        # the window is centred on the coarse peak pose
    en_inj = _en_fine(t_fine, EN_COARSE)
    assert np.allclose(r["en_fine"], en_inj, atol=1e-6), (r["en_fine"], en_inj)
    assert abs(r["pose_fine_m"] - np.linalg.norm(en_inj - en_gt)) < 1e-6
    assert r["yaw_fine_deg"] < 1e-6
    assert not r["fallback_fine"] and abs(r["fine_shift_m"] - np.hypot(1.0, 1.0)) < 1e-6
    assert r["pose_fine_gated_m"] == r["pose_fine_m"]
    # the same fine pose measured in the window frame against the window's own GT H agrees
    sf = fine.ds.item(0, ref_centre_en=torch.tensor(EN_COARSE))
    from bevloc.eval.metrics import pose_errors
    H_est = np.eye(3)
    H_est[:2, 2] = t_fine
    assert abs(pose_errors(H_est, sf["H"].numpy().astype(float), 224, 0.0625)["position_m"] - r["pose_fine_m"]) < 1e-5


def test_gate_keeps_the_coarse_pose_when_the_fine_pose_jumps(tmp_path):
    t_fine = (352.0 + 16 * 8, 320.0)                          # 9 m east, 1 m north of the coarse pose: > 6 m
    ev, ds, query, matcher, cons, cfg, fine = _setup(tmp_path, t_fine)
    r = ev.score(ds, query, matcher, cons, cfg, "cpu", fine=fine, fine_gate=6.0)[0]
    assert np.allclose(r["en_fine"], _en_fine(t_fine, EN_COARSE), atol=1e-6)
    assert r["fallback_fine"] and r["fine_shift_m"] > 6.0
    assert r["pose_fine_gated_m"] == pytest.approx(r["pose_peak_m"], abs=1e-4)
    r2 = ev.score(ds, query, matcher, cons, cfg, "cpu", fine=fine, fine_gate=10.0)[0]   # a wider gate accepts it
    assert not r2["fallback_fine"] and r2["pose_fine_gated_m"] == r2["pose_fine_m"]


def test_gate_falls_back_when_the_fine_pass_finds_no_pose(tmp_path):
    ev, ds, query, matcher, cons, cfg, fine = _setup(tmp_path, (352.0, 320.0))

    class NoPose:
        cfg = fine.cfg

        def __call__(self, i, centre_en):
            return Match(None, None, 0, 0, 0, 0.0), fine.ds.item(i, ref_centre_en=centre_en)
    r = ev.score(ds, query, matcher, cons, cfg, "cpu", fine=NoPose(), fine_gate=6.0)[0]
    assert r["pose_fine_m"] is None and r["en_fine"] is None and r["fallback_fine"]
    assert r["pose_fine_gated_m"] == pytest.approx(r["pose_peak_m"], abs=1e-4)


def test_fine_picture_is_rebuilt_at_the_fine_gsd(tmp_path):
    """The picture of the fine sample is a new ipm_erp at 0.0625 m/px: 14 m wide, the 1.2 m blind disc 4x the pixels."""
    root = _make(tmp_path)
    a = VigorPairs(root, _cfg(0.125), cities=["Chicago"])[0]
    b = VigorPairs(root, _cfg(0.0625, window=56.0), cities=["Chicago"]).item(0, ref_centre_en=torch.zeros(2))
    blind_a, blind_b = int((~a["bev_valid"]).sum()), int((~b["bev_valid"]).sum())
    assert abs(blind_b / blind_a - 4.0) < 0.15, (blind_a, blind_b)
    assert 224 * 0.0625 == 14.0


def test_erp_depth_placement_is_in_fine_virtual_bev_pixels():
    """Same panorama and depth, fine config: the placed points sit twice as far from the BEV centre in pixels."""
    batch = dict(erp=torch.zeros(1, 1, 3, 448, 896), depth=torch.full((1, 1, 448, 896), 10.0),
                 R_w2c=torch.from_numpy(R_NORTH.copy())[None, None])
    xy = {}
    for cell in (0.125, 0.0625):
        cfg = C.load(str(C.REPO / "configs/default.yaml"))
        cfg.grid.cell_m = cell
        xy[cell], valid = ErpDepthQuery(cfg).placement(batch)
    assert bool(valid.any())
    v = valid[0]
    d_c, d_f = xy[0.125][0][v] - O, xy[0.0625][0][v] - O
    assert torch.allclose(d_f, 2.0 * d_c, atol=1e-3)


def test_run_end_to_end_writes_the_fine_rows_and_summary(tmp_path, monkeypatch):
    """eval_vigor.run with --fine-config/--fine-ckpt on the synthetic sample: the model constructors are swapped for
    the planted toy decoders (first built = coarse, second = fine); the JSON carries both passes."""
    import json

    ev = _eval_vigor()
    root = _make(tmp_path / "vigor")
    built = []

    def fake_matcher(ckpt, dev, train_decoder=False):
        m = tiny_matcher(plant_translation(tiny_decoder(), T_COARSE if not built else (352.0, 320.0)))
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
        "--fine-ckpt", str(ck), "--out", str(tmp_path / "out"), "--tag", "t"])
    ev.run(a, ev.decoder_fine)
    d = json.loads((tmp_path / "out" / "eval_vigor_t_crossarea.json").read_text())
    r = d["frames"][0]
    assert len(built) == 2 and d["meta"]["fine"]["cell_m"] == 0.0625 and d["meta"]["fine"]["window_m"] == 56.0
    assert np.allclose(r["en_fine"], _en_fine((352.0, 320.0), EN_COARSE), atol=1e-6)
    s = d["summary"]["all"]
    assert s["fine"]["n"] == 1 and s["fallback_fine_gated"] == 0 and s["nopose_fine"] == 0 and s["no_coarse_fine"] == 0
    assert abs(s["fine"]["median_m"] - r["pose_fine_m"]) < 1e-9 and "fine_gated" in s


def _loftr_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "loftr_fine_vigor.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("bevloc_scripts_loftr_fine_vigor", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeLoftr:
    """Stands in for kornia's LoFTR: matches every 8th picture pixel to itself + t (reference px), plus outliers."""

    def __init__(self, t):
        self.t = np.asarray(t, float)

    def __call__(self, data):
        g = np.stack(np.meshgrid(np.arange(4, 224, 8), np.arange(4, 224, 8)), -1).reshape(-1, 2).astype(float)
        k1 = g + self.t
        k1[::7] += 300.0                                        # 1 in 7 far outliers
        return dict(keypoints0=torch.from_numpy(g), keypoints1=torch.from_numpy(k1),
                    confidence=torch.ones(len(g)))


@pytest.mark.parametrize("solver", ["se2", "srt"])
def test_loftr_second_pass_plumbing_recovers_an_injected_translation(tmp_path, solver):
    """No kornia needed: a fake LoFTR with a known sub-pixel translation goes through LoftrFine, the shared consensus
    and fine_pass_row; pose_fine is the injected pose, and matches on invalid picture pixels are dropped."""
    lm = _loftr_module()
    ev, ds, query, matcher, cons, cfg, fine = _setup(tmp_path, (352.0, 320.0), solver)
    t = (340.3, 329.7)                                         # sub-cell, sub-pixel
    cfg_f = fine.cfg
    cons_f = SatRoMa.from_wrapper(NS(im_a_size=224, im_b_size=896), cfg_f, use_means=False, min_valid_frac=0.05)
    lf = lm.LoftrFine(fine.ds, cons_f, cfg_f, "cpu", _FakeLoftr(t))
    r = lm.EV.score(ds, query, matcher, cons, cfg, "cpu", fine=lf, fine_gate=6.0)[0]
    assert np.allclose(r["en_fine"], _en_fine(t, EN_COARSE), atol=1e-6), (r["en_fine"], _en_fine(t, EN_COARSE))
    n_all = len(np.arange(4, 224, 8)) ** 2
    assert r["nmatch_fine"] == n_all and r["nused_fine"] < n_all       # the blind disc under the camera is invalid
    assert r["inliers_fine"] == pytest.approx(6 / 7, abs=0.03)


def test_decoder_fine_config_uses_the_coarse_matcher_options(tmp_path):
    ev = _eval_vigor()
    cfg = _cfg(0.125, solver="se2")
    cfg.matcher.reproj_cells = 2.5
    a = NS(fine_config=str(C.REPO / "configs/vigor_cell00625_fine.yaml"))
    cfg_f = ev.fine_config(a, cfg)
    assert cfg_f.grid.cell_m == 0.0625 and cfg_f.vigor.ref_window_m == 56 and cfg_f.vigor.ref_jitter_m == 6
    assert cfg_f.matcher.solver == "se2" and cfg_f.matcher.reproj_cells == 2.5
    assert int(cfg_f.grid.n * cfg_f.reference.scale) * cfg_f.grid.cell_m == 56.0   # the canvas IS the window


# ---- review follow-ups ------------------------------------------------------------------------------------------

def test_a_failing_fine_pass_keeps_the_coarse_row_and_records_the_error(tmp_path):
    ev, ds, query, matcher, cons, cfg, fine = _setup(tmp_path, (352.0, 320.0))
    plain = ev.score(ds, query, matcher, cons, cfg, "cpu")

    class Boom:
        cfg = fine.cfg

        def __call__(self, i, centre_en):
            raise ValueError("bad window")
    rows = ev.score(ds, query, matcher, cons, cfg, "cpu", fine=Boom(), fine_gate=6.0)
    assert len(rows) == 1
    r = rows[0]
    assert {k: r[k] for k in plain[0]} == plain[0]
    assert r["fine_error"] == "ValueError: bad window"
    assert all(r[k] is None for k in ev.FINE_KEYS) and r["fallback_fine"] is None

    class Oom(Boom):
        def __call__(self, i, centre_en):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    with pytest.raises(RuntimeError, match="out of memory"):
        ev.score(ds, query, matcher, cons, cfg, "cpu", fine=Oom(), fine_gate=6.0)
    ok = ev.score(ds, query, matcher, cons, cfg, "cpu", fine=fine, fine_gate=6.0)[0]
    assert ok["fine_error"] is None


def test_fine_checkpoint_at_another_gsd_is_refused():
    ev = _eval_vigor()
    cfg_f = _cfg(0.0625, window=56.0)
    ev.check_fine_grid({"train": {"grid": {"cell_m": 0.0625, "ref_window_m": 56.0}}}, cfg_f, "a.pt", "f.yaml")
    ev.check_fine_grid({"train": {"pose_nll_weight": 0.5}}, cfg_f, "old.pt", "f.yaml")      # no record: warned only
    with pytest.raises(SystemExit, match="cell_m 0.125"):
        ev.check_fine_grid({"train": {"grid": {"cell_m": 0.125}}}, cfg_f, "coarse.pt", "f.yaml")


@pytest.mark.parametrize("flags,train", [(dict(train_split=False), False), (dict(train_split=True), True),
                                         (dict(train_split=False, calib=True), True)])
def test_fine_dataset_reads_the_label_lists_of_the_coarse_draw(monkeypatch, flags, train):
    ev = _eval_vigor()
    seen = {}

    class Fake:
        def __init__(self, root, cfg, cities=None, split=None, train=False, **kw):
            seen.update(cities=cities, train=train)
            self.labels = []
    monkeypatch.setattr(ev, "VigorPairs", Fake)
    a = NS(root="r", split="crossarea", cities=None, **flags)
    ds_f = ev.fine_dataset(a, None, NS(row_sign=1.0, height=2.5, labels=["x"]))
    assert seen["train"] is train and ds_f.labels == ["x"]
    assert seen["cities"] == (["NewYork", "Seattle"] if train else ["SanFrancisco", "Chicago"])


def test_report_lists_the_fine_rows_after_the_peak_row(tmp_path):
    import json
    spec = importlib.util.spec_from_file_location(
        "bevloc_scripts_report_vigor", Path(__file__).resolve().parents[1] / "scripts" / "report_vigor.py")
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)

    def p(med):
        return {"median_m": med, "median_ci": [med - 0.1, med + 0.1], "recall@5m": 0.8, "recall@10m": 0.9, "n": 3000}
    s = {"peak": p(1.8), "means": p(1.9), "centre_guess": p(14.3), "mean_peak_m": 4.0, "fine": p(1.2),
         "mean_fine_m": 3.9, "fine_gated": p(1.3), "mean_fine_gated_m": 3.8, "nopose_fine": 0}
    f = tmp_path / "eval_vigor_x_twopass_samearea.json"
    f.write_text(json.dumps(dict(meta={}, summary={"all": s})))
    rows = rep.rows_of(f)
    assert [r["label"] for r in rows] == ["vigor_x_twopass", "vigor_x_twopass, fine", "vigor_x_twopass, fine_gated"]
    other = dict(rows[0], file="b", order=0, median=1.5, label="other")
    assert [r["label"] for r in rep.sort_rows(rows[::-1] + [other])] == \
        ["other", "vigor_x_twopass", "vigor_x_twopass, fine", "vigor_x_twopass, fine_gated"]
