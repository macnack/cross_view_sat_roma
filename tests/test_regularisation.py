"""Regularisation options of scripts/train_vigor.py (2026-09-30): label-consistent pair augmentation on a planted
synthetic scene, photometric jitter, label smoothing, weight-decay groups, `_best.pt` rule, DDP loss shares.
Synthetic data, CPU, no dataset needed."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bevloc import config as C
from bevloc.data import augment as AUG
from bevloc.data.vigor import CITY_RES, DEPTH_DIR, R_NORTH, VigorPairs, collate_vigor, pose_en
from bevloc.model import regularise as REG
from bevloc.model.coarse import roma_coarse_loss, smoothed_cross_entropy
from bevloc.model.depth_query import depth_placement_metric_at, metric_to_bev_px
from vigor_labels import corrected  # noqa: E402

for _d in ("scripts", "tests"):
    if str(C.REPO / _d) not in sys.path:
        sys.path.insert(0, str(C.REPO / _d))

# ---- a planted scene: camera + one landmark, seen in the tile AND in the panorama with its depth ----------------

DY, DX = 40.0, -24.0                  # camera: 40 tile px south, 24 px EAST of the tile centre (dx > 0 = west)
LM_EN = np.array([6.0, 4.0])          # landmark: 6 m east, 4 m north of the camera
ERP_W, ERP_H = 896, 448


def _make_scene(tmp_path, city="Chicago"):
    res = CITY_RES[city]
    (tmp_path / city / "panorama").mkdir(parents=True)
    (tmp_path / city / "satellite").mkdir(parents=True)
    (tmp_path / city / DEPTH_DIR).mkdir(parents=True)
    c = (640 - 1) / 2.0
    cam = np.array([c - DX, c + DY])                                  # tile px (col_sign -1, row_sign +1)
    lm = cam + np.array([LM_EN[0], -LM_EN[1]]) / res
    yy, xx = np.mgrid[0:640, 0:640].astype(np.float64)
    sat = np.full((640, 640, 3), 60.0)                               # BGR
    sat[..., 2] += 195.0 * np.exp(-((xx - lm[0]) ** 2 + (yy - lm[1]) ** 2) / (2 * 3.0 ** 2))   # red landmark
    sat[..., 1] += 195.0 * np.exp(-((xx - cam[0]) ** 2 + (yy - cam[1]) ** 2) / (2 * 3.0 ** 2))  # green camera
    cv2.imwrite(str(tmp_path / city / "satellite" / "s1.png"), np.round(sat).astype(np.uint8))
    # panorama at the ERP size (resize = identity): a red band at the landmark's azimuth (clockwise from north)
    alpha = np.degrees(np.arctan2(LM_EN[0], LM_EN[1]))
    col = int(np.floor((alpha / 360.0 + 0.5) * ERP_W))
    pano = np.full((ERP_H, ERP_W, 3), 128, np.uint8)
    pano[:, col - 1:col + 2] = (0, 0, 255)                           # BGR red, 3 columns
    cv2.imwrite(str(tmp_path / city / "panorama" / "p1,1.0,.png"), pano)
    dep = np.full((ERP_H, ERP_W), 60000, np.uint16)                  # 60 m = invalid (>= 35 m)
    dep[:, col - 1:col + 2] = int(round(float(np.hypot(*LM_EN)) * 1000))
    cv2.imwrite(str(tmp_path / city / DEPTH_DIR / "p1,1.0,.png"), dep)
    lab = tmp_path / "splits" / "VIGOR" / city
    lab.mkdir(parents=True)
    (lab / "satellite_list.txt").write_text("s1.png\n")
    line = f"p1,1.0,.png s1.png {DY} {DX} s1.png 0 0 s1.png 0 0 s1.png 0 0\n"
    for f in ("pano_label_balanced.txt", "same_area_balanced_test.txt", "same_area_balanced_train.txt"):
        (lab / f).write_text(line)
    corrected(lab)
    return tmp_path


def _cfg(config="configs/vigor_cell0125.yaml", mode="erp_depth"):
    cfg = C.load(str(C.REPO / config))
    cfg.lift.query_mode = mode
    cfg.erp_depth.erp_size = [ERP_W, ERP_H]
    return cfg


def _centroid(ref, ch, bg=60.0, r=None, near=None):
    img = ref[ch].numpy().astype(np.float64) * 255.0 - bg
    img[img < 30.0] = 0.0
    if near is not None:
        m = np.zeros_like(img)
        u0, v0 = int(round(near[0])), int(round(near[1]))
        m[max(0, v0 - r):v0 + r + 1, max(0, u0 - r):u0 + r + 1] = 1
        img = img * m
    vv, uu = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return np.array([(img * uu).sum() / img.sum(), (img * vv).sum() / img.sum()])


def _landmark_on_canvas(s, cfg):
    """Where the depth-placed panorama landmark lands on the canvas through the sample's H (the label)."""
    erp = s["erp"][0]
    H_, W_ = erp.shape[-2:]
    red = (erp[0] - erp[1]).mean(0)                                  # red band: R >> G
    cols = torch.nonzero(red > 0.5)[:, 0].double()
    uv = torch.tensor([[float(cols.mean()) + 0.5, H_ / 2.0]], dtype=torch.float64)
    xy, valid = depth_placement_metric_at(s["depth"][None], uv, (H_, W_), torch.from_numpy(R_NORTH)[None], 35.0)
    assert bool(valid.all())
    bev = metric_to_bev_px(xy.double(), int(cfg.grid.n), float(cfg.grid.cell_m))[0, 0].numpy()
    Hm = s["H"].numpy().astype(np.float64)
    p = Hm @ np.array([bev[0], bev[1], 1.0])
    return p[:2] / p[2]


def _camera_on_canvas(s, n=224):
    o = (n - 1) / 2.0
    p = s["H"].numpy().astype(np.float64) @ np.array([o, o, 1.0])
    return p[:2] / p[2]


def _check_label(s, cfg, tol=1.0):
    """The label (H) puts the camera on the green marker and the depth-placed landmark on the red one."""
    cam = _camera_on_canvas(s)
    lm = _landmark_on_canvas(s, cfg)
    g = _centroid(s["ref"], 1, near=cam, r=12)
    r = _centroid(s["ref"], 0, near=lm, r=12)
    assert np.allclose(cam, g, atol=tol), (cam, g)
    assert np.allclose(lm, r, atol=tol), (lm, r)
    # the metric bookkeeping stays self-consistent: pose_en(H, ref_centre_en) == en
    S = s["ref"].shape[-1]
    assert np.allclose(pose_en(s["H"].numpy(), s["ref_centre_en"].numpy(), int(cfg.grid.n), float(cfg.grid.cell_m), S),
                       s["en"].numpy(), atol=1e-3)


def test_planted_scene_is_consistent_without_augmentation(tmp_path):
    root = _make_scene(tmp_path)
    cfg = _cfg()
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="samearea", train=True)
    _check_label(ds[0], cfg)


@pytest.mark.parametrize("geo,rot_deg", [(("rot90",), 0.0), (("rot",), 0.0), ((), 20.0), (("flip",), 0.0),
                                         (("shift",), 0.0), (("rot90", "flip", "shift"), 10.0),
                                         (("rot", "flip", "shift"), 0.0)])
def test_every_geometric_transform_keeps_the_planted_label(tmp_path, geo, rot_deg):
    root = _make_scene(tmp_path)
    cfg = _cfg()
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="samearea", train=True)
    ds.set_aug(AUG.PairAug(geometric=geo, rot_deg=rot_deg))
    seen = dict(k=set(), flip=set(), delta=0.0, shift=0)
    torch.manual_seed(3)
    for _ in range(12):
        s = ds[0]
        d = s["_aug"]
        seen["k"].add(d["k"])
        seen["flip"].add(d["flip"])
        seen["delta"] = max(seen["delta"], abs(d["delta"]))
        seen["shift"] = max(seen["shift"], abs(d["shift"][0]) + abs(d["shift"][1]))
        _check_label(s, cfg, tol=1.2)
        assert abs(d["theta"] - d["s"] * 360.0 / ERP_W) < 1e-9          # the roll IS the rotation
        # the whole tile stays on the canvas: its four edges are not cut (data within the canvas border rows)
        ref = s["ref"].sum(0) > 0
        assert ref.any()
        if "shift" in geo or d["delta"]:
            assert not ref[0].all() and not ref[-1].all() and not ref[:, 0].all() and not ref[:, -1].all()
    if "rot90" in geo or "rot" in geo:
        assert len(seen["k"]) > 1
    if rot_deg or "rot" in geo:
        assert seen["delta"] > 0
    if "flip" in geo:
        assert seen["flip"] == {True, False}
    if "shift" in geo:
        assert seen["shift"] > 0


def test_rotation_direction_east_goes_north(tmp_path):
    """A quarter turn counter-clockwise: the landmark 6 m east / 4 m north of the camera ends up 6 m north / 4 m
    west of it, both on the canvas and (through the depth placement) in the rolled panorama."""
    root = _make_scene(tmp_path)
    cfg = _cfg()
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="samearea", train=True)
    s0 = ds[0]
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s0.items()}
    rng = np.random.default_rng(0)

    class _Fixed:                                                    # draw exactly k = 1, no flip
        def integers(self, lo, hi=None):
            return 1

        def random(self):
            return 0.9
    d = AUG.geometric(out, _Fixed(), AUG.PairAug(geometric=("rot90",)), 224, 0.125)
    assert d["k"] == 1 and d["s"] == ERP_W // 4 and not d["flip"]
    rel = (_landmark_on_canvas(out, cfg) - _camera_on_canvas(out)) * 0.125     # canvas px -> metres, (right, down)
    assert np.allclose(rel, [-4.0, -6.0], atol=0.1), rel            # 4 m west (left), 6 m north (up)
    _check_label(out, cfg)
    del rng


def test_window_mode_allows_rot90_and_flip_only_and_keeps_the_label(tmp_path):
    root = _make_scene(tmp_path)
    cfg = _cfg("configs/vigor_cell00625_fine.yaml")
    cfg.vigor.ref_window_m, cfg.vigor.ref_jitter_m = 56.0, 6.0
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="samearea", train=True)
    for bad in (AUG.PairAug(geometric=("shift",)), AUG.PairAug(geometric=("rot",)), AUG.PairAug(rot_deg=5.0)):
        with pytest.raises(ValueError):
            ds.set_aug(bad)
    ds.set_aug(AUG.PairAug(geometric=("rot90", "flip")))
    torch.manual_seed(1)
    for _ in range(8):
        _check_label(ds[0], cfg, tol=1.2)


def test_ipm_query_refuses_geometric_augmentation(tmp_path):
    root = _make_scene(tmp_path)
    ds = VigorPairs(root, _cfg(mode="ipm"), cities=["Chicago"], split="samearea", train=True)
    with pytest.raises(ValueError):
        ds.set_aug(AUG.PairAug(geometric=("flip",)))
    ds.set_aug(AUG.PairAug(photometric=1.0))                        # photometric is fine (the picture is jittered)


# ---- off = bit-identical ------------------------------------------------------------------------------------------

def test_augmentation_off_draws_nothing_and_changes_nothing(tmp_path):
    root = _make_scene(tmp_path)
    cfg = _cfg()
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="samearea", train=True)
    assert ds.aug is None and AUG.from_config(cfg) is None          # the configs' defaults are all off
    torch.manual_seed(0)
    st = torch.get_rng_state()
    a = ds[0]
    assert torch.equal(torch.get_rng_state(), st)                    # no random number drawn
    ds.set_aug(None)
    b = ds[0]
    assert "_aug" not in a and set(a) == set(b)
    for k, v in a.items():
        assert (torch.equal(v, b[k]) if torch.is_tensor(v) else v == b[k]), k
    # the validation copy of train_vigor.py never augments
    import copy
    ds.set_aug(AUG.PairAug(photometric=1.0, geometric=("rot90",)))
    va = copy.copy(ds)
    va.aug = None
    c = va[0]
    assert all(torch.equal(v, a[k]) for k, v in c.items() if torch.is_tensor(v))


# ---- photometric --------------------------------------------------------------------------------------------------

def test_photometric_keeps_the_black_canvas_black_and_stays_in_range(tmp_path):
    root = _make_scene(tmp_path)
    cfg = _cfg()
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="samearea", train=True)
    s0 = ds[0]
    ds.set_aug(AUG.PairAug(photometric=1.0, blur_p=1.0))
    torch.manual_seed(0)
    s = ds[0]
    black0 = s0["ref"].sum(0) == 0
    assert bool(black0.any()) and bool((s["ref"].sum(0)[black0] == 0).all())
    assert not torch.equal(s["ref"], s0["ref"]) and not torch.equal(s["erp"], s0["erp"])
    for k in ("ref", "erp"):
        assert float(s[k].min()) >= 0.0 and float(s[k].max()) <= 1.0 and s[k].dtype == torch.float32
    assert torch.equal(s["H"], s0["H"]) and torch.equal(s["depth"], s0["depth"])   # geometry untouched
    b = collate_vigor([s, s])
    assert b["ref"].shape[0] == 2 and "_aug" not in b


def test_colour_jitter_identity_at_zero_magnitude():
    rng = np.random.default_rng(0)
    img = np.random.default_rng(1).random((20, 30, 3)).astype(np.float32)
    a = AUG.PairAug(photometric=1.0, brightness=0.0, contrast=0.0, saturation=0.0, hue_deg=0.0, blur_p=0.0)
    assert np.allclose(AUG.colour_jitter(img, rng, a), img, atol=1e-6)


# ---- label smoothing --------------------------------------------------------------------------------------------

def test_label_smoothing_matches_torch_and_the_formula_on_toy_logits():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(5, 16, generator=g)
    tgt = torch.tensor([0, 3, 7, 15, 3])
    per, nll = smoothed_cross_entropy(logits, tgt, 0.1)
    assert torch.allclose(per.mean(), F.cross_entropy(logits, tgt, label_smoothing=0.1), atol=1e-6)
    assert torch.allclose(nll.mean(), F.cross_entropy(logits, tgt), atol=1e-6)
    # support = the valid cells only (the target always kept): a hand-computed value on one row
    sup = torch.zeros(5, 16, dtype=torch.bool)
    sup[:, 8:] = True
    per_s, _ = smoothed_cross_entropy(logits, tgt, 0.2, sup)
    lp = F.log_softmax(logits[1], -1)
    cls = [3] + list(range(8, 16))
    want = 0.8 * -lp[3] + 0.2 * -lp[cls].mean()
    assert torch.allclose(per_s[1], want, atol=1e-6)
    # toy value: uniform logits over K classes -> loss log K whatever eps
    per_u, _ = smoothed_cross_entropy(torch.zeros(2, 8), torch.tensor([1, 2]), 0.3)
    assert torch.allclose(per_u, torch.full((2,), float(np.log(8))), atol=1e-6)


def test_coarse_loss_with_smoothing_logs_the_plain_ce_and_its_parts_sum_to_the_loss():
    g = torch.Generator().manual_seed(1)
    B, k, h, w = 2, 4, 3, 5
    gm = torch.randn(B, k * k, h, w, generator=g)
    idx = torch.randint(0, k * k, (B, h, w), generator=g)
    use = torch.rand(B, h, w, generator=g) > 0.3
    cert = torch.randn(B, 1, h, w, generator=g)
    rv = torch.rand(B, k, k, generator=g) > 0.4
    l0, s0 = roma_coarse_loss(gm, idx, use, cert, neighbour_radius=1, neighbour_weight=0.5)
    l1, s1 = roma_coarse_loss(gm, idx, use, cert, neighbour_radius=1, neighbour_weight=0.5, label_smoothing=0.0)
    assert torch.equal(l0, l1) and s0 == s1                          # eps 0 = the old code path
    parts = {}
    l2, s2 = roma_coarse_loss(gm, idx, use, cert, neighbour_radius=1, neighbour_weight=0.5, parts=parts,
                              label_smoothing=0.1, smooth_support=rv)
    assert s2["ce"] == pytest.approx(s0["ce"], rel=1e-6) and s2["ce_smooth"] != s2["ce"]
    assert float(sum(t for t, _ in parts.values())) == pytest.approx(float(l2), rel=1e-6)
    assert parts["ce"][1] == s2["n"] == int(use.sum())               # still a mean over the matchable tokens
    with pytest.raises(ValueError):
        roma_coarse_loss(gm, idx, use, cert, local_radius=2, label_smoothing=0.1)


# ---- weight decay groups ------------------------------------------------------------------------------------------

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.bn = nn.BatchNorm2d(8)
        self.gn = nn.GroupNorm(2, 8)
        self.lin = nn.Linear(8, 4)
        self.ln = nn.LayerNorm(4)
        self.gamma = nn.Parameter(torch.ones(4))                    # LayerScale-like 1-D tensor
        self.frozen = nn.Linear(2, 2)
        for p in self.frozen.parameters():
            p.requires_grad = False


def test_weight_decay_groups_exclude_biases_and_norms():
    m = _Net()
    assert REG.no_decay_names(m) >= {"conv.bias", "bn.weight", "bn.bias", "gn.weight", "gn.bias", "lin.bias",
                                     "ln.weight", "ln.bias", "gamma"}
    dec, nod = REG.split_decay(m)
    assert {id(p) for p in dec} == {id(m.conv.weight), id(m.lin.weight)}
    assert len(nod) == 9 and all(p.requires_grad for p in dec + nod)
    g0 = REG.module_groups(m, 1e-4)                                  # unset = one group, the optimiser's decay
    assert len(g0) == 1 and "weight_decay" not in g0[0] and len(g0[0]["params"]) == 11
    assert [id(p) for p in g0[0]["params"]] == [id(p) for p in m.parameters() if p.requires_grad]
    g1 = REG.module_groups(m, 1e-4, 0.05, "decoder")
    opt = torch.optim.AdamW(g1, weight_decay=0.01)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.05, 0.0]
    assert "decoder_decay" in REG.groups_report(opt)


def test_projection_head_decay_group_is_weights_only():
    from bevloc.model.depth_query import ProjectionHead
    h = ProjectionHead(dim=32, heads=4)
    dec, nod = REG.split_decay(h)
    assert all(p.ndim >= 2 for p in dec) and all(p.ndim < 2 for p in nod)
    names = {n for n, p in h.named_parameters() if any(p is q for q in dec)}
    assert not any("norm" in n or n.endswith("bias") for n in names)


# ---- selection rule, settings record ------------------------------------------------------------------------------

def test_select_score_and_resolve_and_resume_mismatch():
    v = dict(vce_pose_m=3.1, ce=4.2)
    assert REG.select_score("pose", v, 5.0, True) == 3.1             # the rule of every earlier run
    assert REG.select_score("pose", dict(vce_pose_m=float("nan"), ce=1.0), 5.0, True) == 5.0
    assert REG.select_score("pose", v, 5.0, False) == 5.0
    assert REG.select_score("loss", v, 5.0, True) == 4.2
    cfg = C.load(str(C.REPO / "configs/vigor_cell0125.yaml"))

    class _A:
        label_smoothing = weight_decay_decoder = weight_decay_head = feat_dropout = select_by = None
    r = REG.resolve(cfg, _A())
    assert r == dict(label_smoothing=0.0, weight_decay_decoder=None, weight_decay_head=None, feat_dropout=0.0,
                     select_by="pose")
    r["aug"] = None
    assert REG.mismatch(None, r) == []                               # an old resume file = all defaults
    a = _A()
    a.label_smoothing = 0.1
    r2 = REG.resolve(cfg, a)
    r2["aug"] = None
    assert REG.mismatch(r, r2) and REG.mismatch(None, r2)


# ---- DDP: the smoothed CE keeps the global-count share accounting ----------------------------------------------

def _ls_worker(rank, world, port, out):
    from test_ddp import _run_step, _step_setup
    from bevloc.model.ddp import Dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    D = Dist(rank, world, backend="gloo")
    try:
        cfg, q, mt, batch, params, kw = _step_setup()
        kw["label_smoothing"] = 0.1
        b = batch["erp"].shape[0] // world
        mine = {k: v[rank * b:(rank + 1) * b] for k, v in batch.items()}
        parts = {}
        loss, st = _run_step(q, mt, mine, cfg, kw, parts)
        s_parts = sum(t for t, _ in parts.values() if t is not None)
        D.global_loss(loss, parts).backward()
        bad, _ = D.reduce_step(params, False, st, average=False)
        out[rank] = dict(grads=[None if p.grad is None else p.grad.clone() for p in params],
                         parts_sum_err=float((s_parts - loss).detach().abs()), bad=bad)
    finally:
        D.close()


def test_label_smoothed_step_under_ddp_equals_the_single_process_union_gradient():
    import torch.multiprocessing as mp
    from test_ddp import _free_port, _run_step, _step_setup
    out = mp.Manager().dict()
    mp.start_processes(_ls_worker, args=(2, _free_port(), out), nprocs=2, start_method="spawn")
    res = [out[r] for r in range(2)]
    cfg, q, mt, batch, params, kw = _step_setup()
    kw["label_smoothing"] = 0.1
    loss, _ = _run_step(q, mt, batch, cfg, kw)
    loss.backward()
    ref = [p.grad for p in params]
    assert all(r["parts_sum_err"] < 1e-5 and r["bad"] is False for r in res)
    scale = max(float(g.abs().max()) for g in ref if g is not None)
    err = max(float((a - g).abs().max()) for a, g in zip(res[0]["grads"], ref) if g is not None)
    assert err <= 2e-5 * scale, (err, scale)


def test_feat_dropout_changes_training_only():
    from test_ddp import _run_step, _step_setup
    cfg, q, mt, batch, params, kw = _step_setup(B=2)
    torch.manual_seed(0)
    l0, _ = _run_step(q, mt, batch, cfg, dict(kw, vce_weight=0.0))
    torch.manual_seed(0)
    l1, _ = _run_step(q, mt, batch, cfg, dict(kw, vce_weight=0.0, feat_dropout=0.0))
    assert torch.equal(l0, l1)
    torch.manual_seed(0)
    l2, _ = _run_step(q, mt, batch, cfg, dict(kw, vce_weight=0.0, feat_dropout=0.3))
    assert not torch.equal(l0, l2)
