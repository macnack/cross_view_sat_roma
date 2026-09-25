"""Sub-cell refinement (task 04): warp -> correspondences -> RANSAC, RoMa's fine loss, refiner freezing / saving,
bit-identity of --refine 0, and per-pixel ERP placement against the token placement. CPU, no weights, no data."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from bevloc.data.vigor import R_NORTH
from bevloc.eval.metrics import pose_errors
from bevloc.match.satroma import SatRoMa, consensus_for_query, refined_for_query
from bevloc.model.depth_query import ErpDepthQuery, depth_placement_metric, depth_placement_metric_at
from bevloc.model.erp_query import erp_placement, erp_placement_at
from bevloc.model.refine import (
    RefinerTap, charbonnier, decoder_state, fine_certainty_target, fine_loss, norm_to_px, px_to_norm,
    refiner_displacement, set_refiner_trainable, token_centres_px, warp_samples,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_satroma import (  # noqa: E402
    C_ENC, StubPictureQuery, cfg_stub, plant_translation, tiny_decoder, tiny_matcher,
)

R_N = torch.from_numpy(R_NORTH.copy())[None]


def _cons(solver, reproj=3.0):
    return NS(m=NS(im_a_size=224, im_b_size=896), use_means=False, reproj=reproj, seed=0, solver=solver)


def _translation_flow(t, h=14, w=14):
    q = token_centres_px(h, w, dtype=torch.float64)
    return px_to_norm(q + torch.tensor(t, dtype=torch.float64), 896).permute(2, 0, 1)


# ---- warp -> correspondences -> RANSAC ---------------------------------------------------------------------------

@pytest.mark.parametrize("solver", ["se2", "sim"])
@pytest.mark.parametrize("stride", [16, 8, 4])
def test_known_translation_warp_is_recovered_through_the_ransac(solver, stride):
    t = (301.37, 257.81)                                     # sub-cell: 18.8 / 16.1 cells
    uv, idx, wv, cv = warp_samples(_translation_flow(t), torch.zeros(14, 14), (224, 224), stride)
    centres = (np.arange(224 // stride) + 0.5) * stride - 0.5            # the stride grid's cell centres
    assert len(uv) == int(((centres >= 7.5) & (centres <= 215.5)).sum()) ** 2   # inside the token-centre hull only
    r = norm_to_px(wv.double(), 896).numpy()
    assert np.abs(r - (uv.numpy() + np.array(t))).max() < 1e-3        # the samples themselves are exact
    m, n = SatRoMa.refined_consensus(_cons(solver), uv.numpy(), r)
    assert n == len(uv) and m.H is not None and m.inlier_ratio == 1.0
    H = m.H / m.H[2, 2]
    assert np.abs(H[:2, :2] - np.eye(2)).max() < 1e-6
    assert np.abs(H[:2, 2] - np.array(t)).max() < 1e-3


def test_seed_gate_drops_correspondences_far_from_the_coarse_pose():
    t = np.array([301.37, 257.81])
    uv, _, wv, _ = warp_samples(_translation_flow(tuple(t)), torch.zeros(14, 14), (224, 224), 16)
    r = norm_to_px(wv.double(), 896).numpy()
    r[:20] += 200.0                                           # 20 gross outliers
    seed = np.eye(3)
    seed[:2, 2] = t + 5.0                                     # coarse pose 5 px off
    m, n = SatRoMa.refined_consensus(_cons("se2"), uv.numpy(), r, H_seed=seed, gate_cells=3.0)
    assert n == 196 - 20
    assert np.abs(m.H[:2, 2] - t).max() < 1e-3


# ---- RoMa's fine loss ---------------------------------------------------------------------------------------------

def test_fine_loss_is_zero_at_the_ground_truth_and_grows_with_the_offset():
    gt = torch.rand(2, 2, 5, 7) * 2 - 1
    ok = torch.ones(2, 5, 7, dtype=torch.bool)
    cert = torch.zeros(2, 5, 7)
    losses = []
    for off in (0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5):
        warp = gt + off * torch.tensor([0.6, 0.8])[None, :, None, None]
        loss, st = fine_loss(warp, cert, gt, gt, ok, ok, cell_norm=2 / 56, certainty_weight=0.0)
        losses.append(float(loss))
    assert losses[0] == 0.0
    assert all(b > a for a, b in zip(losses, losses[1:]))
    # the formula: generalised Charbonnier, alpha 0.5, s = c * stride
    s = 1e-4 * 16
    assert math.isclose(losses[3], s ** 0.5 * ((0.01 / s) ** 2 + 1) ** 0.25 - s ** 0.5, rel_tol=1e-5)
    assert math.isclose(float(charbonnier(torch.tensor(0.1), s)), losses[4], rel_tol=1e-5)


def test_fine_loss_ignores_unsupervised_tokens():
    gt = torch.zeros(1, 2, 2, 2)
    warp = gt.clone()
    warp[0, :, 0, 0] = 5.0                                    # huge error on a token that is not supervised
    ok = torch.ones(1, 2, 2, dtype=torch.bool)
    ok[0, 0, 0] = False
    loss, st = fine_loss(warp, torch.zeros(1, 2, 2), gt, gt, ok, ok, cell_norm=2 / 56, certainty_weight=0.0)
    assert float(loss) == 0.0 and st["fine_n"] == 3


def test_certainty_target_is_one_inside_the_cell_and_zero_outside():
    cell = 2 / 56
    gt = torch.zeros(1, 2, 1, 5)
    w_in = torch.zeros(1, 2, 1, 5)
    w_in[0, 0, 0] = torch.tensor([0.0, 0.3, 0.49, 0.51, 2.0]) * cell       # x offsets in cells
    ok = torch.ones(1, 1, 5, dtype=torch.bool)
    tgt = fine_certainty_target(w_in, gt, ok, cell, cells=0.5)
    assert tgt[0, 0].tolist() == [True, True, True, False, False]
    w_in[0, 1, 0, 1] = 0.7 * cell                             # Chebyshev: a y offset beyond the cell counts too
    ok[0, 0, 0] = False                                       # unsupervised token: 0 whatever the offset
    tgt = fine_certainty_target(w_in, gt, ok, cell, cells=0.5)
    assert tgt[0, 0].tolist() == [False, False, True, False, False]
    # and the BCE pulls the certainty toward exactly that target
    logit = torch.zeros(1, 1, 5, requires_grad=True)
    _, st = fine_loss(gt, logit, w_in, gt, ok, torch.ones_like(ok), cell_norm=cell)
    assert abs(st["fine_cert_pos"] - 1 / 5) < 1e-6


# ---- the refiner inside the decoder --------------------------------------------------------------------------------

def _decode(dec, h=14, w=14, seed=3, tap=None):
    g = torch.Generator().manual_seed(seed)
    f_q = torch.randn(1, C_ENC, h, w, generator=g)
    f_r = torch.randn(1, C_ENC, 56, 56, generator=g)
    sf = float(((h * 16) * (w * 16)) ** 0.5 / 560.0)
    dec.expose_intermediates = True
    import contextlib
    with (tap or contextlib.nullcontext()):
        out = dec({16: f_q}, {16: f_r}, scale_factor=sf)
    return out[16]


def test_tap_reproduces_the_decoders_refined_warp_and_rerun_matches_the_decoder():
    dec = plant_translation(tiny_decoder(), (300.0, 280.0))
    with torch.no_grad():
        tap = RefinerTap(dec)
        o = _decode(dec, tap=tap)
        assert torch.allclose(o["flow"] - o["flow_pre_delta"], refiner_displacement(o["delta_flow"], 14, 14), atol=1e-6)
        assert torch.allclose(tap.refined_warp(), o["flow"], atol=1e-6)
        assert torch.allclose(o["certainty"], o["gm_certainty"] + tap.delta_certainty, atol=1e-6)
        w2, dc2 = tap.rerun(o["flow_pre_delta"])             # same input warp -> same output
        assert torch.allclose(w2, o["flow"], atol=1e-6) and torch.allclose(dc2, tap.delta_certainty, atol=1e-6)
        w3, _ = tap.rerun(o["flow_pre_delta"] + 0.05)        # another warp -> another refinement
        assert not torch.allclose(w3, o["flow"] + 0.05, atol=1e-4)
    # the planted classifier: ToWarp puts every token within a cell of its translated centre
    q = token_centres_px(14, 14)
    err = (norm_to_px(o["flow_pre_delta"][0].permute(1, 2, 0), 896) - (q + torch.tensor([300.0, 280.0]))).abs()
    assert float(err.max()) < 8.0


@pytest.mark.parametrize("on", [False, True])
def test_refiner_frozen_and_excluded_when_off_trainable_and_saved_when_on(on):
    from train_lift_splat import refine_step_loss
    dec = plant_translation(tiny_decoder(), (300.0, 280.0))
    for p in dec.parameters():
        p.requires_grad = True                               # FeatureQueryMatcher(train_decoder=True)
    n = set_refiner_trainable(dec, on)
    assert n == sum(p.numel() for k, p in dec.named_parameters() if "conv_refiner" in k) > 0
    assert all(p.requires_grad == on for k, p in dec.named_parameters() if "conv_refiner" in k)
    assert all(p.requires_grad for k, p in dec.named_parameters() if "conv_refiner" not in k)
    saved = decoder_state(dec, include_refiner=on)
    assert any("conv_refiner" in k for k in saved) == on
    assert all(k in saved for k in dec.state_dict() if "conv_refiner" not in k)
    if not on:
        return
    # fine loss: gradients reach the refiner only (its inputs are detached, RoMa's coarse/fine cut)
    tap = RefinerTap(dec, detach_inputs=True)
    o = _decode(dec, tap=tap)
    H = torch.eye(3)[None].clone()
    H[0, 0, 2], H[0, 1, 2] = 300.0, 280.0
    ok = torch.ones(1, 14, 14, dtype=torch.bool)
    loss, st = refine_step_loss(o, tap, H, ok, ok, None, 896, 56, 0.01,
                                dict(alpha=0.5, c=1e-4, cert_cells=0.5))
    assert float(loss.detach()) > 0 and st["fine_n"] == 196
    loss.backward()
    ref_grads = [p.grad for k, p in dec.named_parameters() if "conv_refiner" in k]
    other = {k: p.grad for k, p in dec.named_parameters() if "conv_refiner" not in k}
    assert any(g is not None and float(g.abs().sum()) > 0 for g in ref_grads)
    assert all(g is None or float(g.abs().sum()) == 0 for g in other.values()), [k for k, g in other.items() if g is not None]


# ---- eval: --refine 0 is the old path, bit for bit -----------------------------------------------------------------

class _DS:
    def __init__(self, t, n=3):
        g = torch.Generator().manual_seed(7)
        self.items = []
        for i in range(n):
            valid = torch.ones(224, 224)
            valid[:60, :90] = 0                               # a masked corner, as an IPM picture has
            H = torch.eye(3, dtype=torch.float64)
            H[0, 2], H[1, 2] = t[0] + 3.3 * i, t[1] - 2.1 * i
            self.items.append(dict(id=f"s{i}", city="Chicago", H=H, bev=torch.rand(3, 224, 224, generator=g),
                                   bev_valid=valid, ref=torch.rand(3, 896, 896, generator=g)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def centre_guess_m(self, i):
        return 0.0


def _old_score(ds, query, matcher, cons, cfg):
    """scripts/eval_vigor.score as it was before the sub-cell stage (the coarse rows only)."""
    rows, n = [], int(cfg.grid.n)
    for i in range(len(ds)):
        s = ds[i]
        batch = {k: (v[None] if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            f_q, frac = query(batch, matcher)
            f_s = matcher.reference_features(batch["ref"])
            sf = float(((f_q.shape[-2] * 16) * (f_q.shape[-1] * 16)) ** 0.5 / 560.0)
            with matcher.model.exposed_intermediates():
                gm = matcher.model.decoder({16: f_q}, f_s, scale_factor=sf)[16]["gm_cls"][0]
        H = s["H"].numpy().astype(float)
        row = dict(id=s["id"], city=s["city"], centre_guess_m=ds.centre_guess_m(i))
        for tag, c in cons.items():
            m = consensus_for_query(c, gm, query, batch, frac, n, min_frac=0.05)
            err = pose_errors(m.H, H, n, float(cfg.grid.cell_m)) if m.H is not None else None
            row[f"pose_{tag}_m"] = None if err is None else err["position_m"]
            row[f"yaw_{tag}_deg"] = None if err is None else err["yaw_deg"]
            row[f"inliers_{tag}"] = m.inlier_ratio
        rows.append(row)
    return rows


@pytest.mark.parametrize("solver", ["se2", "srt"])
def test_refine_zero_is_bit_identical_and_refined_rows_do_not_touch_the_coarse_rows(solver):
    from eval_vigor import score
    t = (300.0, 280.0)
    cfg = cfg_stub(solver)
    matcher = tiny_matcher(plant_translation(tiny_decoder(), t))
    query, ds = StubPictureQuery(), _DS(t)
    cons = {tag: SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=means, min_valid_frac=0.05)
            for tag, means in (("peak", False), ("means", True))}
    old = _old_score(ds, query, matcher, cons, cfg)
    new0 = score(ds, query, matcher, cons, cfg, "cpu", refine=0)
    assert new0 == old                                       # every key, every float, bit for bit
    assert any(r["pose_peak_m"] is not None for r in old)
    new = score(ds, query, matcher, cons, cfg, "cpu", refine=16, refine_inits=("none", "coarse", "ransac"),
                refine_gate=3.0)
    for r_old, r_new in zip(old, new):
        assert {k: r_new[k] for k in r_old} == r_old
        for init in ("none", "coarse", "ransac"):
            assert f"pose_refined_{init}_m" in r_new and f"inliers_refined_{init}" in r_new
            assert r_new[f"ncorr_refined_{init}"] > 0
    new1 = score(ds, query, matcher, cons, cfg, "cpu", refine=8, refine_inits=("coarse",), refine_gate=3.0)
    assert all("pose_refined_m" in r for r in new1)
    # the planted classifier localises; the (random, small) refiner keeps it within a few cells
    assert all(r["pose_refined_m"] is not None and r["pose_refined_m"] < 5 * 16 * cfg.grid.cell_m for r in new1)


# ---- placed tokens: a refined ERP pixel goes through its own depth ray -----------------------------------------------

def _erp_batch(h=28, w=56, seed=0):
    g = torch.Generator().manual_seed(seed)
    depth = 1.0 + 30.0 * torch.rand(1, 1, h * 16, w * 16, generator=g)
    depth[..., :40, :] = 80.0                                 # sky: beyond max depth, invalid
    return dict(erp=torch.zeros(1, 1, 3, h * 16, w * 16), depth=depth, R_w2c=R_N[:, None].float())


def test_erp_depth_pixel_placement_matches_the_token_placement_at_token_centres():
    h, w = 28, 56
    b = _erp_batch(h, w)
    xy_tok, v_tok = depth_placement_metric(b["depth"], h, w, b["R_w2c"][:, 0])
    uv_c = token_centres_px(h, w) + 0.5                        # pixel-centred -> continuous ERP coords
    xy_px, v_px = depth_placement_metric_at(b["depth"], uv_c.reshape(-1, 2), (h * 16, w * 16), b["R_w2c"][:, 0])
    assert torch.equal(v_px.reshape(1, h, w), v_tok)
    assert torch.equal(xy_px.reshape(1, h, w, 2), xy_tok)      # same rays, same depth pixel: identical
    # the flat-ground placement too
    xy_g, vg = erp_placement(h, w, (h * 16, w * 16), b["R_w2c"][:, 0], 2.5, 224, 0.125)
    xy_g2, vg2 = erp_placement_at(uv_c.reshape(-1, 2), (h * 16, w * 16), b["R_w2c"][:, 0], 2.5, 224, 0.125)
    assert torch.equal(xy_g2.reshape(1, h, w, 2), xy_g) and torch.equal(vg2.reshape(1, h, w), vg)


def test_erp_depth_pixel_placement_uses_that_pixels_depth_along_that_pixels_ray():
    h, w = 28, 56
    H, W = h * 16, w * 16
    b = _erp_batch(h, w, seed=1)
    uv = torch.tensor([[W / 2 + 37.5, H / 2 + 20.5], [W * 0.75 + 3.25, H / 2 + 60.5], [101.0, 300.75]])
    xy, valid = depth_placement_metric_at(b["depth"], uv, (H, W), b["R_w2c"][:, 0])
    for k, (u, v) in enumerate(uv.tolist()):
        d = float(b["depth"][0, 0, int(v), int(u)])           # nearest pixel = floor of the continuous coords
        az = (u / W - 0.5) * 2 * math.pi                       # clockwise from north (the centre column)
        el = (0.5 - v / H) * math.pi
        assert bool(valid[0, k])
        assert abs(float(xy[0, k, 0]) - d * math.cos(el) * math.cos(az)) < 1e-3     # x forward (north)
        assert abs(float(xy[0, k, 1]) + d * math.cos(el) * math.sin(az)) < 1e-3     # y left: east is -y
    # ErpDepthQuery.placement_at = the same metric point on the virtual BEV grid
    cfg = NS(grid=NS(n=224, cell_m=0.125), erp_depth=NS(max_depth_m=35.0, head=False))
    q = ErpDepthQuery(cfg)
    bev, v2 = q.placement_at(b, uv)
    assert torch.equal(v2, valid)
    assert torch.allclose(bev[0, :, 0], 112 - xy[0, :, 1] / 0.125 - 0.5)
    assert torch.allclose(bev[0, :, 1], 112 - xy[0, :, 0] / 0.125 - 0.5)


def test_placed_query_refinement_runs_on_erp_pixels():
    """erp_depth path of refined_for_query: ERP pixels -> own depth ray -> virtual BEV px; runs and gives a model."""
    h, w = 28, 56
    cfg = NS(grid=NS(n=224, cell_m=0.125), erp_depth=NS(max_depth_m=35.0, head=False))
    q = ErpDepthQuery(cfg)
    b = _erp_batch(h, w, seed=2)
    xy, valid = q.placement(b)
    # a coarse warp consistent with a translation of the placed points, then the (random) refiner
    t = torch.tensor([400.0, 390.0])
    dec = tiny_decoder()
    tap = RefinerTap(dec)
    with torch.no_grad():
        o = _decode(dec, h, w, tap=tap)
        w_in = px_to_norm(xy[0] + t, 896).permute(2, 0, 1)[None]
        flow, dc = tap.rerun(w_in)
    o16 = dict(flow=w_in, certainty=o["gm_certainty"], flow_pre_delta=w_in, gm_certainty=o["gm_certainty"],
               gm_cls=o["gm_cls"])
    cons = _cons("se2")
    Hc = np.eye(3)
    Hc[:2, 2] = t.numpy()
    coarse = NS(H=Hc)
    for init, stride in (("none", 16), ("coarse", 16), ("coarse", 8), ("ransac", 16)):
        m, info = refined_for_query(cons, o16, tap, q, b, stride, init=init, coarse=coarse, gate_cells=3.0)
        assert m.H is not None and info["n_corr"] > 50, (init, stride, info)
        if init != "ransac":                                  # the warp IS the translation of the placed points
            tol = 1e-3 if stride == 16 else 16.0              # stride 8 pairs interpolated warps with own-pixel depth
            assert np.abs(m.H[:2, 2] - t.numpy()).max() < tol, (init, stride, m.H)


def test_null_refiner_on_a_cell_aligned_translation_localises_exactly():
    """With the refiner's output zeroed the refined warp is the package's ToWarp soft-argmax; on a classifier planted
    at a cell-aligned translation that is exact, so the refined row must be 0 m: the px / normalised / grid / pixel-
    homography conventions agree end to end. (Samples 1, 2 carry a known GT shift of 3.9 px = 0.489 m per step.)"""
    from eval_vigor import score
    t = (304.0, 288.0)                                        # 19 and 18 cells: every token lands on a cell centre
    dec = plant_translation(tiny_decoder(), t)
    with torch.no_grad():
        dec.conv_refiner["16"].out_conv.weight.zero_()
        dec.conv_refiner["16"].out_conv.bias.zero_()
    cfg = cfg_stub("se2")
    matcher = tiny_matcher(dec)
    cons = {"peak": SatRoMa.from_wrapper(matcher.wrapper, cfg, use_means=False, min_valid_frac=0.05)}
    rows = score(_DS(t), StubPictureQuery(), matcher, cons, cfg, "cpu", refine=16,
                 refine_inits=("none", "coarse", "ransac"), refine_gate=3.0)
    shift_m = math.hypot(3.3, 2.1) * cfg.grid.cell_m
    for i, r in enumerate(rows):
        for init in ("none", "coarse", "ransac"):
            assert abs(r[f"pose_refined_{init}_m"] - i * shift_m) < 1e-3, (i, init, r)
