"""Loc² VCE pose loss on a differentiable weighted Procrustes, the decoder on a non-square ERP token grid, and the
training step of --query erp_depth end to end on a small randomly initialised decoder (no checkpoint, CPU)."""
from __future__ import annotations

import math
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from bevloc import config as C
from bevloc.data.vigor import R_NORTH
from bevloc.model.coarse import (
    cell_centres, procrustes_2d, resolve_vce_weight, vce_distance, vce_options, vce_pose_loss, virtual_points,
)

N_BEV, CELL, REF, K = 224, 0.25, 896, 56


def _rot(th):
    return torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def test_procrustes_recovers_a_rigid_transform_exactly():
    g = torch.Generator().manual_seed(0)
    A = torch.rand(3, 50, 2, generator=g) * 200
    R_true = torch.stack([_rot(a) for a in (0.0, 0.4, -2.5)])
    t_true = torch.tensor([[10.0, -5.0], [300.0, 250.0], [-40.0, 7.0]])
    B = A @ R_true.transpose(1, 2) + t_true[:, None]
    w = torch.rand(3, 50, generator=g)
    R, t = procrustes_2d(A, B, w)
    assert torch.allclose(R, R_true, atol=1e-4) and torch.allclose(t, t_true, atol=1e-2)
    assert torch.allclose(torch.linalg.det(R), torch.ones(3), atol=1e-5)   # proper rotation, no reflection
    # zero-weight outliers do not move the solution
    B2 = B.clone()
    B2[:, :5] += 500.0
    w2 = w.clone()
    w2[:, :5] = 0.0
    R2, t2 = procrustes_2d(A, B2, w2)
    assert torch.allclose(R2, R_true, atol=1e-4) and torch.allclose(t2, t_true, atol=1e-2)


def test_vce_distance_is_zero_at_gt_and_the_offset_for_a_translation():
    X = virtual_points(N_BEV, CELL, 5.0, 10)
    assert X.shape == (100, 2)
    assert torch.allclose(X.mean(0), torch.full((2,), (N_BEV - 1) / 2.0), atol=1e-4)            # centred on the camera
    assert abs(float(X[:, 0].max() - X[:, 0].min()) * CELL - 5.0) < 1e-4                          # 5 m square
    H = torch.tensor([[[1.0, 0.0, 336.0], [0.0, 1.0, 300.0], [0.0, 0.0, 1.0]]])
    R, t = torch.eye(2)[None], H[:, :2, 2]
    assert float(vce_distance(R, t, H, X, CELL)) < 1e-5
    for dm in (1.0, 3.0, 10.0):
        e = float(vce_distance(R, t + torch.tensor([[dm / CELL * 0.6, dm / CELL * 0.8]]), H, X, CELL))
        assert abs(e - dm) < 1e-4
    # a rotation about the camera moves the virtual points by 2 r sin(θ/2): > 0, larger for a larger angle
    o = torch.full((2,), (N_BEV - 1) / 2.0)
    e = [float(vce_distance(_rot(a)[None], (H[0, :2, 2] + o - _rot(a) @ o)[None], H, X, CELL)) for a in (0.05, 0.2)]
    assert 0 < e[0] < e[1]


def _synthetic_logits(xy, valid, H_true, sharp=30.0):
    """gm_cls (1, K², h, w): each valid token votes (sharply) for the cell its placed point maps to under H_true."""
    h, w = xy.shape[1:3]
    p = torch.cat([xy.reshape(-1, 2), torch.ones(h * w, 1)], -1) @ H_true.T
    s = REF / K
    col = torch.floor((p[:, 0] + 0.5) / s).long().clamp(0, K - 1)
    row = torch.floor((p[:, 1] + 0.5) / s).long().clamp(0, K - 1)
    gm = torch.zeros(1, K * K, h * w)
    gm[0, row * K + col, torch.arange(h * w)] = sharp
    return gm.reshape(1, K * K, h, w)


@pytest.mark.parametrize("mode", ["sample", "expect"])
def test_vce_loss_is_zero_at_gt_and_grows_with_translation_error(mode):
    h, w = 12, 16
    g = torch.Generator().manual_seed(0)
    # placed points up to ~25 m around the camera (virtual BEV px), a few invalid tokens
    xy = (N_BEV - 1) / 2.0 + (torch.rand(1, h, w, 2, generator=g) - 0.5) * (50.0 / CELL)
    valid = torch.rand(1, h, w, generator=g) > 0.2
    H_gt = torch.tensor([[[1.0, 0.0, 340.0], [0.0, 1.0, 330.0], [0.0, 0.0, 1.0]]])
    losses = []
    for dm in (0.0, 2.0, 5.0, 10.0, 20.0):
        H_vote = H_gt[0].clone()
        H_vote[0, 2] += dm / CELL * 0.6                     # the matcher's votes are off by dm metres
        H_vote[1, 2] -= dm / CELL * 0.8
        gm = _synthetic_logits(xy, valid, H_vote)
        loss, st = vce_pose_loss(gm, xy, valid, H_gt, CELL, N_BEV, ref_size=REF, mode=mode, n_samples=1024,
                                 generator=torch.Generator().manual_seed(1))
        assert st["n_vce"] == 1
        losses.append(float(loss))
        assert abs(float(loss) - dm) < 0.35, (dm, float(loss))            # 4 m cells: quantisation averages out
        assert abs(st["vce_pose_m"] - dm) < 0.35
    assert losses[0] < 0.35
    assert all(b > a for a, b in zip(losses, losses[1:])), losses


@pytest.mark.parametrize("mode", ["sample", "expect"])
def test_vce_loss_has_gradients_for_logits_and_certainty(mode):
    h, w = 6, 8
    g = torch.Generator().manual_seed(2)
    xy = (N_BEV - 1) / 2.0 + (torch.rand(2, h, w, 2, generator=g) - 0.5) * 120
    valid = torch.ones(2, h, w, dtype=torch.bool)
    valid[1] = False
    valid[1, 0, :2] = True                                 # 2 < min_tokens: sample 1 is left out
    gm = torch.randn(2, K * K, h, w, generator=g, requires_grad=True)
    cert = torch.randn(2, 1, h, w, generator=g, requires_grad=True)
    H_gt = torch.eye(3).repeat(2, 1, 1)
    H_gt[:, 0, 2], H_gt[:, 1, 2] = 250.0, 200.0             # camera ~35 m from the canvas centre
    loss, st = vce_pose_loss(gm, xy, valid, H_gt, CELL, N_BEV, ref_size=REF, gm_certainty=cert, mode=mode,
                             n_samples=256, generator=torch.Generator().manual_seed(0))
    # random logits: "expect" collapses every token onto the canvas centre, "sample" scatters: both far off
    assert st["n_vce"] == 1 and torch.isfinite(loss) and float(loss.detach()) > 5.0
    loss.backward()
    assert torch.isfinite(gm.grad).all() and float(gm.grad[0].abs().sum()) > 0
    assert float(gm.grad[1].abs().sum()) == 0.0
    assert torch.isfinite(cert.grad).all() and float(cert.grad[0].abs().sum()) > 0


def test_vce_loss_without_enough_tokens_is_zero_with_a_graph():
    gm = torch.randn(1, K * K, 4, 4, requires_grad=True)
    loss, st = vce_pose_loss(gm, torch.zeros(1, 4, 4, 2), torch.zeros(1, 4, 4, dtype=torch.bool),
                             torch.eye(3)[None], CELL, N_BEV)
    assert float(loss) == 0.0 and st["n_vce"] == 0
    loss.backward()


def test_cell_centres_follow_the_coarse_target_binning():
    c = cell_centres(K, REF)
    assert torch.allclose(c[0], torch.tensor([7.5, 7.5])) and torch.allclose(c[K + 2], torch.tensor([39.5, 23.5]))


def test_vce_weight_auto_and_options():
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    assert resolve_vce_weight(cfg, "erp_depth") == 1.0 and resolve_vce_weight(cfg, "ipm") == 0.0
    cfg.train.vce_weight = 0.3
    assert resolve_vce_weight(cfg, "erp") == 0.3
    o = vce_options(cfg)
    assert o == dict(mode="sample", n_samples=1024, grid_m=5.0, points=10, use_certainty=True)


# ---- (d) the decoder on a non-square token grid, and the placed-query consensus on it ---------------------------

def _small_decoder(proj_dim=32, in_dim=1024):
    """The released decoder's architecture (sat_roma.build) at a small width, randomly initialised, without the
    conv refiner (the only consumer of scale_factor; gm_cls / gm_certainty are produced before it)."""
    from bevloc.match import satroma  # noqa: F401  (puts sat-roma-infer on sys.path)
    from sat_roma.matcher import CosKernel, Decoder, GP
    from sat_roma.transformer import Block, MemEffAttention, TransformerDecoder
    td = TransformerDecoder(nn.Sequential(Block(2 * proj_dim, 8, attn_class=MemEffAttention)), 2 * proj_dim,
                            K * K + 1, is_classifier=True, amp=False, pos_enc=False, amp_dtype=torch.float32)
    gps = nn.ModuleDict({"16": GP(CosKernel, T=0.2, learn_temperature=False, only_attention=False, gp_dim=proj_dim,
                                  basis="fourier", no_cov=True, solver="lu", sigma_noise=0.1)})
    proj = nn.ModuleDict({"16": nn.Sequential(nn.Conv2d(in_dim, proj_dim, 1), nn.BatchNorm2d(proj_dim))})
    dec = Decoder(td, gps, proj, nn.ModuleDict(), detach=True, scales=["16"], displacement_dropout_p=0.0,
                  gm_warp_dropout_p=0.0, amp_dtype=torch.float32)
    dec.expose_intermediates = True
    return dec.eval()


def test_decoder_accepts_a_non_square_token_grid_and_the_placed_consensus_runs():
    from bevloc.match.satroma import SatRoMa
    from bevloc.model.depth_query import depth_placement
    torch.manual_seed(0)
    dec = _small_decoder()
    f_q, f_s = torch.randn(2, 1024, 28, 56), torch.randn(2, 1024, 56, 56)
    with torch.no_grad():
        out = dec({16: f_q}, {16: f_s}, scale_factor=float((448 * 896) ** 0.5 / 560))[16]
    assert out["gm_cls"].shape == (2, K * K, 28, 56) and out["gm_certainty"].shape == (2, 1, 28, 56)

    class Stub:                                            # consensus_from_gm reads only these fields
        m = type("m", (), {"im_a_size": N_BEV, "im_b_size": REF})()
        use_means, reproj, seed = False, 3.0, 0

        def __init__(self, solver):
            self.solver = solver

    # depth-placed tokens of a 28 x 56 panorama: 8-30 m below/around the horizon, sky invalid
    rng = np.random.default_rng(0)
    dep = torch.from_numpy(rng.uniform(8.0, 30.0, (28, 56)).astype(np.float32))
    dep[:10] = 100.0
    depth = dep.repeat_interleave(16, 0).repeat_interleave(16, 1)[None, None]
    xy, valid = depth_placement(depth, 28, 56, torch.from_numpy(R_NORTH.copy())[None], N_BEV, CELL, 35.0)
    # the random decoder's logits: the path runs and returns a Match
    m = SatRoMa.consensus_from_gm(Stub("se2"), out["gm_cls"][0], xy[0], valid[0])
    assert m.n_modes >= 0
    # logits voting for a known pose (camera 6 m east / 9 m south of the canvas centre): recovered to < 1 m
    H_true = torch.tensor([[1.0, 0.0, 447.5 - 111.5 + 24.0], [0.0, 1.0, 447.5 - 111.5 + 36.0], [0.0, 0.0, 1.0]])
    gm = _synthetic_logits(xy, valid, H_true)[0]
    for solver in ("se2", "sim"):
        r = SatRoMa.consensus_from_gm(Stub(solver), gm, xy[0], valid[0])
        assert r.H is not None, solver
        o = np.array([111.5, 111.5, 1.0])
        est, gt = r.H @ o, H_true.double().numpy() @ o
        assert np.linalg.norm(est[:2] / est[2] - gt[:2]) * CELL < 1.0, solver
    # the evaluators' shared entry point takes the placed path for an erp_depth query
    from bevloc.match.satroma import consensus_for_query
    from bevloc.model.query import build_query
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    q = build_query(cfg, "erp_depth")
    batch = dict(erp=torch.rand(1, 1, 3, 448, 896), R_w2c=torch.from_numpy(R_NORTH.copy())[None, None], depth=depth)
    r2 = consensus_for_query(Stub("se2"), gm, q, batch, valid.float(), N_BEV)
    r1 = SatRoMa.consensus_from_gm(Stub("se2"), gm, xy[0], valid[0])
    assert r2.H is not None and np.allclose(r2.H, r1.H)


# ---- (3) the training step of --query erp_depth, end to end ---------------------------------------------------

class _TinyMatcher(nn.Module):
    """FeatureQueryMatcher stand-in: fake frozen encoder (1024 ch at stride 16) + the small real decoder."""

    def __init__(self):
        super().__init__()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = _small_decoder()

            @staticmethod
            def encoder(x):
                return {16: torch.nn.functional.avg_pool2d(x.repeat(1, 342, 1, 1)[:, :1024], 16)}

        self.model = Model()

    def reference_features(self, ref):
        with torch.no_grad():
            return self.model.encoder(ref)


def test_training_step_with_erp_depth_and_vce_backpropagates():
    sys.path.insert(0, str(C.REPO / "scripts"))
    from train_lift_splat import step, validate
    from bevloc.model.query import build_query
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.lift.query_mode = "erp_depth"
    cfg.erp_depth.head = True
    cfg.erp_depth.head_dim = 32
    cfg.erp_depth.vce_samples = 128
    q = build_query(cfg, "erp_depth")
    mt = _TinyMatcher()
    rng = np.random.default_rng(3)
    dep = torch.from_numpy(rng.uniform(5.0, 30.0, (2, 28, 56)).astype(np.float32))
    dep[:, :8] = 60.0
    H = torch.eye(3).repeat(2, 1, 1)
    H[:, 0, 2], H[:, 1, 2] = 336.0, 330.0
    batch = dict(erp=torch.rand(2, 1, 3, 448, 896), R_w2c=torch.from_numpy(R_NORTH.copy())[None, None].repeat(2, 1, 1, 1),
                 depth=dep.repeat_interleave(16, 1).repeat_interleave(16, 2)[:, None],
                 ref=torch.rand(2, 3, REF, REF), H=H, negative=torch.zeros(2, dtype=torch.bool))
    loss, st = step(q, mt, batch, cfg, 0.05, 0, 4, 0.5, certainty_weight=0.01, pose_nll_weight=0.5,
                    vce_weight=1.0, vce_opts=vce_options(cfg), generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss) and st["vce_m"] == st["vce_m"] and st["n"] > 0
    loss.backward()
    dec_grads = [p.grad for p in mt.model.decoder.parameters() if p.grad is not None]
    assert dec_grads and all(torch.isfinite(g_).all() for g_ in dec_grads)
    assert q.head.up.weight.grad is not None and float(q.head.up.weight.grad.abs().sum()) > 0
    # validation with a fixed match draw is deterministic
    v1 = validate(q, mt, [batch], cfg, 0.05, 0, "cpu", vce_weight=1.0, vce_opts=vce_options(cfg))
    v2 = validate(q, mt, [batch], cfg, 0.05, 0, "cpu", vce_weight=1.0, vce_opts=vce_options(cfg))
    assert v1["vce_pose_m"] == v2["vce_pose_m"] and v1["vce_m"] == v1["vce_m"]
    # vce_weight 0 (every other mode): the loss is exactly the previous objective, no VCE stats
    l0, s0 = step(q, mt, batch, cfg, 0.05, 0, 4, 0.5, certainty_weight=0.01, pose_nll_weight=0.5)
    assert s0["vce_m"] != s0["vce_m"]


# ---- review follow-ups: reference-validity mask, degenerate guards, head_dim check ------------------------------

@pytest.mark.parametrize("mode", ["sample", "expect"])
def test_vce_pairs_stay_on_valid_reference_cells(mode, monkeypatch):
    import bevloc.model.coarse as co
    h, w = 6, 8
    g = torch.Generator().manual_seed(4)
    xy = (N_BEV - 1) / 2.0 + (torch.rand(1, h, w, 2, generator=g) - 0.5) * 120
    valid = torch.ones(1, h, w, dtype=torch.bool)
    gm = torch.randn(1, K * K, h, w, generator=g) * 3
    rv = torch.zeros(1, K, K, dtype=torch.bool)
    rv[0, 20:23, 30:33] = True                             # only a 3 x 3 block of the canvas has content
    seen = {}
    real = co.procrustes_2d

    def spy(A, B, wts, *args, **kw):
        seen["B"], seen["w"] = B.detach().clone(), wts.detach().clone()
        return real(A, B, wts, *args, **kw)
    monkeypatch.setattr(co, "procrustes_2d", spy)
    loss, st = vce_pose_loss(gm, xy, valid, torch.eye(3)[None], CELL, N_BEV, ref_size=REF, mode=mode,
                             n_samples=512, generator=torch.Generator().manual_seed(0), ref_valid=rv)
    assert st["n_vce"] == 1 and torch.isfinite(loss)
    s = REF / K
    lo = torch.tensor([30 * s - 0.5, 20 * s - 0.5])        # block edges in reference px (x, y)
    hi = torch.tensor([33 * s - 0.5, 23 * s - 0.5])
    B = seen["B"][0]
    assert bool(((B >= lo) & (B <= hi)).all()), B.min(0).values.tolist() + B.max(0).values.tolist()
    if mode == "sample":                                   # every drawn cell index is one of the block's centres
        cols = torch.floor((B[:, 0] + 0.5) / s).long()
        rows = torch.floor((B[:, 1] + 0.5) / s).long()
        assert bool(((rows >= 20) & (rows < 23) & (cols >= 30) & (cols < 33)).all())
        assert bool((seen["w"] > 0).all())


def test_vce_sample_without_valid_reference_cells_is_left_out():
    gm = torch.randn(2, K * K, 4, 4, requires_grad=True)
    xy = torch.rand(2, 4, 4, 2) * 200
    rv = torch.ones(2, K, K, dtype=torch.bool)
    rv[1] = False
    loss, st = vce_pose_loss(gm, xy, torch.ones(2, 4, 4, dtype=torch.bool), torch.eye(3).repeat(2, 1, 1), CELL, N_BEV,
                             n_samples=64, generator=torch.Generator().manual_seed(0), ref_valid=rv)
    assert st["n_vce"] == 1 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(gm.grad).all() and float(gm.grad[1].abs().sum()) == 0.0


def test_procrustes_degenerate_input_gives_identity_rotation():
    A = torch.full((2, 5, 2), 3.0)                          # coincident points
    B = torch.full((2, 5, 2), 7.0)
    R, t = procrustes_2d(A, B, torch.ones(2, 5))
    assert torch.allclose(R, torch.eye(2).expand(2, 2, 2)) and torch.allclose(t, torch.full((2, 2), 4.0))
    R0, t0 = procrustes_2d(torch.rand(1, 5, 2), torch.rand(1, 5, 2), torch.zeros(1, 5))   # all weights zero
    assert torch.allclose(R0, torch.eye(2)[None]) and torch.isfinite(t0).all()


def test_vce_sample_with_zero_token_weight_is_left_out_not_an_error():
    gm = torch.randn(2, K * K, 4, 4, requires_grad=True)
    cert = torch.zeros(2, 1, 4, 4)
    cert[1] = -1e4                                          # sigmoid underflows to exactly 0: no token weight
    xy = torch.rand(2, 4, 4, 2) * 200
    for mode in ("sample", "expect"):
        loss, st = vce_pose_loss(gm, xy, torch.ones(2, 4, 4, dtype=torch.bool), torch.eye(3).repeat(2, 1, 1), CELL,
                                 N_BEV, gm_certainty=cert, mode=mode, n_samples=64,
                                 generator=torch.Generator().manual_seed(0))
        assert st["n_vce"] == 1 and torch.isfinite(loss)
    cert[0] = -1e4
    loss, st = vce_pose_loss(gm, xy, torch.ones(2, 4, 4, dtype=torch.bool), torch.eye(3).repeat(2, 1, 1), CELL, N_BEV,
                             gm_certainty=cert, n_samples=64)
    assert st["n_vce"] == 0 and float(loss) == 0.0


def test_projection_head_rejects_a_width_not_divisible_by_8():
    from bevloc.model.depth_query import ProjectionHead
    with pytest.raises(ValueError, match="divisible by 8"):
        ProjectionHead(dim=100)
    ProjectionHead(dim=64, heads=4)
