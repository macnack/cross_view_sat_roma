"""pillar query (task 07): FG²-style pillar lift geometry, sampling, depth-taught height targets, the module, wiring."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from bevloc.data.vigor import R_NORTH
from bevloc.model.erp_query import rays_at
from bevloc.model.pillar_query import (
    grid_xy, height_targets, pillar_points, project_erp, sample_pillars, slice_validity,
)

R_N = torch.from_numpy(R_NORTH.copy())[None]          # (1, 3, 3) world ENU -> camera (x east, y down, z north)
H, W = 448, 896
h, w = H // 16, W // 16


# ---- (a) ERP projection: the exact inverse of rays_at ------------------------------------------------------------

def test_project_erp_inverts_rays_at_on_the_full_sphere():
    g = torch.Generator().manual_seed(0)
    d = torch.randn(500, 3, generator=g)
    d = d / d.norm(dim=-1, keepdim=True)
    u, v, rng = project_erp(d * 7.0, H, W)
    assert torch.allclose(rng, torch.full((500,), 7.0), atol=1e-5)
    assert bool((u >= 0).all()) and bool((u <= W).all()) and bool((v >= 0).all()) and bool((v <= H).all())
    back, _ = rays_at(u, v, H, W)
    assert torch.allclose(back, d, atol=1e-5)


def test_project_erp_inverts_rays_at_inside_an_elevation_band():
    band = (0.30, -0.45)                                   # lat_top, lat_bottom (radians), the erp_band convention
    Hb = 192
    g = torch.Generator().manual_seed(1)
    u0 = torch.rand(300, generator=g) * W
    v0 = torch.rand(300, generator=g) * Hb
    d, _ = rays_at(u0, v0, Hb, W, band)
    u, v, _ = project_erp(d, Hb, W, band)
    assert torch.allclose(u, u0, atol=1e-3) and torch.allclose(v, v0, atol=1e-3)


def test_ground_slice_of_the_cell_10m_ahead_hits_the_centre_column_below_the_horizon():
    hc = 2.5
    xy = torch.tensor([[10.0, 0.0], [0.0, -10.0]])        # 10 m ahead (north); 10 m to the right (east, y = -10)
    p_cam = pillar_points(xy, [0.0], hc, R_N)              # (B, N, M, 3), ground slice only
    assert p_cam.shape == (1, 2, 1, 3)
    u, v, rng = project_erp(p_cam, H, W)
    assert abs(float(u[0, 0, 0]) - W / 2) < 1e-4                         # forward = centre column
    assert abs(float(v[0, 0, 0]) - (0.5 + math.atan(hc / 10.0) / math.pi) * H) < 1e-3   # below the horizon
    assert abs(float(rng[0, 0, 0]) - math.hypot(10.0, hc)) < 1e-4
    assert abs(float(u[0, 1, 0]) - 3 * W / 4) < 1e-4                     # east = 3/4 of the width


def test_slices_above_ground_rise_in_the_image_and_a_yawed_camera_shifts_the_column():
    xy = torch.tensor([[10.0, 0.0]])
    z = [0.0, 5.0, 20.0]
    u, v, _ = project_erp(pillar_points(xy, z, 2.5, R_N), H, W)
    assert float(v[0, 0, 0]) > float(v[0, 0, 1]) > float(v[0, 0, 2])     # higher slice = smaller row
    assert float(v[0, 0, 1]) < H / 2 < float(v[0, 0, 0])                  # 5 m above ground is above the horizon
    yaw = torch.tensor([[math.cos(math.pi / 2), -math.sin(math.pi / 2), 0.0],
                        [math.sin(math.pi / 2), math.cos(math.pi / 2), 0.0], [0.0, 0.0, 1.0]])
    R_yawed = R_N[0] @ yaw                                 # the camera turned 90 deg: world north is now at its left
    u2, _, _ = project_erp(pillar_points(xy, [0.0], 2.5, R_yawed[None], ego=False), H, W)
    assert abs(float(u2[0, 0, 0]) - W / 4) < 1e-3          # a WORLD-frame point 10 m north lands 1/4 width (left)


def test_grid_xy_is_the_cell_centres_convention_row0_forward_col0_left():
    xy = grid_xy(35, 2.0)                                  # (n, n, 2) metres, x forward / y left
    assert xy.shape == (35, 35, 2)
    assert torch.allclose(xy[0, 17], torch.tensor([34.0, 0.0])) and torch.allclose(xy[17, 0], torch.tensor([0.0, 34.0]))
    assert torch.allclose(xy[12, 17], torch.tensor([10.0, 0.0]))


# ---- (b) sampling the token map at the projected points ------------------------------------------------------

def _token_of(u, v):
    return int(v // 16), int(u // 16)


def test_sample_pillars_reads_the_token_under_the_ground_point():
    xy = torch.tensor([[10.0, 0.0]])
    p_cam = pillar_points(xy, [0.0, 3.0], 2.5, R_N)
    u, v, _ = project_erp(p_cam, H, W)
    i, j = _token_of(float(u[0, 0, 0]), float(v[0, 0, 0]))
    f = torch.zeros(1, 4, h, w)
    f[0, 0, i, j] = 1.0
    s = sample_pillars(f, torch.stack([u, v], -1), H, W)   # (B, N, M, C)
    assert s.shape == (1, 1, 2, 4)
    assert float(s[0, 0, 0, 0]) >= 0.25 and float(s[0, 0, 1, 0]) < 0.25   # the 3 m slice looks at another token


def test_sample_pillars_wraps_the_azimuth_seam():
    f = torch.zeros(1, 1, h, w)
    f[0, 0, 5, 0] = 1.0                                    # column 0 = azimuth -180 deg
    uv = torch.tensor([[[[0.2, 5 * 16 + 8.0]], [[W - 0.2, 5 * 16 + 8.0]]]])   # (1, 2, 1, 2): just inside both edges
    s = sample_pillars(f, uv, H, W)
    assert float(s[0, 0, 0, 0]) > 0.4 and float(s[0, 1, 0, 0]) > 0.4   # the far edge sees column 0 through the wrap


def test_slice_validity_masks_the_ego_cut_and_the_band():
    xy = torch.tensor([[2.0, 0.0], [30.0, 0.0]])
    p_cam = pillar_points(xy, [0.0, 10.0], 2.5, R_N)
    ok = slice_validity(p_cam, H, W, band=None, max_depression_deg=20.0)
    assert ok.shape == (1, 2, 2)
    assert not ok[0, 0, 0] and ok[0, 0, 1] and ok[0, 1, 0] and ok[0, 1, 1]   # ground 2 m ahead is 51 deg down
    assert bool(slice_validity(p_cam, H, W, None, 90.0).all())
    okb = slice_validity(p_cam, 192, W, band=(0.30, -0.05), max_depression_deg=90.0)
    assert not okb[0, 1, 0] and okb[0, 1, 1]                               # ground 30 m ahead (-4.8 deg) is below -2.9 deg


# ---- (c) depth-taught height targets ------------------------------------------------------------------------

def test_height_targets_pick_the_slice_whose_range_matches_the_depth():
    xy = grid_xy(9, 4.0).reshape(-1, 2)
    z = [0.0, 1.0, 3.0, 8.0, 15.0]
    p_cam = pillar_points(xy, z, 2.5, R_N)
    u, v, rng = project_erp(p_cam, H, W)
    uv = torch.stack([u, v], -1)
    ok = slice_validity(p_cam, H, W, None, 90.0)
    k = 2                                                  # render slice 2's range into the depth map at its pixels
    depth = torch.full((1, 1, H, W), 60.0)
    for n in range(uv.shape[1]):
        c, r = int(uv[0, n, k, 0]) % W, int(uv[0, n, k, 1])   # u = W is the seam (the cell straight behind)
        depth[0, 0, r, c] = rng[0, n, k]
    tgt, has = height_targets(rng, uv, ok, depth, H, W, tol_m=1.0, tol_frac=0.1)
    assert tgt.shape == (1, 81) and has.shape == (1, 81)
    assert bool(has.all()) and bool((tgt == k).all())
    far = torch.full((1, 1, H, W), 60.0)                   # 60 m everywhere: no slice agrees anywhere
    _, has2 = height_targets(rng, uv, ok, far, H, W, tol_m=1.0, tol_frac=0.1)
    assert not has2.any()


# ---- (d) the module: factory, head, placement, height loss, checkpoint round trip ----------------------------

from bevloc import config as C                                                   # noqa: E402
from bevloc.model.depth_query import metric_to_bev_px                            # noqa: E402
from bevloc.model.pillar_query import PILLAR_ARCH_KEYS, PillarQuery              # noqa: E402
from bevloc.model.query import apply_query_cfg, build_query, load_query_state    # noqa: E402

O = (224 - 1) / 2.0


class _FakeMatcher:
    class model:  # noqa: N801
        @staticmethod
        def encoder(x):
            return {16: torch.nn.functional.avg_pool2d(x.repeat(1, 342, 1, 1)[:, :1024], 16)}


def _cfg(head=True, cell=0.125, **pillar):
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    cfg.grid.cell_m = cell
    cfg.lift.query_mode = "pillar"
    cfg.pillar.head = head
    for k, v in pillar.items():
        setattr(cfg.pillar, k, v)
    return cfg


def _batch(B=1, depth=None):
    b = dict(erp=torch.rand(B, 1, 3, H, W), R_w2c=R_N[None].repeat(B, 1, 1, 1))
    if depth is not None:
        b["depth"] = depth
    return b


def test_build_query_pillar_forward_shapes_and_identity_head_at_init():
    cfg = _cfg(head=True)
    q = build_query(cfg, "pillar")
    assert isinstance(q, PillarQuery) and q.nq == 35                     # 2 m cells over +-35 m
    b = _batch()
    f_q, frac = q(b, _FakeMatcher)
    assert f_q.shape == (1, 1024, 35, 35) and frac.shape == (1, 35, 35)
    assert float(frac.sum()) == 35 * 35 - 1                               # the blind disc removes the ego cell only
    q0 = build_query(_cfg(head=False), "pillar")
    q0.load_state_dict({k: v for k, v in q.state_dict().items() if k in q0.state_dict()})
    f0, _ = q0(b, _FakeMatcher)
    assert torch.allclose(f_q, f0, atol=1e-6)                             # zero-initialised head output: identity
    assert 1e5 <= sum(p.numel() for p in q.parameters()) <= 3e6


def test_pillar_placement_is_the_cell_centres_in_virtual_bev_pixels():
    q = build_query(_cfg(), "pillar")
    xy, valid = q.placement(_batch(B=2))
    assert xy.shape == (2, 35, 35, 2) and valid.shape == (2, 35, 35) and valid.dtype == torch.bool
    exp = metric_to_bev_px(grid_xy(35, 2.0), 224, 0.125)
    assert torch.allclose(xy[1], exp)
    assert torch.allclose(xy[0, 12, 17], torch.tensor([O, O - 10.0 / 0.125]))   # 10 m ahead = 80 px up
    assert not valid[0, 17, 17] and bool(valid[0, 12, 17])
    # a narrow elevation band (a KITScenes-like crop, +-2.9 deg): the near cells have no slice inside it
    b = _batch()
    b["erp_band"] = torch.tensor([[0.05, -0.05]])
    _, vb = build_query(_cfg(cam_height_m=2.5), "pillar").placement(b)
    assert bool(vb.any()) and int(vb.sum()) < int(valid[0].sum()) and not vb[0, 15, 17]   # 4 m ahead: nothing in the band


def test_pillar_placement_in_fine_pixels_is_twice_as_far_from_the_centre():
    xy_c, _ = build_query(_cfg(cell=0.125), "pillar").placement(_batch())
    xy_f, _ = build_query(_cfg(cell=0.0625, cell_m=1.0, range_m=28.0), "pillar").placement(_batch())
    assert xy_f.shape[1:3] == (56, 56)
    assert torch.allclose(xy_c[0, 12, 17] - O, torch.tensor([0.0, -80.0]))          # 10 m ahead at 0.125 m/px
    at_coarse_gsd = metric_to_bev_px(grid_xy(56, 1.0), 224, 0.125)                 # the fine cells placed at the coarse GSD
    assert torch.allclose(xy_f[0] - O, 2.0 * (at_coarse_gsd - O), atol=1e-4)      # twice as far at 0.0625 m/px


def test_height_loss_supervises_the_slice_the_depth_agrees_with():
    cfg = _cfg(head=False, cell_m=4.0, range_m=18.0)
    q = build_query(cfg, "pillar")
    geo = q.geometry(_batch())
    k = 2
    depth = torch.full((1, 1, H, W), 60.0)
    uv, rng = geo["uv"], geo["rng"]
    for n in range(uv.shape[1]):
        c, r = int(uv[0, n, k, 0]) % W, min(int(uv[0, n, k, 1]), H - 1)
        depth[0, 0, r, c] = rng[0, n, k]
    b = _batch(depth=depth)
    q(b, _FakeMatcher)
    loss, n_sup = q.height_loss(b)
    assert n_sup == 9 * 9 - 1 and torch.isfinite(loss) and float(loss.detach()) > 0   # every cell but the blind one
    loss.backward()
    assert q.scorer[0].weight.grad is not None and float(q.scorer[0].weight.grad.abs().sum()) > 0
    with pytest.raises(KeyError):
        q.height_loss(_batch())
    _, n0 = q.height_loss(_batch(depth=torch.full((1, 1, H, W), 60.0)))
    assert n0 == 0


def test_pillar_state_round_trips_and_the_checkpoint_restores_architecture_keys_only():
    cfg = _cfg(head=True)
    cfg.pillar.head_dim = 64
    q = build_query(cfg, "pillar")
    with torch.no_grad():
        for p in q.parameters():
            p.add_(0.01 * torch.randn_like(p))
    state = {"query": q.state_dict(), "mode": "pillar", "pillar": {k: getattr(cfg.pillar, k) for k in PILLAR_ARCH_KEYS}}
    assert "cell_m" not in state["pillar"] and "head_dim" in state["pillar"]
    cfg_eval = apply_query_cfg(_cfg(head=False, cell_m=1.0, range_m=28.0), state)
    assert cfg_eval.pillar.head is True and cfg_eval.pillar.head_dim == 64
    assert cfg_eval.pillar.cell_m == 1.0 and cfg_eval.pillar.range_m == 28.0   # grid keys stay the evaluator's
    q2 = build_query(cfg_eval, "pillar")
    assert load_query_state(q2, state) == (len(q.state_dict()),) * 2
    for k, v in q.state_dict().items():
        assert torch.equal(v, q2.state_dict()[k])
    assert apply_query_cfg(_cfg(head=False), {"query": {}, "mode": "ipm"}).pillar.head is False   # other ckpts: untouched


# ---- (e) wiring: training step, VCE options, DDP parts, refine refusal, dataset depth flag ----------------------

import sys                                                                        # noqa: E402

from bevloc.model.coarse import resolve_vce_weight, vce_options                   # noqa: E402
from bevloc.model.ddp import PART_KEYS                                            # noqa: E402


def test_vce_weight_and_options_come_from_the_pillar_block():
    cfg = _cfg()
    cfg.pillar.vce_samples = 77
    cfg.erp_depth.vce_samples = 99
    assert resolve_vce_weight(cfg, "pillar") == 1.0 and resolve_vce_weight(cfg, "ipm") == 0.0
    assert vce_options(cfg, "pillar")["n_samples"] == 77
    assert vce_options(cfg, "erp_depth")["n_samples"] == 99 and vce_options(cfg)["n_samples"] == 99


def test_training_step_with_pillar_registers_the_height_part_and_backpropagates():
    sys.path.insert(0, str(C.REPO / "scripts"))
    from train_lift_splat import step, validate
    from test_vce import REF, _TinyMatcher
    cfg = _cfg()
    cfg.pillar.head_dim = 32
    cfg.pillar.vce_samples = 128
    q = build_query(cfg, "pillar")
    mt = _TinyMatcher()
    depth = torch.full((2, 1, H, W), 60.0)
    geo = q.geometry(dict(erp=torch.zeros(2, 1, 3, H, W), R_w2c=R_N[None].repeat(2, 1, 1, 1)))
    uv, rng = geo["uv"], geo["rng"]
    for b_ in range(2):                                    # render slice 0 (the ground) into both depth maps
        for n in range(uv.shape[1]):
            c, r = int(uv[b_, n, 0, 0]) % W, min(int(uv[b_, n, 0, 1]), H - 1)
            depth[b_, 0, r, c] = rng[b_, n, 0]
    Hm = torch.eye(3).repeat(2, 1, 1)
    Hm[:, 0, 2], Hm[:, 1, 2] = 336.0, 330.0
    batch = dict(erp=torch.rand(2, 1, 3, H, W), R_w2c=R_N[None].repeat(2, 1, 1, 1), depth=depth,
                 ref=torch.rand(2, 3, REF, REF), H=Hm, negative=torch.zeros(2, dtype=torch.bool))
    parts = {}
    loss, st = step(q, mt, batch, cfg, 0.05, 0, 4, 0.5, certainty_weight=0.01, pose_nll_weight=0.5,
                    vce_weight=1.0, vce_opts=vce_options(cfg, "pillar"), generator=torch.Generator().manual_seed(0),
                    height_weight=1.0, parts=parts)
    assert torch.isfinite(loss) and st["n"] > 0 and st["vce_m"] == st["vce_m"]
    assert "height" in parts and "height" in PART_KEYS
    assert st["height_ce"] == st["height_ce"] and st["height_frac"] > 0.5 and st["n_height"] > 0
    s_parts = sum(t.detach() for t, _ in parts.values() if t is not None)
    assert torch.allclose(s_parts, loss.detach(), rtol=1e-4, atol=1e-5)
    loss.backward()
    assert float(q.scorer[0].weight.grad.abs().sum()) > 0 and float(q.head.up.weight.grad.abs().sum()) > 0
    v = validate(q, mt, [batch], cfg, 0.05, 0, "cpu", vce_weight=1.0, vce_opts=vce_options(cfg, "pillar"),
                 height_weight=1.0)
    assert v["height_ce"] == v["height_ce"] and v["height_frac"] > 0.5
    # Row A: height_weight 0 -> no height term, nan stats, no depth needed
    parts0 = {}
    b0 = {k: v_ for k, v_ in batch.items() if k != "depth"}
    l0, s0 = step(q, mt, b0, cfg, 0.05, 0, 4, 0.5, certainty_weight=0.01, pose_nll_weight=0.5, parts=parts0)
    assert torch.isfinite(l0) and "height" not in parts0 and s0["height_ce"] != s0["height_ce"]


def test_sub_cell_refinement_refuses_a_bev_query():
    from bevloc.match.satroma import check_refinable, refined_for_query
    q = build_query(_cfg(), "pillar")
    with pytest.raises(ValueError, match="placement_at"):
        check_refinable(q)
    with pytest.raises(ValueError, match="placement_at"):
        refined_for_query(None, None, None, q, {}, 16)
    check_refinable(build_query(_cfg(), "erp_depth"))        # erp_depth / picture queries: fine


def test_vigor_pairs_in_pillar_mode_loads_depth_only_when_asked(tmp_path):
    from bevloc.data.vigor import VigorPairs
    from test_depth_query import _png_mm, _vigor_layout
    root = _vigor_layout(tmp_path, 0.0, 0.0, _png_mm(np.full((448, 896), 10.0, np.float32)))
    cfg = _cfg()
    ds = VigorPairs(root, cfg, cities=["Chicago"], split="samearea", train=False)
    assert not ds.depth and "depth" not in ds[0] and (ds.erp_w, ds.erp_h) == (896, 448)
    cfg.pillar.erp_size = [448, 224]
    ds2 = VigorPairs(root, cfg, cities=["Chicago"], split="samearea", train=False, depth=True)
    assert ds2.depth and (ds2.erp_w, ds2.erp_h) == (448, 224) and ds2[0]["depth"].shape == (1, 224, 448)


# ---- (f) a Sat-RoMa decoder from a local export directory (the DINOv3-LVD row) -------------------------------

def _fake_dinov3(monkeypatch, seen):
    """sat-roma-infer's own offline stub: a stride-16 conv in timm's features_only shape; records the model type."""
    import sat_roma.encoders as enc

    class Inner(torch.nn.Module):
        num_prefix_tokens = 1

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(3, 1024, 16, 16)

        def forward_intermediates(self, x, indices=None, norm=False, output_fmt="NCHW", intermediates_only=True,
                                  attn_mask=None, is_causal=False):
            return [self.proj(x)]

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.out_indices, self.norm, self.output_fmt = Inner(), (-1,), False, "NCHW"

        def forward(self, x):
            return self.model.forward_intermediates(x)

    def build(model_type, compile=True, label="Sat Encoder", freeze=True):
        seen.append(model_type)
        return Net().eval(), 1024, enc._DINOV3_MODEL_NAMES.get(model_type, model_type)

    monkeypatch.setattr(enc, "_build_frozen_dinov3", build)


def test_load_matcher_builds_from_an_export_directory(tmp_path, monkeypatch):
    import json
    from safetensors.torch import save_file
    from bevloc.match.satroma import SatRoMaMatcher, load_matcher
    from sat_roma.build import build_sat_roma
    seen = []
    _fake_dinov3(monkeypatch, seen)
    cfg = dict(name="lvdtest", encoder_type="satdinov3", pretrain_model="vit_large_patch16_dinov3_lvd",
               resolution="medium", h_resized=224, w_resized=224, map_to_img_ratio=4, im_b_size=896, proj_dim=512,
               gp_solver="lu", gp_sigma_noise=0.1, sat_normalize=True, cls_to_coord_res=56, decoder_scales=["16"])
    src = build_sat_roma(pretrain_model=cfg["pretrain_model"], compile_backbone=False)
    with torch.no_grad():
        for p in src.decoder.parameters():
            p.add_(0.01 * torch.randn_like(p))
    dec = {k: v.contiguous() for k, v in src.state_dict().items() if not k.startswith("encoder.")}
    d = tmp_path / "satroma_lvdtest"
    d.mkdir()
    save_file(dec, d / "sat_roma_lvdtest_decoder.safetensors")
    (d / "sat_roma_lvdtest_config.json").write_text(json.dumps(cfg))
    seen.clear()
    m = load_matcher(str(d), device="cpu")
    assert isinstance(m, SatRoMaMatcher) and m.name == "lvdtest" and (m.im_a_size, m.im_b_size) == (224, 896)
    assert seen == ["vit_large_patch16_dinov3_lvd"]                     # the JSON's encoder, not the default
    got = m.model.state_dict()
    assert all(torch.equal(got[k], v) for k, v in dec.items())
    # a plain name goes to the hub resolver as before
    monkeypatch.setattr(SatRoMaMatcher, "from_pretrained", classmethod(lambda cls, name, device=None: ("hub", name)))
    assert load_matcher("0t1q66hy", device="cpu") == ("hub", "0t1q66hy")
    with pytest.raises(FileNotFoundError):
        load_matcher(str(tmp_path / "missing_dir"), device="cpu")
    # the training-side wrapper takes the same path
    from bevloc.model.coarse import FeatureQueryMatcher
    fq = FeatureQueryMatcher(str(d), "cpu")
    assert fq.wrapper.name == "lvdtest"


# ---- (g) the selected-height map (FG² Fig. 4 style overlay) -------------------------------------------------

def test_selected_height_is_the_argmax_slice_per_valid_cell():
    q = build_query(_cfg(head=False), "pillar")
    b = _batch()
    q(b, _FakeMatcher)
    z_idx, z_m = q.selected_height()
    assert z_idx.shape == (1, 35, 35) and z_m.shape == (1, 35, 35)
    assert int(z_idx[0, 17, 17]) == -1 and z_m[0, 17, 17] != z_m[0, 17, 17]      # the blind cell: no selection
    ok = z_idx >= 0
    assert bool(ok[0, 12, 17]) and int(ok.sum()) == 35 * 35 - 1
    zs = torch.tensor(q.z)
    assert torch.allclose(z_m[ok], zs[z_idx[ok]])                                   # metres above ground of the slice
