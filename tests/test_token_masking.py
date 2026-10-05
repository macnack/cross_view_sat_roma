"""Masking of tokens without image (the black rows of a partial panorama, the ego car) in the projection head and the query."""
from __future__ import annotations

import types

import torch

from bevloc.model.depth_query import ErpDepthQuery, ProjectionHead
from test_erp_band import _cfg


def _head():
    torch.manual_seed(0)
    h = ProjectionHead(in_dim=32, dim=16, heads=2, n_convs=2, ff_mult=2).eval()
    torch.nn.init.normal_(h.up.weight, std=0.3)                              # a trained head: the zero init would hide everything
    return h


def test_all_valid_mask_equals_no_mask():
    h = _head()
    f = torch.randn(2, 32, 6, 10)
    with torch.no_grad():
        assert torch.allclose(h(f), h(f, torch.ones(2, 6, 10, dtype=torch.bool)), atol=1e-5)


def test_invalid_tokens_do_not_influence_valid_ones():
    h = _head()
    f = torch.randn(2, 32, 6, 10)
    valid = torch.ones(2, 6, 10, dtype=torch.bool)
    valid[:, :3] = False                                                      # the top half carries no image
    g = f.clone()
    g[:, :, :3] = torch.randn(2, 32, 3, 10) * 50.0                            # arbitrary content where nothing is valid
    with torch.no_grad():
        a, b = h(f, valid), h(g, valid)
        assert torch.allclose(a[:, :, 3:], b[:, :, 3:], atol=1e-4)
        assert not torch.allclose(h(f)[:, :, 3:], h(g)[:, :, 3:], atol=1e-3)  # without the mask they do


def test_a_sample_without_any_valid_token_gives_no_nan():
    h = _head()
    with torch.no_grad():
        out = h(torch.randn(1, 32, 6, 10), torch.zeros(1, 6, 10, dtype=torch.bool))
    assert torch.isfinite(out).all()


def test_query_masks_tokens_with_zero_depth_only(monkeypatch):
    cfg = _cfg()
    cfg.erp_depth.head = True
    cfg.erp_depth.head_dim = 16
    cfg.erp_depth.head_heads = 2
    monkeypatch.setenv("BEVLOC_MASK_INVALID_TOKENS", "all")
    q = ErpDepthQuery(cfg).eval()
    assert q.mask_invalid == "all"
    torch.nn.init.normal_(q.head.up.weight, std=0.3)
    B, h, w = 1, 6, 10
    erp = torch.rand(B, 1, 3, h * 16, w * 16)
    depth = torch.full((B, 1, h * 16, w * 16), 12.0)
    depth[:, :, :32] = 0.0                                                    # token rows 0-1: no depth (no image)
    depth[:, :, 32:48] = 80.0                                                 # token row 2: far (sky): still image
    batch = {"erp": erp, "depth": depth, "R_w2c": torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])[None, None]}
    matcher = types.SimpleNamespace(model=types.SimpleNamespace(encoder=lambda x: {16: torch.randn(B, 1024, h, w)}))
    with torch.no_grad():
        f_q, valid = q(batch, matcher)
    assert (f_q[:, :, :2] == 0).all()                                         # "all": zero features where there is no depth
    assert f_q[:, :, 2:].abs().sum() > 0                                      # the far row keeps its features
    assert valid[:, :3].sum() == 0 and valid[:, 3:].sum() == B * 3 * w        # far (> 35 m) and no-depth tokens are not placed
    monkeypatch.setenv("BEVLOC_MASK_INVALID_TOKENS", "bogus")
    try:
        ErpDepthQuery(cfg)
    except ValueError:
        return
    raise AssertionError("an unknown mask mode must raise")
