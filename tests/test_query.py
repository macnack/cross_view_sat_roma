"""Query factory: every mode returns decoder-shaped tokens."""
from __future__ import annotations

import torch

from bevloc import config as C
from bevloc.model.query import IpmQuery, LiftQuery, build_query, load_query_state


class _FakeMatcher:
    """Stands in for FeatureQueryMatcher: encoder = 1024 channels at stride 16, no weights needed."""

    def __init__(self):
        self.conv = torch.nn.Conv2d(3, 1024, 16, stride=16)

    def image_query_features(self, img):
        with torch.no_grad():
            return self.conv(img)

    class model:  # noqa: N801 — mimics matcher.model.encoder(x)[16]
        @staticmethod
        def encoder(x):
            return {16: torch.nn.functional.avg_pool2d(x.repeat(1, 342, 1, 1)[:, :1024], 16)}


def test_build_query_modes_and_shapes():
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    m = _FakeMatcher()
    lift = build_query(cfg, "lift")
    assert isinstance(lift, LiftQuery)
    batch = dict(erp=torch.rand(1, 1, 3, 448, 896), R_w2c=torch.eye(3)[None, None], se2=torch.zeros(1, 1, 3))
    f_q, frac = lift(batch, m)
    assert f_q.shape == (1, 1024, 14, 14) and frac.shape == (1, 14, 14)
    ipm = build_query(cfg, "ipm")
    assert isinstance(ipm, IpmQuery)
    batch = dict(bev=torch.rand(1, 3, 224, 224), bev_valid=torch.ones(1, 224, 224, dtype=torch.bool))
    f_q, frac = ipm(batch, m)
    assert f_q.shape == (1, 1024, 14, 14) and float(frac.min()) == 1.0


def test_load_query_state_accepts_old_lift_checkpoints():
    cfg = C.load(str(C.REPO / "configs/default.yaml"))
    q = build_query(cfg, "lift")
    old = {"lift": q.lift.state_dict()}
    load_query_state(q, old)
    new = {"query": q.state_dict(), "mode": "lift"}
    load_query_state(q, new)
