"""vote_heatmap is a normalised soft vote; pose_heatmap_nll is unchanged by the refactor."""
from __future__ import annotations

import torch

from bevloc.model.coarse import pose_heatmap_nll, vote_heatmap


def _reference_nll(gm_cls, H, matchable, cert, local_radius=6, query_size=224, ref_size=896):
    """The pre-refactor computation (2026-09-23), kept verbatim as the oracle."""
    import torch.nn.functional as F
    from bevloc.model.coarse import window_logits
    B, K2, h, w = gm_cls.shape
    k = int(round(K2 ** 0.5))
    log_p = F.log_softmax(gm_cls.float().permute(0, 2, 3, 1), dim=-1)
    c = cert[:, 0] if cert.ndim == 4 else cert
    weight = torch.sigmoid(c.float()) * matchable.float()
    wsum = weight.reshape(B, -1).sum(-1).clamp_min(1e-6)
    heat = (log_p * weight[..., None]).reshape(B, -1, K2).sum(1) / wsum[:, None]
    centre = torch.tensor([query_size / 2 - 0.5, query_size / 2 - 0.5, 1.0], dtype=H.dtype)
    p = H.float() @ centre
    xy = p[:, :2] / p[:, 2:3]
    s = float(ref_size) / k
    col = torch.floor((xy[:, 0] + 0.5) / s).long()
    row = torch.floor((xy[:, 1] + 0.5) / s).long()
    tgt = row.clamp(0, k - 1) * k + col.clamp(0, k - 1)
    pred = window_logits(heat, tgt, local_radius)
    return F.cross_entropy(pred, tgt), heat.argmax(1)


def test_vote_heatmap_is_a_log_distribution_peaked_at_the_common_vote():
    B, k, h, w = 2, 8, 3, 3
    gm = torch.zeros(B, k * k, h, w)
    gm[:, 27] = 6.0                                   # every patch votes for cell 27 = (row 3, col 3)
    heat = vote_heatmap(gm)
    assert heat.shape == (B, k, k)
    assert torch.allclose(heat.exp().sum(dim=(1, 2)), torch.ones(B), atol=1e-5)
    assert int(heat[0].reshape(-1).argmax()) == 27


def test_pose_heatmap_nll_matches_pre_refactor_oracle():
    torch.manual_seed(0)
    B, k, h, w = 2, 56, 14, 14
    gm = torch.randn(B, k * k, h, w) * 3
    cert = torch.randn(B, 1, h, w)
    matchable = torch.rand(B, h, w) > 0.3
    H = torch.tensor([[[1.0, 0.0, 300.0], [0.0, 1.0, 400.0], [0.0, 0.0, 1.0]]] * B)
    nll, st = pose_heatmap_nll(gm, H, matchable=matchable, gm_certainty=cert, local_radius=6)
    ref_nll, ref_am = _reference_nll(gm, H, matchable, cert)
    assert abs(float(nll) - float(ref_nll)) < 1e-4
    assert st["n"] == B
