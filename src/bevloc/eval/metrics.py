"""Pose metrics from query->reference pixel homographies."""
from __future__ import annotations

import numpy as np


def _apply(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(H, float).T
    return p[:, :2] / p[:, 2:3]


def pose_errors(H_est, H_gt, query_size: int, gsd_m: float) -> dict:
    """Errors of H_est against H_gt, both query px -> reference px.

    position: displacement of the query centre (= vehicle) in metres.
    corner:   mean displacement of the four query corners in metres.
    yaw:      absolute difference of the in-plane rotation at the centre, degrees.
    """
    s = query_size - 1
    corners = np.array([[0, 0], [s, 0], [s, s], [0, s]], float)
    c = np.array([[s / 2, s / 2]])
    e = np.array([[s / 2 + 1, s / 2]])  # unit step along query +u to read rotation
    out = {}
    out["corner_m"] = float(np.linalg.norm(_apply(H_est, corners) - _apply(H_gt, corners), axis=1).mean() * gsd_m)
    out["position_m"] = float(np.linalg.norm(_apply(H_est, c) - _apply(H_gt, c)) * gsd_m)
    ang = []
    for H in (H_est, H_gt):
        d = (_apply(H, e) - _apply(H, c))[0]
        ang.append(np.arctan2(d[1], d[0]))
    out["yaw_deg"] = float(abs(np.degrees((ang[0] - ang[1] + np.pi) % (2 * np.pi) - np.pi)))
    return out


def mean_removed_cosine(query_tokens, ref_tokens) -> float:
    """Mean cosine after subtracting each side's mean token.

    query_tokens, ref_tokens: (N, C). A shared mean makes plain cosine high
    even when the residual that identifies a cell is unrelated. Rows whose
    residual is ~0 (a constant token) are dropped.
    """
    q = np.asarray(query_tokens, np.float64)
    r = np.asarray(ref_tokens, np.float64)
    q = q - q.mean(0, keepdims=True)
    r = r - r.mean(0, keepdims=True)
    qn = np.linalg.norm(q, axis=1)
    rn = np.linalg.norm(r, axis=1)
    ok = (qn > 1e-8) & (rn > 1e-8)
    if not np.any(ok):
        return float("nan")
    return float(np.mean(np.sum(q[ok] * r[ok], axis=1) / (qn[ok] * rn[ok])))


def recall(position_errors_m, thresholds=(1.0, 5.0, 10.0), n_total=None) -> dict:
    """Recall over ALL pairs: failed matches (None/NaN) count as misses."""
    e = np.asarray([np.inf if (x is None or not np.isfinite(x)) else x for x in position_errors_m])
    n = n_total or len(e)
    return {f"recall@{t:g}m": float((e <= t).sum() / n) for t in thresholds}
