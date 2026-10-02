"""Per-frame statistics of the Sat-RoMa vote map (task 06, item 2): the inputs of a calibrated confidence.

The vote map of a frame is the per-token softmax over the K x K reference cells (the decoder's `gm_cls`), summed over
the valid query tokens and normalised to one: the soft version of what the GMM modes and the RANSAC vote for (the
unweighted sibling of `bevloc.model.coarse.vote_heatmap`, which averages log-probabilities for the pose NLL). Nothing
here runs a model; `bevloc.match.satroma` calls these on the logits it already has.

Two maps. The *token* vote map (`vote_map`, keys vote_*) is where the tokens land: for a correct match it is a blob the
size of the query footprint around the pose, for the layout-prior failure a diffuse blob centred on the tile. The *ego*
vote map (`ego_vote_map`, keys ego_*) is the Hough version: each token's categorical shifted by that token's own
offset from the vehicle (known orientation: query and reference north-aligned at the same GSD, as in VIGOR and every
query of this repo), i.e. "where would the vehicle be if this token sat at that cell"; consistent tokens pile up at
one cell, a wrong-but-diffuse or self-contradicting vote map does not.

Units: cells are reference classifier cells (16 reference px); poses are reference px. Conversion to metres happens
in the evaluator, which knows the ground sampling distance.
"""
from __future__ import annotations

import numpy as np

# keys of a frame row written by scripts/eval_vigor.py (`stats_row`), in this order; HYP_KEYS only with --hyp K > 0
STAT_KEYS = ("vote_entropy", "vote_support_cells", "vote_top1", "vote_mass_2cells", "n_modes", "n_inlier_modes",
             "peak_means_m", "pose_centre_m", "cert_mean", "cert_median", "n_valid_tokens", "placed_depth_m",
             "vote_offtile_mass",
             "ego_entropy", "ego_support_cells", "ego_top1", "ego_mass_2cells", "ego_peak_pose_m")
HYP_KEYS = ("pose_hyp_med_m", "spread_hyp_m", "n_hyp_ok")


def token_valid(gm, valid=None):
    """(h, w) bool numpy: the tokens that vote. valid None = every token whose logit column is not identically zero
    (the evaluators zero the logits of masked tokens, `consensus_for_query`)."""
    import torch
    if valid is not None:
        return torch.as_tensor(valid).detach().bool().cpu().numpy()
    return (gm.detach() != 0).any(dim=0).cpu().numpy()


def vote_map(gm, valid=None):
    """gm (K*K, h, w) logits; valid (h, w) bool or None (see `token_valid`).

    Returns ((K, K) float64 probabilities summing to 1, number of valid tokens); (None, 0) without a valid token."""
    import torch
    v = token_valid(gm, valid)
    n = int(v.sum())
    if n == 0:
        return None, 0
    k = int(round(gm.shape[0] ** 0.5))
    p = torch.softmax(gm.detach().float(), dim=0).cpu().numpy()[:, v].astype(np.float64).sum(1)
    return (p / p.sum()).reshape(k, k), n


def token_centres(h, w, query_size):
    """(h, w, 2) query px (x, y) of the token centres of an h x w grid over a query_size picture."""
    sx, sy = float(query_size) / w, float(query_size) / h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.stack([(xx + 0.5) * sx - 0.5, (yy + 0.5) * sy - 0.5], -1)


def ego_vote_map(gm, q_xy, valid=None, query_size=224, ref_px=896, min_p=1e-7):
    """Hough map of the vehicle position: sum over valid tokens t of p_t(c + d_t), d_t = (query point of t - vehicle)
    in reference cells (1 query px = 1 reference px), bilinearly splatted, normalised to 1 over the mass that lands on
    the K x K grid. gm (K*K, h, w) logits; q_xy (h, w, 2) query px of each token (picture: `token_centres`; placed
    queries: the placed point); the vehicle is the query centre ((query_size - 1) / 2, as in `pose_px`).
    Returns (K, K) float64 or None (no valid token / nothing lands)."""
    import torch
    v = token_valid(gm, valid)
    if not v.any():
        return None
    k = int(round(gm.shape[0] ** 0.5))
    s = float(ref_px) / k
    prob = torch.softmax(gm.detach().float(), dim=0).cpu().numpy()[:, v].T.astype(np.float64)    # (T, K*K)
    off = (np.asarray(q_xy, np.float64)[v] - (float(query_size) - 1.0) / 2.0) / s                  # (T, 2) cells
    t_idx, c_idx = np.nonzero(prob > min_p)
    w0 = prob[t_idx, c_idx]
    X = (c_idx % k) - off[t_idx, 0]
    Y = (c_idx // k) - off[t_idx, 1]
    x0, y0 = np.floor(X), np.floor(Y)
    fx, fy = X - x0, Y - y0
    acc = np.zeros(k * k)
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = (x0 + dx).astype(int), (y0 + dy).astype(int)
        ok = (xi >= 0) & (xi < k) & (yi >= 0) & (yi < k)
        acc += np.bincount((yi * k + xi)[ok], weights=(w0 * wt)[ok], minlength=k * k)
    tot = acc.sum()
    return None if tot <= 0 else (acc / tot).reshape(k, k)


def ref_cell_valid(ref, k, min_frac=0.5):
    """(k, k) bool: reference cells whose pixels are non-black for at least min_frac of the cell (the VIGOR canvas is
    black outside the tile). ref (3, S, S) tensor/array, S divisible by k."""
    import torch
    r = torch.as_tensor(ref).detach().float()
    if r.ndim == 4:
        r = r[0]
    nz = (r.abs().sum(0) > 0).float()[None, None]
    s = nz.shape[-1] // k
    return (torch.nn.functional.avg_pool2d(nz, s)[0, 0] >= min_frac).cpu().numpy()


def px_to_cell(xy_px, ref_px, k):
    """Reference px -> continuous cell coordinates (x = column, y = row), cell-centre convention: cell c spans px
    [c * s, (c + 1) * s) with s = ref_px / k, its centre at c."""
    s = float(ref_px) / float(k)
    return (np.asarray(xy_px, np.float64) + 0.5) / s - 0.5


def vote_stats(p, ref_valid=None, pose_cell=None, radius=2.0):
    """Shape statistics of one vote map p (K, K), probabilities.

    ref_valid (K, K) bool or None (= all cells): the map is restricted to the valid cells and renormalised.
    vote_entropy       Shannon entropy / log(number of valid cells): 0 = one cell, 1 = uniform over the tile.
    vote_support_cells exp(entropy in nats): the number of equally likely cells with that entropy (1 .. n valid).
    vote_top1          mass of the most-voted cell.
    vote_mass_2cells   mass of the cells whose centres lie within `radius` cells of pose_cell ((x, y) cell coords,
                       `px_to_cell`; radius = cfg.certainty.stats_radius_cells in the evaluator); None without a pose.
    vote_offtile_mass  mass on the invalid cells (the black canvas) BEFORE the restriction and renormalisation, as a
                       fraction of the map's total; 0 when every cell is valid."""
    p = np.asarray(p, np.float64)
    k = p.shape[0]
    rv = np.ones_like(p, bool) if ref_valid is None else np.asarray(ref_valid, bool)
    q = np.where(rv, p, 0.0)
    tot = q.sum()
    n_valid = int(rv.sum())
    out = dict(vote_entropy=None, vote_support_cells=None, vote_top1=None, vote_mass_2cells=None,
               vote_offtile_mass=None)
    if p.sum() > 0:
        out["vote_offtile_mass"] = float(max(0.0, 1.0 - tot / p.sum()))
    if tot <= 0 or n_valid == 0:
        return out
    q = q / tot
    nz = q[q > 0]
    h = float(-(nz * np.log(nz)).sum())
    out["vote_entropy"] = float(h / np.log(n_valid)) if n_valid > 1 else 0.0
    out["vote_support_cells"] = float(np.exp(h))
    out["vote_top1"] = float(q.max())
    if pose_cell is not None:
        yy, xx = np.mgrid[0:k, 0:k]
        d = np.hypot(xx - float(pose_cell[0]), yy - float(pose_cell[1]))
        out["vote_mass_2cells"] = float(q[d <= float(radius) + 1e-9].sum())
    return out


def apply_h(H, xy):
    """(3, 3) homography applied to (N, 2) points."""
    p = np.c_[np.asarray(xy, np.float64).reshape(-1, 2), np.ones(len(np.asarray(xy).reshape(-1, 2)))] @ np.asarray(H, np.float64).T
    return p[:, :2] / p[:, 2:3]


def pose_px(H, query_size):
    """The vehicle (query centre, the convention of `bevloc.eval.metrics.pose_errors`) in reference px: (2,)."""
    c = (float(query_size) - 1.0) / 2.0
    return apply_h(H, [[c, c]])[0]


def medoid(points):
    """points (N, 2) -> (index of the medoid = the point with the least summed distance to the others,
    median distance of the OTHER N - 1 points from it (the medoid's own zero excluded); None when N < 2)."""
    P = np.asarray(points, np.float64).reshape(-1, 2)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    i = int(np.argmin(D.sum(1)))
    return i, (float(np.median(np.delete(D[i], i))) if len(P) >= 2 else None)


def stats_row(peak, means_pose_px=None, gsd_m=1.0):
    """The frame-row keys STAT_KEYS from the `Match.stats` of the peak row, lengths converted from reference /
    virtual-BEV px to metres with gsd_m (both grids share it). means_pose_px: the means row's pose in reference px
    (`pose_px` of its H; None = no pose) for the agreement of the two poses; the means row gets no statistics of its
    own (they would only cost time)."""
    s = peak or {}
    g = float(gsd_m)

    def metres(v):
        return None if v is None else float(v) * g
    pm = None
    if s.get("pose_px") is not None and means_pose_px is not None:
        pm = float(np.linalg.norm(np.subtract(s["pose_px"], means_pose_px))) * g
    return dict(vote_entropy=s.get("vote_entropy"), vote_support_cells=s.get("vote_support_cells"),
                vote_top1=s.get("vote_top1"), vote_mass_2cells=s.get("vote_mass_2cells"),
                n_modes=s.get("n_modes"), n_inlier_modes=s.get("n_inlier_modes"), peak_means_m=pm,
                pose_centre_m=metres(s.get("pose_centre_px")), cert_mean=s.get("cert_mean"),
                cert_median=s.get("cert_median"), n_valid_tokens=s.get("n_valid_tokens"),
                placed_depth_m=metres(s.get("placed_range_px")),
                ego_entropy=s.get("ego_entropy"), ego_support_cells=s.get("ego_support_cells"),
                ego_top1=s.get("ego_top1"), ego_mass_2cells=s.get("ego_mass_2cells"),
                ego_peak_pose_m=metres(s.get("ego_peak_pose_px")), vote_offtile_mass=s.get("vote_offtile_mass"))
