"""Sat-RoMa coarse objective for a query that arrives as features, not pixels.

`gm_cls` is (B, K*K, h, w): for each of the h x w query patches, logits over the K x K
reference cells (K = 56 for an 896 px reference, 16 px per cell), class = row * K + col.
The layout is verified against the released checkpoint in scripts/train_fusion.py --check-targets.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def coarse_targets(H, patch_valid, ref_valid=None, query_size=224, ref_size=896, patch=16, cells=56,
                   query_xy=None):
    """GT reference cell of every query patch centre.

    H: (B, 3, 3) query px -> reference px. patch_valid: (B, h, w) bool.
    ref_valid: (B, cells, cells) bool, True where the reference itself has real content at that
    cell (see `ref_cell_validity`). A reference mosaic can be black/no-data in patches (a hole
    between tiles, a river with no LiDAR return, ...) without the WHOLE frame being unusable --
    only the query patches whose match target lands on that hole need excluding, not every patch
    in the frame. Whole-frame filtering (bevloc.data.pairs.fully_covered_names) is still worth a
    lenient pre-filter (skip frames with ~no overlap at all, save the compute), but should not be
    the strict, all-or-nothing gate: it would throw away a mostly-good frame for one bad corner.
    Returns class index (B, h, w) long and the mask of patches that are valid, land inside, AND
    (if ref_valid given) target real reference content.
    """
    B, h, w = patch_valid.shape
    if query_xy is not None:
        # placed query points (B, h, w, 2) in query pixels, e.g. ERP tokens placed on the virtual BEV
        uv = query_xy.to(H.dtype).reshape(B, -1, 2)
        p = torch.cat([uv, torch.ones_like(uv[..., :1])], -1) @ H.transpose(1, 2)
    else:
        cy = (torch.arange(h, device=H.device, dtype=H.dtype) + 0.5) * patch - 0.5    # patch-centre pixels
        cx = (torch.arange(w, device=H.device, dtype=H.dtype) + 0.5) * patch - 0.5
        v, u = torch.meshgrid(cy, cx, indexing="ij")
        p = torch.stack([u, v, torch.ones_like(u)], -1).reshape(1, -1, 3) @ H.transpose(1, 2)
    xy = p[..., :2] / p[..., 2:3]
    s = ref_size / cells
    col, row = torch.floor((xy[..., 0] + 0.5) / s).long(), torch.floor((xy[..., 1] + 0.5) / s).long()
    inside = (col >= 0) & (col < cells) & (row >= 0) & (row < cells)
    rowc, colc = row.clamp(0, cells - 1), col.clamp(0, cells - 1)
    idx = (rowc * cells + colc).reshape(B, h, w)
    use = inside.reshape(B, h, w) & patch_valid
    if ref_valid is not None:
        tgt_valid = ref_valid[torch.arange(B, device=H.device)[:, None], rowc, colc].reshape(B, h, w)
        use = use & tgt_valid
    return idx, use


def ref_cell_validity(ref, cells=56, min_frac=0.5):
    """Per-reference-cell content mask: (B, cells, cells) bool, True where at least `min_frac`
    of that cell's pixels are non-black. `ref`: (B, 3, ref_size, ref_size) in [0, 1]; a cell is
    ref_size / cells px (16 for the standard 896 px / 56 cells)."""
    nonblack = (ref.sum(1, keepdim=True) > 0).float()
    frac = F.avg_pool2d(nonblack, ref.shape[-1] // cells)
    return (frac[:, 0] >= min_frac)


def window_logits(logits, tgt, radius):
    """(N, K*K) logits with everything outside a Chebyshev radius of `tgt` set to -1e4.

    The GT pose defines the window, so an argmax of the result is not a localization
    metric. Use it as a training objective only; score the unmasked logits.
    """
    k = int(round(logits.shape[-1] ** 0.5))
    cells = torch.arange(k, device=logits.device)
    rr, cc = torch.meshgrid(cells, cells, indexing="ij")
    gy, gx = rr.reshape(-1), cc.reshape(-1)
    far = (gy[None] - (tgt // k)[:, None]).abs().gt(radius) | (gx[None] - (tgt % k)[:, None]).abs().gt(radius)
    return logits.masked_fill(far, -1e4)


def neighbour_hinge(logits, tgt, radius, margin=1.0):
    """Push the true cell's logit `margin` above every other cell inside `radius`.

    RoMa's R2 does this for the 8 neighbours, and only once the mode is already local.
    A wider radius is the same hinge while the mode is still several cells away.
    Self-hits from clipping at the border are dropped (they have zero gradient).
    """
    k = int(round(logits.shape[-1] ** 0.5))
    own = logits.gather(1, tgt[:, None])
    rows, cols = tgt // k, tgt % k
    terms = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            nb = (rows + di).clamp(0, k - 1) * k + (cols + dj).clamp(0, k - 1)
            gap = own - logits.gather(1, nb[:, None])
            h = F.relu(margin - gap)
            terms.append(torch.where(nb[:, None] == tgt[:, None], torch.zeros_like(h), h))
    return torch.cat(terms, 1).mean()


def vote_heatmap(gm_cls, gm_certainty=None, matchable=None):
    """Certainty-weighted soft vote of all query patches: (B, K, K) log-probabilities over the
    reference grid. This is the quantity RANSAC ultimately votes for and the observation a
    particle filter reads (bevloc.track)."""
    B, K2, h, w = gm_cls.shape
    k = int(round(K2 ** 0.5))
    if k * k != K2:
        raise ValueError(f"gm_cls channels {K2} not a square")
    log_p = F.log_softmax(gm_cls.float().permute(0, 2, 3, 1), dim=-1)      # (B, h, w, K2)
    if gm_certainty is not None:
        c = gm_certainty[:, 0] if gm_certainty.ndim == 4 else gm_certainty
        weight = torch.sigmoid(c.float())
    else:
        weight = gm_cls.new_ones(B, h, w, dtype=torch.float32)
    if matchable is not None:
        weight = weight * matchable.float()
    wsum = weight.reshape(B, -1).sum(-1).clamp_min(1e-6)
    heat = (log_p * weight[..., None]).reshape(B, -1, K2).sum(1) / wsum[:, None]
    return F.log_softmax(heat, dim=-1).reshape(B, k, k)


def pose_heatmap_nll(gm_cls, H, matchable=None, gm_certainty=None, local_radius=6,
                     query_size=224, patch=16, ref_size=896):
    """OrienterNet-style NLL of the GT vehicle cell under a soft pose heatmap.

    Sat-RoMa's ``gm_cls`` is per-patch over a fixed K×K reference grid (K=56). We
    certainty-weight the per-patch softmaxes into one (B, K, K) heatmap (``vote_heatmap``),
    optionally window it around the GT centre cell, and take −log p(GT). This trains the
    quantity RANSAC ultimately votes for, not each patch in isolation.

    Returns (nll scalar, stats dict). Empty / no-match batches return 0 loss.
    """
    B, K2, h, w = gm_cls.shape
    k = int(round(K2 ** 0.5))
    if k * k != K2:
        raise ValueError(f"gm_cls channels {K2} not a square")
    if matchable is not None and not bool(matchable.any()):
        z = gm_cls.sum() * 0.0
        return z, dict(pose_nll=0.0, pose_err=float("nan"), n=0)
    heat = vote_heatmap(gm_cls, gm_certainty, matchable).reshape(B, K2)   # log-probs, constant shift vs raw votes
    # GT vehicle centre (query image centre) → reference cell
    centre = torch.tensor([query_size / 2 - 0.5, query_size / 2 - 0.5, 1.0],
                          device=H.device, dtype=H.dtype)
    p = (H.float() @ centre)  # (B, 3)
    xy = p[:, :2] / p[:, 2:3]
    s = float(ref_size) / k
    col = torch.floor((xy[:, 0] + 0.5) / s).long()
    row = torch.floor((xy[:, 1] + 0.5) / s).long()
    inside = (col >= 0) & (col < k) & (row >= 0) & (row < k)
    if not bool(inside.any()):
        z = gm_cls.sum() * 0.0
        return z, dict(pose_nll=0.0, pose_err=float("nan"), n=0)
    tgt = (row.clamp(0, k - 1) * k + col.clamp(0, k - 1))
    pred = window_logits(heat, tgt, local_radius) if local_radius else heat
    # Only supervise frames whose GT centre lands on the crop
    nll_all = F.cross_entropy(pred, tgt, reduction="none")
    nll = nll_all[inside].mean()
    with torch.no_grad():
        am = heat.argmax(1)
        err = torch.hypot((am // k - tgt // k).float(), (am % k - tgt % k).float())
        pose_err = float(err[inside].mean()) if bool(inside.any()) else float("nan")
        n = int(inside.sum())
    return nll, dict(pose_nll=float(nll.detach()), pose_err=pose_err, n=n)


def roma_coarse_loss(gm_cls, idx, matchable, gm_certainty=None, certainty_weight=0.01,
                     local_radius=0, neighbour_radius=0, neighbour_weight=0.1, patch_weight=None):
    """Sat-RoMa scale-16 objective: classify the reference cell, and say if the patch is matchable.

    RoMa's coarse loss is a categorical over reference cells, taken only where the
    correspondence is real (`prob > 0.99` in their code), plus a binary certainty
    head trained on that same mask. It is not a cosine between descriptors.
    `certainty_weight` is their `ce_weight` (default 0.01): classification dominates.
    `matchable`: (B, h, w) bool. `gm_certainty`: (B, 1, h, w) or (B, h, w) logits.
    """
    logits = gm_cls.permute(0, 2, 3, 1)[matchable].float()
    tgt = idx[matchable]
    hinge = gm_cls.new_zeros(())
    if tgt.numel() == 0:
        ce = gm_cls.sum() * 0.0
        acc = cell_err = top5 = float("nan")
        n = 0
    else:
        pred = window_logits(logits, tgt, local_radius) if local_radius else logits
        if patch_weight is None:
            ce = F.cross_entropy(pred, tgt)
        else:
            w = patch_weight[matchable].float()
            per = F.cross_entropy(pred, tgt, reduction="none")
            ce = (per * w).sum() / w.sum().clamp(min=1.0)
        if neighbour_radius:
            hinge = neighbour_hinge(logits, tgt, neighbour_radius)
        with torch.no_grad():
            k = int(round(gm_cls.shape[1] ** 0.5))
            am = logits.argmax(1)
            acc = float((am == tgt).float().mean())
            cell_err = float(torch.hypot((am // k - tgt // k).float(), (am % k - tgt % k).float()).mean())
            top5 = float((logits.topk(min(5, logits.shape[-1]), dim=-1).indices == tgt[:, None]).any(-1).float().mean())
            n = int(tgt.numel())
    cert = gm_cls.new_zeros(())
    if gm_certainty is not None and certainty_weight:
        c = gm_certainty[:, 0] if gm_certainty.ndim == 4 else gm_certainty
        cert = F.binary_cross_entropy_with_logits(c.float(), matchable.float())
    loss = ce + certainty_weight * cert + neighbour_weight * hinge
    return loss, dict(
        ce=float(ce.detach()), cert=float(cert.detach()), acc=acc, cell_err=cell_err, top5=top5, n=n)


def coarse_loss(gm_cls, idx, use):
    """Cross-entropy over reference cells, averaged over usable patches. Also returns top-1 accuracy
    and the mean cell distance of the argmax (1 cell = 16 ref px = 4 m at 0.25 m/px)."""
    B, K2, h, w = gm_cls.shape
    logits = gm_cls.permute(0, 2, 3, 1)[use].float()
    tgt = idx[use]
    if tgt.numel() == 0:
        z = gm_cls.sum() * 0.0
        return z, dict(acc=float("nan"), cell_err=float("nan"), n=0)
    loss = F.cross_entropy(logits, tgt)
    with torch.no_grad():
        k = int(round(K2 ** 0.5))
        am = logits.argmax(1)
        d = torch.hypot((am // k - tgt // k).float(), (am % k - tgt % k).float())
    return loss, dict(acc=float((am == tgt).float().mean()), cell_err=float(d.mean()), n=int(tgt.numel()))


class FeatureQueryMatcher(nn.Module):
    """Sat-RoMa with the query branch fed by features: reference -> frozen DINOv3 ViT-L,
    query features -> `f_q_pyramid[16]`, then the released decoder (frozen or fine-tuned)."""

    def __init__(self, checkpoint, device, train_decoder=True):
        super().__init__()
        from bevloc.match.satroma import SatRoMaMatcher
        m = SatRoMaMatcher.from_pretrained(checkpoint, device=device)
        self.wrapper, self.model = m, m.model
        self.model.decoder.expose_intermediates = True     # gm_cls in eval mode too (BatchNorm stays frozen)
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.decoder.parameters():
            p.requires_grad = bool(train_decoder)
        self.train_decoder = bool(train_decoder)

    def train(self, mode=True):
        """The whole matcher stays in eval mode even when the decoder is fine-tuned: with batch 2
        its BatchNorm must keep the released running statistics, or train and val disagree."""
        super().train(mode)
        self.model.eval()
        return self

    def reference_features(self, ref):
        with torch.no_grad():
            return self.model.encoder(ref)

    def forward(self, f_q, ref, scale_factor=0.4):
        """f_q: (B, 1024, 14, 14); ref: (B, 3, 896, 896) in [0, 1]. Returns gm_cls (B, 3136, 14, 14).
        scale_factor = sqrt(224 * 224) / 560, as in bevloc.match.satroma."""
        out = self.model.decoder({16: f_q}, self.reference_features(ref), scale_factor=scale_factor)
        return out[16]["gm_cls"]

    def image_query_features(self, query):
        """The ordinary RGB query path, for checks: (B, 3, 224, 224) -> (B, 1024, 14, 14)."""
        with torch.no_grad():
            return self.model.encoder(query)[16]


def to_tensor(img_uint8, device):
    return torch.from_numpy(np.ascontiguousarray(img_uint8)).to(device).permute(2, 0, 1).float().div(255.0)[None]


# ---- Loc² VCE pose loss on a differentiable weighted Procrustes (task 04, Step 2) ------------------------------
# Written from Loc² (2509.09792v3) Eq. 2-6; nothing copied from third_party/Loc2 (AGPL-3.0).


def procrustes_2d(A, B, w, eps=1e-8):
    """Weighted rigid 2-D fit B ≈ R·A + t with the scale fixed to 1 (metric depth, shared GSD).

    A, B: (Bt, N, 2); w: (Bt, N) >= 0. Closed form of the 2-D Procrustes (no SVD, smooth gradients):
    with weighted centroids removed, cos θ ∝ Σ w a·b and sin θ ∝ Σ w a×b. Returns R (Bt, 2, 2), t (Bt, 2).
    Degenerate input (both sums ~0: coincident points or zero weights) gives the identity rotation, not zeros."""
    wn = w / w.sum(1, keepdim=True).clamp_min(eps)
    ma = (wn[..., None] * A).sum(1)
    mb = (wn[..., None] * B).sum(1)
    ac, bc = A - ma[:, None], B - mb[:, None]
    s = (wn * (ac[..., 0] * bc[..., 1] - ac[..., 1] * bc[..., 0])).sum(1)
    c = (wn * (ac * bc).sum(-1)).sum(1)
    r = torch.sqrt(s * s + c * c + eps * eps)
    degenerate = (s.abs() <= eps) & (c.abs() <= eps)
    cos = torch.where(degenerate, torch.ones_like(c), c / r)
    sin = torch.where(degenerate, torch.zeros_like(s), s / r)
    R = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
    t = mb - (R @ ma[..., None])[..., 0]
    return R, t


def virtual_points(n, cell_m, grid_m=5.0, points=10, device=None):
    """(points², 2) virtual BEV px on a grid_m square centred on the camera (Loc²: 10 x 10 over 5 m)."""
    o = (n - 1) / 2.0
    ax = torch.linspace(-grid_m / 2, grid_m / 2, int(points), device=device) / float(cell_m)
    gu, gv = torch.meshgrid(ax, ax, indexing="ij")
    return torch.stack([gu.reshape(-1) + o, gv.reshape(-1) + o], -1)


def vce_distance(R, t, H_gt, X, cell_m):
    """Loc² Eq. 6: mean Euclidean distance (metres) between the virtual points X (P, 2) BEV px mapped by the
    estimate (R, t) and by the ground truth H_gt (Bt, 3, 3) BEV px -> reference px. Returns (Bt,)."""
    Xh = torch.cat([X, torch.ones_like(X[:, :1])], -1)                       # (P, 3)
    g = Xh[None] @ H_gt.float().transpose(1, 2)                              # (Bt, P, 3)
    g = g[..., :2] / g[..., 2:3]
    p = X[None] @ R.transpose(1, 2) + t[:, None]
    return (p - g).norm(dim=-1).mean(-1) * float(cell_m)


def cell_centres(cells, ref_size, device=None):
    """(cells², 2) reference px (x, y) of every class centre, class = row * cells + col (coarse_targets' binning)."""
    s = float(ref_size) / cells
    c = (torch.arange(cells, device=device, dtype=torch.float32) + 0.5) * s - 0.5
    rr, cc = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([cc.reshape(-1), rr.reshape(-1)], -1)


def vce_pose_loss(gm_cls, query_xy, token_valid, H_gt, cell_m, n, ref_size=896, gm_certainty=None,
                  mode="sample", n_samples=1024, grid_m=5.0, points=10, use_certainty=True, generator=None,
                  min_tokens=3, ref_valid=None, min_weight=1e-8):
    """Loc²'s VCE pose loss with the Sat-RoMa decoder as the matcher.

    gm_cls (B, K², h, w) logits; query_xy (B, h, w, 2) placed query points (virtual BEV px); token_valid (B, h, w);
    H_gt (B, 3, 3) BEV px -> reference px. Reference and BEV share the GSD cell_m, so the pose is rigid (scale 1).

    mode "sample": n_samples (token, cell) pairs drawn in two stages, with replacement: a token from p(token) ∝
    valid·sigmoid(certainty), then a cell from that token's p(cell | token). Each pair is weighted in the Procrustes by
    its probability p(token)·p(cell | token), so the gradient reaches the logits (and certainty) through the weights.
    The pairs have the joint's distribution, but this is NOT Loc²'s code, which draws one multinomial over the flat
    (aerial × ground) matching matrix (and torch caps a multinomial at 2^24 categories: 1568 × 3136 is close for
    larger grids). mode "expect": one pair per valid token, its reference point = the expected cell centre under
    p(cell | token), weight = p(token).
    ref_valid (B, K, K) bool: reference cells with content (`ref_cell_validity`, the mask the coarse CE uses);
    the logits of the others are -inf before the softmax, so no pair or expectation lands on the black canvas.
    Samples are left out (like the coarse CE's unusable patches) when they have < min_tokens valid tokens, token
    weights summing to <= min_weight, or no valid reference cell.
    Returns (loss scalar = mean VCE in metres over the samples kept, stats)."""
    B, K2, h, w = gm_cls.shape
    k = int(round(K2 ** 0.5))
    dev = gm_cls.device
    N = h * w
    logits = gm_cls.float().reshape(B, K2, N).transpose(1, 2)                           # (B, N, K2)
    ok = token_valid.reshape(B, N).sum(1) >= min_tokens
    if ref_valid is not None:
        rv = ref_valid.reshape(B, K2).bool()
        has = rv.any(1)
        ok = ok & has
        rv = rv | ~has[:, None]                              # excluded samples: unmasked, only to stay finite
        logits = logits.masked_fill(~rv[:, None, :], float("-inf"))
    logp = F.log_softmax(logits, dim=-1)
    tw = token_valid.reshape(B, N).float()
    if use_certainty and gm_certainty is not None:
        c = gm_certainty[:, 0] if gm_certainty.ndim == 4 else gm_certainty
        tw = tw * torch.sigmoid(c.float().reshape(B, N))
    ok = ok & (tw.sum(1) > min_weight)
    if not bool(ok.any()):
        z = gm_cls.sum() * 0.0
        return z, dict(vce_m=float("nan"), vce_pose_m=float("nan"), n_vce=0)
    tw = torch.where(ok[:, None], tw, torch.ones_like(tw))                             # keep rows samplable
    ptok = tw / tw.sum(1, keepdim=True).clamp_min(1e-12)
    xy = query_xy.reshape(B, N, 2).float()
    centres = cell_centres(k, ref_size, dev)                                            # (K2, 2)
    if mode == "sample":
        # two-stage draw = a draw from the joint, without a (N·K²)-way multinomial (torch caps categories at 2^24)
        S = int(n_samples)
        tok = torch.multinomial(ptok.detach(), S, replacement=True, generator=generator)         # (B, S)
        lp_tok = torch.gather(logp, 1, tok[..., None].expand(-1, -1, K2))                        # (B, S, K2)
        cls = torch.multinomial(lp_tok.detach().exp().reshape(B * S, K2), 1, generator=generator).reshape(B, S)
        A = torch.gather(xy, 1, tok[..., None].expand(-1, -1, 2))
        Bp = centres[cls]
        wts = torch.gather(ptok, 1, tok) * torch.gather(lp_tok, 2, cls[..., None])[..., 0].exp()  # joint prob of the pair
    elif mode == "expect":
        A = xy
        Bp = logp.exp() @ centres                                                       # (B, N, 2); exp(-inf) = 0
        wts = ptok
    else:
        raise ValueError(f"vce mode must be 'sample' or 'expect', got {mode!r}")
    R, t = procrustes_2d(A, Bp, wts)
    X = virtual_points(n, cell_m, grid_m, points, dev)
    vce = vce_distance(R, t, H_gt, X, cell_m)
    loss = vce[ok].mean()
    with torch.no_grad():
        o = torch.full((1, 2), (n - 1) / 2.0, device=dev)
        pose = vce_distance(R, t, H_gt, o, cell_m)
    return loss, dict(vce_m=float(loss.detach()), vce_pose_m=float(pose[ok].mean()), n_vce=int(ok.sum()))


def resolve_vce_weight(cfg, mode):
    """cfg.train.vce_weight: a number, or "auto"/missing = 1.0 for the erp_depth query and 0 otherwise."""
    v = getattr(getattr(cfg, "train", None), "vce_weight", "auto")
    if v is None or (isinstance(v, str) and v.lower() == "auto"):
        return 1.0 if mode == "erp_depth" else 0.0
    return float(v)


def vce_options(cfg):
    """VCE keyword arguments from cfg.erp_depth (defaults = Loc²)."""
    E = getattr(cfg, "erp_depth", None)
    g = (lambda k, d: getattr(E, k, d) if E is not None else d)
    return dict(mode=str(g("vce_mode", "sample")), n_samples=int(g("vce_samples", 1024)),
                grid_m=float(g("vce_grid_m", 5.0)), points=int(g("vce_points", 10)),
                use_certainty=bool(g("vce_use_certainty", True)))
