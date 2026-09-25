"""Sub-cell refinement with the Sat-RoMa decoder's own conv refiner (task 04, sub-cell stage).

What the decoder actually has (sat_roma/build.py, sat_roma/matcher.py `Decoder.forward`): ONE refinement stage,
``conv_refiner["16"]``, at the same stride-16 token grid as the coarse classifier. There are no stride-8/4/2/1
refiners and no fine features: ``decoder_scales = ["16"]`` and the DINOv3 encoder emits only ``{16: ...}``. At
stride 16 the decoder does, per query token,

    f_q, f_r   = proj["16"](tokens)                               (the same projected features the GP/Transformer see)
    gm_cls     -> W_in = cls_to_flow_refine(gm_cls)               ("ToWarp": 5-neighbour soft-argmax around the argmax,
                                                                  normalised reference coords, align_corners=False)
    delta, dc  = conv_refiner["16"](f_q, f_r, W_in, scale_factor)  (grid_sample of f_r at W_in, a displacement
                                                                  embedding of W_in - own coords, and a 15x15 local
                                                                  correlation of f_q with f_r around W_in)
    W          = W_in + 16 * (delta_x / (4 w), delta_y / (4 h))    (refine_init = 4; h, w = query token grid)
    certainty  = gm_certainty + dc

and exposes ``flow_pre_delta`` (= W_in), ``delta_flow``, ``flow`` (= W) and ``certainty`` under
``exposed_intermediates``. `RefinerTap` records the refiner's inputs/outputs through module hooks (nothing in the
package is edited), so the refiner can be re-run on an injected warp (``--refine-init ransac``) and, in training,
fed detached inputs (RoMa: gradients are cut between the coarse and fine stages).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

REFINER_KEY = "conv_refiner"


# ---- coordinates -----------------------------------------------------------------------------------------------

def px_to_norm(p, size):
    """Pixel coordinates (pixel k centred at k) -> normalised [-1, 1], align_corners=False."""
    return (p + 0.5) * (2.0 / size) - 1.0


def norm_to_px(x, size):
    return (x + 1.0) * (size / 2.0) - 0.5


def refiner_displacement(delta_flow, h, w, stride=16, refine_init=4):
    """The decoder's residual: delta (B, 2, h, w) -> displacement in normalised reference coordinates."""
    return stride * torch.stack((delta_flow[:, 0].float() / (refine_init * w),
                                 delta_flow[:, 1].float() / (refine_init * h)), dim=1)


def token_centres_px(h, w, patch=16, device=None, dtype=torch.float32):
    """(h, w, 2) query-pixel (u, v) of every token centre, the convention of `coarse.coarse_targets`."""
    v = (torch.arange(h, device=device, dtype=dtype) + 0.5) * patch - 0.5
    u = (torch.arange(w, device=device, dtype=dtype) + 0.5) * patch - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([uu, vv], -1)


def apply_h(H, xy):
    """H (3, 3) or (B, 3, 3) applied to the tensor xy (..., 2) or (B, ..., 2), in xy's dtype."""
    H = torch.as_tensor(H).to(device=xy.device, dtype=xy.dtype)
    p = torch.cat([xy, torch.ones_like(xy[..., :1])], -1)
    if H.ndim == 2:
        q = p @ H.T
    else:
        q = (p.reshape(H.shape[0], -1, 3) @ H.transpose(1, 2)).reshape(p.shape)
    return q[..., :2] / q[..., 2:3]


# ---- driving the frozen (or trained) refiner ---------------------------------------------------------------------

class RefinerTap:
    """Context manager: hooks on ``decoder.conv_refiner[scale]`` that record its inputs and outputs.

    detach_inputs=True replaces the refiner's (f_q, f_r, W_in) by detached copies before it runs, so a fine loss
    trains only the refiner (RoMa §3.3: the refiners take the detached coarse warp; the coarse and fine features
    are decoupled). The decoder's own ``flow`` output then still carries the non-detached W_in (it is
    ``W_in + displacement``): use `refined_warp` for anything trained.
    """

    def __init__(self, decoder, scale="16", detach_inputs=False):
        self.decoder, self.scale, self.detach = decoder, str(scale), bool(detach_inputs)
        self.refiner = decoder.conv_refiner[self.scale]
        self.x = self.y = self.warp_in = self.delta_flow = self.delta_certainty = None
        self.scale_factor = 1.0
        self._handles = []

    def _pre(self, module, args, kwargs):
        x, y, warp = args[:3]
        if self.detach:
            x, y, warp = x.detach(), y.detach(), warp.detach()
            args = (x, y, warp) + tuple(args[3:])
        self.x, self.y, self.warp_in = x, y, warp
        self.scale_factor = kwargs.get("scale_factor", 1.0)
        return args, kwargs

    def _post(self, module, args, output):
        self.delta_flow, self.delta_certainty = output

    def __enter__(self):
        self._handles = [self.refiner.register_forward_pre_hook(self._pre, with_kwargs=True),
                         self.refiner.register_forward_hook(self._post)]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def refined_warp(self):
        """W_in + displacement from the recorded refiner call (normalised reference coords, (B, 2, h, w))."""
        h, w = self.x.shape[-2:]
        return self.warp_in.float() + refiner_displacement(self.delta_flow, h, w, int(self.scale),
                                                           self.decoder.refine_init)

    def rerun(self, warp):
        """Run the refiner again on the recorded features with another input warp (B, 2, h, w), e.g. the
        RANSAC-consistent coarse warp. Returns (refined warp, delta certainty)."""
        warp = warp.to(self.warp_in.dtype)
        delta, dc = self.refiner(self.x, self.y, warp, scale_factor=self.scale_factor)
        h, w = self.x.shape[-2:]
        return warp.float() + refiner_displacement(delta, h, w, int(self.scale), self.decoder.refine_init), dc


def set_refiner_trainable(decoder, on):
    """requires_grad of every conv-refiner parameter; returns how many parameters that is."""
    n = 0
    for name, p in decoder.named_parameters():
        if REFINER_KEY in name:
            p.requires_grad = bool(on)
            n += p.numel()
    return n


def decoder_state(decoder, include_refiner=False):
    """The checkpoint's decoder dict: the refiner is saved only when it was trained (refine_weight > 0)."""
    return {k: v for k, v in decoder.state_dict().items() if include_refiner or REFINER_KEY not in k}


# ---- RoMa's fine loss (2305.15404v2 §3.4, Eq. 16-18), written from the equations ----------------------------------

def charbonnier(x, s, alpha=0.5):
    """Generalised Charbonnier with scale s: s^a((x/s)^2 + 1)^(a/2) - s^a  (= (x^2 + s^2)^(a/2) - s^a).

    RoMa Eq. 16 is -log p = (||mu - x||^2 + s^2)^(1/4) (alpha = 0.5); the constant s^a is subtracted so the loss is
    zero at the ground truth (gradients unchanged)."""
    return s ** alpha * (((x / s) ** 2 + 1.0) ** (alpha / 2.0) - 1.0)


def fine_certainty_target(warp_in, gt, ok, cell_norm, cells=0.5):
    """1 where the ground-truth match lies within `cells` reference cells (Chebyshev) of the warp the refiner
    started from, i.e. inside the refiner's cell, and the token is supervised (`ok`); 0 elsewhere.
    warp_in, gt: (B, 2, h, w) normalised; ok (B, h, w) bool; cell_norm = one cell in normalised units."""
    d = (warp_in - gt).abs().amax(1)
    return ok & (d <= float(cells) * cell_norm + 1e-9)


def fine_loss(warp, cert_logit, warp_in, gt, ok, tok_valid, cell_norm, stride=16, alpha=0.5, c=1e-4,
              cert_cells=0.5, certainty_weight=0.01):
    """RoMa's fine-stage objective at one stride.

    warp (B, 2, h, w): refined warp (normalised reference coords; W_in detached, see `RefinerTap`);
    cert_logit (B, h, w): refined certainty logit; warp_in: the refiner's input warp; gt: ground-truth warp;
    ok (B, h, w): tokens whose ground truth is usable (valid, lands inside the reference on real content);
    tok_valid (B, h, w): valid query tokens (the certainty BCE is taken over these only).
    Regression: generalised Charbonnier (alpha, s = c * stride, RoMa's code units: normalised coordinates, c = 1e-4)
    of the end-point error over `ok`. Certainty: BCE toward `fine_certainty_target`. Returns (loss, stats)."""
    epe = (warp.float() - gt.float()).norm(dim=1)                      # (B, h, w)
    s = float(c) * float(stride)
    if bool(ok.any()):
        reg = charbonnier(epe[ok], s, alpha).mean()
        with torch.no_grad():
            epe_in = (warp_in.float() - gt.float()).norm(dim=1)[ok]
            st_epe, st_epe_in = float(epe[ok].mean()), float(epe_in.mean())
            st_med, st_med_in = float(epe[ok].median()), float(epe_in.median())
    else:
        reg = warp.sum() * 0.0
        st_epe = st_epe_in = st_med = st_med_in = float("nan")
    tgt = fine_certainty_target(warp_in.float(), gt.float(), ok, cell_norm, cert_cells)
    if bool(tok_valid.any()) and certainty_weight:
        bce = F.binary_cross_entropy_with_logits(cert_logit.float()[tok_valid], tgt.float()[tok_valid])
    else:
        bce = warp.sum() * 0.0
    loss = reg + float(certainty_weight) * bce
    return loss, dict(fine_reg=float(reg.detach()), fine_cert=float(bce.detach()), fine_epe=st_epe,
                      fine_epe_in=st_epe_in, fine_epe_med=st_med, fine_epe_in_med=st_med_in,
                      fine_cert_pos=float(tgt[tok_valid].float().mean()) if bool(tok_valid.any()) else float("nan"),
                      fine_n=int(ok.sum()))


def gt_warp(H, query_xy, ref_size):
    """Ground-truth warp (B, 2, h, w), normalised reference coords, of query points query_xy (B, h, w, 2) in query
    pixels (token centres for the picture modes, placed points for erp / erp_depth) under H (B, 3, 3)."""
    p = apply_h(H.to(query_xy.dtype), query_xy)                        # (B, h, w, 2) reference px
    return px_to_norm(p, ref_size).permute(0, 3, 1, 2)


# ---- dense warp -> correspondences (evaluation) -------------------------------------------------------------------

def _sample(field, xn, yn):
    """field (C, h, w); xn, yn (N,) normalised -> (N, C), bilinear, border padding, align_corners=False."""
    g = torch.stack([xn, yn], -1).to(field.dtype)[None, None]         # (1, 1, N, 2)
    return F.grid_sample(field[None], g, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T


def warp_samples(flow, cert, img_hw, stride, patch=16):
    """Sample the token-grid warp at a query-pixel grid of stride `stride`.

    flow (2, h, w) normalised reference coords; cert (h, w) logits; the query image is img_hw = (H, W) pixels with
    h = H / patch token rows. Grid pixel centres u_k = (k + 0.5) * stride - 0.5 (pixel k centred at k); at
    stride == patch these are exactly the token centres and the values are read, not interpolated. Grid points
    outside the hull of the token centres (a strip of patch/2 px at the border) are dropped: the warp is not
    defined there, and border padding would pair them with the edge token's match. Returns uv (N, 2) query
    pixels, idx (N, 2) long integer pixel (col, row) for validity lookups, warp (N, 2) normalised, cert (N,)."""
    H, W = int(img_hw[0]), int(img_hw[1])
    h, w = flow.shape[-2:]
    dev = flow.device
    s = int(stride)
    ny, nx = H // s, W // s
    ky, kx = torch.arange(ny, device=dev), torch.arange(nx, device=dev)
    vv, uu = torch.meshgrid((ky + 0.5) * s - 0.5, (kx + 0.5) * s - 0.5, indexing="ij")
    uv = torch.stack([uu, vv], -1).reshape(-1, 2).float()
    col = ((kx.float() + 0.5) * s).floor().long().clamp(0, W - 1)
    row = ((ky.float() + 0.5) * s).floor().long().clamp(0, H - 1)
    rr, cc = torch.meshgrid(row, col, indexing="ij")
    idx = torch.stack([cc, rr], -1).reshape(-1, 2)
    if s == int(patch) and (h, w) == (H // s, W // s):
        wv = flow.reshape(2, -1).T.float()
        cv = cert.reshape(-1).float()
    else:
        lo_u, hi_u = W / w / 2 - 0.5, W - W / w / 2 - 0.5                  # first / last token centre, pixels
        lo_v, hi_v = H / h / 2 - 0.5, H - H / h / 2 - 0.5
        keep = (uv[:, 0] >= lo_u - 1e-6) & (uv[:, 0] <= hi_u + 1e-6) & (uv[:, 1] >= lo_v - 1e-6) & (uv[:, 1] <= hi_v + 1e-6)
        uv, idx = uv[keep], idx[keep]
        xn, yn = px_to_norm(uv[:, 0], W), px_to_norm(uv[:, 1], H)
        wv = _sample(flow.float(), xn, yn)
        cv = _sample(cert.float()[None], xn, yn)[:, 0]
    return uv, idx, wv, cv
