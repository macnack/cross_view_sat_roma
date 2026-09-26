"""Thin wrapper around the released Sat-RoMa inference package.

Adds what the kick-off report needs and `MatchResult` does not expose:
number of extracted GMM modes, patches with >1 mode, RANSAC inlier ratio.
The package itself is used unchanged (third_party rule: wrap, never edit).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_CANDIDATES = [
    os.environ.get("SATROMA_INFER_DIR", ""),
    str(Path(__file__).resolve().parents[3] / "third_party/sat_roma_infer"),
    str(Path.home() / "Github/sat-roma-infer"),
]
PACKAGE_DIR = next((c for c in _CANDIDATES if c and (Path(c) / "sat_roma/api.py").exists()), None)
if PACKAGE_DIR is None:
    raise ImportError("sat-roma-infer not found; set SATROMA_INFER_DIR or clone it to ~/Github/sat-roma-infer")
sys.path.insert(0, PACKAGE_DIR)

from sat_roma import SatRoMaMatcher  # noqa: E402
from sat_roma.ransac import estimate_homography  # noqa: E402
from sat_roma.ransac.correspondence import find_gaussians  # noqa: E402
from sat_roma.ransac.ransac_init import ransac_init  # noqa: E402
from sat_roma.ransac.transforms import convert_to_pixel_homography  # noqa: E402


@dataclass
class Match:
    H: np.ndarray | None        # (3,3) query px -> reference px, None = no match
    corners: np.ndarray | None  # (4,2) query corners in reference px
    n_modes: int                # GMM modes extracted over all query patches
    n_patches: int              # query patches that produced >= 1 mode
    n_multimodal: int           # query patches with > 1 mode
    inlier_ratio: float         # modes within the RANSAC threshold of H
    argmax_cells: float | None = None   # mean |argmax - GT cell|, when H_gt was passed
    n_inliers: int = 0                  # modes within the RANSAC threshold of H ("kept")
    # vote-map statistics of the frame (task 06; `attach_stats`), in cells / reference px; None = not computed
    stats: dict | None = field(default=None, compare=False)
    # the correspondences the consensus saw: (placed (N, 2), tgt (N, 2), in_dim, out_dim) grid units, for
    # `hypothesis_spread`; None when there were none
    modes: tuple | None = field(default=None, repr=False, compare=False)


class SatRoMa:
    def __init__(self, checkpoint="0t1q66hy", device=None, use_means=False,
                 reproj_cells=3.0, seed=0, min_valid_frac=1.0, solver="srt"):
        """use_means=False is the published configuration: RANSAC targets are the
        per-patch peak (identical for all modes of a patch). use_means=True feeds
        the distinct per-mode means, i.e. genuine competing hypotheses.
        solver: "srt" = the package's 4-DoF similarity RANSAC + weighted refinement (published);
        "se2" = fixed-scale 2-point SE(2) RANSAC on the same hypothesis set (kick-off H7)."""
        self.m = SatRoMaMatcher.from_pretrained(checkpoint, device=device)
        self.use_means = use_means
        # NOTE (2026-09-23): with the published refine=False configuration the package's model="sRT"
        # never reaches a solver: pipeline.estimate_homography returns the cv2.findHomography RANSAC
        # init, i.e. an 8-DoF homography. "srt" here therefore means "package default = homography";
        # "sim" is a 4-DoF similarity RANSAC (cv2.estimateAffinePartial2D via ransac_init) and "se2"
        # the 3-DoF fixed-scale solver of bevloc.match.se2 (kick-off H7).
        if solver not in ("srt", "sim", "se2"):
            raise ValueError(f"solver must be 'srt', 'sim' or 'se2', got {solver!r}")
        self.solver = solver
        self.reproj = float(reproj_cells)
        self.seed = seed
        # fraction of a 16 px query patch that must be valid for the patch to vote.
        # Package default 1.0 leaves a median of 10/196 patches for the ipm_cl query.
        self.min_valid_frac = float(min_valid_frac)

    @classmethod
    def from_wrapper(cls, wrapper, cfg, use_means=False, min_valid_frac=None):
        """A consensus-only instance sharing an existing SatRoMaMatcher (no second model copy).

        Evaluation scripts decode once with their FeatureQueryMatcher and run `_ransac` /
        `consensus_from_gm` here for the "peak" and "means" rows on the same logits."""
        self = cls.__new__(cls)
        m = cfg.matcher
        self.m = wrapper
        self.use_means = bool(use_means)
        self.reproj = float(m.reproj_cells)
        self.seed = int(m.seed)
        self.solver = str(getattr(m, "solver", "srt"))
        self.min_valid_frac = float(m.min_valid_frac if min_valid_frac is None else min_valid_frac)
        return self

    @classmethod
    def from_config(cls, cfg, **override):
        m = cfg.matcher
        kw = dict(checkpoint=m.checkpoint, use_means=m.use_means, reproj_cells=m.reproj_cells,
                  seed=m.seed, min_valid_frac=m.min_valid_frac, solver=getattr(m, "solver", "srt"))
        kw.update(override)
        return cls(**kw)

    def encode(self, image, size):
        """Scale-16 encoder tokens, (1, C, H/16, W/16). The backbone normalizes internally."""
        import torch
        m = self.m
        t = m.load_image(image, size).unsqueeze(0).to(m.device)
        with torch.no_grad():
            return m.model.encoder(t)[16]

    def match_encoded(self, f_q, f_s, scale_factor, mask=None, H_gt=None) -> Match:
        """Decoder + RANSAC on tokens already produced by `encode`. One reference
        encode can be reused across query variants."""
        import torch
        m = self.m
        with m.model.exposed_intermediates(), torch.no_grad():
            out = m.model.decoder({16: f_q}, {16: f_s}, scale_factor=float(scale_factor))
        gm = out[16]["gm_cls"][0].clone()
        if mask is not None:
            gm[:, ~self.query_patches(mask)] = 0.0
        cell_err = self._argmax_cells(gm, mask, H_gt) if H_gt is not None else None
        return self._ransac(gm, cell_err)

    def query_patches(self, mask):
        """(h, w) bool: a query patch votes when at least min_valid_frac of its pixels are valid.

        Implemented here so a validity mask does not depend on a private method of
        whichever sat-roma-infer checkout is on the path.
        """
        import torch
        import torch.nn.functional as F
        m = self.m
        patch, size = int(m.model.encoder.patch_size), int(m.im_a_size)
        arr = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
        valid = arr != 0 if arr.ndim == 2 else np.any(arr != 0, axis=-1 if arr.shape[-1] <= 4 else 0)
        t = torch.from_numpy(np.ascontiguousarray(valid, dtype=np.float32))[None, None]
        if t.shape[-2:] != (size, size):
            t = F.interpolate(t, size=(size, size), mode="nearest")
        frac = F.avg_pool2d(t, kernel_size=patch, stride=patch)
        return (frac >= self.min_valid_frac - 1e-6)[0, 0].to(m.device)

    def match(self, query, reference, mask=None, H_gt=None) -> Match:
        import torch
        m = self.m
        a = m.load_image(query, m.im_a_size).unsqueeze(0).to(m.device)
        b = m.load_image(reference, m.im_b_size).unsqueeze(0).to(m.device)
        sf = float(((a.shape[-2] * a.shape[-1]) / (560.0 ** 2)) ** 0.5)
        with torch.no_grad():
            f_q, f_s = m.model.encoder(a)[16], m.model.encoder(b)[16]
        return self.match_encoded(f_q, f_s, sf, mask, H_gt)

    # ---- ERP-token query: placement after matching (task 03, Task 6) -------------------------------

    def match_placed(self, f_q, f_s, xy, valid, scale_factor, H_gt=None) -> Match:
        """Decoder on an ERP token map, then consensus on the tokens' placed ground points.

        xy (h, w, 2): each token's placed point in virtual query pixels (224 px BEV); valid (h, w) bool.
        Returns the same Match as `match_encoded`: H maps virtual query px -> reference px."""
        import torch
        m = self.m
        with m.model.exposed_intermediates(), torch.no_grad():
            out = m.model.decoder({16: f_q}, {16: f_s}, scale_factor=float(scale_factor))
        gm = out[16]["gm_cls"][0].clone()
        return SatRoMa.consensus_from_gm(self, gm, xy, valid, H_gt=H_gt)

    @staticmethod
    def consensus_from_gm(self, gm, xy, valid, H_gt=None, certainty=None, ref_valid=None, stats=True) -> Match:
        """gm (K*K, h, w) logits; xy (h, w, 2) placed query px; valid (h, w).
        certainty (h, w) decoder certainty logits and ref_valid (K, K) valid reference cells, both optional, only
        feed the vote-map statistics (`attach_stats`; stats=False skips them, Match.stats None); the pose does not
        depend on any of the three.

        Static with an explicit ``self`` so a test can pass a stub carrying only
        m.im_a_size / m.im_b_size, use_means, reproj, seed and solver."""
        import torch
        valid_t = torch.as_tensor(valid, dtype=torch.bool, device=gm.device)
        gm = gm.clone()
        gm[:, ~valid_t] = 0.0                                          # invalid tokens contribute no mode
        xy_np = torch.as_tensor(xy).detach().float().cpu().numpy()
        m = SatRoMa._consensus_placed(self, gm, xy_np)
        if not stats:
            return m
        vt_np = valid_t.cpu().numpy()
        c = (float(self.m.im_a_size) - 1.0) / 2.0                      # the ego in virtual BEV px
        extra = dict(placed_range_px=float(np.linalg.norm(xy_np[vt_np] - c, axis=-1).mean()) if vt_np.any() else None)
        return SatRoMa.attach_stats(self, m, gm, vt_np, certainty=certainty, ref_valid=ref_valid, extra=extra,
                                    q_xy=xy_np)

    @staticmethod
    def _consensus_placed(self, gm, xy_np) -> Match:
        pts_A, means_B, peaks_B, covs_B = find_gaussians(
            gm.detach().float().cpu(), adaptive_gauss_fit=False, log_missing_gaussians=False,
            fixed_threshold=0.008, fixed_window_size=4)
        n_modes = int(pts_A.shape[0])
        argmax_cells = None
        if n_modes == 0:
            return Match(None, None, 0, 0, 0, 0.0, argmax_cells)
        cols, rows = pts_A[:, 0].astype(int), pts_A[:, 1].astype(int)
        placed_px = xy_np[rows, cols]                                  # (N, 2) virtual query pixels
        s = float(self.m.im_a_size) / 14.0                             # 16 px per virtual patch
        placed = ((placed_px - (s / 2 - 0.5)) / s).astype(np.float64)  # cell-centre convention of convert_to_pixel_homography
        _, counts = np.unique(np.round(pts_A, 3), axis=0, return_counts=True)
        n_patches, n_multi = int(len(counts)), int((counts > 1).sum())
        tgt = np.asarray(means_B if self.use_means else peaks_B, np.float64)
        out_dim = int(round(gm.shape[0] ** 0.5))
        m = SatRoMa._fit_grid(self, placed, tgt, 14, out_dim, n_patches, n_multi)
        m.modes = (placed, tgt, 14, out_dim)
        return m

    # ---- vote-map statistics and hypothesis spread (task 06) ------------------------------------------------------

    @staticmethod
    def attach_stats(self, m, gm, valid=None, certainty=None, ref_valid=None, extra=None, q_xy=None) -> Match:
        """Set m.stats from the logits the consensus already had (no forward pass): vote-map shape
        (`bevloc.match.vote_stats.vote_stats`, mass around this match's pose), modes seen / kept, the pose in reference
        px and its distance from the reference centre, and the certainty logits over the valid tokens. gm (K*K, h, w)
        with the masked tokens' logits zeroed; valid (h, w) bool or None (= the non-zero columns); q_xy (h, w, 2) query
        px of each token for the ego (Hough) map: None = the token centres of a picture query, placed queries pass
        their placed points. The pose-mass radius is `self.stats_radius` (cells; the evaluator sets it from
        cfg.certainty.stats_radius_cells), 2 when the instance has none."""
        import torch
        from bevloc.match import vote_stats as V
        vt = V.token_valid(gm, valid)
        p, n_tok = V.vote_map(gm, vt)
        k = int(round(gm.shape[0] ** 0.5))
        ref_px = int(self.m.im_b_size)
        pose = None if m.H is None else V.pose_px(m.H, int(self.m.im_a_size))
        radius = float(getattr(self, "stats_radius", 2.0))
        pose_cell = None if pose is None else V.px_to_cell(pose, ref_px, k)
        if p is None:
            st = dict(vote_entropy=None, vote_support_cells=None, vote_top1=None, vote_mass_2cells=None,
                      vote_offtile_mass=None)
        else:
            st = V.vote_stats(p, ref_valid, pose_cell, radius)
        st.update(n_modes=int(m.n_modes), n_inlier_modes=int(m.n_inliers), n_valid_tokens=n_tok,
                  pose_px=None if pose is None else [float(pose[0]), float(pose[1])],
                  pose_centre_px=None if pose is None else float(np.linalg.norm(pose - (ref_px - 1) / 2.0)),
                  cert_mean=None, cert_median=None, placed_range_px=None)
        qa = int(self.m.im_a_size)
        pe = None if p is None else V.ego_vote_map(
            gm, V.token_centres(gm.shape[-2], gm.shape[-1], qa) if q_xy is None else q_xy, vt, qa, ref_px)
        es = V.vote_stats(pe if pe is not None else np.zeros((k, k)), ref_valid, pose_cell, radius)
        st.update({f"ego_{key[5:]}": val for key, val in es.items()})           # vote_x -> ego_x
        st["ego_peak_pose_px"] = None
        if pe is not None and pose is not None:
            r, c = np.unravel_index(int(np.argmax(pe)), pe.shape)
            s_ref = ref_px / k
            st["ego_peak_pose_px"] = float(np.hypot((c + 0.5) * s_ref - 0.5 - pose[0], (r + 0.5) * s_ref - 0.5 - pose[1]))
        if certainty is not None and n_tok:
            c = torch.as_tensor(certainty).detach().float().cpu().numpy().reshape(vt.shape)[vt].astype(np.float64)
            st.update(cert_mean=float(c.mean()), cert_median=float(np.median(c)))
        st.update(extra or {})
        m.stats = st
        return m

    @staticmethod
    def hypothesis_spread(self, m, k, seed=0):
        """K RANSAC runs (this instance's solver / threshold / seed) on bootstrap resamples (with replacement, same
        size) of the correspondences m's consensus saw. Returns dict(H = the medoid hypothesis (query px -> reference
        px) or None, spread_px = median distance of the other successful poses from their medoid (reference px; the
        medoid's own zero distance excluded; None when fewer than 2 hypotheses produced a model, since one pose has
        no spread), n_ok = hypotheses that produced a model), or None when k <= 0 or m had no modes.
        For the package path (`_ransac`) the "srt" resamples go through the cv2 homography RANSAC of `_fit_grid`
        (the package's own init is the same cv2.findHomography call)."""
        from bevloc.match import vote_stats as V
        if k <= 0 or m.modes is None:
            return None
        placed, tgt, in_dim, out_dim = m.modes
        n = len(placed)
        rng = np.random.default_rng(seed)
        Hs = []
        for _ in range(int(k)):
            idx = rng.integers(0, n, n)
            h = SatRoMa._fit_grid(self, placed[idx], tgt[idx], in_dim, out_dim, 0, 0).H
            if h is not None:
                Hs.append(h)
        if not Hs:
            return dict(H=None, spread_px=None, n_ok=0)
        i, spread = V.medoid([V.pose_px(h, int(self.m.im_a_size)) for h in Hs])
        return dict(H=Hs[i], spread_px=spread, n_ok=len(Hs))

    @staticmethod
    def _fit_grid(self, placed, tgt, in_dim, out_dim, n_patches, n_multi, argmax_cells=None) -> Match:
        """Consensus of grid-unit correspondences placed (N, 2) query patches -> tgt (N, 2) reference cells
        (cell-centre convention), with this instance's solver / threshold / seed; H converted to query px ->
        reference px. The tail of `consensus_from_gm`, shared with `refined_consensus`."""
        n_modes = int(placed.shape[0])
        cv2.setRNGSeed(self.seed)
        if self.solver == "se2":
            from bevloc.match.se2 import se2_ransac
            if n_modes < 2:
                return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells)
            Hf, _ = se2_ransac(placed, tgt, thresh=self.reproj, n_iter=500, seed=self.seed)
        else:
            if n_modes < 4:
                return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells)
            # "srt" mirrors the package default on grid points (cv2.findHomography); "sim" is 4-DoF
            Hf, _, _ = ransac_init(placed, tgt, method=cv2.RANSAC, reproj_threshold=self.reproj,
                                   max_iters=5000, confidence=0.995, quiet=True,
                                   estimator="homography" if self.solver == "srt" else "similarity")
            Hf = None if Hf is None else np.asarray(Hf, np.float64)
            if Hf is not None and (not np.isfinite(Hf).all() or np.array_equal(Hf, np.eye(3))):
                Hf = None
        if Hf is None:
            return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells)
        p = np.c_[placed, np.ones(n_modes)] @ Hf.T
        err = np.linalg.norm(p[:, :2] / p[:, 2:3] - tgt, axis=1)
        inl = float((err <= self.reproj).mean())
        ha = wa = int(self.m.im_a_size)
        hb = wb = int(self.m.im_b_size)
        H = np.asarray(convert_to_pixel_homography(
            Hf, in_patch_dim=in_dim, out_patch_dim=out_dim,
            crop_res=(ha, wa), map_res=(hb, wb), cell_convention="center"), dtype=np.float64)
        c = np.array([[0, 0, 1], [wa - 1, 0, 1], [wa - 1, ha - 1, 1], [0, ha - 1, 1]], float) @ H.T
        return Match(H, c[:, :2] / c[:, 2:3], n_modes, n_patches, n_multi, inl, argmax_cells,
                     n_inliers=int((err <= self.reproj).sum()))

    @staticmethod
    def refined_consensus(self, q_px, r_px, cells=56, H_seed=None, gate_cells=None, min_corr=None):
        """Consensus on dense (sub-cell) correspondences: q_px (N, 2) query px (the 224 px picture, or virtual BEV
        px of placed ERP pixels) -> r_px (N, 2) reference px. Same solver, threshold (in reference cells) and seed
        as the coarse rows; H is query px -> reference px as there.

        H_seed (3, 3) query px -> reference px, gate_cells: keep only correspondences within gate_cells reference
        cells of H_seed's prediction before the RANSAC ("seeded by the coarse pose": cv2's RANSAC takes no initial
        model, so the seed acts as a guided-matching gate). min_corr: fewer surviving correspondences than this
        -> no model (H None), rather than a 2- or 3-point fit. Returns (Match, n_used)."""
        q_px = np.asarray(q_px, np.float64).reshape(-1, 2)
        r_px = np.asarray(r_px, np.float64).reshape(-1, 2)
        sa = float(self.m.im_a_size) / 14.0
        sb = float(self.m.im_b_size) / float(cells)
        if H_seed is not None and gate_cells is not None and len(q_px):
            p = np.c_[q_px, np.ones(len(q_px))] @ np.asarray(H_seed, np.float64).T
            keep = np.linalg.norm(p[:, :2] / p[:, 2:3] - r_px, axis=1) <= float(gate_cells) * sb
            q_px, r_px = q_px[keep], r_px[keep]
        placed = (q_px - (sa / 2 - 0.5)) / sa                       # grid units, cell-centre convention
        tgt = (r_px - (sb / 2 - 0.5)) / sb
        n = int(len(placed))
        if min_corr is not None and n < int(min_corr):
            return Match(None, None, n, n, 0, 0.0), n
        return SatRoMa._fit_grid(self, placed, tgt, 14, int(cells), n, 0), n

    def _argmax_cells(self, gm, mask, H_gt):
        """Mean coarse-cell error of the per-patch argmax against H_gt (query px -> ref px)."""
        import torch
        from bevloc.model.coarse import coarse_loss, coarse_targets
        dev = gm.device
        if mask is None:
            pv = torch.ones(gm.shape[-2:], dtype=torch.bool, device=dev)
        else:
            pv = self.query_patches(mask)
        H = torch.as_tensor(H_gt, dtype=torch.float32, device=dev)[None]
        idx, use = coarse_targets(H, pv[None])
        _, st = coarse_loss(gm[None].float(), idx, use)
        return None if st["n"] == 0 else st["cell_err"]

    def _ransac(self, gm, argmax_cells, valid=None, certainty=None, ref_valid=None, stats=True) -> Match:
        """Package consensus on (masked) logits; valid / certainty / ref_valid only feed the statistics (stats=False:
        none, Match.stats None)."""
        mt = self._ransac_fit(gm, argmax_cells)
        if not stats:
            return mt
        return SatRoMa.attach_stats(self, mt, gm, valid, certainty=certainty, ref_valid=ref_valid)

    def _ransac_fit(self, gm, argmax_cells) -> Match:
        m = self.m

        cv2.setRNGSeed(self.seed)
        r = estimate_homography(
            gm.detach().float().cpu(), backend="numpy", model="sRT",
            use_means_for_ransac=self.use_means, ransac_method=cv2.RANSAC,
            ransac_reproj_threshold=self.reproj, ransac_max_iters=5000,
            ransac_confidence=0.995, refine=False, return_details=True)

        n_modes = int(r.pts_A.shape[0])
        if n_modes:
            _, counts = np.unique(np.round(r.pts_A, 3), axis=0, return_counts=True)
            n_patches, n_multi = int(len(counts)), int((counts > 1).sum())
        else:
            n_patches = n_multi = 0
        Hf = np.asarray(r.H, dtype=np.float64)
        tgt = r.means_B if self.use_means else r.peaks_B
        modes = (np.asarray(r.pts_A, np.float64), np.asarray(tgt, np.float64), int(gm.shape[-1]),
                 int(round(gm.shape[0] ** 0.5))) if n_modes else None
        if self.solver == "sim":
            if n_modes < 4:
                return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells, modes=modes)
            Hs, _, _ = ransac_init(np.asarray(r.pts_A, np.float64), np.asarray(tgt, np.float64),
                                   method=cv2.RANSAC, reproj_threshold=self.reproj, max_iters=5000,
                                   confidence=0.995, quiet=True, estimator="similarity")
            # ransac_init substitutes np.eye(3) when cv2 finds no model: treat that as a miss, like `srt`
            if Hs is None or not np.isfinite(np.asarray(Hs)).all() or np.array_equal(np.asarray(Hs), np.eye(3)):
                return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells, modes=modes)
            Hf = np.asarray(Hs, np.float64)
        elif self.solver == "se2":
            # Same hypothesis set, but the consensus model has no scale (metric query, shared GSD).
            from bevloc.match.se2 import se2_ransac
            if n_modes < 2:
                return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells, modes=modes)
            H2, _ = se2_ransac(r.pts_A, tgt, thresh=self.reproj, n_iter=500, seed=self.seed)
            if H2 is None:
                return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells, modes=modes)
            Hf = H2
        elif n_modes < 4 or np.array_equal(Hf, np.eye(3)) or not np.isfinite(Hf).all():
            return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells, modes=modes)

        p = np.c_[r.pts_A, np.ones(n_modes)] @ Hf.T
        err = np.linalg.norm(p[:, :2] / p[:, 2:3] - tgt, axis=1)
        inl = float((err <= self.reproj).mean())
        n_inl = int((err <= self.reproj).sum())

        ha = wa = int(m.im_a_size)
        hb = wb = int(m.im_b_size)
        H = np.asarray(convert_to_pixel_homography(
            Hf, in_patch_dim=int(gm.shape[-1]), out_patch_dim=int(round(gm.shape[0] ** 0.5)),
            crop_res=(ha, wa), map_res=(hb, wb),
            cell_convention="center"), dtype=np.float64)
        c = np.array([[0, 0, 1], [wa - 1, 0, 1], [wa - 1, ha - 1, 1], [0, ha - 1, 1]], float) @ H.T
        return Match(H, c[:, :2] / c[:, 2:3], n_modes, n_patches, n_multi, inl, argmax_cells,
                     n_inliers=n_inl, modes=modes)


def consensus_for_query(cons, gm, query, batch, frac, n, min_frac=0.05, certainty=None, ref_valid=None, stats=True):
    """Consensus for one decoded sample, whatever the query mode (evaluators share this).

    Queries with `placement` (erp, erp_depth): the tokens' placed points go through `consensus_from_gm`
    (= `match_placed` without a second decode). BEV queries (lift, ipm, hybrid): the per-patch validity
    `frac` (1, h, w) masks the 14 x 14 patch grid, then the package RANSAC (`_ransac`).
    gm (K*K, h, w) logits of the sample; batch holds that one sample with a leading batch axis.
    certainty (h, w) decoder certainty logits, ref_valid (K, K) valid reference cells: optional, only for the
    statistics in Match.stats (the pose does not depend on them); stats=False skips the statistics (the evaluators
    compute them for the peak row only)."""
    import torch
    import torch.nn.functional as F
    if hasattr(query, "placement"):
        xy, valid = query.placement(batch)
        return SatRoMa.consensus_from_gm(cons, gm, xy[0], valid[0], certainty=certainty, ref_valid=ref_valid,
                                         stats=stats)
    mask = F.interpolate(frac[:, None].float(), size=(n, n), mode="nearest")[0, 0] >= min_frac
    gmm = gm.clone()
    qp = cons.query_patches(mask)
    gmm[:, ~qp] = 0.0
    return cons._ransac(gmm, None, valid=qp, certainty=certainty, ref_valid=ref_valid, stats=stats)


REFINE_INITS = ("none", "coarse", "ransac")


def refined_for_query(cons, o16, tap, query, batch, stride, init="coarse", coarse=None, gate_cells=None,
                      min_cert=0.0, frac=None, min_frac=0.05, min_corr=8):
    """Sub-cell consensus for one decoded sample from the decoder's stride-16 conv refiner (bevloc.model.refine).

    o16: the decoder's scale-16 output dict (``flow``, ``certainty``, ``flow_pre_delta``, ``gm_certainty``,
    ``gm_cls``; batch of one); tap: the `RefinerTap` that was active during that decode (its recorded features let
    the refiner re-run on an injected warp); coarse: this sample's coarse `Match` (the "peak" row), the seed.
    init:
      none   - refiner fed the package's own coarse warp (cls_to_flow_refine of gm_cls, i.e. o16["flow"] as decoded);
               RANSAC from scratch on the refined correspondences.
      coarse - same refined warp; correspondences gated to within gate_cells reference cells of the coarse pose
               before the RANSAC (seeding).
      ransac - the refiner re-run on the RANSAC-consistent coarse warp W_in(token) = H_coarse(token's query point)
               (valid tokens; others keep the package's warp), then gated as `coarse`.
    Correspondences: the refined warp sampled on a query grid of stride `stride` px (bilinear between token
    centres; read exactly at stride 16), kept where the query is valid: picture modes at stride 16 = the token set
    of the coarse rows (`query_patches` of the valid fraction >= min_frac, as in `consensus_for_query`), at
    stride < 16 the pixel's own bev_valid; erp_depth / erp = the pixel's own depth / ground placement (its query
    point is the placed ego point of that ERP pixel in virtual BEV px). If min_cert > 0, also
    sigmoid(refined certainty) >= min_cert. Seeded inits fall back to the coarse match when fewer than min_corr
    correspondences survive the gate or the gated set gives no model. Returns (Match, info)."""
    import torch
    from bevloc.model.refine import apply_h, norm_to_px, px_to_norm, token_centres_px, warp_samples
    if init not in REFINE_INITS:
        raise ValueError(f"refine init must be one of {REFINE_INITS}, got {init!r}")
    ref_size = int(cons.m.im_b_size)
    cells = int(round(o16["gm_cls"].shape[1] ** 0.5))
    placed = hasattr(query, "placement")
    seeded = init in ("coarse", "ransac")
    H_c = None if coarse is None else coarse.H
    info = dict(n_corr=0, n_used=0, fallback=False)
    if seeded and H_c is None:                                      # nothing to seed from: the coarse miss stands
        return Match(None, None, 0, 0, 0, 0.0), info
    flow, cert = o16["flow"][0].float(), o16["certainty"][0, 0].float()
    if init == "ransac":
        h, w = flow.shape[-2:]
        if placed:
            xy, vt = query.placement(batch)
            q, vt = xy[0].double(), vt[0]
        else:
            q = token_centres_px(h, w, device=flow.device, dtype=torch.float64)
            vt = torch.ones(h, w, dtype=torch.bool, device=flow.device)
        w_inj = px_to_norm(apply_h(torch.as_tensor(H_c), q), ref_size).permute(2, 0, 1).float()
        w_inj = torch.where(vt[None], w_inj, o16["flow_pre_delta"][0].float())
        flow_r, dc = tap.rerun(w_inj[None])
        flow, cert = flow_r[0].float(), (o16["gm_certainty"] + dc)[0, 0].float()
    if placed:
        uv, _, wv, cv = warp_samples(flow, cert, batch["erp"].shape[-2:], stride)
        xy, ok = query.placement_at(batch, uv + 0.5)                # continuous ERP coords: pixel k spans [k, k+1)
        q_px, ok = xy[0], ok[0]
    elif int(stride) == 16:                                         # one per token: the coarse rows' token set
        n_px = int(cons.m.im_a_size)
        mask = torch.nn.functional.interpolate(frac[:, None].float(), size=(n_px, n_px), mode="nearest")[0, 0] >= min_frac
        uv, _, wv, cv = warp_samples(flow, cert, (n_px, n_px), stride)
        q_px, ok = uv, cons.query_patches(mask).to(flow.device).reshape(-1)
    else:
        if "bev_valid" in batch:
            valid = batch["bev_valid"][0] != 0
        else:                                                       # lift / hybrid: the patch mask of the coarse rows
            n_px = int(cons.m.im_a_size)
            valid = torch.nn.functional.interpolate(
                (frac[:, None].float() >= min_frac).float(), size=(n_px, n_px), mode="nearest")[0, 0] > 0
        uv, idx, wv, cv = warp_samples(flow, cert, valid.shape[-2:], stride)
        q_px, ok = uv, valid[idx[:, 1], idx[:, 0]]
    if min_cert > 0:
        ok = ok & (torch.sigmoid(cv) >= float(min_cert))
    r_px = norm_to_px(wv, ref_size)
    ok_np = ok.cpu().numpy().astype(bool)
    q_np = q_px.detach().double().cpu().numpy()[ok_np]
    r_np = r_px.detach().double().cpu().numpy()[ok_np]
    info["n_corr"] = int(ok_np.sum())
    m, info["n_used"] = SatRoMa.refined_consensus(cons, q_np, r_np, cells=cells,
                                                  H_seed=H_c if seeded else None,
                                                  gate_cells=gate_cells if seeded else None,
                                                  min_corr=min_corr if seeded else None)
    if seeded and m.H is None:
        info["fallback"] = True
        return coarse, info
    return m, info
