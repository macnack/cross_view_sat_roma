"""Thin wrapper around the released Sat-RoMa inference package.

Adds what the kick-off report needs and `MatchResult` does not expose:
number of extracted GMM modes, patches with >1 mode, RANSAC inlier ratio.
The package itself is used unchanged (third_party rule: wrap, never edit).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
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
        if solver not in ("srt", "se2"):
            raise ValueError(f"solver must be 'srt' or 'se2', got {solver!r}")
        self.solver = solver
        self.reproj = float(reproj_cells)
        self.seed = seed
        # fraction of a 16 px query patch that must be valid for the patch to vote.
        # Package default 1.0 leaves a median of 10/196 patches for the ipm_cl query.
        self.min_valid_frac = float(min_valid_frac)

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

    def _ransac(self, gm, argmax_cells) -> Match:
        import torch
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
        if self.solver == "se2":
            # Same hypothesis set, but the consensus model has no scale (metric query, shared GSD).
            from bevloc.match.se2 import se2_ransac
            if n_modes < 2:
                return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells)
            H2, _ = se2_ransac(r.pts_A, tgt, thresh=self.reproj, n_iter=500, seed=self.seed)
            if H2 is None:
                return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells)
            Hf = H2
        elif n_modes < 4 or np.array_equal(Hf, np.eye(3)) or not np.isfinite(Hf).all():
            return Match(None, None, n_modes, n_patches, n_multi, 0.0, argmax_cells)

        p = np.c_[r.pts_A, np.ones(n_modes)] @ Hf.T
        err = np.linalg.norm(p[:, :2] / p[:, 2:3] - tgt, axis=1)
        inl = float((err <= self.reproj).mean())

        ha = wa = int(m.im_a_size)
        hb = wb = int(m.im_b_size)
        H = np.asarray(convert_to_pixel_homography(
            Hf, in_patch_dim=int(gm.shape[-1]), out_patch_dim=int(round(gm.shape[0] ** 0.5)),
            crop_res=(ha, wa), map_res=(hb, wb),
            cell_convention="center"), dtype=np.float64)
        c = np.array([[0, 0, 1], [wa - 1, 0, 1], [wa - 1, ha - 1, 1], [0, ha - 1, 1]], float) @ H.T
        return Match(H, c[:, :2] / c[:, 2:3], n_modes, n_patches, n_multi, inl, argmax_cells)
