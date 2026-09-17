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


class SatRoMa:
    def __init__(self, checkpoint="4adxis71", device=None, use_means=False,
                 reproj_cells=3.0, seed=0, min_valid_frac=1.0):
        """use_means=False is the published configuration: RANSAC targets are the
        per-patch peak (identical for all modes of a patch). use_means=True feeds
        the distinct per-mode means, i.e. genuine competing hypotheses."""
        self.m = SatRoMaMatcher.from_pretrained(checkpoint, device=device)
        self.use_means = use_means
        self.reproj = float(reproj_cells)
        self.seed = seed
        # fraction of a 16 px query patch that must be valid for the patch to vote.
        # Package default 1.0 leaves a median of 10/196 patches for the ipm_cl query.
        self.min_valid_frac = float(min_valid_frac)

    @classmethod
    def from_config(cls, cfg, **override):
        m = cfg.matcher
        kw = dict(checkpoint=m.checkpoint, use_means=m.use_means, reproj_cells=m.reproj_cells,
                  seed=m.seed, min_valid_frac=m.min_valid_frac)
        kw.update(override)
        return cls(**kw)

    def match(self, query, reference, mask=None) -> Match:
        import torch
        m = self.m
        a = m.load_image(query, m.im_a_size).unsqueeze(0).to(m.device)
        b = m.load_image(reference, m.im_b_size).unsqueeze(0).to(m.device)
        sf = float(((a.shape[-2] * a.shape[-1]) / (560.0 ** 2)) ** 0.5)
        with m.model.exposed_intermediates(), torch.no_grad():
            out = m.model.forward({"im_A": a, "im_B": b}, batched=False, scale_factor=sf)
        gm = out[16]["gm_cls"][0].clone()
        if mask is not None:
            gm[:, ~m._query_valid_patches(mask, min_valid_frac=self.min_valid_frac)] = 0.0

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
        if n_modes < 4 or np.array_equal(Hf, np.eye(3)) or not np.isfinite(Hf).all():
            return Match(None, None, n_modes, n_patches, n_multi, 0.0)

        tgt = r.means_B if self.use_means else r.peaks_B
        p = np.c_[r.pts_A, np.ones(n_modes)] @ Hf.T
        err = np.linalg.norm(p[:, :2] / p[:, 2:3] - tgt, axis=1)
        inl = float((err <= self.reproj).mean())

        ha, wa = a.shape[-2:]
        hb, wb = b.shape[-2:]
        H = np.asarray(convert_to_pixel_homography(
            Hf, in_patch_dim=int(gm.shape[-1]), out_patch_dim=int(round(gm.shape[0] ** 0.5)),
            crop_res=(int(ha), int(wa)), map_res=(int(hb), int(wb)),
            cell_convention="center"), dtype=np.float64)
        c = np.array([[0, 0, 1], [wa - 1, 0, 1], [wa - 1, ha - 1, 1], [0, ha - 1, 1]], float) @ H.T
        return Match(H, c[:, :2] / c[:, 2:3], n_modes, n_patches, n_multi, inl)
