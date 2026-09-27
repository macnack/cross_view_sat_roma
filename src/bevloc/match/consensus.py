"""Consensus settings of the Sat-RoMa multi-hypothesis RANSAC, and the mode extraction they act on.

What the package (sat-roma-infer, sat_roma/ransac) fixes and this module exposes as settings (package values in []):
  mode_thr    `find_gaussians(fixed_threshold=)` [0.008]: a mode is a local maximum (cv2.dilate, 4 x 4 kernel) of the
              token's categorical over the K x K reference cells AFTER a 5 x 5 Gaussian blur (cv2.GaussianBlur, sigma
              auto = 1.1) whose blurred height exceeds the threshold. Not a raw probability: a one-hot peak of mass p
              blurs to 0.14 p, so 0.008 needs ~0.06 of the mass concentrated in one cell.
  window      `fixed_window_size` [4]: dilation kernel of the peak test and (half 2) the 5 x 5 moment window of the
              mean. Not swept (it changes the mode set itself, not a filter on it).
  target      `use_means_for_ransac` [False = "peak"]: RANSAC target per mode = the token's global argmax cell (RAW,
              unblurred; identical for every mode of a token) or the mode's own moment mean ("means").
  max_modes   no package parameter [all]: keep the max_modes highest (blurred height) modes per token.
  reproj      `ransac_reproj_threshold` [3.0, in reference cells] -- this repo's `matcher.reproj_cells`.
  ransac      `ransac_max_iters` [5000] / `ransac_confidence` [0.995]; "4x" = max_iters x 4 and confidence
              1 - (1 - 0.995)^4, i.e. the adaptive stopping rule also runs four times as long. se2 (ours): 500 trials
              [x 4]. Weighted sampling (cert="weight") uses fixed trial counts: se2 500, srt/sim 2000 [x 4].
  cert        no package parameter [off]: "weight" = sigmoid(token certainty logit) as RANSAC sampling weight
              (srt: the package's weighted 4-point DLT RANSAC `_ransac_init_weighted`; sim: its weighted 2-point
              similarity `_ransac_init_weighted_srt`; se2: `se2_ransac(weights=)`), "filter" = drop the tokens whose
              certainty is below the 25th percentile of the frame's valid tokens.
  solver      srt (package default: cv2.findHomography, 8 DoF) | sim (4 DoF) | se2 (3 DoF, bevloc.match.se2).

`extract_modes` runs the package's own extraction (softmax_heatmaps + process_patches + assemble_correspondences, i.e.
find_gaussians) once at a loose threshold and records each mode's blurred height, so every stricter threshold is an
exact filter of that list (the peak test's local-maximum part and each mode's mean do not depend on the threshold).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

PACKAGE_MODE_THR = 0.008
PACKAGE_WINDOW = 4
PACKAGE_MAX_ITERS = 5000
PACKAGE_CONFIDENCE = 0.995
SE2_TRIALS = 500
WEIGHTED_TRIALS = 2000
CERT_FILTER_Q = 0.25
RANSAC_MULT = {"default": 1, "4x": 4}
TARGETS = ("peak", "means")
CERTS = ("off", "weight", "filter")
SOLVERS = ("srt", "sim", "se2")


@dataclass(frozen=True)
class ConsensusCfg:
    reproj_cells: float = 3.0
    target: str = "peak"
    max_modes: int = 0            # 0 = all
    mode_thr: float = PACKAGE_MODE_THR
    cert: str = "off"
    solver: str = "srt"
    ransac: str = "default"

    def __post_init__(self):
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {self.target!r}")
        if self.cert not in CERTS:
            raise ValueError(f"cert must be one of {CERTS}, got {self.cert!r}")
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {self.solver!r}")
        if self.ransac not in RANSAC_MULT:
            raise ValueError(f"ransac must be one of {tuple(RANSAC_MULT)}, got {self.ransac!r}")
        if int(self.max_modes) < 0 or float(self.mode_thr) <= 0 or float(self.reproj_cells) <= 0:
            raise ValueError(f"bad consensus settings {self}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown consensus settings {sorted(unknown)}")
        kw = dict(d)
        if "max_modes" in kw:
            kw["max_modes"] = 0 if kw["max_modes"] in (None, "all") else int(kw["max_modes"])
        for k in ("reproj_cells", "mode_thr"):
            if k in kw:
                kw[k] = float(kw[k])
        return cls(**kw)

    def key(self):
        mm = "all" if not self.max_modes else str(self.max_modes)
        return (f"reproj={self.reproj_cells:g} target={self.target} max_modes={mm} thr={self.mode_thr:g} "
                f"cert={self.cert} solver={self.solver} ransac={self.ransac}")


def settings_of(cons) -> ConsensusCfg:
    """The settings a SatRoMa consensus instance runs with (attributes absent = the package defaults)."""
    return ConsensusCfg(reproj_cells=float(cons.reproj), target="means" if cons.use_means else "peak",
                        max_modes=int(getattr(cons, "max_modes", 0) or 0),
                        mode_thr=float(getattr(cons, "mode_thr", PACKAGE_MODE_THR)),
                        cert=str(getattr(cons, "cert_mode", "off")), solver=str(cons.solver),
                        ransac=str(getattr(cons, "ransac", "default")))


def apply_settings(cons, c: ConsensusCfg, target=True):
    """Set c on a SatRoMa consensus instance (in place). target=False keeps the instance's own peak/means choice
    (the evaluator's two rows)."""
    cons.reproj = float(c.reproj_cells)
    cons.solver = c.solver
    cons.max_modes = int(c.max_modes)
    cons.mode_thr = float(c.mode_thr)
    cons.cert_mode = c.cert
    cons.ransac = c.ransac
    if target:
        cons.use_means = c.target == "means"
    return cons


def ransac_budget(ransac: str, weighted: bool, solver: str):
    """(max_iters or trials, confidence) of one RANSAC run."""
    k = RANSAC_MULT[ransac]
    if solver == "se2":
        return SE2_TRIALS * k, None
    if weighted:
        return WEIGHTED_TRIALS * k, None
    conf = PACKAGE_CONFIDENCE if k == 1 else 1.0 - (1.0 - PACKAGE_CONFIDENCE) ** k
    return PACKAGE_MAX_ITERS * k, conf


# ---- modes --------------------------------------------------------------------------------------------------------

def extract_modes(gm, thr=PACKAGE_MODE_THR, window=PACKAGE_WINDOW):
    """The package's mode extraction (find_gaussians, fixed window) on logits gm (K*K, h, w), plus each mode's
    blurred peak height. Returns dict of arrays in the package's order (tokens row-major, modes within a token in
    np.where order): tok (N, 2) float32 (col, row) = pts_A; means (N, 2), peaks (N, 2) float32 (reference cells,
    cell-centre convention); mass (N,) float32 (moment-window mass); height (N,) float32 (blurred height at the mode's
    local maximum, the value the threshold is tested on)."""
    import cv2
    import torch
    from sat_roma.ransac.gaussian_extraction import assemble_correspondences, process_patches, softmax_heatmaps
    t = gm.detach().float().cpu() if isinstance(gm, torch.Tensor) else torch.as_tensor(np.asarray(gm, np.float32))
    sm, k = softmax_heatmaps(t)
    d = process_patches(sm, adaptive_gauss_fit=False, fixed_threshold=thr, fixed_window_size=window,
                        log_missing_gaussians=False)
    pts_A, means, peaks, _, mass = assemble_correspondences(d, log_missing_gaussians=False, return_weights=True)
    height = []
    for (px, py), gs in d.items():                        # assemble_correspondences' iteration order
        if not gs:
            continue
        blur = cv2.GaussianBlur(sm[:, py, px].reshape(k, k), (5, 5), 0)   # the package's blur of that heatmap
        for g in gs:
            x, y = g["peaks"]
            height.append(blur[int(y), int(x)])
    n = int(pts_A.shape[0])
    as2 = (lambda a: np.asarray(a, np.float32).reshape(n, 2))
    return dict(tok=as2(pts_A), means=as2(means), peaks=as2(peaks), mass=np.asarray(mass, np.float32).reshape(n),
                height=np.asarray(height, np.float32).reshape(n))


def select_modes(modes, c: ConsensusCfg, certainty=None, valid=None):
    """Indices (increasing, i.e. the package order) of the modes the consensus c uses: blurred height > mode_thr
    (the package's strict test, in float32), tokens below the certainty quartile dropped (cert="filter"), then the
    max_modes highest per token. certainty (h, w) logits, valid (h, w) bool tokens (the quartile is over these)."""
    h = modes["height"]
    keep = h > np.float32(c.mode_thr)
    tok = modes["tok"].astype(np.int64)
    if c.cert == "filter":
        if certainty is None:
            raise ValueError("cert='filter' needs the token certainty")
        cert = np.asarray(certainty, np.float64)
        vt = np.ones(cert.shape, bool) if valid is None else np.asarray(valid, bool)
        if vt.any():
            q = np.quantile(cert[vt], CERT_FILTER_Q)
            keep &= cert[tok[:, 1], tok[:, 0]] >= q
    idx = np.flatnonzero(keep)
    if c.max_modes and len(idx):
        tid = tok[idx, 1] * (int(tok[:, 0].max()) + 1) + tok[idx, 0]
        order = np.lexsort((np.arange(len(idx)), -h[idx].astype(np.float64), tid))   # by token, height desc, stable
        ts = tid[order]
        start = np.r_[0, np.flatnonzero(np.diff(ts)) + 1]
        rank = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
        idx = np.sort(idx[order[rank < int(c.max_modes)]])
    return idx


def mode_weights(modes, idx, certainty):
    """sigmoid(token certainty logit) of the selected modes."""
    tok = modes["tok"][idx].astype(np.int64)
    cert = np.asarray(certainty, np.float64)[tok[:, 1], tok[:, 0]]
    return 1.0 / (1.0 + np.exp(-cert))


# ---- sweep json -> evaluator ----------------------------------------------------------------------------------------

def load_chosen(paths):
    """(coarse ConsensusCfg or None, fine ConsensusCfg or None) from one or more sweep jsons (later files override):
    the `chosen` block sets the coarse pass, `chosen_fine` the fine pass."""
    coarse = fine = None
    for p in ([paths] if isinstance(paths, (str, Path)) else paths):
        d = json.loads(Path(p).read_text())
        if d.get("chosen"):
            coarse = ConsensusCfg.from_dict(d["chosen"]["config"])
        if d.get("chosen_fine"):
            fine = ConsensusCfg.from_dict(d["chosen_fine"]["config"])
    return coarse, fine
