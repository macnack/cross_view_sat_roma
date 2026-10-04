"""Constant per-city pose offset of a VIGOR model: fitted on held-out TRAINING frames, subtracted at test time.

Why (experiments/17_corrected_labels/OFFSET.md): a model trained with rot90 / flip augmentation predicts the camera a
roughly constant vector b (about 0.4 m, north) away from VIGOR's label. The offset is a property of the model + labels,
not of the test draw, so it is measured on frames the model never trained on (the --val-frac part of the training list,
`eval_vigor.py --calib`) and removed from the test predictions. Nothing about the test frames is used.

Positions are (east, north) metres from the tile centre, the frame of the `en_gt` / `en_coarse` / `en_fine` keys that
`scripts/eval_vigor.py` writes per frame. The estimate is the median of the residual (prediction - label) over the frames
within `max_err_m` of the label (a median over a gated set: a handful of wrong-place frames do not move it).
"""
from __future__ import annotations

import numpy as np


def final_en(row):
    """The position the two-pass pipeline reports for a frame (east, north), or None when it reports none: the fine pose
    unless the 6 m gate fell back to the coarse one (`fallback_fine`); a frame with no coarse pose has none."""
    coarse, fine = row.get("en_coarse"), row.get("en_fine")
    if coarse is None:
        return None
    if row.get("fallback_fine") or fine is None:
        return coarse
    return fine


def residuals(rows, key="final"):
    """(city, residual (n, 2)) per frame with a position: prediction - label in (east, north) metres."""
    out = []
    for r in rows:
        p = final_en(r) if key == "final" else r.get(key)
        if p is None or r.get("en_gt") is None:
            continue
        out.append((r["city"], np.subtract(p, r["en_gt"], dtype=np.float64)))
    return out


def fit_offsets(rows, key="final", max_err_m=5.0, min_frames=100):
    """{city: dict(b_en = [east, north], n, scatter_en)}: the per-city median residual of the frames within max_err_m.
    scatter_en = 1.4826 * MAD per axis, the offset's standard error is about scatter / sqrt(n). A city with fewer than
    min_frames usable frames raises ValueError (no silent global fallback)."""
    by = {}
    for city, d in residuals(rows, key):
        if np.linalg.norm(d) < max_err_m:
            by.setdefault(city, []).append(d)
    out = {}
    for city, ds in sorted(by.items()):
        d = np.array(ds)
        if len(d) < min_frames:
            raise ValueError(f"{city}: only {len(d)} frames within {max_err_m} m, need {min_frames}")
        b = np.median(d, axis=0)
        out[city] = dict(b_en=[float(b[0]), float(b[1])], n=int(len(d)),
                         scatter_en=[float(1.4826 * np.median(np.abs(d[:, i] - b[i]))) for i in (0, 1)])
    return out


def corrected_errors(rows, offsets, key="final"):
    """Per-row error (m) of the prediction with its city's offset subtracted, np.inf for a frame with no position.
    A city without a fitted offset raises KeyError (never silently uncorrected)."""
    errs = []
    for r in rows:
        p = final_en(r) if key == "final" else r.get(key)
        if p is None or r.get("en_gt") is None:
            errs.append(np.inf)
            continue
        b = np.asarray(offsets[r["city"]]["b_en"], np.float64)
        errs.append(float(np.linalg.norm(np.subtract(p, r["en_gt"], dtype=np.float64) - b)))
    return np.array(errs)
