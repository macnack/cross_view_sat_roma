"""Summary statistics with percentile-bootstrap intervals for pose evaluations."""
from __future__ import annotations

import numpy as np


def _finite(values):
    return np.asarray([np.inf if (v is None or not np.isfinite(v)) else float(v) for v in values], float)


def bootstrap_ci(values, stat, n_boot=1000, seed=0, alpha=0.05):
    """(stat(values), (lo, hi)) percentile bootstrap. Misses (None/inf) stay inf, so they
    count against recall and pull the median up, never silently dropped."""
    v = _finite(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.array([stat(v[i]) for i in idx], float)
    # "nearest" avoids interpolating between two inf bootstrap medians (inf - inf = nan) when misses
    # dominate a resample; an unbounded upper limit is then reported as inf, which is the truth.
    return float(stat(v)), (float(np.quantile(boots, alpha / 2, method="nearest")),
                            float(np.quantile(boots, 1 - alpha / 2, method="nearest")))


def summarise_pose(errors_m, n_boot=1000, seed=0):
    v = _finite(errors_m)
    med, med_ci = bootstrap_ci(v, np.median, n_boot, seed)
    out = {"n": int(len(v)), "matched": int(np.isfinite(v).sum()), "median_m": med, "median_ci": med_ci}
    for t in (5.0, 10.0):
        r, ci = bootstrap_ci(v, lambda a, t=t: float((a <= t).mean()), n_boot, seed)
        out[f"recall@{t:g}m"], out[f"recall@{t:g}m_ci"] = r, ci
    out["frac_gt_30m"] = float((v > 30.0).mean())
    return out


def centre_guess_errors(entries):
    """Error of 'predict the crop centre' per manifest entry; crop_offset_m is (right, up)."""
    return [float(np.hypot(*e["crop_offset_m"])) for e in entries]
