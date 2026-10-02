"""Bootstrap summaries for pose evaluations."""
from __future__ import annotations

import numpy as np

from bevloc.eval.report import bootstrap_ci, centre_guess_errors, summarise_pose


def test_bootstrap_ci_brackets_the_statistic():
    v = np.random.default_rng(0).normal(10.0, 2.0, 500)
    med, (lo, hi) = bootstrap_ci(v, np.median, n_boot=300)
    assert lo <= med <= hi
    assert hi - lo < 1.0


def test_summarise_counts_misses_as_inf():
    s = summarise_pose([1.0, None, 3.0, 20.0], n_boot=50)
    assert s["n"] == 4 and s["matched"] == 3
    assert abs(s["recall@5m"] - 0.5) < 1e-9
    assert abs(s["recall@10m"] - 0.5) < 1e-9
    assert s["frac_gt_30m"] == 0.25          # the miss counts as > 30 m


def test_centre_guess_is_norm_of_crop_offset():
    assert centre_guess_errors([{"crop_offset_m": [3.0, 4.0]}]) == [5.0]
