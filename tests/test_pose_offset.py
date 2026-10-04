"""Per-city constant pose offset: fitted on one set of frames, removed from another."""
from __future__ import annotations

import numpy as np
import pytest

from bevloc.eval import pose_offset as P


def _rows(city, n, b, seed, gate_fallback=0, wrong=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        gt = rng.uniform(-15, 15, 2)
        fine = gt + np.asarray(b) + rng.normal(0, 0.5, 2)
        rows.append(dict(city=city, en_gt=gt.tolist(), en_coarse=(fine + 0.3).tolist(), en_fine=fine.tolist(),
                         fallback_fine=False))
    for r in rows[:gate_fallback]:
        r["fallback_fine"] = True                                  # the final position is then the coarse one
    for r in rows[gate_fallback:gate_fallback + wrong]:
        r["en_fine"] = (np.asarray(r["en_gt"]) + 40.0).tolist()   # a wrong-place frame, 56 m away
    return rows


def test_final_position_follows_the_gate_and_missing_poses():
    r = dict(city="C", en_gt=[0, 0], en_coarse=[1.0, 2.0], en_fine=[3.0, 4.0], fallback_fine=False)
    assert P.final_en(r) == [3.0, 4.0]
    assert P.final_en({**r, "fallback_fine": True}) == [1.0, 2.0]
    assert P.final_en({**r, "en_fine": None}) == [1.0, 2.0]
    assert P.final_en({**r, "en_coarse": None, "en_fine": None}) is None


def test_offset_is_recovered_per_city_despite_wrong_place_frames():
    calib = _rows("A", 600, (0.0, 0.4), 0, wrong=30) + _rows("B", 600, (0.1, 0.3), 1, gate_fallback=50)
    off = P.fit_offsets(calib)
    assert np.allclose(off["A"]["b_en"], (0.0, 0.4), atol=0.1)
    assert np.allclose(off["B"]["b_en"], (0.1, 0.3), atol=0.15)         # 50 gated frames report the coarse pose (+0.3)
    assert off["A"]["n"] <= 600 - 30                                 # the 56 m frames are outside the 5 m gate


def test_correction_removes_the_offset_on_other_frames_and_does_not_touch_missing_ones():
    off = P.fit_offsets(_rows("A", 800, (0.0, 1.0), 2))
    test = _rows("A", 400, (0.0, 1.0), 3)
    raw = np.array([np.linalg.norm(np.subtract(P.final_en(r), r["en_gt"])) for r in test])
    cor = P.corrected_errors(test, off)
    assert np.median(cor) < np.median(raw) - 0.3
    test.append(dict(city="A", en_gt=[0, 0], en_coarse=None, en_fine=None, fallback_fine=False))
    assert np.isinf(P.corrected_errors(test, off)[-1])


def test_missing_city_and_too_few_frames_raise():
    off = P.fit_offsets(_rows("A", 300, (0.0, 0.4), 4))
    with pytest.raises(KeyError):
        P.corrected_errors(_rows("Z", 5, (0, 0), 5), off)
    with pytest.raises(ValueError):
        P.fit_offsets(_rows("A", 20, (0.0, 0.4), 6))
