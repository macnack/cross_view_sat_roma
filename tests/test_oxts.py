import numpy as np

from bevloc.data.oxts import (CONVENTIONS, OxtsTrack, circ_stats, convention_test,
                              grid_convergence, wrap180)


def _track(convention, yaw_offset_deg=0.0, n=400, seed=0):
    """A circle driven counter-clockwise, with the stored yaw encoded in `convention`."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 10.0
    a = 2 * np.pi * t / t[-1]
    r = 60.0
    en = np.c_[427000 + r * np.cos(a), 541000 + r * np.sin(a)]     # speed ~ 3.8 m/s
    travel = np.degrees(np.arctan2(-np.sin(a), np.cos(a)))         # bearing of d(en)/da, cw from north
    bearing = travel + yaw_offset_deg
    inv = {"ccw_from_east": lambda b: 90.0 - b, "ccw_from_north": lambda b: -b,
           "cw_from_north": lambda b: b, "cw_from_east": lambda b: b - 90.0}[convention]
    yaw = np.radians(wrap180(inv(bearing)))
    lla = np.c_[np.full(n, 54.77), np.full(n, -1.57), np.zeros(n)]
    return OxtsTrack([f"{i:010d}" for i in range(n)], t, lla, np.c_[np.zeros((n, 2)), yaw], en)


def test_convention_test_recovers_each_convention():
    for k in CONVENTIONS:
        scores, use = convention_test(_track(k), min_speed=2.0, smooth=2)
        assert use.all()
        best = min(scores, key=lambda c: (scores[c]["std_deg"], abs(scores[c]["mean_deg"])))
        assert best == k, (k, {c: round(v["std_deg"], 1) for c, v in scores.items()})
        assert abs(scores[k]["mean_deg"]) < 1.0 and scores[k]["std_deg"] < 1.0


def test_convention_test_reports_mounting_offset():
    """A constant INS-to-vehicle yaw offset shows up in the mean, not the std."""
    scores, _ = convention_test(_track("ccw_from_east", yaw_offset_deg=7.0), min_speed=2.0, smooth=2)
    assert abs(scores["ccw_from_east"]["mean_deg"] - 7.0) < 1.0
    assert scores["ccw_from_east"]["std_deg"] < 1.0
    # the 90-deg sibling is separated by the mean alone, not by the spread
    assert abs(scores["ccw_from_north"]["std_deg"] - scores["ccw_from_east"]["std_deg"]) < 1e-6


def test_wrap_and_circular_stats():
    assert np.allclose(wrap180([190, -190, 180, -180]), [-170, 170, -180, -180])
    m, s = circ_stats([179, -179, 180])            # straddles the cut
    assert abs(abs(m) - 180) < 1.0 and s < 2.0


def test_static_frames_are_excluded():
    tr = _track("ccw_from_east")
    tr.en[:] = tr.en[0]                            # parked: no direction of travel
    _, use = convention_test(tr, min_speed=2.0, smooth=2)
    assert not use.any()


def test_grid_convergence_sign_at_durham():
    """Durham is east of BNG's central meridian (-2 deg), so grid north is west of true
    north there: convergence > 0 and bearing_grid = bearing_true - convergence."""
    c = grid_convergence(-1.57, 54.77)
    assert 0.1 < float(c) < 0.6
    assert float(grid_convergence(-2.0, 54.77)) == 0.0 or abs(float(grid_convergence(-2.0, 54.77))) < 1e-6
