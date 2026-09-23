"""Coordinate / SE(2) recovery checks for the baseline comparison protocol."""
from __future__ import annotations

import numpy as np

from bevloc.baselines.common import (
    inject_and_recover,
    pose_from_homography,
    se2_ego_to_map,
    se2_map_error,
)
from bevloc.data.ortho import Oriented, gt_homography


def test_se2_ego_to_map_facing_north():
    # up-bearing 0° = image-up = grid north. Ego +x → +N, ego +y (left) → −E.
    en, yaw = se2_ego_to_map((100.0, 200.0), 0.0, dx=5.0, dy=3.0, dyaw_deg=7.0)
    assert abs(en[0] - (100.0 - 3.0)) < 1e-9
    assert abs(en[1] - (200.0 + 5.0)) < 1e-9
    assert abs(yaw - 7.0) < 1e-9


def test_se2_ego_to_map_facing_east():
    # up-bearing 90° = image-up = grid east. Ego +x → +E, ego +y (left) → +N.
    # dy=-3 moves opposite left → −N.
    en, yaw = se2_ego_to_map((0.0, 0.0), 90.0, dx=5.0, dy=-3.0, dyaw_deg=0.0)
    assert abs(en[0] - 5.0) < 1e-9
    assert abs(en[1] - (-3.0)) < 1e-9
    assert abs(yaw - 90.0) < 1e-9


def test_injected_pose_5m_m3m_7deg_recovers():
    """Task checkpoint: injected (5 m, −3 m, 7°) recovers to numerical tolerance."""
    out = inject_and_recover(
        en=(357_000.0, 505_000.0),
        up_bearing_deg=35.0,
        dx=5.0,
        dy=-3.0,
        dyaw_deg=7.0,
    )
    assert out["ok"], out
    assert out["err_m"] < 1e-3
    assert out["err_deg"] < 1e-2


def test_homography_roundtrip_identity():
    q = Oriented((10.0, 20.0), 12.0, 224, 0.25)
    r = Oriented((12.0, 18.0), 15.0, 896, 0.25)
    H = gt_homography(q, r)
    en, yaw = pose_from_homography(H, 224, r)
    pe, ye = se2_map_error(en, yaw, q.centre_en, q.up_bearing_deg)
    assert pe < 1e-6
    assert ye < 1e-4


def test_pixel_metric_map_chain():
    """Synthetic translation on the reference recovers the metric offset."""
    gsd = 0.25
    q = Oriented((0.0, 0.0), 0.0, 224, gsd)
    # Reference shares orientation; vehicle sits 8 m east / 4 m north of crop centre
    # → in ref axes with up=north: right=+E so offset_right=8, offset_up=4.
    r = Oriented((-8.0, -4.0), 0.0, 896, gsd)
    H = gt_homography(q, r)
    # Query centre maps to ref pixel displaced by (8/gsd, -4/gsd) from ref centre
    # (v increases opposite to up).
    s_q = (q.size - 1) / 2.0
    s_r = (r.size - 1) / 2.0
    p = H @ np.array([s_q, s_q, 1.0])
    u, v = p[0] / p[2], p[1] / p[2]
    assert abs(u - (s_r + 8.0 / gsd)) < 1e-6
    assert abs(v - (s_r - 4.0 / gsd)) < 1e-6
