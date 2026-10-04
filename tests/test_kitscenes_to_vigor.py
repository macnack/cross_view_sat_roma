"""KITScenes -> VIGOR layout: the label, bearing and roll conventions (no scene, no network)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kitscenes_to_vigor.py"
spec = importlib.util.spec_from_file_location("kitscenes_to_vigor", SCRIPT)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def rz(yaw_deg):
    a = np.radians(yaw_deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_bearing_is_clockwise_from_north():
    assert np.isclose(T.bearing_cw_from_north(rz(0)), 90.0)        # forward = east
    assert np.isclose(T.bearing_cw_from_north(rz(90)), 0.0)        # forward = north
    assert np.isclose(T.bearing_cw_from_north(rz(-90)), 180.0)     # forward = south
    assert np.isclose(T.bearing_cw_from_north(rz(180)), 270.0)     # forward = west


def test_label_inverts_vigors_offset_rule():
    """VigorPairs: en = (col_sign * dx * res, -row_sign * dy * res) with col_sign -1, row_sign +1 = the camera's metres
    east / north of the tile centre."""
    for e, n in ((0.0, 0.0), (5.0, -3.0), (-12.5, 17.0)):
        dy, dx = T.label_from_offset(e, n)
        assert np.allclose((-dx * T.RES, -dy * T.RES), (e, n))
    dy, dx = T.label_from_offset(0.0, 10.0)                        # 10 m NORTH of the tile centre: dy < 0 (north = up)
    assert dy < 0 and dx == 0
    dy, dx = T.label_from_offset(10.0, 0.0)                        # 10 m EAST: dx < 0 (dx > 0 = WEST)
    assert dx < 0 and dy == 0


def test_rolling_by_the_bearing_puts_the_forward_axis_at_that_bearing():
    W = 2048
    erp = np.zeros((4, W), np.uint8)
    erp[:, W // 2] = 255                                           # the ego's forward axis = the centre column
    for b in (0.0, 90.0, 144.8, 270.0):
        rolled = np.roll(erp, int(round(b / 360.0 * W)), axis=1)
        col = int(np.argmax(rolled[0]))
        az_cw_from_north = ((col / W) - 0.5) * 360.0               # VIGOR: centre column = north, azimuth grows to the right
        assert np.isclose(az_cw_from_north % 360.0, b % 360.0, atol=360.0 / W)


def test_wms_choice_by_scene_origin():
    assert T.default_wms(50.1104, 8.6821) == "hessen"              # Frankfurt
    assert T.default_wms(49.0094, 8.4044) == "lgl"                 # Karlsruhe
    assert T.default_wms(48.7134, 9.0021) == "lgl"                 # Sindelfingen
