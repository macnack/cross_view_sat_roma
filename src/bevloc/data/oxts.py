"""OxTS RT3000v3 track: positions in British National Grid and vehicle heading.

Dur360BEV stores one `lat lon alt roll pitch yaw` line per frame. The stored yaw
convention is settled empirically against the direction of travel (`convention_test`),
not taken from the header — see docs/decisions.md.

Headings here are BEARINGS: degrees clockwise from north, 0 = north, which is what
`bevloc.data.ortho.Oriented.up_bearing_deg` wants. Bearings are TRUE north unless
converted with `grid_convergence` (BNG grid north differs by ~0.35 deg in Durham).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Stored yaw [rad] -> bearing [deg cw from north]. All four readings of "yaw around z".
CONVENTIONS = {
    "ccw_from_east": lambda y: 90.0 - np.degrees(y),    # dataformat.txt / Dur360BEV map_api
    "ccw_from_north": lambda y: -np.degrees(y),
    "cw_from_north": lambda y: np.degrees(y),           # RT3000 device convention (NED)
    "cw_from_east": lambda y: 90.0 + np.degrees(y),
}


def wrap180(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def circ_stats(d_deg):
    """Circular mean and std [deg] of wrapped angular residuals."""
    a = np.radians(np.asarray(d_deg, float))
    if a.size == 0:
        return np.nan, np.nan
    c, s = np.cos(a).mean(), np.sin(a).mean()
    r = np.hypot(c, s)
    return np.degrees(np.arctan2(s, c)), np.degrees(np.sqrt(max(-2 * np.log(max(r, 1e-12)), 0.0)))


@dataclass
class OxtsTrack:
    names: list[str]
    t: np.ndarray        # (N,) seconds since the first record
    lla: np.ndarray      # (N, 3) lat [deg], lon [deg], alt [m]
    rpy: np.ndarray      # (N, 3) roll, pitch, yaw as stored [rad]
    en: np.ndarray       # (N, 2) easting, northing [m], EPSG:27700

    def __len__(self):
        return len(self.names)

    def speed(self, smooth=1):
        """Ground speed [m/s] from central differences over +-`smooth` samples."""
        i = np.arange(len(self))
        a, b = np.clip(i - smooth, 0, len(self) - 1), np.clip(i + smooth, 0, len(self) - 1)
        dt = np.maximum(self.t[b] - self.t[a], 1e-6)
        return np.linalg.norm(self.en[b] - self.en[a], axis=1) / dt

    def travel_bearing(self, smooth=1):
        """Bearing of motion [deg cw from north] over +-`smooth` samples; NaN when static."""
        i = np.arange(len(self))
        a, b = np.clip(i - smooth, 0, len(self) - 1), np.clip(i + smooth, 0, len(self) - 1)
        d = self.en[b] - self.en[a]
        out = np.degrees(np.arctan2(d[:, 0], d[:, 1]))       # atan2(east, north)
        return np.where(np.linalg.norm(d, axis=1) > 1e-6, out, np.nan)

    def bearing(self, convention, grid=False):
        """Stored yaw as a bearing [deg cw from north] under `convention`."""
        b = CONVENTIONS[convention](self.rpy[:, 2])
        return (b - grid_convergence(self.lla[:, 1], self.lla[:, 0])) % 360.0 if grid else b % 360.0


def read_track(root, names=None, timestamps=True) -> OxtsTrack:
    """Read oxts/data/*.txt (all of them, or just `names`) plus oxts/timestamps.txt."""
    root = Path(root)
    d = root / "oxts/data"
    names = sorted(p.stem for p in d.glob("*.txt")) if names is None else [str(n) for n in names]
    v = np.array([np.loadtxt(d / f"{n}.txt").reshape(-1) for n in names], float)
    t = np.zeros(len(names))
    if timestamps and (root / "oxts/timestamps.txt").exists():
        all_t = read_timestamps(root / "oxts/timestamps.txt")
        t = all_t[[int(n) for n in names]] - all_t[0]
    e, n = lonlat_to_bng(v[:, 1], v[:, 0])
    return OxtsTrack(names, t, v[:, :3], v[:, 3:6], np.c_[e, n])


def read_timestamps(path) -> np.ndarray:
    """ISO timestamps -> seconds since the epoch (float64 keeps ~0.1 us here)."""
    s = np.loadtxt(path, dtype="U32")
    return np.array(s, dtype="datetime64[ns]").astype("int64") / 1e9


def lonlat_to_bng(lon, lat):
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True).transform(lon, lat)


def grid_convergence(lon, lat):
    """BNG grid north minus true north [deg]: bearing_grid = bearing_true - convergence."""
    from pyproj import Proj
    f = Proj("EPSG:27700").get_factors(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(f.meridian_convergence, float)


def convention_test(track: OxtsTrack, min_speed=2.0, smooth=2):
    """Score every reading of the stored yaw against the direction of travel.

    The correct convention has BOTH a small circular std (right handedness and zero
    axis) and a small |mean| (the residual mean is the INS-to-vehicle yaw mounting
    offset, expected to be a few degrees). Conventions differing by a constant 90 deg
    are separated by the mean alone, so both numbers are reported.
    Moving frames only: a parked vehicle's GNSS jitter has no direction.
    """
    trav = track.travel_bearing(smooth)
    use = np.isfinite(trav) & (track.speed(smooth) > min_speed)
    out = {}
    for k in CONVENTIONS:
        r = wrap180(track.bearing(k)[use] - trav[use])
        m, s = circ_stats(r)
        out[k] = dict(mean_deg=float(m), std_deg=float(s),
                      within_10deg=float(np.mean(np.abs(wrap180(r - m)) < 10)) if r.size else float("nan"))
    return out, use
