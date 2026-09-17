"""Dur360BEV calibration used by this project (values live in configs/default.yaml).

Dur360BEV ships no camera<->LiDAR<->INS extrinsics; ours were measured on the data
(experiments/00_calib, docs/decisions.md) and are provisional until cross-checked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Calib:
    R_cl: np.ndarray          # camera <- LiDAR rotation (both frames x fwd / y left / z up)
    t_cl: np.ndarray          # camera <- LiDAR translation [m]:  p_cam = R_cl @ p_lidar + t_cl
    lidar_height: float       # LiDAR origin above the road [m]

    @property
    def camera_height(self) -> float:
        return float(self.lidar_height - self.t_cl[2])

    @property
    def kw(self) -> dict:
        """Keyword arguments for the projection functions."""
        return dict(R_cl=self.R_cl, t_cl=self.t_cl)

    @classmethod
    def from_config(cls, cfg) -> "Calib":
        c = cfg.calib
        R = Rotation.from_euler("xyz", c.cam_from_lidar_rpy_deg, degrees=True).as_matrix()
        return cls(R, np.asarray(c.cam_from_lidar_t_m, float), float(c.lidar_height_m))
