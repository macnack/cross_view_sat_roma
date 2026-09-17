"""The kick-off BEV query variants for one frame.

  ipm_raw  : flat-ground IPM of every valid ERP pixel (lower bound; smears objects)
  ipm_cl   : IPM of ground pixels only = below the LiDAR-derived contact line v_c(u).
             Camera geometry + ORACLE contact line (the label the geometry head must predict).
             Primary kick-off query (docs/decisions.md).
  ipm_edge : ipm_cl + the contact line of tall objects drawn as thin wall edges
  oracle_a : LiDAR points coloured from the ERP; cells without a return are invalid
  oracle_b : ipm_cl ground + LiDAR object points at their true cells
  ipm_fp   : ipm_cl + facades collapsed into radial bands (REJECTED: fans; kept for the record)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import BevGrid, ipm_bev, level_from_body, lidar_bev
from .ground import (Contact, collapse_above_contact, contact_edges, contact_line, ground_pixel_mask,
                     segment_ground)

EDGE_RGB = (15, 15, 15)      # wall edge drawn near-black (roof/ground boundaries are dark in orthophotos)
VARIANTS = ("ipm_raw", "ipm_cl", "ipm_edge", "oracle_a", "oracle_b", "ipm_fp")


@dataclass
class BevSet:
    name: str
    erp: np.ndarray
    contact: Contact                   # per-column contact foot (row, xy, object height)
    edge: np.ndarray                   # (n, n) bool wall-edge cells
    ground_fraction: float             # share of LiDAR returns labelled ground
    images: dict                       # variant -> (n, n, 3) uint8
    valid: dict                        # variant -> (n, n) bool

    def get(self, variant):
        return self.images[variant], self.valid[variant]


def build_variants(erp, points, erp_valid, calib, cfg, name="", roll=0.0, pitch=0.0,
                   variants=VARIANTS) -> BevSet:
    grid = BevGrid(cfg.grid.n, cfg.grid.cell_m)
    g, e = cfg.ground, cfg.edges
    h, w = erp.shape[:2]
    pts = np.asarray(points, np.float64)
    H0, kw = calib.lidar_height, calib.kw

    R_lb = level_from_body(roll, pitch)
    P = pts @ R_lb.T
    is_ground, floor_z = segment_ground(P, H0, cell=g.floor_cell_m, ground_tol=g.ground_tol_m,
                                        max_dev=g.max_floor_dev_m, max_range=g.max_range_m, return_floor=True)
    foot_z_body = (np.c_[P[:, :2], floor_z] @ R_lb)[:, 2]          # local floor under each point, body frame
    contact = contact_line(pts, is_ground, w, h, foot_z_body=foot_z_body,
                           contact_max_height=g.contact_max_height_m, **kw)
    edge = contact_edges(grid, contact, R_lb, min_obj_height=e.min_obj_height_m, max_jump_m=e.max_jump_m)

    img, ok = {}, {}
    img["ipm_cl"], ok["ipm_cl"] = ipm_bev(erp, grid, roll, pitch, H0,
                                          erp_valid=ground_pixel_mask(contact.row, h, erp_valid), **kw)
    if "ipm_raw" in variants:
        img["ipm_raw"], ok["ipm_raw"] = ipm_bev(erp, grid, roll, pitch, H0, erp_valid=erp_valid, **kw)
    if "ipm_edge" in variants:
        im = img["ipm_cl"].copy()
        im[edge] = EDGE_RGB
        img["ipm_edge"], ok["ipm_edge"] = im, ok["ipm_cl"] | edge
    if "oracle_a" in variants:
        img["oracle_a"], ok["oracle_a"], _ = lidar_bev(erp, pts, grid, roll, pitch, erp_valid=erp_valid,
                                                       select="mean", **kw)
    if "oracle_b" in variants:
        o, o_ok, _ = lidar_bev(erp, pts[~is_ground], grid, roll, pitch, erp_valid=erp_valid, select="mean", **kw)
        im = img["ipm_cl"].copy()
        im[o_ok] = o[o_ok]
        img["oracle_b"], ok["oracle_b"] = im, ok["ipm_cl"] | o_ok
    if "ipm_fp" in variants:
        f, f_ok = collapse_above_contact(erp, grid, contact, H0, erp_valid, **kw)
        im = img["ipm_cl"].copy()
        fill = f_ok & ~ok["ipm_cl"]
        im[fill] = f[fill]
        img["ipm_fp"], ok["ipm_fp"] = im, ok["ipm_cl"] | f_ok
    return BevSet(name, erp, contact, edge, float(is_ground.mean()), img, ok)
