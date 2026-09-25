"""Thin read-only wrapper around ``third_party/Loc2`` (AGPL-3.0: imported from its own directory, never copied
into ours) and its depth model ``third_party/Loc2/external/unik3d``.

Loads the released VIGOR matcher (``VigorCrossViewMatcher``) with the frozen DINOv2 extractor Loc² shares
between views, and UniK3D for the metric depth maps Loc² lifts its matches with. The two mmcv layers Loc²'s
``models.modules`` imports come from ``bevloc.baselines.mmcv_shim`` when mmcv is not installed, exactly as
for FG².
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch

from bevloc import config as C
from bevloc.baselines.common import sha256_file
from bevloc.baselines.fg2 import ensure_mmcv

LOC2_ROOT = C.REPO / "third_party" / "Loc2"
UNIK3D_ROOT = LOC2_ROOT / "external" / "unik3d"
CKPT_ROOT = C.REPO / "checkpoints" / "baselines" / "loc2" / "vigor"

# Native settings (third_party/Loc2/config.ini, eval_vigor.py, preprocess/infer_depth_vigor.py).
NATIVE = dict(
    ground_image_size=(714, 1428),   # H, W fed to DINOv2 (14 px patches -> 51 x 102 ground tokens)
    satellite_image_size=(630, 630),
    sat_bev_res=41,                  # aerial points per side; grid = 640 px * city GSD (about 71 m)
    num_samples_matches=1024,
    max_depth_m=35.0,                # matches beyond this depth are masked out
    depth_png_max_m=65.0,            # depth PNGs: uint16 millimetres, clipped here
    panorama_size=(2048, 1024),      # W, H of the VIGOR panoramas UniK3D sees
)
_LOC2_PKGS = ("models", "dataloaders", "att_layers", "DINO_modules", "utils", "unik3d")


def ensure_loc2_on_path():
    """Loc2 (then UniK3D) first on sys.path, with any cached top-level package of the same name that does not
    live in the Loc2 tree dropped (FG² ships its own ``models`` / ``utils``)."""
    roots = [str(LOC2_ROOT), str(UNIK3D_ROOT)]
    resolved = {Path(r).resolve() for r in roots}
    sys.path[:] = roots + [p for p in sys.path
                           if not p or (Path(p).resolve() not in resolved and "third_party/FG2" not in p.replace("\\", "/"))]
    for k in list(sys.modules):
        top = k.split(".")[0]
        if top in _LOC2_PKGS:
            f = getattr(sys.modules[k], "__file__", None) or ""
            if str(LOC2_ROOT) not in f:
                del sys.modules[k]
    ensure_mmcv()


def checkpoint_path(area: str = "samearea", orientation: str = "known_ori") -> Path:
    p = CKPT_ROOT / area / orientation / "model.pt"
    if not p.is_file():
        raise FileNotFoundError(
            f"Loc² checkpoint missing: {p}\n"
            "Download from https://drive.google.com/drive/folders/1JQHSxN-IRViKdFri2m9JtLMR_JdFLBIO")
    return p


def _import_from_loc2(fn):
    ensure_loc2_on_path()
    prev = os.getcwd()
    try:
        os.chdir(LOC2_ROOT)
        return fn()
    finally:
        os.chdir(prev)


def load_matcher(device, area="samearea", orientation="known_ori"):
    """Released ``VigorCrossViewMatcher`` with its weights (strict). Returns (model, meta)."""
    def imp():
        from models.vigor_matcher import VigorCrossViewMatcher  # noqa: WPS433
        return VigorCrossViewMatcher
    VigorCrossViewMatcher = _import_from_loc2(imp)
    ckpt = checkpoint_path(area, orientation)
    model = VigorCrossViewMatcher(device, sat_bev_res=NATIVE["sat_bev_res"], embed_dim=1024)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device).eval()
    meta = {
        "checkpoint": str(ckpt.relative_to(C.REPO)),
        "sha256": sha256_file(ckpt),
        "area": area,
        "orientation": orientation,
        "native": dict(NATIVE),
        "license": "AGPL-3.0 (third_party/Loc2/LICENSE)",
    }
    return model, meta


def load_dino(device):
    def imp():
        from models.modules import DinoExtractor  # noqa: WPS433
        return DinoExtractor
    return _import_from_loc2(imp)().to(device).eval()


def load_unik3d(device, name="unik3d-vitl", resolution_level=9, interpolation_mode="bilinear"):
    """UniK3D from the HF cache (``lpiccinelli/<name>``), configured as Loc²'s preprocess script does."""
    ensure_loc2_on_path()
    from unik3d.models import UniK3D  # noqa: WPS433
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
    model.resolution_level = resolution_level
    model.interpolation_mode = interpolation_mode
    return model.to(device).eval()


def spherical_camera(width: int, height: int):
    """Loc²'s docs/equirectangular.json for a W x H panorama: full 360 x 180 degrees."""
    ensure_loc2_on_path()
    from unik3d.utils.camera import Spherical  # noqa: WPS433
    return Spherical(params=torch.tensor([1.0, 1.0, 1.0, 1.0, float(width), float(height), math.pi, math.pi / 2]))


def depth_png_path(root, city: str, pano: str) -> Path:
    """Where Loc²'s dataloader looks for the depth of ``<root>/<city>/panorama/<pano>`` (one definition, shared
    with the VigorPairs depth reader)."""
    from bevloc.data.vigor import depth_png_path as _p
    return _p(root, city, pano)
