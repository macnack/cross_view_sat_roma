"""Thin read-only wrapper around ``third_party/FG2`` (no edits to that tree).

Loads the released VIGOR CVM weights + frozen DINOv2 extractor. Adapting
Fixtor panoramas / Poznań orthophotos to FG²'s native tensor sizes happens
here; the third-party code is imported from its own directory on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from bevloc import config as C
from bevloc.baselines.common import sha256_file

FG2_ROOT = C.REPO / "third_party" / "FG2"
MMCV_SHIM = Path(__file__).resolve().parent / "mmcv_shim"   # see mmcv_shim/mmcv/__init__.py
CKPT_ROOT = C.REPO / "checkpoints" / "baselines" / "fg2" / "VIGOR"

# Native VIGOR settings from third_party/FG2/config.ini (do not force Sat-RoMa sizes).
NATIVE = dict(
    ground_image_size=(714, 1428),  # H, W
    satellite_image_size=(630, 630),
    grid_size_h=71.0,
    grid_size_v=20.0,
    grd_bev_res=41,
    grd_height_res=11,
    sat_bev_res=41,
    num_samples_matches=1024,
)


def ensure_fg2_on_path():
    """Put FG2 first and neutralise the sat_roma ``utils`` editable install.

    ``bev-patch-pf`` is editable-installed into the local conda env and owns a
    top-level ``utils`` package. FG2's ``utils/`` has no ``__init__.py``, so
    Python resolves ``utils`` to sat_roma and ``from utils.utils import …``
    fails. We do not edit ``third_party/FG2``; instead we drop the conflicting
    path and register a package shim for FG2's utils directory.
    """
    root = str(FG2_ROOT)
    # Drop sat_roma's editable src (and any prior FG2 entry) then prepend FG2.
    cleaned = []
    for p in sys.path:
        if not p:
            cleaned.append(p)
            continue
        if "sat_roma/third_party/bev-patch-pf" in p.replace("\\", "/"):
            continue
        if Path(p).resolve() == FG2_ROOT.resolve():
            continue
        cleaned.append(p)
    sys.path[:] = [root] + cleaned
    for k in list(sys.modules):
        if k == "utils" or k.startswith("utils."):
            del sys.modules[k]
    # FG2 ships utils/*.py without __init__.py — make it a real package at runtime.
    import types
    utils_dir = FG2_ROOT / "utils"
    pkg = types.ModuleType("utils")
    pkg.__path__ = [str(utils_dir)]  # type: ignore[attr-defined]
    pkg.__file__ = str(utils_dir / "__init__.py")
    sys.modules["utils"] = pkg
    ensure_mmcv()


def ensure_mmcv():
    """FG² imports two mmcv layers; use the pure-PyTorch shim when mmcv's compiled ops are unavailable."""
    try:
        from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention  # noqa: F401
        return
    except Exception:  # ImportError, or mmcv-lite without ops
        for k in list(sys.modules):
            if k == "mmcv" or k.startswith("mmcv."):
                del sys.modules[k]
        if str(MMCV_SHIM) not in sys.path:
            sys.path.insert(0, str(MMCV_SHIM))


def checkpoint_path(area: str = "samearea", orientation: str = "known_ori",
                    first_run: bool | None = None) -> Path:
    """Resolve a released FG² VIGOR checkpoint under ``checkpoints/baselines/fg2``."""
    if orientation == "known_ori":
        p = CKPT_ROOT / area / "known_ori" / "model.pt"
    elif orientation == "unknown_ori":
        stage = "first_run" if first_run else "second_run"
        p = CKPT_ROOT / area / "unknown_ori" / stage / "model.pt"
    else:
        raise ValueError(orientation)
    if not p.is_file():
        raise FileNotFoundError(
            f"FG² checkpoint missing: {p}\n"
            "Download from https://drive.google.com/drive/folders/1WUViQcX9m0PE9FePbWklisBS88sHOyMK"
        )
    return p


def load_cvm(device, area="samearea", orientation="known_ori", first_run=None):
    """Build CVM + load released weights. Returns (model, meta)."""
    ensure_fg2_on_path()
    # FG2 modules read ./config.ini relative to CWD; chdir into FG2 for import.
    import os
    prev = os.getcwd()
    try:
        os.chdir(FG2_ROOT)
        from models.model_vigor import CVM  # noqa: WPS433
    finally:
        os.chdir(prev)

    ckpt = checkpoint_path(area, orientation, first_run)
    model = CVM(
        device,
        grd_bev_res=NATIVE["grd_bev_res"],
        grd_height_res=NATIVE["grd_height_res"],
        sat_bev_res=NATIVE["sat_bev_res"],
        grid_size_h=NATIVE["grid_size_h"],
        grid_size_v=NATIVE["grid_size_v"],
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    meta = {
        "checkpoint": str(ckpt.relative_to(C.REPO)),
        "sha256": sha256_file(ckpt),
        "area": area,
        "orientation": orientation,
        "first_run": first_run,
        "native": dict(NATIVE),
        "license": "GPL-3.0 (third_party/FG2/LICENSE)",
    }
    return model, meta


def load_dino(device):
    ensure_fg2_on_path()
    import os
    prev = os.getcwd()
    try:
        os.chdir(FG2_ROOT)
        from models.modules import DinoExtractor  # noqa: WPS433
    finally:
        os.chdir(prev)
    ext = DinoExtractor().to(device).eval()
    return ext
