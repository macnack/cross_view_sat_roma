"""Thin read-only wrapper around ``third_party/Loc2`` (AGPL-3.0: imported from its own directory, never copied
into ours) and its depth model ``third_party/Loc2/external/unik3d``.

Loads the released VIGOR matcher (``VigorCrossViewMatcher``) with the frozen DINOv2 extractor Loc² shares
between views, and UniK3D for the metric depth maps Loc² lifts its matches with. The two mmcv layers Loc²'s
``models.modules`` imports come from ``bevloc.baselines.mmcv_shim`` when mmcv is not installed, exactly as
for FG².
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
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


def spherical_camera(width: int, height: int, vfov_half: float = 1.57080):
    """Loc²'s docs/equirectangular.json for a W x H panorama: full 360 x 180 degrees.

    Single use: UniK3D.infer mutates the camera it is given (BatchCamera.from_camera shares its params, .to(device)
    rebinds them to the device copy, crop / resize then scale fx..H in place by the resize factor), so a camera
    reused across calls decays to W = H = 0 within ~40 panoramas and the depth map turns into image-independent row
    stripes. Use `infer_distance`, which builds a fresh one per call."""
    ensure_loc2_on_path()
    from unik3d.utils.camera import Spherical  # noqa: WPS433
    # half-FoVs exactly as docs/equirectangular.json (3.14159, 1.5708), not math.pi: math.pi changes 0.7 % of the
    # pixels by > 1 cm (up to 0.45 m at depth edges) against Loc²'s released preprocessing (measured 2026-10-01)
    return Spherical(params=torch.tensor([1.0, 1.0, 1.0, 1.0, float(width), float(height), 3.14159, float(vfov_half)]))


def infer_distance(model, rgb, vfov_half: float = 1.57080):
    """UniK3D metric distance along the ray (H, W) float32 numpy for one ERP panorama ``rgb`` (H, W, 3) uint8, as
    third_party/Loc2/preprocess/infer_depth_vigor.py: full-sphere Spherical camera of the image size, normalize=True,
    |points|. A new camera every call (see `spherical_camera`)."""
    h, w = rgb.shape[:2]
    with torch.no_grad():
        out = model.infer(rgb=torch.from_numpy(rgb).permute(2, 0, 1), camera=spherical_camera(w, h, vfov_half), normalize=True,
                          rays=None)
    return out["points"][0].norm(dim=0).detach().float().cpu().numpy()


class DepthWriter:
    """Writes UniK3D distance maps as Loc²'s uint16-millimetre PNGs, with the row-stripe guard.

    A map whose `depth_stripe_score` exceeds ``stripe_max`` is not written: it is reported, collected and listed by
    `finish` (and in ``rejects_json``), and generation continues. ``abort_after`` rejects in a row mean the camera /
    model is broken, not the images: RuntimeError. Files are written to ``<name>.png.tmp<pid>`` and renamed, so an
    existing PNG is always complete (a killed or requeued job resumes by skipping existing files, and two jobs writing
    the same panorama cannot leave a torn file)."""

    def __init__(self, stripe_max=None, abort_after: int = 20):
        from bevloc.data.vigor import STRIPE_MAX
        self.stripe_max = STRIPE_MAX if stripe_max is None else float(stripe_max)
        self.abort_after = int(abort_after)
        self.max_mm = NATIVE["depth_png_max_m"] * 1000.0
        self.n_written = 0
        self.rejects = []                                  # (name, score)
        self._in_a_row = 0

    def write(self, depth, path, name) -> bool:
        from bevloc.data.vigor import depth_stripe_score
        score = depth_stripe_score(depth)
        if not score <= self.stripe_max:                   # also catches NaN
            self.rejects.append((str(name), float(score)))
            self._in_a_row += 1
            print(f"  REJECT {name}: row-stripe score {score:.3g} > {self.stripe_max:.3g}, not written", flush=True)
            if self._in_a_row >= self.abort_after:
                raise RuntimeError(f"{self._in_a_row} striped depth maps in a row: camera reuse bug?")
            return False
        from PIL import Image  # noqa: WPS433
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")              # not *.png: globs skip it
        png = np.clip(np.asarray(depth, np.float32) * 1000.0, 0.0, self.max_mm).astype(np.uint16)
        Image.fromarray(png).save(tmp, format="PNG")
        os.replace(tmp, path)
        self.n_written += 1
        self._in_a_row = 0
        return True

    def finish(self, rejects_json=None) -> int:
        """Print the reject list (and write it to ``rejects_json`` when given, removing a stale one when there are
        none); returns the number of rejects."""
        print(f"depth maps written: {self.n_written}, rejected by the stripe guard: {len(self.rejects)}", flush=True)
        for name, score in self.rejects:
            print(f"  rejected {name}  score {score:.3g}", flush=True)
        if rejects_json is not None and not self.rejects:
            Path(rejects_json).unlink(missing_ok=True)      # a stale list from an earlier run of the same draw
        if rejects_json is not None and self.rejects:
            import json  # noqa: WPS433
            rejects_json = Path(rejects_json)
            rejects_json.parent.mkdir(parents=True, exist_ok=True)
            rejects_json.write_text(json.dumps(dict(stripe_max=self.stripe_max,
                                                    rejects=[dict(name=n, score=s) for n, s in self.rejects]),
                                               indent=1))
            print(f"wrote {rejects_json} (rerun with --stripe-max <higher> to force these after looking at them)",
                  flush=True)
        return len(self.rejects)


def depth_png_path(root, city: str, pano: str) -> Path:
    """Where our readers (VigorPairs, and Loc²'s dataloader through eval_loc2_vigor.py's override) find the depth of
    ``<root>/<city>/panorama/<pano>``: ``<root>/<city>/<DEPTH_DIR>/<stem>.png`` (bevloc.data.vigor.DEPTH_DIR)."""
    from bevloc.data.vigor import depth_png_path as _p
    return _p(root, city, pano)
