"""UniK3D depth generation: the camera handed to UniK3D.infer must be fresh on every call (v1 depth bug), and the row
stripe detector separates correct ERP depth from the striped maps that bug produced.

The model is a stub that keeps UniK3D's real `infer` (padding, resize, the in-place camera crop / resize, the
camera rays, the post-processing) and replaces only the network: distance = 2 + ray_y, so the output depends on the
camera exactly as the real decoder's ray conditioning does, without weights."""
import math

import numpy as np
import pytest
import torch

from bevloc.baselines import loc2 as loc2_wrap
from bevloc.data.vigor import DEPTH_DIR, STRIPE_MAX_M, depth_png_path, depth_stripe_score

pytestmark = pytest.mark.skipif(not (loc2_wrap.UNIK3D_ROOT / "unik3d" / "models").is_dir(),
                                reason="third_party/Loc2/external/unik3d not present")


def _stub_model():
    loc2_wrap.ensure_loc2_on_path()
    from unik3d.models import UniK3D  # noqa: WPS433

    class Stub:
        infer = UniK3D.infer
        shape_constraints = {"ratio_bounds": [0.5, 2.5], "pixels_min": 200000.0, "pixels_max": 600000.0}
        resolution_level = 0
        interpolation_mode = "bilinear"
        device = torch.device("cpu")

        def encode_decode(self, inputs, image_metas):
            rays = inputs["rays"]                                   # (B, 3, h, w) unit rays of the given camera
            dist = 2.0 + rays[:, 1:2]
            return None, {"points": rays * dist, "rays": rays, "confidence": dist, "lowres_features": None}

    return Stub()


def _expected(h, w):
    """2 + sin(latitude) of UniK3D's Spherical camera (y down) at each row, for the image size."""
    v = np.arange(h, dtype=np.float64)
    lat = (v - (h - 1) / 2) / (h - 1) * math.pi
    return np.repeat((2.0 + np.sin(lat))[:, None], w, axis=1)


def test_infer_distance_is_identical_across_calls():
    model = _stub_model()
    rgb = np.random.default_rng(0).integers(0, 255, (64, 128, 3), dtype=np.uint8)
    outs = [loc2_wrap.infer_distance(model, rgb) for _ in range(6)]
    for d in outs:
        np.testing.assert_allclose(d, outs[0], atol=1e-6)
    # interior rows; 0.05: half-pixel offsets of the resampled grid (a mutated camera is off by O(1))
    np.testing.assert_allclose(outs[-1][2:-2], _expected(64, 128)[2:-2], atol=0.05)
    assert depth_stripe_score(outs[-1]) < STRIPE_MAX_M


def test_reused_camera_is_mutated_by_unik3d_infer():
    """The v1 bug: one Spherical reused across calls (scripts/loc2_depth_vigor.py's per-size cache,
    scripts/unik3d_depth_poznan.py's single camera) is rescaled in place on every call."""
    model = _stub_model()
    rgb = np.zeros((64, 128, 3), np.uint8)
    cam = loc2_wrap.spherical_camera(128, 64)
    before = cam.params.clone()
    with torch.no_grad():
        model.infer(rgb=torch.from_numpy(rgb).permute(2, 0, 1), camera=cam, normalize=True, rays=None)
    assert not torch.allclose(cam.params.cpu().flatten(), before.flatten()), \
        "UniK3D no longer mutates the camera: the fresh-camera rule is still right, but update this test"


def test_stripe_score_separates_flat_ground_from_row_alternation():
    h, w = 1024, 2048
    el = (np.arange(h) + 0.5 - h / 2) / h * math.pi                 # > 0 below the horizon
    ground = np.where(el > 0.02, 2.5 / np.sin(np.clip(el, 0.02, None)), 65.0)
    d = np.repeat(np.minimum(ground, 65.0)[:, None], w, axis=1)
    assert depth_stripe_score(d) < 0.01
    striped = d * (1.0 + 0.8 * np.sin(np.arange(h) * 2 * math.pi * 232 / 512))[:, None]   # the v1 PNGs' period
    assert depth_stripe_score(striped) > 1.0


@pytest.mark.skipif(bool(__import__("os").environ.get("BEVLOC_DEPTH_DIR")), reason="BEVLOC_DEPTH_DIR overrides")
def test_depth_folder_is_versioned():
    assert DEPTH_DIR == "unik3d_depth_v2"
    assert depth_png_path("/r", "Chicago", "p.jpg").parent.name == "unik3d_depth_v2"
