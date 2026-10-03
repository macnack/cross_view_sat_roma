"""Review of the geometric pair augmentation (2026-09-30), independent of tests/test_regularisation.py: the check goes
through the TRAINING path of the erp_depth query, token by token, and never uses bevloc.data.augment's own geometry.

A planted world: five landmarks (discs on the tile, each with its own red level) seen in the panorama as bands of the
same colour, with their depth along the ray. For every depth-valid token of the augmented sample, `ErpDepthQuery
.placement` (token-level placement, as in training) -> H (the label) -> the canvas pixel must show the landmark whose
colour the token sees in the augmented panorama; and `coarse_targets` / `ref_cell_validity` on the augmented batch
never select a black (off-tile, e.g. rotated-corner) cell. A wrong roll sign, a missing mirror of either side or a
shift not applied to H sends the tokens to the wrong disc or to background, and the match fraction collapses."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from bevloc import config as C
from bevloc.data import augment as AUG
from bevloc.data.vigor import CITY_RES, DEPTH_DIR, VigorPairs, collate_vigor
from bevloc.model.coarse import coarse_targets, ref_cell_validity
from bevloc.model.depth_query import ErpDepthQuery
from vigor_labels import corrected  # noqa: E402

ERP_W, ERP_H = 896, 448
DY, DX = -30.0, 50.0                  # camera 30 tile px north, 50 px west of the tile centre
# landmarks (east, north) metres from the camera; red level = identity
LMS = [((9.0, 3.0), 100), ((-4.0, 12.0), 140), ((-11.0, -7.0), 180), ((5.0, -14.0), 220), ((0.5, 20.0), 250)]
R_M = 1.6                             # disc radius on the ground


def _scene(tmp_path, city="Chicago"):
    res = CITY_RES[city]
    for d in ("panorama", "satellite", DEPTH_DIR):
        (tmp_path / city / d).mkdir(parents=True)
    c = (640 - 1) / 2.0
    cam = np.array([c - DX, c + DY])                                 # dx > 0 = west (col_sign -1), dy > 0 = south
    sat = np.full((640, 640, 3), 60, np.uint8)                       # BGR grey
    pano = np.full((ERP_H, ERP_W, 3), 128, np.uint8)
    dep = np.full((ERP_H, ERP_W), 60000, np.uint16)                  # 60 m: invalid
    for (e, n), red in LMS:
        p = cam + np.array([e, -n]) / res                            # tile px: x east, y south
        cv2.circle(sat, (int(round(p[0])), int(round(p[1]))), int(R_M / res), (0, 0, red), -1)
        # azimuth clockwise from north, INDEPENDENT derivation: a panorama seen from inside, north at the centre
        # column, east a quarter turn to the right
        alpha = np.degrees(np.arctan2(e, n))
        d = float(np.hypot(e, n))
        half = 0.5 * np.degrees(np.arcsin(R_M / d))                  # half the disc's angular half-width
        u0 = (alpha / 360.0 + 0.5) * ERP_W
        du = half / 360.0 * ERP_W
        cols = np.arange(int(np.floor(u0 - du)), int(np.ceil(u0 + du))) % ERP_W
        pano[:, cols] = (0, 0, red)
        dep[ERP_H // 2 - 16:ERP_H // 2 + 16, cols] = int(round(d * 1000))   # token rows 13, 14 (|elev| 3 deg)
    cv2.imwrite(str(tmp_path / city / "satellite" / "s1.png"), sat)
    cv2.imwrite(str(tmp_path / city / "panorama" / "p1,1.0,.png"), pano)
    cv2.imwrite(str(tmp_path / city / DEPTH_DIR / "p1,1.0,.png"), dep)
    lab = tmp_path / "splits" / "VIGOR" / city
    lab.mkdir(parents=True)
    (lab / "satellite_list.txt").write_text("s1.png\n")
    line = f"p1,1.0,.png s1.png {DY} {DX} s1.png 0 0 s1.png 0 0 s1.png 0 0\n"
    for f in ("pano_label_balanced.txt", "same_area_balanced_test.txt", "same_area_balanced_train.txt"):
        (lab / f).write_text(line)
    corrected(lab)
    return tmp_path


def _cfg():
    cfg = C.load(str(C.REPO / "configs/vigor_cell0125.yaml"))
    cfg.lift.query_mode = "erp_depth"
    cfg.erp_depth.erp_size = [ERP_W, ERP_H]
    cfg.erp_depth.head = False
    return cfg


def _token_check(s, cfg):
    """(fraction of valid tokens whose canvas target shows the token's own landmark colour, n tokens, per-landmark
    hits); also asserts the targets never select black cells."""
    b = collate_vigor([s])
    q = ErpDepthQuery(cfg)
    xy, valid = q.placement(b)                                       # (1, h, w, 2) BEV px, (1, h, w)
    Hm = b["H"][0].double()
    p = torch.cat([xy[0].double(), torch.ones_like(xy[0][..., :1]).double()], -1) @ Hm.T
    uv = (p[..., :2] / p[..., 2:]).numpy()
    ref = (b["ref"][0].numpy() * 255.0).round()
    erp = (b["erp"][0, 0].numpy() * 255.0).round()
    h, w = valid.shape[1:]
    hits, n, per = 0, 0, {}
    for i, j in zip(*np.nonzero(valid[0].numpy())):
        tok = erp[:, int((i + 0.5) * 16), int((j + 0.5) * 16)]
        assert tok[1] == 0 and tok[0] > 90, tok                     # a valid token looks at a landmark band
        u, v = int(round(uv[i, j, 0])), int(round(uv[i, j, 1]))
        n += 1
        if 0 <= u < ref.shape[2] and 0 <= v < ref.shape[1]:
            got = ref[:, v, u]
            ok = abs(got[0] - tok[0]) <= 12 and got[1] <= 12
            hits += ok
            per[int(tok[0])] = per.get(int(tok[0]), 0) + int(ok)
    rv = ref_cell_validity(b["ref"], cells=56, min_frac=cfg.train.min_ref_cell_valid)
    idx, use = coarse_targets(b["H"], valid, ref_valid=rv, ref_size=b["ref"].shape[-1], cells=56, query_xy=xy)
    assert bool(rv[0].flatten()[idx[use]].all())
    black = (b["ref"][0].sum(0) == 0).float()
    frac_black = torch.nn.functional.avg_pool2d(black[None, None], 16)[0, 0].flatten()
    fb = frac_black[idx[use]]
    assert fb.numel() == 0 or float(fb.max()) <= 1.0 - cfg.train.min_ref_cell_valid + 1e-6
    return hits / max(n, 1), n, per


def test_scene_tokens_hit_their_landmarks_unaugmented(tmp_path):
    cfg = _cfg()
    ds = VigorPairs(_scene(tmp_path), cfg, cities=["Chicago"], split="samearea", train=True)
    frac, n, per = _token_check(ds[0], cfg)
    assert n >= 10 and frac >= 0.95 and len(per) == len(LMS) and all(v > 0 for v in per.values()), (frac, n, per)


@pytest.mark.parametrize("geo,rot_deg", [(("rot",), 0.0), (("rot90", "flip"), 10.0), (("rot", "flip", "shift"), 0.0),
                                         (("flip", "shift"), 0.0)])
def test_augmented_tokens_hit_their_landmarks(tmp_path, geo, rot_deg):
    cfg = _cfg()
    ds = VigorPairs(_scene(tmp_path), cfg, cities=["Chicago"], split="samearea", train=True)
    ds.set_aug(AUG.PairAug(geometric=geo, rot_deg=rot_deg))
    torch.manual_seed(7)
    flips = set()
    for _ in range(10):
        s = ds[0]
        flips.add(s["_aug"]["flip"])
        frac, n, per = _token_check(s, cfg)
        assert n >= 10 and frac >= 0.95, (s["_aug"], frac, n, per)
        # every landmark some token sees is hit (the 20 m one's band is narrower than a token and may fall between
        # two token centres after the roll)
        assert len(per) >= len(LMS) - 1 and all(v > 0 for v in per.values()), (s["_aug"], per)
    if "flip" in geo:
        assert flips == {True, False}


def test_a_wrong_roll_sign_is_detected(tmp_path, monkeypatch):
    """The check has teeth: rolling the panorama the other way (+s) sends the tokens off their landmarks."""
    cfg = _cfg()
    ds = VigorPairs(_scene(tmp_path), cfg, cities=["Chicago"], split="samearea", train=True)
    ds.set_aug(AUG.PairAug(geometric=("rot90",)))
    real_roll = torch.roll
    monkeypatch.setattr(AUG.torch, "roll", lambda t, shifts, dims: real_roll(t, shifts=-shifts, dims=dims))
    torch.manual_seed(0)
    fracs = []
    for _ in range(8):
        s = ds[0]
        if s["_aug"]["k"] in (1, 3):
            fracs.append(_token_check(s, cfg)[0])
    assert fracs and max(fracs) < 0.5, fracs


def test_photometric_keeps_the_no_data_mask_of_a_dark_reference_exactly():
    """Dark shadows under contrast > 1 clip to (0, 0, 0); on the reference that would read as NO DATA."""
    g = np.random.default_rng(2)
    ref = np.zeros((64, 64, 3), np.float32)                          # black canvas border = no data
    ref[8:56, 8:56] = g.uniform(0.004, 0.08, (48, 48, 3))            # a shadowed tile
    ref[8:56, 32:56] = g.uniform(0.3, 0.9, (48, 24, 3))              # and a bright half (contrast mean up)
    out = dict(ref=torch.from_numpy(ref.transpose(2, 0, 1).copy()), erp=torch.rand(1, 3, 8, 16))
    a = AUG.PairAug(photometric=1.0, brightness=0.3, contrast=0.4)
    data0 = out["ref"].sum(0) > 0
    rng = np.random.default_rng(0)
    for _ in range(30):
        o = AUG.photometric({k: v.clone() for k, v in out.items()}, rng, a)
        assert torch.equal(o["ref"].sum(0) > 0, data0)
