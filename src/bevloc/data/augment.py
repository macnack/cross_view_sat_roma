"""Training augmentation of a VIGOR pair (panorama + reference canvas), label-consistent (2026-09-30, regularisation).

Applied by `VigorPairs` to the finished sample, on the CPU inside the DataLoader workers, before any encoding; the
DINOv3 input normalisation (inside the encoder) is untouched. Off (``VigorPairs.aug = None``, the default) nothing
here runs and no random number is drawn: every sample is bit-identical to before.

Photometric (`PairAug.photometric` = p): independently for the query image and for the reference, with probability p
each: brightness, contrast and saturation factors drawn uniformly in [1 - a, 1 + a], a hue shift uniform in ±hue_deg,
then with probability ``blur_p`` a Gaussian blur of sigma uniform in [0.1, blur_sigma] px. The query image is the ERP
(or the IPM picture for query mode "ipm", only where it is valid). On the reference the black canvas (= no data, which
`ref_cell_validity` reads) stays exactly black: jitter and blur are applied, then every pixel that was black is black
again.

Geometric (`PairAug.geometric`, a subset of rot90 / rot / flip / shift, plus `rot_deg`), all label-consistent. The
known-orientation convention (bevloc.data.vigor): canvas up = north = the ERP's centre column, azimuth alpha clockwise
from north, alpha = (u / W - 0.5) * 360 deg for the continuous ERP column u (bevloc.model.erp_query.rays_at; pixel k
spans [k, k + 1)); the query's placement goes through R_w2c = R_NORTH and is never changed here.

* rotation by theta (counter-clockwise on the map) about the canvas centre c = (S - 1) / 2: canvas point p ->
  c + R(theta) (p - c) with R = [[cos, sin], [-sin, cos]] in image coordinates (y down), which is the ordinary
  counter-clockwise rotation in (east, north). A landmark at azimuth alpha from the camera is then at alpha - theta
  from the new north, so the new panorama at alpha' shows the old one at alpha' + theta: new[:, k] = old[:, k + s]
  (np.roll by -s) with s = theta / 360 * W columns. theta is therefore drawn as an integer number of ERP columns
  (0.40 deg at W = 896), so the roll is exact (no resampling of the panorama or its depth). A multiple of 90 deg is an
  exact np.rot90 of the canvas; the remainder (|delta| <= 45 deg) is one bilinear cv2.warpAffine about c, black
  outside. The 71 m tile on the 112 m canvas has a 100 m diagonal, so ANY angle keeps the whole tile on the canvas
  (coarse config); the black border is then no longer axis-aligned, and the rotated tile is resampled once.
  "rot90" draws k * 90 deg, k uniform in 0..3; "rot" draws theta uniform over the full circle (column-quantised);
  rot_deg D > 0 adds a small angle uniform in ±D (column-quantised) on top of either (or alone).
* mirror ("flip", probability 0.5): canvas u -> S - 1 - u (east <-> west); azimuth alpha -> -alpha, i.e. ERP column
  k -> W - 1 - k (np.flip; the centre column boundary W / 2 stays north).
* translation ("shift"): the canvas content moves by an integer (du, dv) px, uniform over the shifts that keep the
  (rotated, mirrored) tile footprint fully on the canvas (at most ``shift_max_m`` per axis if set); with it the
  camera moves by the same (du, dv). The positive-tile protocol puts the camera in the tile's central quarter and the
  tile at the canvas centre, a prior that also holds at test time: translation removes it (an ablation of its own).

Order: rotation, then mirror, then translation, on both the canvas and the panorama (+ depth). H stays a pure
translation BEV px -> canvas px (the query frame is unchanged, the camera just sits elsewhere on the canvas):
H' = [[1, 0, cam'_u - o], [0, 1, cam'_v - o]] with cam' the transformed camera pixel and o = (n - 1) / 2.
``en`` / ``ref_centre_en`` are re-expressed in the augmented canvas' frame (north = canvas up), keeping
`pose_en(H', ref_centre_en') == en'`: rotation and mirror act about the canvas centre (ref_centre_en fixed, en
rotated / mirrored about it); a shift keeps en and moves ref_centre_en by -(du, -dv) * cell_m. Nothing in training
reads en (the targets come from H alone), but the invariant keeps the sample self-consistent.

Window mode (the fine config: a 56 m window filling the whole 896 px canvas at 0.0625 m/px, already jittered around
the true position) has no room to shift and loses its corners under a non-right-angle rotation, so only rot90 and
flip are allowed there (`check_window`). Query mode "ipm" (a BEV picture built from the panorama) is refused with
any geometric transform: its picture would have to rotate instead of roll.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import torch

GEOMETRIC = ("rot90", "rot", "flip", "shift")


@dataclass
class PairAug:
    photometric: float = 0.0          # probability per image (query, reference independently)
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue_deg: float = 5.0
    blur_p: float = 0.2               # probability of a Gaussian blur after the jitter (of an image that was jittered)
    blur_sigma: float = 1.0           # max sigma (px)
    geometric: tuple = field(default_factory=tuple)   # subset of GEOMETRIC
    rot_deg: float = 0.0              # small-angle rotation, uniform in ±rot_deg (column-quantised)
    shift_max_m: float | None = None  # translation cap per axis (None = the whole slack)

    def __post_init__(self):
        g = tuple(str(x) for x in (self.geometric or ()))
        bad = [x for x in g if x not in GEOMETRIC]
        if bad:
            raise ValueError(f"unknown geometric augmentation(s) {bad}; choose from {GEOMETRIC}")
        if "rot" in g and "rot90" in g:
            raise ValueError("'rot' (full circle) already contains 'rot90'; give one of them")
        self.geometric = g
        if not 0.0 <= float(self.photometric) <= 1.0:
            raise ValueError(f"photometric probability {self.photometric} not in [0, 1]")

    @property
    def rotates(self):
        return "rot90" in self.geometric or "rot" in self.geometric or float(self.rot_deg) > 0

    @property
    def active(self):
        return float(self.photometric) > 0 or bool(self.geometric) or float(self.rot_deg) > 0

    def record(self):
        d = asdict(self)
        d["geometric"] = list(self.geometric)
        return d

    def check_window(self, window_mode: bool, query_mode: str):
        if window_mode and ("rot" in self.geometric or "shift" in self.geometric or float(self.rot_deg) > 0):
            raise ValueError("window reference (cfg.vigor.ref_window_m): only rot90 and flip are label-clean "
                             "(the 56 m window fills the canvas: no room to shift, a free angle cuts its corners)")
        if query_mode == "ipm" and (self.geometric or float(self.rot_deg) > 0):
            raise ValueError("geometric augmentation needs a panorama query (erp / erp_depth), not the IPM picture")


def from_config(cfg, photometric=None, geometric=None, rot_deg=None):
    """PairAug from cfg.train.aug_* (missing = off) with CLI overrides; None when everything is off."""
    T = getattr(cfg, "train", None)
    g = (lambda k, d: getattr(T, k, d) if T is not None else d)
    geo = g("aug_geometric", []) if geometric is None else geometric
    if isinstance(geo, str):
        geo = [x for x in geo.replace(",", " ").split() if x and x != "none"]
    aug = PairAug(photometric=float(g("aug_photometric", 0.0) if photometric is None else photometric),
                  brightness=float(g("aug_brightness", 0.2)), contrast=float(g("aug_contrast", 0.2)),
                  saturation=float(g("aug_saturation", 0.2)), hue_deg=float(g("aug_hue_deg", 5.0)),
                  blur_p=float(g("aug_blur_p", 0.2)), blur_sigma=float(g("aug_blur_sigma", 1.0)),
                  geometric=tuple(geo or ()), rot_deg=float(g("aug_rot_deg", 0.0) if rot_deg is None else rot_deg),
                  shift_max_m=g("aug_shift_max_m", None))
    return aug if aug.active else None


# ---- photometric ----------------------------------------------------------------------------------------------

def colour_jitter(img, rng, a: PairAug, keep=None):
    """img (H, W, 3) float32 RGB in [0, 1] -> jittered copy (clipped to [0, 1]); keep (H, W) bool: only these pixels
    change (the others are returned as they were; the contrast mean is taken over them)."""
    x = img.astype(np.float32, copy=True)
    m = np.ones(x.shape[:2], bool) if keep is None else keep
    if not m.any():
        return x
    b = rng.uniform(1 - a.brightness, 1 + a.brightness)
    c = rng.uniform(1 - a.contrast, 1 + a.contrast)
    s = rng.uniform(1 - a.saturation, 1 + a.saturation)
    h = rng.uniform(-a.hue_deg, a.hue_deg)
    y = x * b
    grey = y @ np.array([0.299, 0.587, 0.114], np.float32)
    y = (y - grey[m].mean()) * c + grey[m].mean()
    grey = y @ np.array([0.299, 0.587, 0.114], np.float32)
    y = (y - grey[..., None]) * s + grey[..., None]
    y = np.clip(y, 0.0, 1.0)
    if h:
        hsv = cv2.cvtColor(y, cv2.COLOR_RGB2HSV)                 # float32: H in degrees [0, 360)
        hsv[..., 0] = np.mod(hsv[..., 0] + h, 360.0)
        y = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if a.blur_p and rng.random() < a.blur_p:
        sig = rng.uniform(0.1, max(0.1, a.blur_sigma))
        y = cv2.GaussianBlur(y, (0, 0), sigmaX=sig, sigmaY=sig, borderType=cv2.BORDER_REFLECT)
    y = np.clip(y, 0.0, 1.0)
    out = x.copy()
    out[m] = y[m]
    return out


def _chw_to_hwc(t):
    return np.ascontiguousarray(t.numpy().transpose(1, 2, 0))


def _hwc_to_chw(a, like):
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).to(like.dtype)


def photometric(out, rng, a: PairAug):
    """In place on the sample dict: query image (erp, or bev where bev_valid) and reference, each with prob p."""
    if rng.random() < a.photometric:
        if "bev" in out:
            img = _chw_to_hwc(out["bev"])
            out["bev"] = _hwc_to_chw(colour_jitter(img, rng, a, keep=out["bev_valid"].numpy().astype(bool)), out["bev"])
        else:
            e = out["erp"]                                       # (T, 3, h, w)
            out["erp"] = torch.stack([_hwc_to_chw(colour_jitter(_chw_to_hwc(e[t]), rng, a), e[t])
                                      for t in range(e.shape[0])])
    if rng.random() < a.photometric:
        ref = _chw_to_hwc(out["ref"])
        data = ref.sum(-1) > 0                                   # the black canvas stays exactly black
        out["ref"] = _hwc_to_chw(colour_jitter(ref, rng, a, keep=data), out["ref"])
    return out


# ---- geometric ------------------------------------------------------------------------------------------------

def rot_matrix_img(theta_deg):
    """2x2 counter-clockwise (on the map) rotation in image coordinates (x right, y down)."""
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])


def rot_matrix_en(theta_deg):
    """The same rotation in (east, north) coordinates: the standard counter-clockwise matrix."""
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def rotate_canvas(img, k, delta_deg):
    """(S, S, C) canvas rotated counter-clockwise by k * 90 + delta_deg about its centre ((S - 1) / 2): np.rot90 (exact)
    for the quarter turns, then one bilinear warp for delta (black outside)."""
    out = np.rot90(img, int(k) % 4, axes=(0, 1)) if int(k) % 4 else img
    if delta_deg:
        S = out.shape[0]
        c = (S - 1) / 2.0
        M = cv2.getRotationMatrix2D((c, c), float(delta_deg), 1.0)   # + = counter-clockwise, pixel centres at integers
        out = cv2.warpAffine(np.ascontiguousarray(out), M, (out.shape[1], S), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if out.ndim == 2 and img.ndim == 3:
            out = out[..., None]
    return np.ascontiguousarray(out)


def shift_canvas(img, du, dv):
    """Content moved by integer (du, dv) px (right, down), zeros where nothing moves in."""
    out = np.zeros_like(img)
    S0, S1 = img.shape[:2]
    du, dv = int(du), int(dv)
    ys, yd = (slice(0, S0 - dv), slice(dv, S0)) if dv >= 0 else (slice(-dv, S0), slice(0, S0 + dv))
    xs, xd = (slice(0, S1 - du), slice(du, S1)) if du >= 0 else (slice(-du, S1), slice(0, S1 + du))
    out[yd, xd] = img[ys, xs]
    return out


def transform_points(p, S, theta_deg=0.0, flip=False, shift=(0, 0)):
    """Canvas px (..., 2) through rotation (about (S - 1) / 2), mirror, shift: what the canvas content undergoes."""
    c = (S - 1) / 2.0
    p = np.asarray(p, np.float64)
    q = (p - c) @ rot_matrix_img(theta_deg).T + c
    if flip:
        q = q.copy()
        q[..., 0] = S - 1 - q[..., 0]
    return q + np.asarray(shift, np.float64)


def draw_geometry(rng, a: PairAug, W: int):
    """(s columns, k quarter turns, delta_deg, theta_deg, flip): theta = s * 360 / W exactly, = k * 90 + delta."""
    if W % 4:
        raise ValueError(f"ERP width {W} not divisible by 4: a quarter turn is not a whole number of columns")
    q = W // 4
    s = 0
    if "rot" in a.geometric:
        s = int(rng.integers(0, W))
    elif "rot90" in a.geometric:
        s = int(rng.integers(0, 4)) * q
    if float(a.rot_deg) > 0:
        m = int(math.floor(float(a.rot_deg) / 360.0 * W))
        s += int(rng.integers(-m, m + 1)) if m > 0 else 0
    s %= W
    k = int(round(s / q)) % 4
    ds = s - k * q
    if ds > W // 2:
        ds -= W
    elif ds < -W // 2:
        ds += W
    delta = ds * 360.0 / W
    flip = "flip" in a.geometric and bool(rng.random() < 0.5)
    return s, k, delta, s * 360.0 / W, flip


def geometric(out, rng, a: PairAug, n: int, cell_m: float, footprint=None):
    """In place on the sample dict (erp (T, 3, h, w), optional depth (1, h, w), ref (3, S, S), H, en, ref_centre_en).
    footprint: (x_min, y_min, x_max, y_max) canvas px edges of the tile (for the shift range); None = the canvas.
    Returns a dict of what was drawn (for tests and logging)."""
    erp = out["erp"]
    W = int(erp.shape[-1])
    s, k, delta, theta, flip = draw_geometry(rng, a, W)
    ref = _chw_to_hwc(out["ref"])
    S = ref.shape[0]
    o = (n - 1) / 2.0
    Hm = out["H"].numpy().astype(np.float64)
    cam = (Hm @ np.array([o, o, 1.0]))[:2] / (Hm @ np.array([o, o, 1.0]))[2]
    if s or delta:
        ref = rotate_canvas(ref, k, delta)
    if flip:
        ref = np.ascontiguousarray(ref[:, ::-1])
    du = dv = 0
    if "shift" in a.geometric:
        fp = footprint if footprint is not None else (-0.5, -0.5, S - 0.5, S - 0.5)
        corners = np.array([[fp[0], fp[1]], [fp[2], fp[1]], [fp[2], fp[3]], [fp[0], fp[3]]], np.float64)
        cc = transform_points(corners, S, theta, flip)
        lo = np.floor(cc.min(0)) - (1 if delta else 0)           # one px margin for the bilinear edge
        hi = np.ceil(cc.max(0)) + (1 if delta else 0)
        rmin, rmax = -0.5 - lo, (S - 0.5) - hi                   # allowed shift range per axis
        cap = None if a.shift_max_m is None else float(a.shift_max_m) / float(cell_m)
        sh = []
        for i in range(2):
            a0, a1 = math.ceil(rmin[i]), math.floor(rmax[i])
            if cap is not None:
                a0, a1 = max(a0, -int(cap)), min(a1, int(cap))
            sh.append(int(rng.integers(a0, a1 + 1)) if a1 >= a0 else 0)
        du, dv = sh
        if du or dv:
            ref = shift_canvas(ref, du, dv)
    out["ref"] = _hwc_to_chw(ref, out["ref"])

    def pano_op(t):                                              # (..., w): roll by -s, then mirror
        if s:
            t = torch.roll(t, shifts=-s, dims=-1)
        if flip:
            t = torch.flip(t, dims=(-1,))
        return t.contiguous()
    out["erp"] = pano_op(erp)
    if "depth" in out:
        if int(out["depth"].shape[-1]) != W:
            raise ValueError("depth and ERP widths differ: the azimuth roll would not be the same on both")
        out["depth"] = pano_op(out["depth"])
    cam2 = transform_points(cam, S, theta, flip, (du, dv))
    H2 = np.array([[1.0, 0.0, cam2[0] - o], [0.0, 1.0, cam2[1] - o], [0.0, 0.0, 1.0]])
    out["H"] = torch.from_numpy(H2.astype(np.float32))
    rc = out["ref_centre_en"].numpy().astype(np.float64)
    en = out["en"].numpy().astype(np.float64)
    en2 = rc + rot_matrix_en(theta) @ (en - rc)
    if flip:
        en2[0] = rc[0] - (en2[0] - rc[0])
    rc2 = rc - np.array([du, -dv], np.float64) * float(cell_m)
    out["en"] = torch.from_numpy(en2).to(out["en"].dtype)
    out["ref_centre_en"] = torch.from_numpy(rc2).to(out["ref_centre_en"].dtype)
    return dict(s=s, k=k, delta=delta, theta=theta, flip=flip, shift=(du, dv))


def apply(out, a: PairAug, n: int, cell_m: float, footprint=None, rng=None):
    """Photometric then geometric augmentation of one sample (in place, returned). rng None = a numpy Generator
    seeded from ONE torch draw (per-worker seeded by the DataLoader; the main-process RNG with workers 0)."""
    if rng is None:
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
    if float(a.photometric) > 0:
        photometric(out, rng, a)
    if a.geometric or float(a.rot_deg) > 0:
        out["_aug"] = geometric(out, rng, a, n, cell_m, footprint)
    return out
