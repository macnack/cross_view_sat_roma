"""Pose-correctness head (task 06, step 4): P(coarse pose error < tau) from the frozen matcher's own evidence.

Inputs per frame, all produced once by scripts/certainty_cache_vigor.py from the frozen matcher (docs/tasks/06_certainty.md,
"Step 4"):
  maps        (4, K, K) float, K = 56 reference cells, in MAP_KEYS order: the ego (Hough) vote map, the token vote
              map, the pose footprint (+1 inlier token / -1 other valid token / 0, rasterised at the cell the coarse
              H sends each token to) and the reference-cell validity (1 on the tile, 0 on the black canvas).
  tokens      (T, 8) float in TOKEN_KEYS order, one row per query token (T = h * w of the token grid, 56 x 28 = 1568
              for the panorama query): the token's categorical mass at the cell H sends it to, its top-1 mass, its
              normalised entropy, the decoder's certainty logit, the distance (cells) of its peak from that
              pose-consistent cell, its placed range (m), and its grid position as normalised (azimuth column, row),
              both (index + 0.5) / size so the grid centre is 0.5 and the D4 augmentation is exact.
  token_valid (T,) bool: the tokens that voted (invalid rows are zero and ignored).
  frame       (F,) float, the frame statistics of scripts/certainty_vigor.py (F = 19: the RANSAC inlier ratio and
              `bevloc.match.vote_stats.STAT_KEYS`), raw (None -> NaN); the head standardises them itself.

Streams (each optional, `streams`): map = conv(32, s2) > conv(64, s2) > conv(64, s2) > global average pool (64);
tokens = per-token MLP 8 > 64 > 64, masked attention pooling with a learned query (64) + mean and max over the valid
tokens of the pose-consistency mass (2); frame = the standardised statistics > 32. Fusion: concat > 128 > n_targets
logits, one per tau of cfg.certainty.targets_m. Under 1 M parameters (about 0.1 M with the defaults).

Standardisation of the frame statistics is a set of buffers (`set_standardisation`, from the training frames: the
log1p transform of certainty_vigor.FEATURES, median imputation of missing values, mean / std), and so is the
per-target temperature of `probabilities` (fitted on the validation frames by scripts/train_certainty_head.py);
both travel in the state dict.

Augmentation (`augment`): the label "error < tau" does not depend on the world's orientation, so the four maps may
be rotated by multiples of 90 degrees and mirrored (the dihedral group of the square, 8 elements) as long as the
token rows follow. Maps are in the reference frame (rows south-down, columns east-right, the vehicle at the
centre); a counter-clockwise quarter turn of that picture moves what was east to north, i.e. every azimuth drops
by 90 degrees, so the panorama's azimuth column shifts by a quarter of its width (`grid` "erp"); a left-right mirror
(east <-> west) sends azimuth alpha to -alpha, i.e. column c to w - 1 - c. For a picture query ("bev" grid: the
token rows are BEV grid coordinates, `token_centres`) the (column, row) pair rotates and mirrors with the map. The
other token features (masses, entropy, certainty, distances) are rotation invariant by construction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STREAMS = ("map", "tokens", "frame")
MAP_KEYS = ("ego", "vote", "footprint", "ref_valid")
TOKEN_KEYS = ("p_pose", "top1", "entropy", "cert", "peak_pose_cells", "depth_m", "azimuth", "row")
GRIDS = ("erp", "bev")
P_POSE, AZIMUTH, ROW = TOKEN_KEYS.index("p_pose"), TOKEN_KEYS.index("azimuth"), TOKEN_KEYS.index("row")


def canonical_streams(streams):
    """The streams in STREAMS order; unknown names or an empty set are refused."""
    s = tuple(x for x in STREAMS if x in set(streams))
    unknown = set(streams) - set(STREAMS)
    if unknown:
        raise ValueError(f"unknown streams {sorted(unknown)}; choose from {STREAMS}")
    if not s:
        raise ValueError("at least one stream is needed")
    return s


def streams_tag(streams):
    """File-name tag of a stream set: 'map-tokens-frame', 'frame', ..."""
    return "-".join(canonical_streams(streams))


class CertaintyHead(nn.Module):
    def __init__(self, streams=STREAMS, n_targets=3, n_frame=19, n_token=len(TOKEN_KEYS), map_channels=(32, 64, 64),
                 token_dim=64, frame_dim=32, fusion_dim=128, dropout=0.0):
        super().__init__()
        self.streams = canonical_streams(streams)
        self.n_targets, self.n_frame, self.n_token = int(n_targets), int(n_frame), int(n_token)
        self.map_channels, self.token_dim, self.frame_dim, self.fusion_dim, self.dropout = \
            tuple(int(c) for c in map_channels), int(token_dim), int(frame_dim), int(fusion_dim), float(dropout)
        d = 0
        if "map" in self.streams:
            layers, c_in = [], len(MAP_KEYS)
            for c in self.map_channels:
                layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.GELU()]
                c_in = c
            self.map_net = nn.Sequential(*layers)
            d += c_in
        if "tokens" in self.streams:
            self.token_mlp = nn.Sequential(nn.Linear(self.n_token, self.token_dim), nn.GELU(),
                                           nn.Linear(self.token_dim, self.token_dim), nn.GELU())
            self.token_key = nn.Linear(self.token_dim, self.token_dim, bias=False)
            self.token_query = nn.Parameter(torch.randn(self.token_dim) / self.token_dim ** 0.5)
            d += self.token_dim + 2
        if "frame" in self.streams:
            self.frame_net = nn.Sequential(nn.Linear(self.n_frame, self.frame_dim), nn.GELU())
            d += self.frame_dim
        self.fusion = nn.Sequential(nn.Linear(d, self.fusion_dim), nn.GELU(), nn.Dropout(self.dropout),
                                    nn.Linear(self.fusion_dim, self.n_targets))
        # frame standardisation (set_standardisation) and the per-target temperature (probabilities)
        self.register_buffer("frame_log", torch.zeros(self.n_frame, dtype=torch.bool))
        self.register_buffer("frame_fill", torch.zeros(self.n_frame))
        self.register_buffer("frame_mean", torch.zeros(self.n_frame))
        self.register_buffer("frame_std", torch.ones(self.n_frame))
        self.register_buffer("temperature", torch.ones(self.n_targets))

    def config(self):
        """The constructor arguments (a checkpoint stores them next to the state dict)."""
        return dict(streams=list(self.streams), n_targets=self.n_targets, n_frame=self.n_frame, n_token=self.n_token,
                    map_channels=list(self.map_channels), token_dim=self.token_dim, frame_dim=self.frame_dim,
                    fusion_dim=self.fusion_dim, dropout=self.dropout)

    # ---- frame statistics ----------------------------------------------------------------------------------------

    def _transform_frame(self, x):
        """log1p(max(x, 0)) on the log-transformed columns (as certainty_vigor.matrix), NaN kept."""
        x = x.float()
        return torch.where(self.frame_log[None], torch.log1p(x.clamp_min(0.0)), x)

    @torch.no_grad()
    def set_standardisation(self, frame, log_mask):
        """frame (N, n_frame) raw statistics of the TRAINING frames (NaN = missing); log_mask (n_frame,) bool = the
        log1p columns. Sets the imputation value (median of the finite values, 0 without any), mean and std (1 where
        the column is constant)."""
        x = torch.as_tensor(np.asarray(frame, np.float64), dtype=torch.float32)
        self.frame_log.copy_(torch.as_tensor(np.asarray(log_mask, bool)))
        t = self._transform_frame(x)
        fill = torch.zeros(self.n_frame)
        for j in range(self.n_frame):
            col = t[:, j]
            fin = col[torch.isfinite(col)]
            fill[j] = fin.median() if len(fin) else 0.0
        t = torch.where(torch.isfinite(t), t, fill[None])
        std = t.std(0, unbiased=False) if len(t) > 1 else torch.ones(self.n_frame)
        self.frame_fill.copy_(fill)
        self.frame_mean.copy_(t.mean(0) if len(t) else torch.zeros(self.n_frame))
        self.frame_std.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))
        return self

    def prepare_frame(self, frame):
        """(B, n_frame) raw statistics -> standardised (transform, imputation, mean / std)."""
        t = self._transform_frame(torch.as_tensor(frame))
        t = torch.where(torch.isfinite(t), t, self.frame_fill[None])
        return (t - self.frame_mean[None]) / self.frame_std[None]

    # ---- forward ------------------------------------------------------------------------------------------------

    def forward(self, maps=None, tokens=None, token_valid=None, frame=None):
        """maps (B, 4, K, K), tokens (B, T, 8), token_valid (B, T) bool, frame (B, n_frame) raw statistics; inputs
        of absent streams may be None. Returns the logits (B, n_targets), uncalibrated."""
        parts = []
        if "map" in self.streams:
            parts.append(self.map_net(maps.float()).mean(dim=(-2, -1)))
        if "tokens" in self.streams:
            x = tokens.float()
            v = token_valid.bool()
            h = self.token_mlp(x)                                              # (B, T, d)
            s = (self.token_key(h) @ self.token_query) / self.token_dim ** 0.5  # (B, T)
            a = torch.softmax(s.masked_fill(~v, -1e4), dim=1) * v.float()
            a = a / a.sum(1, keepdim=True).clamp_min(1e-6)                     # no valid token: pooled = 0
            pooled = (a[..., None] * h).sum(1)
            p = torch.where(v, x[..., P_POSE], torch.zeros_like(x[..., P_POSE]))
            n = v.float().sum(1).clamp_min(1.0)
            parts += [pooled, p.sum(1, keepdim=True) / n[:, None], p.max(1, keepdim=True).values]
        if "frame" in self.streams:
            parts.append(self.frame_net(self.prepare_frame(frame)))
        return self.fusion(torch.cat(parts, dim=-1))

    def probabilities(self, *args, **kw):
        """P(error < tau) per target: sigmoid of the temperature-scaled logits, (B, n_targets)."""
        return torch.sigmoid(self(*args, **kw) / self.temperature[None])


def n_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


# ---- augmentation --------------------------------------------------------------------------------------------------

def augment_tokens(tokens, k, flip, grid):
    """The token rows of one D4 element: mirror left-right first (flip), then k counter-clockwise quarter turns, the
    order `augment_maps` applies to the maps. tokens (T, 8) or (B, T, 8); returns a copy."""
    if grid not in GRIDS:
        raise ValueError(f"grid must be one of {GRIDS}, got {grid!r}")
    t = tokens.clone()
    a, r = t[..., AZIMUTH], t[..., ROW]
    if grid == "erp":
        if flip:
            a = 1.0 - a                                              # column c -> w - 1 - c: azimuth alpha -> -alpha
        a = torch.remainder(a - 0.25 * int(k), 1.0)                  # a quarter turn CCW: azimuth - 90 degrees
    else:                                                            # BEV grid coordinates (u = column, v = row)
        if flip:
            a = 1.0 - a
        for _ in range(int(k) % 4):
            a, r = r, 1.0 - a                                        # rot90 CCW: (row, col) -> (w - 1 - col, row)
    t[..., AZIMUTH], t[..., ROW] = a, r
    return t


def augment_maps(maps, k, flip):
    """maps (..., K, K): mirror left-right (flip), then k counter-clockwise quarter turns (torch.rot90)."""
    m = torch.flip(maps, dims=(-1,)) if flip else maps
    return torch.rot90(m, int(k) % 4, dims=(-2, -1))


def augment(maps, tokens, k, flip, grid):
    """One D4 element per sample: maps (B, 4, K, K), tokens (B, T, 8), k (B,) ints in 0..3, flip (B,) bool.
    Returns (maps', tokens') with the token grid positions moved consistently (module doc)."""
    k = torch.as_tensor(k).long().reshape(-1) % 4
    flip = torch.as_tensor(flip).bool().reshape(-1)
    out_m, out_t = maps.clone(), tokens.clone()
    for kk in range(4):
        for ff in (False, True):
            sel = (k == kk) & (flip == ff)
            if sel.any():
                out_m[sel] = augment_maps(maps[sel], kk, ff)
                out_t[sel] = augment_tokens(tokens[sel], kk, ff, grid)
    return out_m, out_t


# ---- cache arrays --------------------------------------------------------------------------------------------------

class CacheArrays:
    """The head's inputs stacked from a certainty cache (scripts/certainty_cache_vigor.py): maps (N, 4, K, K) float16,
    tokens (N, T, 8) float16, token_valid (N, T) bool, frame (N, F) float32 (NaN = missing), err (N,) float64 (inf =
    no pose), ids, cities, plus the header: frame_keys, frame_log (bool per key), grid ('erp' | 'bev'), meta."""

    def __init__(self, cache):
        meta, frames = cache["meta"], cache["frames"]
        self.meta = meta
        self.frame_keys = list(meta["frame_keys"])
        self.frame_log = np.array([bool(x) for x in meta["frame_log"]], bool)
        self.grid = str(meta["token_grid"])
        self.ids = [f["id"] for f in frames]
        self.cities = [f["city"] for f in frames]
        self.maps = np.stack([f["maps"] for f in frames]).astype(np.float16) if frames else np.zeros((0, 4, 1, 1), np.float16)
        self.tokens = np.stack([f["tokens"] for f in frames]).astype(np.float16) if frames else np.zeros((0, 1, 8), np.float16)
        self.token_valid = np.stack([f["token_valid"] for f in frames]).astype(bool) if frames else np.zeros((0, 1), bool)
        self.frame = np.array([[np.nan if f["stats"].get(k) is None else float(f["stats"][k]) for k in self.frame_keys]
                               for f in frames], np.float32).reshape(len(frames), len(self.frame_keys))
        self.err = np.array([np.inf if f["pose_peak_m"] is None else float(f["pose_peak_m"]) for f in frames], np.float64)

    def __len__(self):
        return len(self.ids)

    def labels(self, targets_m):
        """(N, len(targets)) float: 1 where the coarse error is below tau (no pose = 0 for every tau)."""
        return np.stack([(self.err < float(t)).astype(np.float32) for t in targets_m], 1)

    def subset(self, idx):
        import copy
        s = copy.copy(self)
        idx = np.asarray(idx)
        s.ids, s.cities = [self.ids[i] for i in idx], [self.cities[i] for i in idx]
        for k in ("maps", "tokens", "token_valid", "frame", "err"):
            setattr(s, k, getattr(self, k)[idx])
        return s


def load_cache(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# ---- checkpoints -----------------------------------------------------------------------------------------------------

def save_head(path, head, extra=None):
    """state dict (weights, standardisation, temperature) + constructor config + whatever the trainer records."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=head.state_dict(), config=head.config(), **(extra or {})), path)
    return path


def load_head(path, device="cpu"):
    """(head in eval mode on `device`, the checkpoint dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = CertaintyHead(**ck["config"])
    head.load_state_dict(ck["state"])
    return head.to(device).eval(), ck


@torch.no_grad()
def predict(head, arrays, device="cpu", batch=256):
    """P(error < tau) of every frame of a CacheArrays: (N, n_targets) float64. Frames without a pose keep the head's
    output here; the evaluator sets them to 0 (a failure), as for every other calibrator."""
    head = head.to(device).eval()
    out = []
    for lo in range(0, len(arrays), int(batch)):
        sl = slice(lo, lo + int(batch))
        out.append(head.probabilities(
            maps=torch.as_tensor(arrays.maps[sl]).to(device), tokens=torch.as_tensor(arrays.tokens[sl]).to(device),
            token_valid=torch.as_tensor(arrays.token_valid[sl]).to(device),
            frame=torch.as_tensor(arrays.frame[sl]).to(device)).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, head.n_targets))
