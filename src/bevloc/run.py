"""Shared set-up for the command-line scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from bevloc import config as C
from bevloc.data.calib import Calib
from bevloc.data.dur360 import Dur360Frames


def context(cfg, need_mask=True):
    ds, calib = Dur360Frames.from_config(cfg), Calib.from_config(cfg)
    p = C.REPO / cfg.erp.valid_mask
    if p.exists():
        mask = np.load(p)
        if mask.shape != (cfg.erp.size[1], cfg.erp.size[0]):
            sys.exit(f"{p}: mask shape {mask.shape} does not match erp.size {cfg.erp.size}; re-run `make mask`")
    elif need_mask:
        sys.exit(f"{p} not found: run `make mask` first")
    else:
        mask = None
    return ds, calib, mask


def frame_names(ds, frames):
    """Accept 12, 0000000012, or ranges like 200:210 or 500:1400:10."""
    out = []
    for f in frames:
        if ":" in f:
            a = [int(x) for x in f.split(":")]
            out += [f"{k:010d}" for k in range(a[0], a[1] + 1, a[2] if len(a) > 2 else 1)]
        else:
            out.append(f"{int(f):010d}")
    missing = [n for n in out if not ds.has(n)]
    if missing:
        print(f"skipping {len(missing)} frame(s) not on disk, e.g. {missing[:3]}", file=sys.stderr)
    return [n for n in out if n not in missing]
