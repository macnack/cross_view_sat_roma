"""YAML config loading and run snapshots (reproducibility rule in CLAUDE.md)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / "configs/default.yaml"


def _ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    return d


def _plain(o):
    if isinstance(o, SimpleNamespace):
        return {k: _plain(v) for k, v in vars(o).items()}
    return o


def load(path=None) -> SimpleNamespace:
    with open(path or DEFAULT) as f:
        return _ns(yaml.safe_load(f))


def snapshot(cfg, out_dir, extra=None) -> Path:
    """Write the resolved config, command line and git revision into out_dir/config.yaml."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip() or "uncommitted"
    except OSError:
        rev = "unknown"
    meta = dict(command=" ".join(sys.argv), time=time.strftime("%Y-%m-%d %H:%M:%S"), git=rev, **(extra or {}))
    p = out / "config.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(dict(run=meta, config=_plain(cfg)), f, sort_keys=False)
    return p


def add_args(ap):
    ap.add_argument("--config", default=str(DEFAULT), help="YAML config (default: configs/default.yaml)")
    return ap
