"""Thin read-only wrapper around ``third_party/BevSplat``.

BevSplat needs its CUDA rasterizers (``feat_gaussian``, ``pano_gaussian_feat``)
and the authors' OneDrive VIGOR checkpoints. This module only resolves paths
and records hashes; full inference waits on those dependencies.
"""
from __future__ import annotations

from pathlib import Path

from bevloc import config as C
from bevloc.baselines.common import sha256_file

BEVSPLAT_ROOT = C.REPO / "third_party" / "BevSplat"
CKPT_ROOT = C.REPO / "checkpoints" / "baselines" / "bevsplat"

# Released VIGOR panoramic checkpoints (OneDrive; see third_party/BevSplat/README.md).
EXPECTED = {
    "VIGOR_same_no_GPS.pth": "same-area aligned-orientation λ₁=0",
    "VIGOR_same_GPS.pth.pth": "same-area aligned-orientation λ₁=1",
    "VIGOR_cross_no_GPS.pth.pth": "cross-area aligned-orientation λ₁=0",
    "VIGOR_cross_GPS.pth.pth": "cross-area aligned-orientation λ₁=1",
}

ONEDRIVE = (
    "https://1drv.ms/f/c/86d953bfc66eb903/"
    "IgAP7P2tFzChR7rHeMuXIOq8AakOxR02eKMyI2Z7qsMjLxo?e=zaD0Fb"
)


def pinned_commit() -> str:
    head = BEVSPLAT_ROOT / ".git" / "HEAD"
    if not BEVSPLAT_ROOT.is_dir():
        raise FileNotFoundError(f"BevSplat not cloned at {BEVSPLAT_ROOT}")
    import subprocess
    return subprocess.check_output(
        ["git", "-C", str(BEVSPLAT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def checkpoint_path(name: str = "VIGOR_same_no_GPS.pth") -> Path:
    p = CKPT_ROOT / name
    if not p.is_file():
        raise FileNotFoundError(
            f"BevSplat checkpoint missing: {p}\n"
            f"Download from OneDrive: {ONEDRIVE}\n"
            f"Place the six .pth files under {CKPT_ROOT}/"
        )
    return p


def list_available() -> dict:
    found = {}
    for name in EXPECTED:
        p = CKPT_ROOT / name
        if p.is_file():
            found[name] = sha256_file(p)
    return found


def status() -> dict:
    return {
        "repo": str(BEVSPLAT_ROOT.relative_to(C.REPO)) if BEVSPLAT_ROOT.is_dir() else None,
        "commit": pinned_commit() if BEVSPLAT_ROOT.is_dir() else None,
        "checkpoints_dir": str(CKPT_ROOT.relative_to(C.REPO)),
        "available": list_available(),
        "missing": [n for n in EXPECTED if not (CKPT_ROOT / n).is_file()],
        "onedrive": ONEDRIVE,
        "license": "MIT (third_party/BevSplat/LICENSE)",
    }
