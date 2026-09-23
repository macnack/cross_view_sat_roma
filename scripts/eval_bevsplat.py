"""BevSplat evaluation on the shared Fixtor×Poznań manifest (task 02)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bevloc import config as C
from bevloc.baselines import bevsplat as bs
from bevloc.baselines.common import load_manifest


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--manifest", default="experiments/06_fg2_bevsplat/manifest.json")
    ap.add_argument("--out", default="experiments/06_fg2_bevsplat/bevsplat_zero")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = load_manifest(a.manifest)
    st = bs.status()
    metrics = {
        "status": "blocked",
        "stage": "checkpoints",
        "bevsplat": st,
        "manifest": a.manifest,
        "n_smoke": a.n or (20 if a.smoke else None),
        "blocker": (
            "BevSplat VIGOR checkpoints are on OneDrive and were not downloadable "
            f"non-interactively ({st['onedrive']}; HTTP 403 / reauth). "
            f"Missing: {st['missing']}. Also needs CUDA rasterizer build "
            "(bash third_party/BevSplat/scripts/bootstrap_cuda.sh) and the VIGOR "
            "dataset for native reproduction (task §1 stop rule)."
        ),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.yaml").write_text(Path(a.config).read_text())
    print(json.dumps(metrics, indent=2))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
