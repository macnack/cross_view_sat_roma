"""Command line for mapillary_dl.

  python -m mapillary_dl --image 735205558899799
  python -m mapillary_dl --bbox 16.7316,52.2919,17.0717,52.5093 --name poznan
  python -m mapillary_dl --sample-scan data/mapillary/scans/poznan.json --frames 12
"""

import argparse
from pathlib import Path
from typing import Optional

from .dataset import download_sequence, sample_users
from .geo import parse_bbox
from .scan import write_scan

_HELP = """\
Download Mapillary 360 sequences, or list the ones inside a lon/lat box.

OrienterNet cuts each panorama into four perspective crops. This package keeps
the equirectangular image. The access token is read from MAPILLARY_TOKEN and
is never written to disk.
"""


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Mapillary image id; its sequence is downloaded")
    g.add_argument("--sequence", help="Mapillary sequence id")
    g.add_argument(
        "--bbox",
        help="west,south,east,north. List 360 sequences in the rectangle; do not download pixels.",
    )
    g.add_argument(
        "--sample-scan",
        type=Path,
        help="Scan JSON. Download evenly spaced frames from the longest sequence of each user.",
    )
    ap.add_argument("--name", help="filename stem for a --bbox scan, e.g. poznan")
    ap.add_argument("--frames", type=int, default=12, help="frames per user for --sample-scan")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quality", choices=("original", "2048"), default="original")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    if a.sample_scan:
        sample_users(
            a.sample_scan,
            a.out or Path("data/mapillary"),
            a.frames,
            a.quality,
            a.workers,
        )
        return

    if a.bbox:
        write_scan(
            parse_bbox(a.bbox),
            a.name,
            a.out or Path("data/mapillary/scans"),
            a.workers,
        )
        return

    dest = download_sequence(
        image_id=a.image,
        sequence_id=a.sequence,
        out_root=a.out or Path("data/mapillary"),
        quality=a.quality,
        workers=a.workers,
    )
    print(f"wrote {dest}")
