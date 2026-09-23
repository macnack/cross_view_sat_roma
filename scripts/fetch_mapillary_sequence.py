"""Thin CLI for the mapillary_dl package.

  MAPILLARY_TOKEN=... python scripts/fetch_mapillary_sequence.py --image 735205558899799
  python -m mapillary_dl --help
"""

from mapillary_dl.cli import main

if __name__ == "__main__":
    main()
