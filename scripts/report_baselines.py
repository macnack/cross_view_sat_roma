"""Aggregate baseline run metrics into experiments/06_fg2_bevsplat/REPORT.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bevloc import config as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="experiments/06_fg2_bevsplat")
    a = ap.parse_args()
    root = Path(a.root)
    status = (root / "STATUS.md").read_text() if (root / "STATUS.md").exists() else ""
    chunks = ["# Task 02 — FG² / BevSplat transfer report\n", status, "\n## Run metrics\n"]
    for p in sorted(root.glob("*/metrics.json")):
        chunks.append(f"### `{p.parent.name}`\n```json\n{p.read_text().strip()}\n```\n")
    (root / "REPORT.md").write_text("\n".join(chunks))
    print(f"wrote {root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
