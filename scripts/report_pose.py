"""Collect experiments/05_lift_splat/eval/eval_*.json into experiments/05_lift_splat/REPORT.md.

  make pose-report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bevloc import config as C

HEADER = """# Pose evaluation on immutable manifests

Protocol (docs/decisions.md, 2026-09-23): train = 4 Fixtor routes; validation/selection = IcRzj
(`manifest.json`, 200 frames); TEST = irAsBUK (`manifest_test.json`, 200 frames, never used for
selection). Local window ±10 % of the 224 m reference edge / ±10°. Mapillary poses are a proxy,
not survey GT. Median and recalls carry 95 % percentile-bootstrap intervals (1000 resamples);
a failed RANSAC counts as a miss (inf). "centre guess" = predict the crop centre: the chance level
of the window. One row per (checkpoint tag, manifest, year, solver); "peak" = published one-peak-
per-patch RANSAC, "means" = distinct GMM means.

"""


def fmt(s):
    lo, hi = s["median_ci"]
    r5lo, r5hi = s["recall@5m_ci"]
    return (f"{s['median_m']:.1f} [{lo:.1f}, {hi:.1f}] | {s['recall@5m']:.2f} [{r5lo:.2f}, {r5hi:.2f}] | "
            f"{s['recall@10m']:.2f} | {s['frac_gt_30m']:.2f} | {s['matched']}/{s['n']}")


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--eval-dir", default="experiments/05_lift_splat/eval")
    ap.add_argument("--out", default="experiments/05_lift_splat/REPORT.md")
    a = ap.parse_args()
    files = sorted(Path(a.eval_dir).glob("eval_*.json"))
    lines = [HEADER, "| tag | manifest | year | solver | row | median m [95 % CI] | R@5 [CI] | R@10 | >30 m | matched |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    chance = {}
    for f in files:
        d = json.loads(f.read_text())
        meta, tag = d["meta"], f.stem[len("eval_"):]
        man = Path(meta["manifest"]).stem
        tag = tag[: -len(man) - 1] if tag.endswith("_" + man) else tag
        for year, s in d["summary"].items():
            for row in ("peak", "means"):
                lines.append(f"| {tag} | {man} | {year} | {meta.get('solver', 'srt')} | {row} | {fmt(s[row])} |")
            chance[(man, year)] = s["centre_guess"]
    for (man, year), s in sorted(chance.items()):
        lines.append(f"| centre guess | {man} | {year} | – | chance | {fmt(s)} |")
    lines.append("")
    lines.append(f"Sources: {', '.join(f.name for f in files) or 'none yet'}.")
    verdict = Path(a.out).parent / "VERDICT.md"          # hand-written interpretation, kept next to the table
    if verdict.exists():
        lines += ["", verdict.read_text().rstrip()]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out} ({len(files)} eval files)")


if __name__ == "__main__":
    main()
