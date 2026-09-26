"""One table per VIGOR split from every eval json in experiments/09_vigor (ours, FG², Loc²), all rows on the
shared sample draws, followed by the hand-written research-ideas ledger (experiments/09_vigor/IDEAS.md) verbatim.

  make vigor-report            # -> experiments/09_vigor/REPORT.md

Rows are labelled from the file tag (LABELS below; unknown tags print as themselves); multi-city files also
get a per-city table. Numbers come from the json summaries only, nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bevloc import config as C

LABELS = {
    "vigor_ipm_long_zeroshot_chicago": "Ours, zero-shot (Poznań IPM checkpoint), Chicago only",
    "vigor_ipm_long_zeroshot": "Ours, zero-shot (Poznań IPM checkpoint)",
    "vigor_ipm_long_zeroshot_crossarea": "Ours, zero-shot (Poznań IPM checkpoint)",
    "vigor_ft_chicago_same": "Ours, Chicago fine-tune, 3k steps",
    "vigor_ft_chicago_same_30k": "Ours, Chicago fine-tune, 30k steps",
    "vigor_ft_chicago_same_30k_nopoznan": "Ours, 30k steps, no Poznań warm start",
    "vigor_ft_chicago_same_30k_sim": "Ours, 30k, 4-DoF solver (sim)",
    "vigor_ft_chicago_same_30k_se2": "Ours, 30k, 3-DoF solver (se2)",
    "vigor_ft_chicago_same_30k_cell0125": "Ours, 30k, 2 m cells (0.125 m/px)",
    "vigor_ft_4city_cell0125_chicago_same": "Ours, four-city fine-tune, 60k steps, 2 m cells",
    "vigor_ft_4city_cell0125_all_same": "Ours, four-city fine-tune, 60k steps, 2 m cells (all cities)",
    "vigor_ft_chicago30k_cell0125_all_same": "Ours, Chicago-only 30k checkpoint, 2 m cells (all cities)",
    "vigor_ft_crossarea_cell0125": "Ours, NY + Seattle fine-tune, 60k steps, 2 m cells",
    "vigor_chicago_same_30k_erp_depth_cell0125_best_se2": "Task 04 erp_depth + head + heat-map loss, 2 m cells, 30k, best, se2",
    "vigor_chicago_same_30k_erp_depth_cell0125_last_se2": "Task 04 erp_depth + head + heat-map loss, 2 m cells, 30k, last, se2",
    "vigor_chicago_same_30k_erp_depth_cell0125_best_srt": "Task 04 erp_depth + head + heat-map loss, 2 m cells, 30k, best, homography",
    "vigor_ft_4city_chicago_same": "Ours, four-city fine-tune, 60k steps",
    "vigor_ft_4city_all_same": "Ours, four-city fine-tune, 60k steps (all cities)",
    "vigor_ft_4city_v2_chicago_same": "Ours, four-city, held-out validation (v2)",
    "vigor_ft_4city_v2_all_same": "Ours, four-city, held-out validation (v2, all cities)",
    "vigor_ft_chicago30k_all_same": "Ours, Chicago-only 30k checkpoint (all cities)",
    "vigor_ft_crossarea": "Ours, NY + Seattle fine-tune, 60k steps",
    "vigor_ft_crossarea_v2": "Ours, NY + Seattle, held-out validation (v2)",
    "vigor_chicago_same_10k_erp_depth_best_srt": "Task 04 erp_depth + head, 10k steps, best (step 8500), homography",
    "vigor_chicago_same_10k_erp_depth_best_se2": "Task 04 erp_depth + head, 10k steps, best (step 8500), se2",
    "vigor_chicago_same_10k_erp_depth_last_srt": "Task 04 erp_depth + head, 10k steps, last (step 10000), homography",
    "vigor_chicago_same_10k_erp_depth_nll_best_srt": "Task 04 erp_depth + head + heat-map loss, 10k, best (step 9500), homography",
    "vigor_chicago_same_10k_erp_depth_nll_best_se2": "Task 04 erp_depth + head + heat-map loss, 10k, best (step 9500), se2",
    "vigor_chicago_same_30k_erp_depth_best_srt": "Task 04 erp_depth + head + heat-map loss, 30k, best, homography",
    "vigor_chicago_same_30k_erp_depth_best_se2": "Task 04 erp_depth + head + heat-map loss, 30k, best, se2",
    "vigor_chicago_same_30k_erp_depth_last_se2": "Task 04 erp_depth + head + heat-map loss, 30k, last, se2",
    "vigor_chicago_same_30k_erp_depth_nohead_best_se2": "Task 04 erp_depth, no head, heat-map loss, 30k, best, se2",
    "vigor_chicago_same_30k_erp_depth_nohead_last_se2": "Task 04 erp_depth, no head, heat-map loss, 30k, last, se2",
    "fg2_chicago_same": "FG² (released same-area checkpoint)",
    "fg2_crossarea": "FG² (released cross-area checkpoint)",
    "loc2_chicago_same": "Loc² (released same-area checkpoint)",
    "loc2_crossarea": "Loc² (released cross-area checkpoint)",
}
SOLVER_LABEL = {"peak": "", "procrustes": ", weighted Procrustes", "ransac": ", RANSAC",
                "hyp_med": ", medoid of the bootstrap RANSAC hypotheses"}
SKIP = ("smoke",)


def fmt(p, mean):
    lo, hi = p["median_ci"]
    return (f"{p['median_m']:.2f} m ({lo:.2f}–{hi:.2f}) | {mean:.2f} m | {100 * p['recall@5m']:.0f} % | "
            f"{100 * p['recall@10m']:.0f} % | {p['n']}")


def rows_of(path: Path):
    d = json.loads(path.read_text())
    m = re.match(r"eval_(.+?)_(samearea|crossarea)\.json$", path.name)
    tag, split = m.group(1), m.group(2)
    meta, summ = d["meta"], d["summary"]
    if meta.get("calib"):                          # eval_vigor.py --calib: held-out TRAINING frames, not a test row
        return []
    method = meta.get("method", "ours")
    solvers = ["peak"] + extra_rows(summ["all"]) if method == "ours" else \
        [k for k in ("ransac", "procrustes") if k in summ["all"]]
    cities = [c for c in summ if c != "all"]
    out = []
    for order, solver in enumerate(solvers):
        for city in ["all"] + (cities if len(cities) > 1 else []):
            s = summ[city]
            p = s[solver]
            mean = s["mean_peak_m"] if solver == "peak" else s.get(f"mean_{solver}_m", float("nan"))
            label = LABELS.get(tag, tag) + SOLVER_LABEL.get(solver, f", {solver}")
            out.append(dict(split=split, city=city, label=label, n=p["n"], line=fmt(p, mean),
                            centre=s["centre_guess"]["median_m"], file=path.name, order=order,
                            median=p["median_m"]))
    return out


def extra_rows(s):
    """Summary rows of an "ours" JSON besides peak: the second pass (fine, fine_gated), sub-cell (refined*), hyp*."""
    keys = [k for k in s if isinstance(s[k], dict) and "median_m" in s[k]
            and (k in ("fine", "fine_gated") or k.startswith("refined") or k.startswith("hyp"))]
    first = [k for k in ("fine", "fine_gated") if k in keys]
    return first + sorted(k for k in keys if k not in first)


def sort_rows(sub):
    """Files ordered by their first (peak) row's median; within a file, the peak row first, then the extra rows."""
    def shown(r):                                   # the printed (2-decimal) median, as the table was always sorted
        return round(r["median"], 2)
    key_of = {r["file"]: shown(r) for r in sub if r["order"] == 0}
    return sorted(sub, key=lambda r: (key_of.get(r["file"], shown(r)), r["file"], r["order"]))


def main():
    ap = C.add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--dir", default="experiments/09_vigor")
    a = ap.parse_args()
    d = Path(a.dir)
    rows = []
    for f in sorted(d.glob("eval_*.json")):
        if any(s in f.name for s in SKIP):
            continue
        rows += rows_of(f)
    head = "| Method | Median (95 % CI) | Mean | ≤ 5 m | ≤ 10 m | n |\n|---|---|---|---|---|---|\n"
    out = ["# VIGOR results (known orientation)\n",
           "Every row is scored on the same sample draw per split (seed 0: 3000 Chicago same-area, 6000 San Francisco + "
           "Chicago cross-area, 12000 across the four cities for the per-city tables); FG² and Loc² run at their native "
           "sizes from their released checkpoints. Generated by `make vigor-report` from the eval jsons in this folder; "
           "hand-written notes live in IDEAS.md, appended below.\n"]
    for split, title in (("samearea", "Same-area"), ("crossarea", "Cross-area (train New York + Seattle, test San Francisco + Chicago)")):
        sub = [r for r in rows if r["split"] == split and r["city"] == "all"]
        if not sub:
            continue
        sub = sort_rows(sub)
        out.append(f"\n## {title}\n\n" + head + "".join(f"| {r['label']} | {r['line']} |\n" for r in sub))
        centre = sorted({round(r["centre"], 2) for r in sub})
        out.append(f"\nCentre-guess chance (predict the tile centre): median {', '.join(f'{c:.2f}' for c in centre)} m.\n")
        per_city = [r for r in rows if r["split"] == split and r["city"] != "all"]
        if per_city:
            out.append(f"\n### Per city\n\n| Method | City | Median (95 % CI) | Mean | ≤ 5 m | ≤ 10 m | n |\n|---|---|---|---|---|---|---|\n"
                       + "".join(f"| {r['label']} | {r['city']} | {r['line']} |\n" for r in sorted(per_city, key=lambda r: (r['label'], r['city']))))
    ideas = d / "IDEAS.md"
    if ideas.exists():
        out.append("\n" + ideas.read_text())
    (d / "REPORT.md").write_text("".join(out))
    print(f"wrote {d / 'REPORT.md'} ({len(rows)} rows from {len(set(r['file'] for r in rows))} files)")


if __name__ == "__main__":
    main()
