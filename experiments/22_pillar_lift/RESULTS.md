# 22 — Pillar feature lift (task 07): gate results

Design: docs/tasks/07_pillar_lift.md. Code: `--query pillar` (`bevloc.model.pillar_query`), branch commit 19eed44.

## Gate (2026-10-13/14): Chicago same-area, 20 epochs, regularisation D, 4× H100

Recipe (both rows): `configs/vigor_cell0125.yaml`, `--query pillar --split samearea --cities Chicago --val-frac 0.2
--val-samples 400 --batch 32 --epochs 20 --label-smoothing 0.1 --weight-decay-decoder 0.3 --weight-decay-head 0.05
--aug-photometric 0.5 --aug-geometric rot90,flip --aug-rot-deg 10 --select-by pose` + H100 flags; 6360 steps
(318/epoch), 45 min each. Evaluation: coarse pass only, `--solver se2 --limit 3000` (the standard Chicago draw),
`last` checkpoint, image-only (no depth read at inference in either row).

| Row | Supervision | Median (95 % CI) | Mean | R@1 | R@5 | R@10 | > 30 m | Yaw median | Jobs |
|---|---|---|---|---|---|---|---|---|---|
| A | pose only (coarse CE + VCE + pose-NLL), pure FG² | **2.56 m** (2.47–2.65) | 4.29 m | 0.125 | 0.794 | 0.904 | 0.4 % | 1.12° | 9090823 → 9090824 |
| B | + depth-taught height CE (`--height-weight 1`) | **2.23 m** (2.15–2.33) | 3.99 m | 0.172 | 0.814 | 0.909 | 0.4 % | 1.15° | 9090825 → 9090826 |
| centre guess | — | 14.71 m | | | | | | | |

References on the same draw (docs/decisions.md): PanoRoMa ERP-token query, Chicago 30k steps at 2 m cells
(≈ the same sample budget, 2026-09-26) **1.76 m** coarse; the IPM picture matcher at 2 m cells (2026-09-25)
**1.89 m**; PanoRoMa 100-epoch regularised D first pass **1.33 m** (the gate threshold as written); FG² 0.98 m,
Loc² 1.21 m (full methods).

Validation curves (held-out 20 %, `train_pillar_*_chi20.csv`): Row A CE 6.75 → 4.00, top-1 0.3 → 7.4 %,
Procrustes pose 11.0 → 3.6 m; Row B CE 6.31 → 3.82, top-1 0.5 → 9.1 %, Procrustes 10.1 → 3.4 m, height CE
1.88 → 1.77 (supervising ≈ 35 % of the valid cells). Both curves were still falling at epoch 20.

## Reading

1. **Gate not passed.** Both rows are above the 1.33 m threshold and also above the like-budget ERP-token
   reference (1.76 m) and the picture matcher (1.89 m). Both are far above chance (14.7 m), no frame is without
   a pose, the > 30 m tail is 0.4 %, and the heading is recovered to ~1.1° — the lift localises; it is less sharp
   than the ERP-token query at this budget (R@1 0.13–0.17 against the ERP query's finer peaks).
2. **The depth-taught height selection helps (kick-off H3, in this setting): Row B beats Row A by 0.33 m median
   with non-overlapping CIs**, lower mean (3.99 vs 4.29 m), better R@1/R@5, and a lower coarse CE throughout
   training at the same step (4.03 vs 4.38 at step 4452). Depth was used only to supervise the height softmax;
   inference is identical and image-only in both rows.
3. What the gate does not settle: the lift's 2.5 M parameters train from scratch while the ERP query matches frozen
   tokens, and both curves were still improving at epoch 20; the design's expected advantages (band input,
   cross-area, Poznań) were not part of the gate. The height overlay (`height_smoke_B.jpg`) shows the selection
   behaving as FG² Fig. 4 describes (ground on roads, high slices toward façades).

## Options after the gate (not yet decided)

- (a) Band-input check with the existing Row B checkpoint: KITScenes Frankfurt (`make eagle-kit-compare` pattern)
  and the VIGOR ±27.6° crop test — no retraining; this is where PanoRoMa collapses (4.70 m / 6.09 m) and where a
  BEV-grid query should not.
- (b) The full budget for Row B (four-city 100 epochs, coarse → fine, as experiment 18) to see whether the gap to
  the ERP query closes with training rather than architecture.
- (c) The ablation ladder from the task doc: FG² deformable offsets → six lift iterations → DINOv3-LVD encoder
  (`configs/vigor_cell0125_lvd.yaml`, needs the `tpupzyfr` export).
