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

## After the gate (Maciej 2026-10-14: "do a and b, both on eagle"); jobs submitted 2026-10-14

- (a) Band-input check with the gate checkpoints, no retraining (where PanoRoMa collapses: 4.70 m / 6.09 m):
  VIGOR Chicago 3000 draw cropped to ±27.6° on the fly (`BEVLOC_CROP_BAND_DEG=27.6`, as experiment 21) for Row B
  (job 9091987, tag `pillar_B_chi20_crop27`) and Row A (9091988); KITScenes Frankfurt 100 frames, cropped dataset
  with its `erp_band.json` (`kit_vigor/frankfurt_142f1419_crop`, 9091989, `..._kit_frankfurt_crop`) and the uncropped
  stitch with black rows (`frankfurt_142f1419_eagle`, 9091990, `..._kit_frankfurt_full`). Image-only; `pillar.cam_height_m`
  stays 2.5 m on KITScenes (the rig height is not recorded; a caveat on that row).
- (b) Full budget for Row B, experiment-18 recipe: four-city same-area, 100 epochs coarse (`pillar_B_e100`, 9091992,
  `--aug-rot-deg 10`) → coarse eval on the 3000 draw (9091993, `pillar_B_e100_eval3000`) and fine
  (`vigor_cell00625_fine.yaml`, 1 m cells over 56 m, `pillar_B_fine_e100`, 9091994, warm-started from the coarse
  `last`) → whole Chicago same-area test two-pass se2 gate 6 m (9091995, `pillar_B_e100_chicago`). A 50-step smoke of
  the fine config from the gate checkpoint runs first (9091991, `pillar_fine_smoke_B`) so a fine-pass failure shows
  before the 14-h coarse pass ends.
### (a) Band-input results (2026-10-14, jobs 9091987–9091990, 28 min each on VIGOR, 1 min on KITScenes)

Same gate checkpoints (`vigor_pillar_{A,B}_chi20_last.pt`, Chicago-only, 20 epochs), coarse pass, `se2`, image-only.
The ±27.6° crop is the experiment-21 protocol (panorama cut on the fly to the KITScenes image band, 896 × 128 px, the
query's rays and slice validity use the band); the uncropped rows are the gate numbers.

| Chicago 3000 draw, first pass | uncropped median (95 % CI) | cropped ±27.6° median (95 % CI) | ratio | cropped mean | cropped R@5 / R@10 | > 30 m |
|---|---|---|---|---|---|---|
| pillar Row B (depth-taught height) | 2.23 m (2.15–2.33) | **3.01 m** (2.91–3.15) | ×1.35 | 5.17 m | 0.70 / 0.86 | 0.6 % |
| pillar Row A (pose only) | 2.56 m (2.47–2.65) | 4.00 m (3.84–4.17) | ×1.56 | 6.13 m | 0.60 / 0.81 | 0.6 % |
| PanoRoMa D (100 ep, four cities, experiment 21) | 1.41 m | 5.20 m (two-pass 1.06 → 6.09 m) | ×3.7 | 7.90 m (two-pass) | 0.38 / 0.77 (two-pass) | — |
| centre guess | 14.71 m | 14.71 m | | | | |

| KITScenes Frankfurt, 100 frames, first pass | median (95 % CI) | mean | R@5 / R@10 | yaw median |
|---|---|---|---|---|
| pillar Row B, cropped dataset (band file) | 9.32 m (8.25–10.41) | 9.82 m | 0.23 / 0.55 | 2.6° |
| pillar Row B, uncropped stitch (black rows) | 7.77 m (6.63–8.97) | 8.75 m | 0.22 / 0.69 | 4.0° |
| PanoRoMa D two-pass, uncropped (experiment 20) | 4.70 m | | 0.55 / — | |
| FG² / Loc² (experiment 20) | 2.34 / 2.44 m | | 0.94 / — | |
| centre guess | 13.28 m | | | |

Reading:
1. **On VIGOR the band claim holds.** Cropping to the KITScenes band costs the pillar query ×1.35 (Row B) and ×1.56
   (Row A) on the median, against ×3.7 for the ERP-token query at its full budget; the cropped pillar Row B (3.01 m) is
   better than the cropped PanoRoMa first pass (5.20 m) and two-pass (6.09 m) despite being 0.8 m worse uncropped and
   trained 20× less. The yaw stays at 1.2°. Depth-taught height degrades less under the crop than pose-only (×1.35 vs
   ×1.56): the slices still visible in the band carry the learned height prior.
2. **KITScenes is not answered by the gate checkpoint**: 7.8–9.3 m (chance 13.3 m), worse than PanoRoMa (4.70 m) and FG²
   (2.34 m). Confounds, all stacked on this row: the checkpoint saw 20 epochs of Chicago only (PanoRoMa / FG²: four cities,
   full budget); the pillar geometry assumes the VIGOR 2.5 m camera height while the KIT ring rig's height is not recorded
   (a wrong height rescales every ground slice radially, which the ERP-token query with depth did not depend on); a
   German DOP20 orthophoto. The cropped dataset is worse than the black-row stitch (9.3 vs 7.8 m, overlapping CIs), unlike
   PanoRoMa (10.2 vs 4.7 m). Decide after the full-budget Row B (b) is evaluated on the same scene; measure the KIT rig
   height from the LiDAR ground plane (`kitscenes_pano` calibration path) before that run.
3. Fine-config smoke (9091991, 50 steps from the gate checkpoint, 1 m cells over 56 m, 56 × 56 × 11 slices, batch 4):
   trains and validates (train CE 5.37 → 4.98, height CE 1.57 at 40 % supervised cells); the chained fine pass is unblocked.

### Follow-ups submitted 2026-10-14 (Maciej: "do all on eagle")

- **KIT rig camera height measured: 1.66 m** (`make kitscenes-cam-height`, `cam_height.json`; LiDAR ground plane, densest
  layer ≥ 1 m below the panorama centre, median of 100 frames, p10 1.65 / p90 1.68; the first pass of the fit picked the
  bridge parapet tops 0.35 m below the camera on 46 frames, hence the 1 m cut). The KITScenes rows above assumed 2.5 m, so
  every ground slice was sampled 1.5× too far out. `configs/vigor_cell0125_kit.yaml` carries 1.66 m; KIT evaluations are
  re-run with it: gate Row B on the cropped and the black-row dataset (jobs 9093575 / 9093577), and the full-budget coarse
  checkpoint on both when 9091992 finishes (9093576 / 9093578).
- **KIT re-evaluation with 1.66 m (jobs 9093575 / 9093577), gate Row B, first pass: cropped dataset 11.07 m (CI 8.88–12.32,
  R@5 0.19, R@10 0.46, yaw 2.7°), black-row stitch 7.45 m (6.87–8.41, R@5 0.25, R@10 0.70, yaw 1.5°)** — against 9.32 /
  7.77 m with 2.5 m assumed. The camera height is not the explanation: the correct geometry leaves the black-row row where
  it was and makes the cropped row worse (the lift was trained with 2.5 m sampling on full spheres, so the sampling
  pattern it learned is the VIGOR one). What remains on this scene is Chicago-only 20-epoch training versus the four-city
  full-budget references, and the German orthophoto; the full-budget coarse checkpoint is evaluated on both datasets with
  1.66 m when 9091992 finishes (9093576 / 9093578). (The eval JSON's `height_m` is `vigor.height_m`, the picture's;
  `pillar.cam_height_m` is not recorded in the meta — a deferred minor.)
- **Cross-area Row B**, same recipe trained on New York + Seattle: coarse 9093545 → 6000-frame coarse eval 9093546
  (`pillar_B_xa_e100_eval6000`); fine 9093547 → two-pass 6000 9093548 (`pillar_B_xa_e100_twopass6000`). References:
  PanoRoMa two-pass 1.94 m (coarse 2.27–2.50), FG² 1.41 m.
- **Row A without pose-NLL** on the gate recipe (9093549 → eval 9093550, `pillar_A_nopose_chi20`): does the pose-NLL
  window around the camera cell hurt the ±35 m grid?
- **DINOv3-LVD row**: `tpupzyfr` exported on Eagle (`checkpoints/satroma_tpupzyfr/`, 336 MB decoder; the config JSON's
  `pretrain_model` patched to `vit_large_patch16_dinov3_lvd`, no training sidecar, `config_source` says so), timm
  `vit_large_patch16_dinov3.lvd1689m` copied to `$P/hf_cache/hub`. Row B gate recipe with `configs/vigor_cell0125_lvd.yaml`
  (9093567 → eval 9093568, `pillar_B_lvd_chi20`). Validation at epoch 20: CE 3.26 / Procrustes 2.44 m against 3.82 / 3.41 m
  for the sat493m Row B (no-pose Row A: 3.54 / 3.29 m against 4.00 / 3.62 m).
- **LVD Row B at the full budget** (Maciej, "run lvd row on full budget too"): experiment-18 recipe with
  `configs/vigor_cell0125_lvd.yaml` / `configs/vigor_cell00625_fine_lvd.yaml`: coarse 9093830 → 3000-draw eval 9093831
  (`pillar_B_lvd_e100_eval3000`), fine 9093832 → whole-Chicago two-pass 9093833 (`pillar_B_lvd_e100_chicago`).

- (c) still open: the ablation ladder from the task doc (FG² deformable offsets → six lift iterations → DINOv3-LVD
  encoder, `configs/vigor_cell0125_lvd.yaml`, needs the `tpupzyfr` export).
