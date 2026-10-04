# PanoRoMa D retrained on SliceMatch's corrected labels, 2026-10-04

Same recipe as `experiments/15_panoroma_v2` run `v2_reg_D` (regularisation D, 100 epochs coarse + 100 epochs fine, 4x H100,
global batch 32, bf16, corrected v2 depth); only the labels differ: root `vigor_corrected` (`__corrected` files only, the
reader refuses raw labels since 2026-10-03). Jobs 8891319 (coarse, 13 h 38) -> 8891320 (fine, 13 h 41) -> 8891362
(evaluation, 3 h 03). Evaluation: whole Chicago same-area test list (12,739), known orientation, two-pass se2, gate 6 m,
bf16 decoder, `last` checkpoints (protocol fixed in docs/decisions.md before the run). `make vigor-compare-corrected`.

| Method (corrected labels) | Median (95 % CI) | Mean | <= 5 m | <= 10 m | > 5 m |
|---|---|---|---|---|---|
| FG² | **0.98 m** (0.96–0.99) | **1.84 m** | **94.7 %** | **96.9 %** | **5.3 %** |
| **PanoRoMa D, retrained on corrected labels** | 1.07 m (1.06–1.09) | 2.40 m | 92.1 % | 95.3 % | 7.9 % |
| PanoRoMa D, original labels (experiment 17) | 1.19 m (1.17–1.21) | 2.53 m | 91.8 % | 95.0 % | 8.2 % |
| Loc² | 1.21 m (1.19–1.23) | 3.03 m | 88.0 % | 93.1 % | 12.0 % |
| PanoRoMa D retrained, first pass only | 1.42 m (1.40–1.45) | 2.72 m | 91.1 % | 95.4 % | 8.9 % |

- The retrain gains 0.12 m on the median (1.19 -> 1.07 m) and 0.13 m on the mean; the tail barely moves (7.9 vs 8.2 % over
  5 m). The remaining gap to FG² is 0.09 m on the median and 0.56 m on the mean, in the tail (7.9 vs 5.3 % over 5 m).
- Failure overlap with FG²: both within 5 m 90.4 %, only FG² fails 1.7 %, only PanoRoMa fails 4.3 %, both fail 3.6 %.
- Offset: the retrained model still sits on average (+0.01 E, +0.42 N) m from the label (fit constant, first pass +0.52 N);
  median offset of frames within 5 m (+0.01, +0.36). Removing it would take the median 1.07 -> 1.01 m (fitted on the test
  split, an upper bound). Same size as before the retrain, so the offset is not a label-scale effect; it comes with the
  rot90/flip augmentation (experiments/17_corrected_labels/OFFSET.md). Not corrected in these rows.
- Partial run during training (job 8895879): epoch-70 checkpoint, first pass only, 1.36 m median / 2.77 m mean on the same
  list (`eval_vigor_cl_ep070_coarse_chicago_samearea.json`; CI 1.34–1.38). The epoch-100 first pass is 1.42 m (CI 1.40–1.45) / 2.72 m:
  the first-pass median did not improve between epoch 70 and 100 (the intervals do not overlap, 0.06 m worse), the mean is level.

## Per-city offset correction (2026-10-04)

`make vigor-offset-calib` (`bevloc.eval.pose_offset`): the constant offset of each city is the median residual
(prediction − label, east/north) of the frames within 5 m on 4,000 held-out TRAINING frames (job 8905031,
`eval_vigor.py --calib --limit 4000`, the --val-frac part of the training list the model never trained on; ~1,000 per city),
then subtracted from every test prediction of that city. No test frame is used in the fit (ids checked disjoint) and the
uncorrected errors are first reproduced from the positions. Offsets (east, north): Chicago (−0.01, +0.37) m, New York
(+0.09, +0.46), San Francisco (−0.18, +0.24), Seattle (−0.07, +0.34).

| Chicago test, corrected labels | Median (95 % CI) | Mean | <= 5 m | <= 10 m | > 5 m |
|---|---|---|---|---|---|
| PanoRoMa D retrained, uncorrected | 1.073 m (1.057–1.089) | 2.399 m | 92.1 % | 95.3 % | 7.9 % |
| **… with per-city offset correction** | **1.013 m (1.000–1.027)** | 2.347 m | 92.1 % | 95.3 % | 7.9 % |
| FG² | 0.98 m (0.96–0.99) | 1.84 m | 94.7 % | 96.9 % | 5.3 % |

The correction improves 59 % of the frames and moves the median by 0.06 m (intervals do not overlap), the mean by 0.05 m;
the recall is unchanged because the offset is far below the 5 m threshold. The median gap to FG² falls from 0.09 to 0.03 m;
the tail (7.9 vs 5.3 % over 5 m) and the 0.5 m mean gap are untouched. The offset estimated on calibration frames (+0.37 m
north in Chicago) agrees with the one measured on the test frames (+0.36 m median of the frames within 5 m), so it
transfers. Result file: `offset_cl_new_chicago.json`.
