# Full Chicago same-area test split on SliceMatch's corrected labels, 2026-10-03

Same 12,739 panoramas, same checkpoints and settings as `experiments/16_full_test` (nothing retrained); only the ground
truth is `splits__corrected` from SliceMatch (Lentsch et al., CVPR 2023, github.com/tudelft-iv/SliceMatch,
`VIGOR_corrected_labels`), which FG²'s README recommends. Eagle root
`/mnt/storage_6/project_data/pl1269-01/krupka_maciej/vigor_corrected` (city symlinks + corrected splits, also under the
plain file names the FG²/Loc² loaders read). Jobs 8890915 (PanoRoMa D) / 8890919 (FG²) / 8890920 (Loc²).
Comparison: `make vigor-label-effect` -> `label_effect.json`.

| Method | Original: median | Corrected: median (95 % CI) | Mean orig -> corr | R@5 corr | R@10 corr |
|---|---|---|---|---|---|
| FG² | 1.05 m | **0.98 m** (0.96–0.99) | 1.87 -> **1.84 m** | **0.947** | **0.969** |
| PanoRoMa D, two-pass | 1.12 m | 1.19 m (1.17–1.21) | 2.44 -> 2.53 m | 0.918 | 0.950 |
| Loc² | 1.26 m | 1.21 m (1.19–1.23) | 3.01 -> 3.03 m | 0.880 | 0.931 |
| PanoRoMa D, first pass | 1.37 m | 1.53 m (1.51–1.55) | 2.66 -> 2.80 m | 0.910 | 0.953 |

- The correction is radial: each label moves outward from the tile centre by ~3 % (ratio 1.030, p5–p95 1.024–1.044);
  shift median 0.45 m, max 0.80 m. Recall at 5/10 m barely moves for anyone.
- Predictions are bit-identical between the two runs (max 0.00 m); only the yardstick changed.
- FG² (trained on corrected labels) gains 0.07 m; PanoRoMa (trained on the original labels) loses 0.07 m; it is closer
  to the original label on 58 % of frames and radially short by 0.40 m on average (first pass 0.68 m). Loc² gains
  0.05 m (its loader reads `splits_new`; which labels those are is unchecked).
- Separate finding: PanoRoMa's final predictions are offset by (+0.12 E, +0.43 N) m from the label on both label sets
  (first pass (−0.10, +0.44)). Removing the constant takes the corrected median 1.19 -> 1.11 m. Not explained by the
  labels; suspected convention offset in our pipeline (half-pixel / row / tile centre). To find before retraining.
- Like-for-like vs FG² needs PanoRoMa retrained on the corrected labels.
