# Full Chicago same-area TEST split (12,739 panoramas), 2026-10-02

The whole VIGOR same-area test list of Chicago (no sampling), known orientation, positives only. All three methods on the
same panoramas; all settings fixed before the run. PanoRoMa D = regularisation D, coarse + second pass trained 100 epochs
each on corrected (v2) UniK3D depth, se2, bf16 decoder, gate 6 m. FG² and Loc² are the released same-area checkpoints
(Loc² reads the v2 depth). Jobs 8888246 / 8888247 / 8888248; files `eval_*.json`.

| Method | Median (95 % CI) | Mean | ≤ 5 m | ≤ 10 m |
|---|---|---|---|---|
| FG² | **1.05 m** (1.04–1.06) | **1.87 m** | **95 %** | **97 %** |
| **PanoRoMa D, two-pass** | 1.12 m (1.10–1.13) | 2.44 m | 92 % | 95 % |
| Loc² | 1.26 m (1.23–1.28) | 3.01 m | 88 % | 93 % |
| PanoRoMa D, first pass only | 1.37 m (1.35–1.39) | 2.66 m | 91 % | 95 % |

Second pass: 17 of 12,739 frames fell back to the coarse pose (2 had no coarse pose).

On the full split PanoRoMa is 0.07 m behind FG² on the median (the intervals do not overlap), 0.57 m behind on the mean
and 3 pp behind within 5 m; it is 0.14 m / 0.57 m ahead of Loc² on median / mean. The earlier 3000-sample draw
(1.11 m / 2.26 m) was slightly optimistic on the mean.
