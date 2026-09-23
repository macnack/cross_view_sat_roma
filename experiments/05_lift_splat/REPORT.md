# Pose evaluation on immutable manifests

Protocol (docs/decisions.md, 2026-09-23): train = 4 Fixtor routes; validation/selection = IcRzj
(`manifest.json`, 200 frames); TEST = irAsBUK (`manifest_test.json`, 200 frames, never used for
selection). Local window ±10 % of the 224 m reference edge / ±10°. Mapillary poses are a proxy,
not survey GT. Median and recalls carry 95 % percentile-bootstrap intervals (1000 resamples);
a failed RANSAC counts as a miss (inf). "centre guess" = predict the crop centre: the chance level
of the window. One row per (checkpoint tag, manifest, year, solver); "peak" = published one-peak-
per-patch RANSAC, "means" = distinct GMM means.


| tag | manifest | year | solver | row | median m [95 % CI] | R@5 [CI] | R@10 | >30 m | matched |
|---|---|---|---|---|---|---|---|---|---|
| aug | manifest | 2025 | srt | peak | 11.5 [9.4, 13.6] | 0.17 [0.12, 0.23] | 0.47 | 0.10 | 200/200 |
| aug | manifest | 2025 | srt | means | 11.3 [9.1, 13.9] | 0.19 [0.14, 0.24] | 0.47 | 0.11 | 200/200 |
| aug | manifest | 2024 | srt | peak | 12.5 [10.6, 14.1] | 0.19 [0.14, 0.24] | 0.41 | 0.09 | 200/200 |
| aug | manifest | 2024 | srt | means | 12.0 [10.6, 13.9] | 0.21 [0.16, 0.28] | 0.41 | 0.10 | 200/200 |
| aug | manifest_test | 2025 | srt | peak | 17.2 [12.8, 19.0] | 0.13 [0.09, 0.18] | 0.34 | 0.17 | 200/200 |
| aug | manifest_test | 2025 | srt | means | 16.5 [12.8, 18.7] | 0.12 [0.08, 0.17] | 0.34 | 0.15 | 200/200 |
| aug | manifest_test | 2024 | srt | peak | 16.0 [13.1, 18.4] | 0.16 [0.11, 0.21] | 0.35 | 0.19 | 200/200 |
| aug | manifest_test | 2024 | srt | means | 16.0 [12.1, 19.1] | 0.14 [0.09, 0.19] | 0.35 | 0.16 | 200/200 |
| hybrid_warm | manifest | 2025 | srt | peak | 11.3 [10.5, 13.0] | 0.18 [0.14, 0.24] | 0.41 | 0.10 | 200/200 |
| hybrid_warm | manifest | 2025 | srt | means | 10.5 [8.9, 11.9] | 0.17 [0.12, 0.22] | 0.47 | 0.10 | 200/200 |
| hybrid_warm | manifest | 2024 | srt | peak | 10.4 [8.9, 13.0] | 0.18 [0.14, 0.24] | 0.49 | 0.10 | 199/200 |
| hybrid_warm | manifest | 2024 | srt | means | 11.6 [9.4, 13.2] | 0.21 [0.16, 0.28] | 0.45 | 0.11 | 199/200 |
| hybrid_warm | manifest_test | 2025 | srt | peak | 15.9 [13.4, 18.4] | 0.14 [0.09, 0.18] | 0.34 | 0.14 | 196/200 |
| hybrid_warm | manifest_test | 2025 | srt | means | 15.1 [12.8, 17.9] | 0.15 [0.10, 0.20] | 0.36 | 0.13 | 198/200 |
| hybrid_warm | manifest_test | 2024 | srt | peak | 16.4 [13.7, 20.4] | 0.14 [0.10, 0.20] | 0.34 | 0.17 | 199/200 |
| hybrid_warm | manifest_test | 2024 | srt | means | 16.0 [13.4, 19.4] | 0.17 [0.11, 0.21] | 0.33 | 0.17 | 199/200 |
| ipm | manifest | 2025 | srt | peak | 5.5 [4.3, 6.9] | 0.46 [0.39, 0.53] | 0.72 | 0.06 | 200/200 |
| ipm | manifest | 2025 | srt | means | 5.9 [4.6, 7.0] | 0.45 [0.38, 0.52] | 0.71 | 0.04 | 200/200 |
| ipm | manifest | 2024 | srt | peak | 5.9 [4.8, 6.7] | 0.45 [0.39, 0.52] | 0.71 | 0.05 | 199/200 |
| ipm | manifest | 2024 | srt | means | 5.8 [5.0, 6.6] | 0.43 [0.37, 0.50] | 0.72 | 0.04 | 199/200 |
| ipm | manifest_test | 2025 | srt | peak | 4.6 [3.9, 6.5] | 0.53 [0.45, 0.59] | 0.67 | 0.07 | 200/200 |
| ipm | manifest_test | 2025 | srt | means | 4.5 [3.9, 5.9] | 0.52 [0.45, 0.59] | 0.69 | 0.07 | 200/200 |
| ipm | manifest_test | 2024 | srt | peak | 5.7 [3.9, 7.5] | 0.48 [0.41, 0.56] | 0.66 | 0.11 | 200/200 |
| ipm | manifest_test | 2024 | srt | means | 5.6 [4.1, 7.7] | 0.46 [0.39, 0.53] | 0.66 | 0.10 | 200/200 |
| multi | manifest | 2025 | srt | peak | 11.9 [10.7, 14.6] | 0.15 [0.10, 0.20] | 0.40 | 0.12 | 200/200 |
| multi | manifest | 2025 | srt | means | 11.5 [10.2, 13.4] | 0.15 [0.10, 0.20] | 0.41 | 0.12 | 200/200 |
| multi | manifest | 2024 | srt | peak | 13.5 [11.6, 15.4] | 0.14 [0.09, 0.18] | 0.37 | 0.12 | 200/200 |
| multi | manifest | 2024 | srt | means | 13.2 [11.2, 15.4] | 0.16 [0.11, 0.21] | 0.39 | 0.10 | 200/200 |
| multi | manifest_test | 2025 | srt | peak | 17.5 [14.3, 20.3] | 0.11 [0.07, 0.15] | 0.29 | 0.17 | 200/200 |
| multi | manifest_test | 2025 | srt | means | 16.7 [13.7, 18.8] | 0.10 [0.06, 0.14] | 0.29 | 0.15 | 200/200 |
| multi | manifest_test | 2024 | srt | peak | 16.3 [12.8, 19.6] | 0.12 [0.07, 0.17] | 0.34 | 0.19 | 200/200 |
| multi | manifest_test | 2024 | srt | means | 15.2 [12.3, 18.1] | 0.11 [0.07, 0.15] | 0.35 | 0.18 | 200/200 |
| pose_nll | manifest | 2025 | srt | peak | 13.6 [10.8, 15.3] | 0.14 [0.09, 0.18] | 0.41 | 0.11 | 200/200 |
| pose_nll | manifest | 2025 | srt | means | 13.3 [10.9, 15.2] | 0.13 [0.08, 0.18] | 0.39 | 0.12 | 200/200 |
| pose_nll | manifest | 2024 | srt | peak | 13.1 [11.7, 15.8] | 0.18 [0.14, 0.24] | 0.39 | 0.13 | 199/200 |
| pose_nll | manifest | 2024 | srt | means | 12.3 [10.8, 13.6] | 0.17 [0.12, 0.23] | 0.39 | 0.12 | 199/200 |
| pose_nll | manifest_test | 2025 | srt | peak | 16.6 [12.9, 19.8] | 0.12 [0.08, 0.17] | 0.32 | 0.17 | 200/200 |
| pose_nll | manifest_test | 2025 | srt | means | 15.2 [12.3, 19.2] | 0.12 [0.09, 0.17] | 0.33 | 0.17 | 200/200 |
| pose_nll | manifest_test | 2024 | srt | peak | 16.5 [14.0, 21.5] | 0.12 [0.07, 0.17] | 0.29 | 0.21 | 200/200 |
| pose_nll | manifest_test | 2024 | srt | means | 16.2 [13.3, 21.5] | 0.09 [0.05, 0.13] | 0.29 | 0.21 | 200/200 |
| seq | manifest | 2025 | srt | peak | 11.3 [9.7, 13.3] | 0.16 [0.12, 0.21] | 0.45 | 0.10 | 200/200 |
| seq | manifest | 2025 | srt | means | 10.8 [9.3, 13.3] | 0.20 [0.14, 0.26] | 0.46 | 0.10 | 200/200 |
| seq | manifest | 2024 | srt | peak | 13.0 [11.2, 15.5] | 0.17 [0.12, 0.23] | 0.40 | 0.10 | 200/200 |
| seq | manifest | 2024 | srt | means | 12.1 [10.1, 14.1] | 0.20 [0.15, 0.26] | 0.43 | 0.10 | 200/200 |
| seq | manifest_test | 2025 | srt | peak | 16.7 [13.0, 21.1] | 0.14 [0.09, 0.18] | 0.34 | 0.20 | 200/200 |
| seq | manifest_test | 2025 | srt | means | 16.0 [13.9, 19.1] | 0.11 [0.07, 0.15] | 0.33 | 0.20 | 200/200 |
| seq | manifest_test | 2024 | srt | peak | 16.2 [13.5, 20.6] | 0.13 [0.09, 0.18] | 0.34 | 0.21 | 200/200 |
| seq | manifest_test | 2024 | srt | means | 15.6 [13.9, 19.0] | 0.10 [0.06, 0.13] | 0.32 | 0.21 | 200/200 |
| seq_single | manifest | 2025 | srt | peak | 12.0 [9.5, 15.1] | 0.12 [0.08, 0.17] | 0.46 | 0.12 | 200/200 |
| seq_single | manifest | 2025 | srt | means | 12.1 [10.1, 14.4] | 0.17 [0.12, 0.23] | 0.43 | 0.11 | 200/200 |
| seq_single | manifest | 2024 | srt | peak | 13.5 [10.2, 15.6] | 0.17 [0.12, 0.21] | 0.41 | 0.14 | 200/200 |
| seq_single | manifest | 2024 | srt | means | 11.7 [10.0, 13.5] | 0.17 [0.12, 0.22] | 0.42 | 0.11 | 200/200 |
| seq_single | manifest_test | 2025 | srt | peak | 16.6 [14.2, 20.1] | 0.12 [0.07, 0.17] | 0.29 | 0.22 | 200/200 |
| seq_single | manifest_test | 2025 | srt | means | 17.3 [14.3, 20.1] | 0.10 [0.07, 0.15] | 0.31 | 0.18 | 200/200 |
| seq_single | manifest_test | 2024 | srt | peak | 17.8 [14.0, 23.8] | 0.11 [0.07, 0.15] | 0.30 | 0.23 | 200/200 |
| seq_single | manifest_test | 2024 | srt | means | 18.2 [12.5, 22.6] | 0.11 [0.07, 0.15] | 0.30 | 0.23 | 200/200 |
| years | manifest | 2025 | srt | peak | 12.4 [9.7, 15.1] | 0.15 [0.10, 0.21] | 0.45 | 0.10 | 200/200 |
| years | manifest | 2025 | srt | means | 12.0 [9.8, 14.8] | 0.16 [0.11, 0.21] | 0.43 | 0.10 | 200/200 |
| years | manifest | 2024 | srt | peak | 13.3 [11.2, 16.8] | 0.18 [0.13, 0.23] | 0.40 | 0.12 | 200/200 |
| years | manifest | 2024 | srt | means | 12.3 [10.7, 15.3] | 0.17 [0.12, 0.22] | 0.39 | 0.12 | 200/200 |
| years | manifest_test | 2025 | srt | peak | 14.9 [11.6, 18.9] | 0.17 [0.12, 0.21] | 0.38 | 0.19 | 200/200 |
| years | manifest_test | 2025 | srt | means | 15.9 [12.5, 18.8] | 0.13 [0.09, 0.18] | 0.36 | 0.18 | 200/200 |
| years | manifest_test | 2024 | srt | peak | 14.5 [12.0, 17.4] | 0.15 [0.10, 0.20] | 0.39 | 0.16 | 200/200 |
| years | manifest_test | 2024 | srt | means | 14.3 [11.6, 17.2] | 0.15 [0.10, 0.20] | 0.37 | 0.16 | 200/200 |
| centre guess | manifest | 2024 | – | chance | 16.9 [15.7, 17.8] | 0.05 [0.03, 0.08] | 0.17 | 0.01 | 200/200 |
| centre guess | manifest | 2025 | – | chance | 16.9 [15.7, 17.8] | 0.05 [0.03, 0.08] | 0.17 | 0.01 | 200/200 |
| centre guess | manifest_test | 2024 | – | chance | 18.3 [16.9, 19.4] | 0.04 [0.01, 0.07] | 0.15 | 0.01 | 200/200 |
| centre guess | manifest_test | 2025 | – | chance | 18.3 [16.9, 19.4] | 0.04 [0.01, 0.07] | 0.15 | 0.01 | 200/200 |

Sources: eval_aug_manifest.json, eval_aug_manifest_test.json, eval_hybrid_warm_manifest.json, eval_hybrid_warm_manifest_test.json, eval_ipm_manifest.json, eval_ipm_manifest_test.json, eval_multi_manifest.json, eval_multi_manifest_test.json, eval_pose_nll_manifest.json, eval_pose_nll_manifest_test.json, eval_seq_manifest.json, eval_seq_manifest_test.json, eval_seq_single_manifest.json, eval_seq_single_manifest_test.json, eval_years_manifest.json, eval_years_manifest_test.json.

## Verdict (running notes; newest first)

**2026-09-23, Task 2 gate: the camera-only RGB-IPM query wins by a wide margin.** Flat-ground IPM of
the panorama (camera height 1.7 m, no depth, no LiDAR, no learned lift) pushed through the frozen
`sat493m` encoder with the decoder fine-tuned by the same recipe as every other run (3000 steps,
cross-year references, hinge, pose NLL): validation 5.5 m median [4.3, 6.9], R@5 0.46, R@10 0.72;
**test route 4.6 m [3.9, 6.5], R@5 0.53, R@10 0.67, >30 m 0.07**, cross-year 2024 5.7 m. Every lifted
query sits at 15–18 m on the same test frames (chance 18.3 m). The intervals do not overlap; the
median is a factor three to four lower; and the number holds on the route no training or selection
saw, which the lifted queries' validation numbers did not. This reproduces BevSplat's Tab. 3 ordering
(IPM ≫ Lift-Splat-Shoot under a fixed matcher) on our data. Consequences: the learned depth-bin lift
is retired as the camera-only method; the "hybrid" reduces to "IPM + something for above-horizon
content", and the question becomes what that something is (ERP-token query, plan Task 6, running);
the training-time proxies (windowed CE 3.47, top-1 12.6 %) were the right early signal.

**2026-09-23, TEST manifest (irAsBUK, reserved route, 200 frames × 2025/2024).** The six lifted-BEV
checkpoints (`multi`, `years`, `aug`, `seq`, `seq_single`, `pose_nll`) score 14.9–17.5 m median at 2025
against a centre-guess of 18.3 m; R@5 0.11–0.17 against 0.04; R@10 0.29–0.38 against 0.15; the `>30 m`
tail is 0.17–0.22 against 0.01. They are better than chance but by only 1–3 m of median, and every
interval overlaps every other. The validation-route table below (11–13 m) was flattered by
checkpoint selection on that route: the same checkpoints lose 3–5 m when moved to a route no
training or selection ever saw. Multi-frame (`seq` vs `seq_single`) and pose NLL (`pose_nll`) change
nothing. Implication: the learned depth-bin lift does not generalise; the query, not the loss, is the
lever (plan Task 6, ERP-token query with placement after matching, is the response). Pending rows:
`ipm`, `hybrid`, `hybrid_warm`.

**2026-09-23, validation manifest (IcRzj, 200 frames × 2025/2024), single-frame re-scores.**
The four Lift-Splat checkpoints (`multi`, `years`, `aug`, `seq`) are statistically indistinguishable:
peak medians 11.5–12.4 m at 2025 and 12.5–13.5 m at 2024, every 95 % interval overlapping every other,
R@5 between 0.13 and 0.19, R@10 between 0.37 and 0.47. All of them beat the centre-guess row
(16.9 m, R@5 0.05, R@10 0.17) by more than the interval width, so the matcher does localise, but
none of the training changes since the first multi-route run (cross-year references, augmentation,
multi-frame + pose NLL) moved the number beyond noise. The earlier "8.7 m, R@5 27 %" for `seq` came
from n = 48 frames selected on this same route; on 200 frames it reads 11.9 m, R@5 0.13.
Note: this table scores `seq` single-frame; the multi-frame (0/2/5 m) re-score is a separate row
(`seq` with `seq_dists 0,2,5`) once it lands. Implication: per plan Task 1 gate, no checkpoint
"wins"; Task 2 (IPM baseline) and Task 3 (dense hybrid query) proceed, and the paper's story cannot
rest on loss or augmentation variants of this query.
