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
| multi | manifest | 2025 | srt | peak | 11.9 [10.7, 14.6] | 0.15 [0.10, 0.20] | 0.40 | 0.12 | 200/200 |
| multi | manifest | 2025 | srt | means | 11.5 [10.2, 13.4] | 0.15 [0.10, 0.20] | 0.41 | 0.12 | 200/200 |
| multi | manifest | 2024 | srt | peak | 13.5 [11.6, 15.4] | 0.14 [0.09, 0.18] | 0.37 | 0.12 | 200/200 |
| multi | manifest | 2024 | srt | means | 13.2 [11.2, 15.4] | 0.16 [0.11, 0.21] | 0.39 | 0.10 | 200/200 |
| seq | manifest | 2025 | srt | peak | 11.9 [9.7, 14.6] | 0.13 [0.09, 0.17] | 0.45 | 0.12 | 200/200 |
| seq | manifest | 2025 | srt | means | 11.7 [9.8, 14.4] | 0.17 [0.12, 0.23] | 0.45 | 0.11 | 200/200 |
| seq | manifest | 2024 | srt | peak | 13.4 [10.4, 15.8] | 0.17 [0.12, 0.21] | 0.41 | 0.14 | 200/200 |
| seq | manifest | 2024 | srt | means | 11.9 [9.4, 13.4] | 0.18 [0.14, 0.23] | 0.45 | 0.10 | 200/200 |
| years | manifest | 2025 | srt | peak | 12.4 [9.7, 15.1] | 0.15 [0.10, 0.21] | 0.45 | 0.10 | 200/200 |
| years | manifest | 2025 | srt | means | 12.0 [9.8, 14.8] | 0.16 [0.11, 0.21] | 0.43 | 0.10 | 200/200 |
| years | manifest | 2024 | srt | peak | 13.3 [11.2, 16.8] | 0.18 [0.13, 0.23] | 0.40 | 0.12 | 200/200 |
| years | manifest | 2024 | srt | means | 12.3 [10.7, 15.3] | 0.17 [0.12, 0.22] | 0.39 | 0.12 | 200/200 |
| centre guess | manifest | 2024 | – | chance | 16.9 [15.7, 17.8] | 0.05 [0.03, 0.08] | 0.17 | 0.01 | 200/200 |
| centre guess | manifest | 2025 | – | chance | 16.9 [15.7, 17.8] | 0.05 [0.03, 0.08] | 0.17 | 0.01 | 200/200 |

Sources: eval_aug_manifest.json, eval_multi_manifest.json, eval_seq_manifest.json, eval_years_manifest.json.

## Verdict (running notes; newest first)

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
