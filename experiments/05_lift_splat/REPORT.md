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
| erp_long_se2 | manifest | 2025 | se2 | peak | 12.4 [10.5, 14.6] | 0.19 [0.14, 0.24] | 0.41 | 0.15 | 200/200 |
| erp_long_se2 | manifest | 2025 | se2 | means | 11.7 [10.0, 13.1] | 0.22 [0.17, 0.28] | 0.43 | 0.12 | 200/200 |
| erp_long_se2 | manifest | 2024 | se2 | peak | 11.9 [10.2, 13.9] | 0.23 [0.18, 0.29] | 0.42 | 0.13 | 199/200 |
| erp_long_se2 | manifest | 2024 | se2 | means | 10.4 [9.1, 13.0] | 0.21 [0.16, 0.28] | 0.48 | 0.12 | 199/200 |
| erp_long_se2 | manifest_test | 2025 | se2 | peak | 13.1 [10.8, 14.9] | 0.14 [0.10, 0.20] | 0.41 | 0.14 | 200/200 |
| erp_long_se2 | manifest_test | 2025 | se2 | means | 12.6 [10.5, 14.4] | 0.16 [0.11, 0.21] | 0.41 | 0.14 | 200/200 |
| erp_long_se2 | manifest_test | 2024 | se2 | peak | 11.2 [9.5, 12.4] | 0.22 [0.17, 0.28] | 0.46 | 0.14 | 200/200 |
| erp_long_se2 | manifest_test | 2024 | se2 | means | 11.2 [9.4, 12.8] | 0.20 [0.14, 0.26] | 0.47 | 0.11 | 200/200 |
| erp_long_srt | manifest | 2025 | srt | peak | 12.3 [10.6, 14.4] | 0.20 [0.15, 0.26] | 0.41 | 0.17 | 200/200 |
| erp_long_srt | manifest | 2025 | srt | means | 11.6 [9.1, 13.2] | 0.23 [0.17, 0.28] | 0.45 | 0.12 | 200/200 |
| erp_long_srt | manifest | 2024 | srt | peak | 11.4 [9.8, 13.9] | 0.23 [0.17, 0.28] | 0.45 | 0.12 | 199/200 |
| erp_long_srt | manifest | 2024 | srt | means | 12.4 [9.8, 14.4] | 0.20 [0.15, 0.27] | 0.45 | 0.14 | 199/200 |
| erp_long_srt | manifest_test | 2025 | srt | peak | 13.6 [11.4, 15.0] | 0.15 [0.10, 0.20] | 0.40 | 0.15 | 199/200 |
| erp_long_srt | manifest_test | 2025 | srt | means | 12.3 [9.9, 14.3] | 0.15 [0.10, 0.20] | 0.45 | 0.14 | 200/200 |
| erp_long_srt | manifest_test | 2024 | srt | peak | 11.8 [10.0, 13.0] | 0.21 [0.16, 0.28] | 0.43 | 0.14 | 200/200 |
| erp_long_srt | manifest_test | 2024 | srt | means | 11.3 [9.2, 13.0] | 0.19 [0.14, 0.24] | 0.47 | 0.12 | 200/200 |
| erp_se2 | manifest | 2025 | se2 | peak | 12.6 [9.9, 16.5] | 0.14 [0.10, 0.20] | 0.43 | 0.17 | 194/200 |
| erp_se2 | manifest | 2025 | se2 | means | 12.4 [10.5, 15.6] | 0.18 [0.13, 0.24] | 0.42 | 0.15 | 194/200 |
| erp_se2 | manifest | 2024 | se2 | peak | 11.6 [9.6, 13.8] | 0.17 [0.12, 0.21] | 0.45 | 0.12 | 197/200 |
| erp_se2 | manifest | 2024 | se2 | means | 11.2 [9.4, 14.2] | 0.14 [0.10, 0.20] | 0.47 | 0.12 | 197/200 |
| erp_se2 | manifest_test | 2025 | se2 | peak | 13.8 [11.3, 17.6] | 0.11 [0.07, 0.15] | 0.34 | 0.18 | 197/200 |
| erp_se2 | manifest_test | 2025 | se2 | means | 13.6 [11.3, 17.6] | 0.12 [0.08, 0.17] | 0.35 | 0.17 | 197/200 |
| erp_se2 | manifest_test | 2024 | se2 | peak | 12.6 [11.2, 16.0] | 0.13 [0.08, 0.17] | 0.38 | 0.15 | 198/200 |
| erp_se2 | manifest_test | 2024 | se2 | means | 13.0 [11.4, 15.7] | 0.15 [0.10, 0.20] | 0.38 | 0.12 | 198/200 |
| erp_srt | manifest | 2025 | srt | peak | 13.5 [10.6, 17.4] | 0.14 [0.10, 0.18] | 0.41 | 0.18 | 191/200 |
| erp_srt | manifest | 2025 | srt | means | 13.1 [10.5, 16.1] | 0.15 [0.10, 0.20] | 0.41 | 0.15 | 193/200 |
| erp_srt | manifest | 2024 | srt | peak | 11.5 [10.1, 15.1] | 0.16 [0.12, 0.21] | 0.43 | 0.15 | 193/200 |
| erp_srt | manifest | 2024 | srt | means | 12.4 [10.3, 14.4] | 0.16 [0.12, 0.21] | 0.41 | 0.12 | 197/200 |
| erp_srt | manifest_test | 2025 | srt | peak | 14.3 [11.7, 18.3] | 0.10 [0.06, 0.14] | 0.32 | 0.22 | 192/200 |
| erp_srt | manifest_test | 2025 | srt | means | 13.3 [11.9, 17.2] | 0.12 [0.08, 0.17] | 0.32 | 0.18 | 197/200 |
| erp_srt | manifest_test | 2024 | srt | peak | 13.8 [11.5, 17.0] | 0.14 [0.10, 0.19] | 0.36 | 0.17 | 196/200 |
| erp_srt | manifest_test | 2024 | srt | means | 12.9 [10.9, 15.4] | 0.14 [0.09, 0.18] | 0.38 | 0.15 | 197/200 |
| hybrid | manifest | 2025 | srt | peak | 14.4 [12.6, 17.3] | 0.10 [0.06, 0.14] | 0.34 | 0.13 | 193/200 |
| hybrid | manifest | 2025 | srt | means | 14.5 [11.8, 16.9] | 0.09 [0.05, 0.14] | 0.35 | 0.14 | 195/200 |
| hybrid | manifest | 2024 | srt | peak | 15.6 [12.8, 18.6] | 0.09 [0.05, 0.12] | 0.30 | 0.17 | 187/200 |
| hybrid | manifest | 2024 | srt | means | 13.9 [12.1, 16.5] | 0.10 [0.07, 0.14] | 0.32 | 0.15 | 193/200 |
| hybrid | manifest_test | 2025 | srt | peak | 19.3 [16.2, 21.9] | 0.12 [0.08, 0.17] | 0.27 | 0.28 | 175/200 |
| hybrid | manifest_test | 2025 | srt | means | 19.2 [15.4, 21.2] | 0.12 [0.08, 0.17] | 0.26 | 0.25 | 182/200 |
| hybrid | manifest_test | 2024 | srt | peak | 17.2 [15.4, 21.2] | 0.12 [0.08, 0.17] | 0.29 | 0.26 | 183/200 |
| hybrid | manifest_test | 2024 | srt | means | 17.3 [14.6, 21.0] | 0.12 [0.07, 0.17] | 0.28 | 0.25 | 189/200 |
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
| ipm_mosaic025 | manifest | 2025 | srt | peak | 5.8 [4.7, 7.2] | 0.46 [0.39, 0.53] | 0.74 | 0.04 | 200/200 |
| ipm_mosaic025 | manifest | 2025 | srt | means | 5.6 [4.6, 7.2] | 0.46 [0.39, 0.53] | 0.73 | 0.04 | 200/200 |
| ipm_mosaic025 | manifest | 2024 | srt | peak | 6.6 [5.4, 7.4] | 0.42 [0.36, 0.49] | 0.71 | 0.07 | 200/200 |
| ipm_mosaic025 | manifest | 2024 | srt | means | 6.3 [5.3, 7.2] | 0.42 [0.35, 0.49] | 0.70 | 0.05 | 200/200 |
| ipm_mosaic025 | manifest_test | 2025 | srt | peak | 5.4 [4.5, 6.1] | 0.46 [0.39, 0.52] | 0.69 | 0.07 | 200/200 |
| ipm_mosaic025 | manifest_test | 2025 | srt | means | 5.5 [4.5, 7.0] | 0.47 [0.41, 0.55] | 0.69 | 0.09 | 200/200 |
| ipm_mosaic025 | manifest_test | 2024 | srt | peak | 5.2 [4.2, 6.8] | 0.49 [0.42, 0.56] | 0.66 | 0.09 | 199/200 |
| ipm_mosaic025 | manifest_test | 2024 | srt | means | 5.4 [4.4, 7.0] | 0.47 [0.41, 0.54] | 0.68 | 0.10 | 199/200 |
| ipm_mosaic0510 | manifest | 2025 | srt | peak | 5.9 [4.6, 6.9] | 0.46 [0.39, 0.53] | 0.76 | 0.03 | 199/200 |
| ipm_mosaic0510 | manifest | 2025 | srt | means | 5.8 [4.9, 6.7] | 0.44 [0.37, 0.51] | 0.77 | 0.03 | 199/200 |
| ipm_mosaic0510 | manifest | 2024 | srt | peak | 5.7 [4.6, 6.7] | 0.47 [0.40, 0.54] | 0.69 | 0.05 | 200/200 |
| ipm_mosaic0510 | manifest | 2024 | srt | means | 5.3 [4.5, 6.2] | 0.47 [0.41, 0.55] | 0.71 | 0.04 | 200/200 |
| ipm_mosaic0510 | manifest_test | 2025 | srt | peak | 6.1 [4.7, 6.8] | 0.46 [0.39, 0.53] | 0.69 | 0.10 | 200/200 |
| ipm_mosaic0510 | manifest_test | 2025 | srt | means | 5.5 [4.7, 7.1] | 0.46 [0.40, 0.53] | 0.68 | 0.10 | 200/200 |
| ipm_mosaic0510 | manifest_test | 2024 | srt | peak | 5.1 [4.3, 7.3] | 0.49 [0.42, 0.56] | 0.66 | 0.10 | 200/200 |
| ipm_mosaic0510 | manifest_test | 2024 | srt | means | 5.5 [4.3, 7.4] | 0.48 [0.41, 0.55] | 0.66 | 0.12 | 200/200 |
| ipm_mosaic_trained | manifest | 2025 | srt | peak | 5.5 [4.8, 6.9] | 0.46 [0.40, 0.53] | 0.77 | 0.03 | 200/200 |
| ipm_mosaic_trained | manifest | 2025 | srt | means | 5.4 [4.6, 6.6] | 0.47 [0.41, 0.54] | 0.77 | 0.03 | 200/200 |
| ipm_mosaic_trained | manifest | 2024 | srt | peak | 5.6 [4.4, 6.7] | 0.46 [0.39, 0.53] | 0.74 | 0.04 | 200/200 |
| ipm_mosaic_trained | manifest | 2024 | srt | means | 5.7 [4.3, 6.9] | 0.47 [0.40, 0.54] | 0.73 | 0.04 | 200/200 |
| ipm_se2 | manifest | 2025 | se2 | peak | 5.7 [4.7, 6.9] | 0.44 [0.37, 0.51] | 0.72 | 0.05 | 200/200 |
| ipm_se2 | manifest | 2025 | se2 | means | 5.5 [4.6, 7.0] | 0.45 [0.39, 0.52] | 0.73 | 0.04 | 200/200 |
| ipm_se2 | manifest | 2024 | se2 | peak | 6.0 [4.8, 6.7] | 0.45 [0.39, 0.52] | 0.73 | 0.06 | 199/200 |
| ipm_se2 | manifest | 2024 | se2 | means | 6.1 [4.9, 6.6] | 0.43 [0.36, 0.50] | 0.72 | 0.04 | 199/200 |
| ipm_se2 | manifest_test | 2025 | se2 | peak | 4.5 [3.7, 5.8] | 0.52 [0.45, 0.59] | 0.68 | 0.09 | 200/200 |
| ipm_se2 | manifest_test | 2025 | se2 | means | 4.8 [3.9, 6.0] | 0.52 [0.45, 0.59] | 0.68 | 0.08 | 200/200 |
| ipm_se2 | manifest_test | 2024 | se2 | peak | 5.6 [4.1, 7.0] | 0.48 [0.41, 0.56] | 0.67 | 0.10 | 200/200 |
| ipm_se2 | manifest_test | 2024 | se2 | means | 5.5 [4.1, 6.9] | 0.47 [0.40, 0.53] | 0.67 | 0.10 | 200/200 |
| ipm_sim | manifest | 2025 | sim | peak | 5.8 [4.4, 6.7] | 0.47 [0.41, 0.55] | 0.72 | 0.05 | 200/200 |
| ipm_sim | manifest | 2025 | sim | means | 5.5 [4.6, 7.0] | 0.46 [0.40, 0.54] | 0.71 | 0.03 | 200/200 |
| ipm_sim | manifest | 2024 | sim | peak | 6.0 [5.0, 6.8] | 0.43 [0.36, 0.50] | 0.71 | 0.06 | 199/200 |
| ipm_sim | manifest | 2024 | sim | means | 6.2 [5.0, 6.8] | 0.43 [0.36, 0.50] | 0.71 | 0.04 | 199/200 |
| ipm_sim | manifest_test | 2025 | sim | peak | 4.5 [3.9, 6.2] | 0.52 [0.44, 0.58] | 0.65 | 0.09 | 200/200 |
| ipm_sim | manifest_test | 2025 | sim | means | 4.7 [3.9, 6.0] | 0.51 [0.43, 0.58] | 0.67 | 0.09 | 200/200 |
| ipm_sim | manifest_test | 2024 | sim | peak | 6.0 [4.3, 7.8] | 0.46 [0.38, 0.53] | 0.65 | 0.11 | 200/200 |
| ipm_sim | manifest_test | 2024 | sim | means | 5.6 [4.1, 7.8] | 0.46 [0.39, 0.53] | 0.65 | 0.10 | 200/200 |
| ipm_v2 | manifest | 2025 | srt | peak | 5.3 [4.6, 6.3] | 0.47 [0.40, 0.54] | 0.74 | 0.03 | 200/200 |
| ipm_v2 | manifest | 2025 | srt | means | 5.5 [4.3, 6.5] | 0.46 [0.39, 0.53] | 0.72 | 0.04 | 200/200 |
| ipm_v2 | manifest | 2024 | srt | peak | 6.1 [5.1, 6.8] | 0.42 [0.35, 0.49] | 0.73 | 0.06 | 200/200 |
| ipm_v2 | manifest | 2024 | srt | means | 6.4 [5.1, 7.1] | 0.41 [0.35, 0.48] | 0.72 | 0.06 | 200/200 |
| ipm_v2 | manifest_test | 2025 | srt | peak | 5.0 [4.1, 6.3] | 0.49 [0.42, 0.56] | 0.67 | 0.09 | 200/200 |
| ipm_v2 | manifest_test | 2025 | srt | means | 4.6 [4.0, 6.3] | 0.52 [0.44, 0.58] | 0.67 | 0.08 | 200/200 |
| ipm_v2 | manifest_test | 2024 | srt | peak | 6.0 [5.0, 7.3] | 0.43 [0.36, 0.50] | 0.63 | 0.12 | 200/200 |
| ipm_v2 | manifest_test | 2024 | srt | means | 6.4 [4.8, 7.9] | 0.46 [0.39, 0.52] | 0.62 | 0.12 | 200/200 |
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

Sources: eval_aug_manifest.json, eval_aug_manifest_test.json, eval_erp_long_se2_manifest.json, eval_erp_long_se2_manifest_test.json, eval_erp_long_srt_manifest.json, eval_erp_long_srt_manifest_test.json, eval_erp_se2_manifest.json, eval_erp_se2_manifest_test.json, eval_erp_srt_manifest.json, eval_erp_srt_manifest_test.json, eval_hybrid_manifest.json, eval_hybrid_manifest_test.json, eval_hybrid_warm_manifest.json, eval_hybrid_warm_manifest_test.json, eval_ipm_manifest.json, eval_ipm_manifest_test.json, eval_ipm_mosaic025_manifest.json, eval_ipm_mosaic025_manifest_test.json, eval_ipm_mosaic0510_manifest.json, eval_ipm_mosaic0510_manifest_test.json, eval_ipm_mosaic_trained_manifest.json, eval_ipm_se2_manifest.json, eval_ipm_se2_manifest_test.json, eval_ipm_sim_manifest.json, eval_ipm_sim_manifest_test.json, eval_ipm_v2_manifest.json, eval_ipm_v2_manifest_test.json, eval_multi_manifest.json, eval_multi_manifest_test.json, eval_pose_nll_manifest.json, eval_pose_nll_manifest_test.json, eval_seq_manifest.json, eval_seq_manifest_test.json, eval_seq_single_manifest.json, eval_seq_single_manifest_test.json, eval_years_manifest.json, eval_years_manifest_test.json.

## Verdict (running notes; newest first)

**2026-09-24, IPM family on the test route (2025, peak row): a plateau.** `ipm` 4.6 m / R@5 0.53 /
R@10 0.67; 10k steps (`ipm_long`) 4.3 / 0.55 / 0.70; trained mosaic 0/2/5 m 4.8 / 0.52 / 0.71;
zero-shot mosaic 5.4 / 0.46 / 0.69; ego cut + dynamic mask (`ipm_v2`) 5.0 / 0.49 / 0.67. All within
one another's intervals; only the mosaic's R@10 moves consistently (+4–5 points on both routes).
Diagnosis (`make pose-diag-semantic` on `ipm_long`, test route): the quartile of frames with the
least above-ground structure in the panorama scores 6.5 m median / R@10 0.62, the two most
structured quartiles 2.8–4.3 m / 0.76–0.80; vegetation fraction is the only class that correlates
with error (+0.23). The hard set is tree-lined straight road: the road picture cannot break the
along-road ambiguity and the orthophoto shows canopy where the panorama shows trunks. Next on the
query side: the camera-only contact-line picture (`ipm_cl`: wall/fence/hedge feet from the semantic
map painted at their ground foot, kick-off §3.1 H2 without LiDAR), queued. Beyond that the levers
are the filter (already 4.4 m along the route) and the reference side, not more picture variants.

**2026-09-24, code review of the branch (`/code-review high`, 10 findings) and what was done.**
Fixed: (1) `sim` solver reported a failed RANSAC (identity from `ransac_init`) as a match — now a miss
like `srt`; the `ipm_sim` rows above were scored before the fix and may contain a few identity
"matches" counted as >30 m, which only makes `sim` look worse, never better; (2) bootstrap intervals
became NaN when a resample's median was inf (`np.quantile` interpolation) — now `method="nearest"`;
(3) `eval_pose` ran the decoder twice per entry and built two extra ViT-L copies just to toggle
`use_means` — now one encoder/decoder pass and two consensus runs on the same logits; dead code and a
hard-coded `scale_factor=0.4` removed; (4) `--query` override crashed on any mode other than the
checkpoint's — now loads the overlapping tensors and prints the mismatch; (5) the IPM mosaic decoded
each neighbour panorama twice with different attitude-noise draws — now one decode, same R as the
ERP stack; (6) `track_route` hard-coded the 224 px geometry (would crash on ERP checkpoints) and also
ran a second model — now uses the query grid's scale factor, the placement path, one model;
(7) CLAUDE.md "current task" pointed at task 01; (8) `ipm.height_m = 1.7` was undocumented (now a
decisions entry with a plan to fit it from the similarity solver's scale). Not fixed: (9) `HybridQuery`
CPU syncs per step — hybrid retired; (10) `argmax_m` is None for placed (ERP) queries by design and now
says so in the code. No finding changes a reported number except (1), in the conservative direction.

**2026-09-24, ERP-token query, 10 000 steps (`erp_long`).** Validation 12.3 m [10.6, 14.4], R@5 0.20;
test 13.6 m [11.4, 15.0], R@5 0.15, >30 m 0.15 (2024: 11.8 m, R@5 0.21); SE(2) solver identical
within noise. Tripling the schedule moved the test median by 0.7 m. Schedule is not the explanation;
the decoder-only fine-tune cannot turn ground-view tokens into overhead-matchable ones. Task 6's gate
fails as specified. A Loc²-style query-side projection head (trained, on top of the frozen encoder)
is the remaining variant of this idea and is a design decision (OPEN in docs/decisions.md).

**2026-09-24, ERP-token query (Task 6), 3000 steps.** Validation 13.5 m [10.6, 17.4], R@5 0.14;
test 14.3 m [11.7, 18.3], R@5 0.10, >30 m 0.22, 192/200 matched. Same band as the lifted queries,
three times worse than `ipm`. With the decoder fine-tuned alone (no query-side head), the ground-view
tokens do not become matchable to overhead tokens in 3000 steps; this is the same lesson as the
hybrid row below (feature placement ≠ picture placement). A 10 000-step run (`erp_long`) is queued to
separate "schedule" from "architecture"; if it does not move, the Loc²-style route needs a trained
projection head on the panorama side, which is a design decision (docs/decisions.md OPEN).

**2026-09-24, H7 solver ablation on the `ipm` query (test route, 2025).** Package default = 8-DoF
homography 4.6 m [3.9, 6.5] / R@5 0.53 / >30 m 0.07; 4-DoF similarity 4.5 [3.9, 6.2] / 0.52 / 0.09;
3-DoF fixed-scale SE(2) 4.5 [3.7, 5.8] / 0.52 / 0.09; validation route 5.5 / 5.8 / 5.7 m. All within
one another's intervals; the tail does not shrink. H7 ("metric scale pays off") is not supported here:
once the query is right, the consensus is not limited by solver freedom. The sheared boxes on the
miss frames are a symptom, not the cause. `srt` stays the reference row; `se2` is kept as an option.

**2026-09-23, hybrid query (dense IPM *features* + learned above-horizon splat, Task 3).** From
scratch: validation 14.4 m, test 19.3 m (= chance, 28 % beyond 30 m, 175/200 matched); warm-started
from the lift: 11.3 m / 15.9 m, i.e. the lift's own numbers. So placing the panorama's *tokens* on the
ground by exact IPM does not work, while placing the panorama's *pixels* on the ground and encoding
that picture (the `ipm` query, 4.6 m on test) does. Reading: what the frozen `sat493m` encoder + decoder
can match is an overhead-looking *image*; ground-view tokens moved to the right place are still
ground-view tokens, and a 3000–4000-step decoder fine-tune does not bridge that view gap. This is a
warning for the ERP-token query (Task 6), which hands ground-view tokens to the decoder directly and
asks it to bridge the gap alone; its early validation CE (4.9 at step 1000 vs 3.7 for `ipm`) is
consistent with that, and Loc² needs a trained projection head per branch plus long schedules for the
same reason. Task 3's gate fails; the dense-ground idea survives only in pixel form (IPM picture).

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
