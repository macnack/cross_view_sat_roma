# Poznań three-way comparison: IPM + Sat-RoMa, FG², Loc², PanoRoMa

Written by `make poznan-three-way` (scripts/report_poznan_three_way.py). Manifest `experiments/06_fg2_bevsplat/manifest.json`, years [2025, 2024]: 200 held-out frames of route IcRzj × 2 orthophoto years; reference = the manifest's crop (224 m at 0.25 m/px for IPM; each method's native extent) centred within ±22 m of the position proxy. FG², Loc² and PanoRoMa are zero-shot (VIGOR-trained); the IPM row's checkpoint was trained on four other Poznań Fixtor routes and selected on this route (IcRzj = bevloc.data.mapillary.VAL_SEQS), so it is not zero-shot and the manifest is its dev set (manifest_test.json, route irAsB, is the untouched one). Error = distance to the Mapillary pose proxy (not survey GT). Median with 95 % bootstrap interval, mean capped at 1 km, recalls over all entries (failures count as misses), heading error median over the entries with a pose.

Heading protocols: **prior** = the panorama is oriented by the manifest's noisy heading (crop_up, U(−10°, 10°) off the proxy) and the method must recover the rest (what the IPM row always did); **gt** = oriented by the proxy heading itself (what the reported FG² / Loc² 'native' rows did: their panorama was rolled by crop_rot_deg, which uses the true heading).

## Protocol `prior`

| method | year | n | median m (95 % CI) | mean m | R@1 | R@5 | R@10 | > 30 m | heading median ° |
|---|---|---|---|---|---|---|---|---|---|
| IPM + Sat-RoMa (existing) | all | 400 | 5.69 (4.94–6.54) | 11.61 | 0.06 | 0.45 | 0.71 | 0.05 | 2.6 |
| IPM + Sat-RoMa (existing) | 2024 | 200 | 5.95 (4.77–6.67) | 14.24 | 0.06 | 0.45 | 0.71 | 0.05 | 2.5 |
| IPM + Sat-RoMa (existing) | 2025 | 200 | 5.51 (4.34–6.85) | 8.98 | 0.06 | 0.46 | 0.72 | 0.06 | 2.7 |
| FG² native | all | 400 | 4.15 (3.82–4.56) | 5.70 | 0.06 | 0.58 | 0.86 | 0.00 | 2.4 |
| FG² native | 2024 | 200 | 4.12 (3.42–4.60) | 5.81 | 0.07 | 0.59 | 0.85 | 0.00 | 2.0 |
| FG² native | 2025 | 200 | 4.15 (3.85–4.94) | 5.60 | 0.06 | 0.57 | 0.86 | 0.00 | 2.6 |
| Loc² native flat depth | all | 400 | 3.91 (3.47–4.25) | 6.80 | 0.06 | 0.62 | 0.83 | 0.03 | 2.3 |
| Loc² native flat depth | 2024 | 200 | 3.65 (3.34–4.07) | 6.66 | 0.06 | 0.66 | 0.84 | 0.03 | 2.3 |
| Loc² native flat depth | 2025 | 200 | 4.17 (3.59–4.70) | 6.93 | 0.07 | 0.59 | 0.83 | 0.03 | 2.3 |
| Loc² native UniK3D | all | 400 | 3.50 (3.26–3.77) | 6.29 | 0.06 | 0.68 | 0.85 | 0.03 | 2.0 |
| Loc² native UniK3D | 2024 | 200 | 3.38 (2.98–3.84) | 6.29 | 0.06 | 0.67 | 0.85 | 0.04 | 1.9 |
| Loc² native UniK3D | 2025 | 200 | 3.62 (3.30–4.18) | 6.29 | 0.06 | 0.69 | 0.85 | 0.03 | 2.1 |
| PanoRoMa coarse (crop) | all | 400 | 5.20 (4.65–5.83) | 7.76 | 0.03 | 0.49 | 0.77 | 0.02 | 5.3 |
| PanoRoMa coarse (crop) | 2024 | 200 | 4.85 (4.45–5.77) | 7.55 | 0.02 | 0.51 | 0.79 | 0.02 | 5.3 |
| PanoRoMa coarse (crop) | 2025 | 200 | 5.49 (4.63–6.43) | 7.98 | 0.03 | 0.47 | 0.76 | 0.02 | 5.1 |
| PanoRoMa two-pass (crop) | all | 400 | 5.01 (4.50–5.57) | 7.67 | 0.03 | 0.50 | 0.77 | 0.03 | 4.4 |
| PanoRoMa two-pass (crop) | 2024 | 200 | 4.50 (3.78–5.06) | 7.26 | 0.03 | 0.56 | 0.78 | 0.03 | 3.9 |
| PanoRoMa two-pass (crop) | 2025 | 200 | 5.57 (4.81–6.38) | 8.08 | 0.02 | 0.44 | 0.77 | 0.03 | 4.8 |
| PanoRoMa fine (ungated) (crop) | all | 400 | 4.97 (4.46–5.57) | 7.66 | 0.03 | 0.50 | 0.77 | 0.03 | 4.4 |
| PanoRoMa fine (ungated) (crop) | 2024 | 200 | 4.45 (3.71–5.05) | 7.24 | 0.03 | 0.56 | 0.78 | 0.03 | 3.9 |
| PanoRoMa fine (ungated) (crop) | 2025 | 200 | 5.57 (4.81–6.38) | 8.08 | 0.02 | 0.44 | 0.77 | 0.03 | 4.8 |
| PanoRoMa coarse (north) | all | 400 | 5.46 (5.20–6.16) | 8.70 | 0.02 | 0.42 | 0.75 | 0.04 | 5.7 |
| PanoRoMa coarse (north) | 2024 | 200 | 5.38 (5.07–6.32) | 8.21 | 0.03 | 0.42 | 0.77 | 0.04 | 5.6 |
| PanoRoMa coarse (north) | 2025 | 200 | 5.59 (5.02–6.56) | 9.18 | 0.01 | 0.42 | 0.73 | 0.05 | 5.8 |
| PanoRoMa two-pass (north) | all | 400 | 4.75 (4.03–5.39) | 8.05 | 0.06 | 0.53 | 0.76 | 0.04 | 4.4 |
| PanoRoMa two-pass (north) | 2024 | 200 | 4.75 (3.98–5.82) | 7.70 | 0.06 | 0.54 | 0.79 | 0.04 | 4.4 |
| PanoRoMa two-pass (north) | 2025 | 200 | 4.75 (3.86–5.74) | 8.39 | 0.06 | 0.52 | 0.74 | 0.04 | 4.6 |
| PanoRoMa fine (ungated) (north) | all | 400 | 4.83 (4.09–5.51) | 8.08 | 0.06 | 0.52 | 0.76 | 0.04 | 4.4 |
| PanoRoMa fine (ungated) (north) | 2024 | 200 | 4.84 (4.02–6.09) | 7.77 | 0.06 | 0.53 | 0.79 | 0.04 | 4.4 |
| PanoRoMa fine (ungated) (north) | 2025 | 200 | 4.83 (3.89–5.74) | 8.39 | 0.06 | 0.52 | 0.74 | 0.04 | 4.7 |
| PanoRoMa coarse (north_ext71) | all | 400 | 5.34 (4.69–6.14) | 8.15 | 0.05 | 0.47 | 0.78 | 0.02 | 4.7 |
| PanoRoMa coarse (north_ext71) | 2024 | 200 | 5.43 (4.38–6.37) | 8.24 | 0.06 | 0.47 | 0.77 | 0.03 | 4.4 |
| PanoRoMa coarse (north_ext71) | 2025 | 200 | 5.27 (4.51–6.24) | 8.05 | 0.04 | 0.47 | 0.79 | 0.01 | 4.9 |
| PanoRoMa two-pass (north_ext71) | all | 400 | 4.53 (3.88–5.12) | 7.87 | 0.05 | 0.54 | 0.76 | 0.03 | 4.2 |
| PanoRoMa two-pass (north_ext71) | 2024 | 200 | 4.31 (3.49–5.46) | 7.95 | 0.04 | 0.54 | 0.76 | 0.04 | 3.8 |
| PanoRoMa two-pass (north_ext71) | 2025 | 200 | 4.64 (3.83–5.20) | 7.78 | 0.06 | 0.55 | 0.77 | 0.02 | 4.6 |
| PanoRoMa fine (ungated) (north_ext71) | all | 400 | 4.63 (3.92–5.15) | 7.90 | 0.05 | 0.54 | 0.76 | 0.03 | 4.2 |
| PanoRoMa fine (ungated) (north_ext71) | 2024 | 200 | 4.57 (3.76–5.64) | 8.01 | 0.04 | 0.53 | 0.76 | 0.04 | 3.8 |
| PanoRoMa fine (ungated) (north_ext71) | 2025 | 200 | 4.67 (3.85–5.21) | 7.79 | 0.06 | 0.54 | 0.77 | 0.02 | 4.6 |

## Protocol `gt`

| method | year | n | median m (95 % CI) | mean m | R@1 | R@5 | R@10 | > 30 m | heading median ° |
|---|---|---|---|---|---|---|---|---|---|
| FG² native (reported, task 02) | all | 400 | 4.24 (3.79–4.61) | 5.57 | 0.07 | 0.59 | 0.85 | 0.00 | – |
| FG² native (reported, task 02) | 2024 | 200 | 3.95 (3.35–4.71) | 5.59 | 0.07 | 0.60 | 0.84 | 0.01 | – |
| FG² native (reported, task 02) | 2025 | 200 | 4.35 (3.82–4.90) | 5.56 | 0.07 | 0.58 | 0.86 | 0.00 | – |
| Loc² native flat depth (reported, task 02) | all | 400 | 4.03 (3.58–4.35) | 6.77 | 0.06 | 0.62 | 0.83 | 0.03 | – |
| Loc² native flat depth (reported, task 02) | 2024 | 200 | 3.67 (3.33–4.19) | 6.68 | 0.06 | 0.65 | 0.83 | 0.03 | – |
| Loc² native flat depth (reported, task 02) | 2025 | 200 | 4.21 (3.61–4.69) | 6.86 | 0.07 | 0.60 | 0.83 | 0.03 | – |
| FG² native | all | 400 | 3.98 (3.63–4.62) | 5.54 | 0.05 | 0.59 | 0.86 | 0.00 | 2.0 |
| FG² native | 2024 | 200 | 3.76 (3.31–4.67) | 5.60 | 0.06 | 0.59 | 0.86 | 0.01 | 2.0 |
| FG² native | 2025 | 200 | 4.32 (3.74–4.91) | 5.49 | 0.04 | 0.59 | 0.86 | 0.00 | 2.0 |
| Loc² native flat depth | all | 400 | 4.01 (3.60–4.28) | 6.83 | 0.06 | 0.61 | 0.83 | 0.03 | 2.3 |
| Loc² native flat depth | 2024 | 200 | 3.76 (3.24–4.18) | 6.66 | 0.05 | 0.64 | 0.83 | 0.03 | 2.2 |
| Loc² native flat depth | 2025 | 200 | 4.26 (3.75–4.91) | 7.01 | 0.07 | 0.58 | 0.82 | 0.03 | 2.6 |
| Loc² native UniK3D | all | 400 | 3.53 (3.22–3.83) | 6.33 | 0.07 | 0.67 | 0.85 | 0.03 | 1.9 |
| Loc² native UniK3D | 2024 | 200 | 3.40 (3.10–3.83) | 6.28 | 0.06 | 0.67 | 0.86 | 0.04 | 1.9 |
| Loc² native UniK3D | 2025 | 200 | 3.63 (3.32–4.21) | 6.38 | 0.08 | 0.67 | 0.84 | 0.03 | 2.0 |
| PanoRoMa coarse (north) | all | 400 | 5.41 (4.93–6.09) | 8.73 | 0.02 | 0.46 | 0.74 | 0.05 | 3.7 |
| PanoRoMa coarse (north) | 2024 | 200 | 5.28 (4.55–6.24) | 8.37 | 0.03 | 0.47 | 0.75 | 0.04 | 3.6 |
| PanoRoMa coarse (north) | 2025 | 200 | 5.54 (4.82–6.28) | 9.09 | 0.01 | 0.45 | 0.74 | 0.05 | 3.9 |
| PanoRoMa two-pass (north) | all | 400 | 4.73 (4.13–5.34) | 8.24 | 0.06 | 0.52 | 0.76 | 0.04 | 2.3 |
| PanoRoMa two-pass (north) | 2024 | 200 | 4.57 (3.89–5.79) | 7.98 | 0.06 | 0.53 | 0.78 | 0.04 | 2.1 |
| PanoRoMa two-pass (north) | 2025 | 200 | 4.97 (4.02–5.61) | 8.50 | 0.05 | 0.51 | 0.73 | 0.04 | 2.6 |
| PanoRoMa fine (ungated) (north) | all | 400 | 4.88 (4.15–5.45) | 8.24 | 0.06 | 0.51 | 0.76 | 0.04 | 2.4 |
| PanoRoMa fine (ungated) (north) | 2024 | 200 | 4.73 (4.12–5.88) | 8.00 | 0.06 | 0.52 | 0.78 | 0.04 | 2.1 |
| PanoRoMa fine (ungated) (north) | 2025 | 200 | 5.00 (4.09–5.63) | 8.48 | 0.05 | 0.50 | 0.73 | 0.04 | 2.6 |

## Entry identity

Every row is on the same (frame id, year) entries.

## UniK3D camera-height check

Implied camera height from road points [-20.0, -8.0]° below the horizon, ±30.0° fore/aft (200 panoramas, 200 restricted to Cityscapes road): median 1.05 m (p10 0.90, p90 1.33); flat-ground proxy of the reported Loc² row 1.65 m, IPM height 1.7 m.

## Error CDF

![CDF](cdf.png)
