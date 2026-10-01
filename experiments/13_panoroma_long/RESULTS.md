# PanoRoMa, 100-epoch coarse decoder: evaluation at epochs 10 / 40 / 80 / 100

Status 2026-09-30. Coarse run 8859503 (4× H100 DDP, global batch 32, TF32, bf16 encoder + bf16 decoder, compiled
encoder; 131,500 steps, 13 h 17 min), tag `samearea_4city_erp_depth_cell0125_e100`. "Two-pass" here still uses the OLD
second-pass decoder (`vigor_samearea_4city_fine00625_erp_depth_last.pt`, 60k steps, fp16); the new 100-epoch fine run is
8867975. Evaluations with `--decoder-dtype bfloat16`. Files: `eval_vigor_e100_ep*_chicago_twopass_se2_samearea.json`,
`poznan_e100_ep*/`; training curve `train_samearea_4city_erp_depth_cell0125_e100.csv` (and W&B run
`cross_view_sat_roma/samearea_4city_erp_depth_cell0125_e100`).

## VIGOR Chicago same-area, 3000 samples (seed 0), se2, fine gate 6 m

| Checkpoint | Coarse median (95 % CI) | Coarse mean | Two-pass median (95 % CI) | Two-pass mean | ≤ 5 m | ≤ 10 m |
|---|---|---|---|---|---|---|
| Old (60k steps, batch 4, fp16) | 1.77 m (1.70–1.82) | 3.68 m | 1.24 m (1.19–1.29) | 3.33 m | 85 % | 91 % |
| Epoch 10 | 1.57 m (1.52–1.63) | 3.37 m | 1.23 m (1.19–1.26) | 3.20 m | 86 % | 92 % |
| Epoch 40 | 1.54 m (1.50–1.60) | 3.35 m | 1.22 m (1.19–1.27) | 3.17 m | 85 % | 92 % |
| Epoch 80 | 1.43 m (1.39–1.48) | 3.08 m | 1.22 m (1.18–1.27) | 3.01 m | 86 % | 92 % |
| **Epoch 100** | **1.37 m (1.33–1.41)** | **2.91 m** | **1.22 m (1.17–1.27)** | **2.89 m** | 87 % | 93 % |
| FG² (released, same draw) | | | 1.06 m | 1.86 m | 95 % | 97 % |
| Loc² (released, same draw) | | | 1.33 m | 2.93 m | 88 % | 93 % |

## Poznań zero-shot, 400 entries, prior heading, 71 m imagery extent

| Checkpoint | Coarse median (95 % CI) | Two-pass median (95 % CI) | Two-pass mean | ≤ 10 m | > 30 m | Heading |
|---|---|---|---|---|---|---|
| Old | 5.34 m | 4.53 m (3.88–5.12) | 7.87 m | 76 % | 3 % | 4.2° |
| Epoch 10 | 6.08 m (5.44–6.65) | 5.13 m (4.51–5.76) | 8.94 m | 70 % | 6 % | 4.7° |
| Epoch 40 | 4.66 m (3.98–5.52) | 4.50 m (4.02–5.08) | 7.74 m | 76 % | 4 % | 4.9° |
| Epoch 80 | 4.84 m (4.33–5.60) | 4.45 m (3.87–5.15) | 8.27 m | 75 % | 6 % | 4.5° |
| **Epoch 100** | **4.66 m (3.99–5.09)** | **4.37 m (3.86–4.90)** | 7.80 m | 77 % | 4 % | 4.3° |
| Loc², UniK3D (zero-shot) | | 3.50 m | 6.29 m | 85 % | 3 % | 2.0° |
| FG² (zero-shot) | | 4.15 m | 5.70 m | 86 % | 0 % | 2.4° |

## Readings

- The coarse pass keeps improving to epoch 100 on VIGOR: median 1.77 → 1.37 m, mean 3.68 → 2.91 m. Validation top-1
  12.0 → 19.9 %, validation pose error 3.65 → 3.13 m; validation cross-entropy rises from epoch ~15 (over-confidence
  on wrong frames) without hurting localisation.
- With the old second-pass decoder the two-pass median is stuck at 1.22 m: the old fine decoder is now the limit. The
  two-pass MEAN still improves (3.33 → 2.89 m, now below Loc²'s 2.93 m) because the better coarse pose puts more
  frames inside the fine window.
- Poznań: epoch 10 was worse than the old checkpoint, by epoch 100 the two-pass median is 4.37 m (old 4.53 m), coarse
  4.66 m (old 5.34 m) — longer VIGOR training does not hurt transfer in the end. Still ~0.9 m behind Loc²-UniK3D.
- Next: the 100-epoch fine decoder (8867975), then two-pass rows with both new decoders on VIGOR (three splits) and
  Poznań.

## Both decoders at 100 epochs (2026-10-01)

Coarse `..._cell0125_e100_last` + fine `vigor_samearea_4city_fine00625_erp_depth_e100_last` (job 8867975, 100 epochs, bf16): **VIGOR Chicago 3000 two-pass 1.05 m (95 % CI 1.02–1.09), mean 2.67 m, ≤ 5 m 89 %, ≤ 10 m 93 %** (job 8878168) — level with FG² (1.06 m) on the median, below Loc² (2.93 m) on the mean; FG²'s mean 1.86 m. Poznań zero-shot two-pass 4.55 m (3.83–5.10), worse than with the old fine decoder (4.37 m): the 100-epoch fine decoder specialises to VIGOR (job 8878170). All-cities row: job 8878169.

## Match figures of frames that work (`make panoroma-viz`, 2026-10-01)

`viz/`: coarse `vigor_samearea_4city_erp_depth_cell0125_e100_last.pt` + fine
`vigor_samearea_4city_fine00625_erp_depth_e100_last.pt`, se2, bf16 decoder, gate 6 m, on the Chicago same-area draw
(3000, seed 0) of `eval_vigor_e100_full_chicago_twopass_se2_samearea.json` (two-pass median 1.05 m, coarse 1.37 m).
Frames re-matched by `scripts/viz_panoroma_matches.py`; recomputed errors equal the json's on all 30 candidates.
Percentile = rank of the final error among the 3000 (lower is better). Six at 5/15/25/35/45/50 %, four more picked by
scene type from a 2-50 % pool (no frame directly under the elevated track was in the pool; #5 has a CTA station).

| # | Percentile | Panorama | Scene | Coarse | Final |
|---|---|---|---|---|---|
| 1 | 4 | MyX5FkowZD0bv8uqKUyy3A | park / formal garden | 1.31 m | 0.24 m |
| 2 | 5 | 4tiB8B4K68H824DmNXro8w | tree-lined street by a school | 0.93 m | 0.26 m |
| 3 | 6 | hPxxLmGcTeuVwEDge65eOA | intersection, traffic lights | 0.93 m | 0.29 m |
| 4 | 15 | -pARrCcO8EO0IWZR8MbK3A | straight residential road | 0.77 m | 0.47 m |
| 5 | 25 | wzC-iVJkVnH_nwaUx-DReA | shopping street, elevated CTA station behind | 1.64 m | 0.63 m |
| 6 | 28 | xz4ZV6_y3CzDt5iuGbTIBw | downtown high-rise intersection | 1.84 m | 0.68 m |
| 7 | 35 | 8WxJW96en5rf7iR39zqbYA | straight road along a long building | 4.24 m | 0.79 m |
| 8 | 45 | nCA8gB-vqKOVvHhz5UaZBg | alley with trees | 1.50 m | 0.95 m |
| 9 | 46 | X5uEYl2qj9_92lhAiesP4A | open parking lot | 1.38 m | 0.97 m |
| 10 | 50 | S0kh4Wsbuf6ZGODaaXLKJA | open plaza | 2.29 m | 1.05 m |

`viz/contact_sheet.jpg` = all ten; `viz/diagram_explained.jpg` = #5 with the parts labelled.

Seen in the figures: the stored UniK3D depth PNGs (`vigor/<City>/unik3d_depth/`) have row stripes: neighbouring rows
differ by ~2 m on average in the lower half (columns: 0.001 m), on both files checked. Sky rows therefore alternate
between "too far" and a few metres, and some sky tokens get placed (visible as dotted sky rows in panel 1). Worth
checking `scripts/loc2_depth_vigor.py` / the UniK3D spherical camera before the next training run.
