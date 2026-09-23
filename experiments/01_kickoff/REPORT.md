# H1 — unchanged Sat-RoMa on EA-covered frames

200 frames, evenly spaced from the 690 whose full 224 m reference footprint is inside the EA 2009 mosaic (`min_frac` 0.999). Checkpoint `0t1q66hy`, published RANSAC (one peak per patch), patch validity 0.5. Reference offset and rotation are the config defaults (±30 % of the edge, ±55°). Roll = pitch = 0. Recall counts a failed RANSAC as a miss. Median position is over the matches only.

| Query | Matched | Recall @1 / 5 / 10 m | Median position | Median yaw | Median argmax |
|---|---:|---:|---:|---:|---:|
| IPM below contact line | 182/200 | 0.5% / 6% / 12.5% | 66 m | 29° | 68 m |
| Oracle A, LiDAR only | 131/200 | 0% / 6.5% / 14% | 58 m | 29° | 66 m |
| Oracle B, ground IPM + LiDAR | 193/200 | 1% / 13% / 20% | 43 m | 28° | 58 m |

Sat-RoMa’s own overhead-to-overhead number is 84% within 5 m. None of these queries is in that regime. Oracle B is the least bad, so metric placement helps, and it is not enough for the frozen matcher.

## Cosine probe

Mean-removed cosine between each valid query token and the reference token at its ground-truth cell. A shared mean is removed first. Chance is about 0.

| Query | sat493m vs EA | sat493m vs NLP | ConvNeXt-T LVD vs EA | ConvNeXt-T LVD vs NLP |
|---|---:|---:|---:|---:|
| IPM below contact line | 0.028 | 0.033 | 0.014 | 0.036 |
| Oracle A | 0.028 | 0.041 | 0.026 | 0.056 |
| Oracle B | 0.056 | 0.043 | 0.030 | 0.050 |

EA versus NLP does not separate. ConvNeXt-Tiny LVD does not agree with the map more than `sat493m` does. This ConvNeXt is the frozen ground-pretrained trunk, not `tudho9xp`.

Per-frame table: `experiments/01_kickoff/h1_ea/frames.csv`. Eight early overlays are in that folder; frame 0’s footprint sits in grass beside a track in the 2009 photo.

## Frozen-decoder ablation

`experiments/02_fusion/nlp_run5_frozen_decoder_groundfill`, 500 steps, NLP intensity, 1760/32 frames, `lr_decoder` 0, ground fill on. Validation argmax 13.1 cells (52 m), cross-entropy 7.29. Run 3 at the same step, decoder fine-tuned and no ground fill, was 11.1 cells (45 m) and 6.87. The two changes were applied together, and the combination is worse than run 3.
