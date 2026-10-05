# PanoRoMa vs FG² (and Loc²) on KITScenes, Frankfurt val scene 142f1419, 2026-10-05

One scene: 100 frames at 10 Hz (10 s, ~50 m of a straight bridge over the Main), `kit_vigor/frankfurt_142f1419_eagle` built ON
Eagle from the sources (`make eagle-fetch-kitscenes`: HuggingFace scene tar + Hessen DOP20 WMS, `scripts/kitscenes_to_vigor.py`),
VIGOR layout, known orientation (panorama rolled to grid north from the INS heading), tile = free 20 cm DOP20 resampled to 0.111262 m/px,
640 px, camera at a random offset inside 17.8 m of the tile centre (seed 0), exact labels. The six ring cameras stitch to a panorama with
image only in ±28° elevation (28 % of the pixels after the car rows are removed). Models: PanoRoMa D retrained on corrected labels
(experiments/18; coarse + second pass, gate 6 m, se2, bf16), the released same-area FG² and Loc² checkpoints; depth UniK3D v2 (checked
against LiDAR on 5 frames: UniK3D / LiDAR range median 1.01; 1.03 at 3–8 m, 1.02 at 8–15 m). Jobs: depth 8918598, PanoRoMa 8918599, FG²
8918600, Loc² 8918601; masked-depth reruns 8920139 / 8920140. Chance level (tile centre): 13.3 m median.

| Method | Median (95 % CI) | Mean | <= 5 m | <= 10 m |
|---|---|---|---|---|
| FG² | **2.34 m** (2.19–2.80) | **2.88 m** | **94 %** | 98 % |
| Loc² | 2.44 m (1.69–3.05) | 4.15 m | 73 % | 91 % |
| Loc², depth masked | 2.04 m (1.81–2.86) | 3.80 m | 77 % | 94 % |
| PanoRoMa, two pass | 4.70 m (3.83–6.02) | 9.41 m | 55 % | 71 % |
| PanoRoMa, two pass, depth masked | 4.57 m (3.78–5.74) | 9.02 m | 56 % | 72 % |
| PanoRoMa, first pass, depth masked | 5.14 m (3.97–6.96) | 9.69 m | 49 % | 67 % |

Depth masking (`scripts/kit_mask_depth.py`, DEPTH_DIR `unik3d_depth_v2_kitmask`): depth 0 where the panorama is black and in rows steeper than
20° below the horizon. It was chosen AFTER the first results (post hoc); the first rows stay the primary ones. Effect: PanoRoMa 4.70 -> 4.57 m,
Loc² 2.44 -> 2.04 m, both inside the intervals of the unmasked runs: black-pixel tokens are not what separates PanoRoMa from the others here.

Robustness: the same three methods on the dataset built on the laptop (identical labels, panoramas within 0.14/255 mean pixel difference,
depth computed locally): FG² 2.29, Loc² 2.44, PanoRoMa 4.71 m: same picture.

PanoRoMa errors: 84 % of the squared error is along the bridge axis (bearing 144°), the error across the bridge has a signed median of
−2.05 m (a registration offset of the orthophoto against the pose, or a PanoRoMa bias: not separated). Median error by third of the drive:
PanoRoMa 6.3 / 5.0 / 3.9 m, FG² 2.9 / 2.4 / 2.1 m, Loc² 2.1 / 3.1 / 2.3 m.

Limits: one 10-second trajectory on one straight bridge: the frames are strongly correlated, the intervals are wide, and a straight bridge deck
is ambiguous along its axis for any matcher. Not evidence about KITScenes in general. Untested explanations for the PanoRoMa gap: its
56 x 28-token grid has only ~3 token rows of ground in the ±28° band (a VIGOR panorama has ~14); its projection head mixes black and image
tokens with self-attention (masking the depth does not change the features); the along-axis ambiguity.
