# Controlled crop test on VIGOR: PanoRoMa D evaluated on panoramas cut to +-27.6 deg of elevation, 2026-10-05

Same checkpoints (PanoRoMa D retrained on corrected labels, `vigor_cl_reg_D_*`, experiments/18), same 3000 Chicago same-area test frames (seed-0 draw),
corrected labels, two passes with the 6 m gate. The panorama AND its UniK3D depth are cut to the rows within +-27.6 deg of the horizon (314 of 1024 rows,
the image band of the KITScenes ring cameras) on the fly (`$BEVLOC_CROP_BAND_DEG`, `VigorPairs._crop`), resized to 896 x 128 px (8 token rows); every
token's ray and placement uses the band's elevations (`erp_band`, equal to the rays of the same rows of the full sphere by test). Jobs 8924264 / 8924265.

| Chicago 3000 | First pass median | Two-pass median (95 % CI) | Mean | <= 5 m | <= 10 m | second pass fell back to the first |
|---|---|---|---|---|---|---|
| uncropped (full sphere) | 1.41 m | **1.06 m** (1.02–1.09) | 2.22 m | 92 % | 95 % | 2 |
| cropped to +-27.6 deg | 5.20 m | 6.09 m (5.95–6.30) | 7.90 m | 38 % | 77 % | 791 |

Reading: a model trained on full-sphere panoramas collapses on VIGOR panoramas that are only cropped (5.2 m first pass vs 1.4 m), with the correct
depth and the band geometry. The depth scale and the black pixels of the KITScenes stitch are therefore not the reason cropped KITScenes panoramas fail
(10.2 m): the checkpoint is not usable on band input. The second pass (also trained on full-sphere panoramas) falls back to the coarse pose on 26 % of
the frames (gate 6 m) and does not repair it. Training on cropped panoramas (or with black rows) is the open question.
