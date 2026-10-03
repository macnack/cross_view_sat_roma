# The 0.43 m north offset of PanoRoMa D (2026-10-03)

On the full Chicago test split PanoRoMa D's predictions sit (+0.12 E, +0.43 N) m from the label on both label sets.
A per-axis fit residual = k * gt + b separates the radial shrinkage (k, the original-vs-corrected label scale, ~-3 % on
corrected labels) from a constant b; b_N = +0.44 m (coarse) / +0.46 m (fine), the same on both label sets.

Per city and per model (constant b of the fit, frames with error < 5 m; `make vigor-pose-offset`, eval jsons in
experiments/13–16):

| Model | geometric aug | coarse b_N per city | fine b_N per city |
|---|---|---|---|
| v2 plain e100 (all cities) | none | Chi +0.05, NY −0.04, SF −0.03, Sea +0.02 | +0.08, +0.05, −0.01, +0.03 |
| v1 e100 (Chicago) | none | +0.15 | +0.11 |
| reg A / C e30 (label smoothing / weight decay) | none | +0.11 / +0.19 | (old fine) +0.21 / +0.21 |
| reg B e30 (photometric + rot90 ±10° + flip) | yes | +0.58 | (old fine) +0.23 |
| reg D e30 (A+B+C) | yes | +0.54 | (old fine) +0.23 |
| v2 reg D e100 (all cities) | yes | Chi +0.42, NY +0.33, SF +0.34, Sea +0.50 | +0.45, +0.35, +0.25, +0.38 |
| v2 cross reg D e100 (cross-area) | yes | Chi +0.13, SF +0.07 | +0.42, +0.14 |

**Cause: the geometric augmentation, not a coordinate bug.** Every model trained without rot90/flip has |b| <= 0.2 m,
every model trained with it +0.25–0.58 m north, in all four cities. The augmentation itself is label-consistent
(rotation about (S−1)/2 on canvas, label and ERP roll; `tests/test_augment_labels_review.py`). Reading: the imagery-
consistent camera position differs from VIGOR's label by a roughly constant vector b (pointing north, i.e. along the
panorama's centre column, which in VIGOR is always north). Without rotation the model learns the labels, b included.
Rotating map + panorama together rotates b with the sample; quarter turns average R·b to zero, so the model learns the
imagery position and misses the labels by b at test time. Whether b lives in the labels (Street View GPS vs tile
registration) or in the panorama (e.g. a systematic placement shift towards the panorama's centre column) cannot be told
apart on VIGOR, where both point north.

Consequence: subtracting b would take the corrected-label median from 1.19 to 1.11 m (fit on the test split itself, an
upper bound). Fix candidates (not decided): (1) estimate b per city on the validation part of the TRAINING split and
subtract it at test time; (2) keep only east–west flips (they preserve b_N) and drop the rotations; (3) learn b as a
per-city offset that is rotated with the augmentation. The retrain on corrected labels (experiments/18) keeps
regularisation D unchanged, so the offset is expected there too and (1) can be applied post hoc.
