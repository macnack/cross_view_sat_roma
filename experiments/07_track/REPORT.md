# Route tracking with an SE(2) particle filter (plan Task 5)

`scripts/track_route.py` (`make track-route`), observation = Sat-RoMa's certainty-weighted vote heatmap
from the **camera-only IPM query** (`checkpoints/05_lift_splat_fixtor_ipm_best.pt`), 512 particles,
reference crop re-rendered every frame around the filter's own estimate (no random window). Motion
between frames = Mapillary proxy relative pose + N(0, 1 m / 1°) noise, which is also the process
noise; it stands in for odometry, so the dead-reckoning row is a strawman that drifts by design.
Errors are against the Mapillary pose proxy, not survey GT. Every frame of the route, in order.

| route | frames | PF median | PF p95 | PF ≤5 m | PF ≤10 m | per-frame RANSAC median / p95 / ≤10 m | prior (prediction) median |
|---|---|---|---|---|---|---|---|
| irAsBUK (test, 2025) | 1265 | **4.4 m** | 21.6 m | 0.56 | 0.83 | 4.5 m / 26.3 m / 0.77 | 4.6 m |
| IcRzj (validation, 2025) | 1980 | **4.2 m** | 14.1 m | 0.59 | 0.88 | 5.1 m / 19.5 m / 0.78 | 4.5 m |

Files: `track_<route>_ipm_y2025.json` (per-frame errors and positions), `track_<route>_ipm_y2025.jpg`
(trajectory over EN with proxy, filter, per-frame RANSAC and dead reckoning; error vs frame).

Reading. The filter keeps lock over the whole 4.3 km without any absolute fix after initialisation:
its prediction alone is already at 4.6 m median, so the observation model is doing steady work every
frame. Against per-frame RANSAC on the same crops the filter gains little at the median (4.4 vs
4.5 m) and more in the tail (p95 21.6 vs 26.3 m, ≤10 m 0.83 vs 0.77): it suppresses isolated wrong
frames but follows runs of consistent wrong observations for 20–50 frames (frames ~50–100, ~250–300,
~620–660, ~870–920 in the plot) before recovering, which is what a heatmap that is wrong in the same
direction for a stretch of road does to a filter with no likelihood flattening. Plan Task 5 gate
(median ≤ 5 m on the test route): met, with the odometry caveat above. Next lever if the tail
matters: BEV-Patch-PF's per-frame uncertainty flattening (their Eq. 5), and real odometry instead of
noised proxies.
