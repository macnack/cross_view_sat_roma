## Verdict (running notes; newest first)

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
