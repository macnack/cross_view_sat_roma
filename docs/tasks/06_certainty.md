# Task 06 — A certainty that means something

**Status (26 Sep 2026):** steps 1–3 implemented, reviewed and run (results below); step 4 (pose-correctness head) pending the four-city Task 04 checkpoint. Written from the measured VIGOR rows
(experiments/09_vigor, 10_loc2_matcher) and the papers in docs/related.

## The question

Can the matcher say, per frame, how likely its pose is to be right — well enough that (a) a route filter can
down-weight bad frames and (b) a single-frame table can report abstentions honestly? Today's signals are weak
gates, not calibrated probabilities.

## What we have measured

- **RANSAC inlier ratio** (Chicago 30k, 3000 samples, decisions 2026-09-25): lowest quartile by inlier ratio has
  5.8 m median and 35 % of errors > 10 m; the rest 2.4 m and 5 %. Useful as a gate, but 65 % of the gated frames are
  fine and 5 % of the "confident" ones are gross. Many of the worst-20 misses have inlier ratios of 0.9–1.0: a
  wrong-but-consistent vote map is fully "inlier".
- **Vote-map shape** (worst-20 sheet): the 31–47 m misses have a diffuse blob centred on the tile, i.e. the decoder
  falls back on a layout prior. The *spread* of the categorical, not its peak, carries the information.
- **Decoder certainty** (`gm_certainty`): trained as "is this token's true cell inside the reference" (coarse CE's
  certainty term, weight 0.01); it says nothing about whether the *pose* is right.
- **Loc² Fig. 4**: mean error falls from 12.5 m at the 10 % inlier quantile to 1.75 m at 90 %; the same monotone but
  soft relation we see. FG²/Loc² report no calibrated confidence either.

## Why the current signals cannot be calibrated

The inlier ratio measures self-consistency of the modes RANSAC kept; a repeated street pattern gives consistent
wrong votes. Cell-membership certainty is a per-token quantity with no notion of the frame's pose. Neither is
trained against the event we care about ("pose error < τ"), so no monotone transform makes them calibrated.

## Designs, cheapest first

1. **Post-hoc calibration on features we already log (no training).** Fit, on the held-out 20 % of the training
   list, a small monotone/logistic model P(error < 5 m | inlier_peak, n_modes, vote entropy, top-1 mass, distance
   of the peak from the tile centre, mean depth of placed tokens). Evaluate with reliability diagrams and the
   abstention curve (median error vs coverage) on the test draw. Expected: an AUROC of ~0.75–0.8 from the inlier
   ratio alone; the entropy of the summed vote map should add the "blob" cases. Cost: one script, an hour of GPU.
2. **Vote-map statistics as first-class outputs.** Entropy and effective support of the per-frame vote map, the
   mass within 2 cells of the RANSAC pose, and the agreement between the peak and GMM-mean poses. All available
   in `consensus_from_gm` without a forward pass. Feeds 1.
3. **Multi-hypothesis spread.** Run RANSAC on K bootstrap subsets of the modes (or keep the top-K RANSAC hypotheses
   by score); the spread of the K poses is an epistemic-style uncertainty. Costs K solver runs, no training. This is
   the natural input to the particle filter (a mixture likelihood instead of a point).
4. **A certainty head trained on pose correctness.** Add a small head on the pooled decoder state (or on the vote
   map) predicting P(pose error < τ) for τ ∈ {2, 5, 10} m, trained with BCE against the RANSAC outcome of the
   *current* model on each training batch (labels come for free from H). This is the "certainty that means something":
   it is supervised by the event, sees the vote-map shape, and can learn the repeated-pattern failure. Risk: the
   label depends on the model that produced it (moving target); train it after the matcher, frozen, on the held-out
   split. Cost: a 30k-step head-only run.
5. **Conformal wrapper.** Whatever score 1 or 4 gives, split-conformal on the held-out list yields sets/abstentions
   with a guaranteed coverage at the chosen error radius — this is the honest single-frame claim for the paper.

## Where it pays

- **Filter (Task 5 of task 03 / idea 7):** a likelihood proportional to the calibrated P(correct) flattens exactly the
  frames that were flattened by hand in BEV-Patch-PF; expected to remove the filter's failure runs on Poznań.
- **Tables:** report "median at 90 % coverage" next to the full-coverage median; FG²'s 3 % gross misses vs our 12 %
  is the number this addresses, and abstaining on the right 10 % could bring the mean to Loc²'s level without
  changing the matcher.
- **Not a fix for the misses themselves**: the misses stay; this makes them known.

## Plan

Do 2 + 1 now (one script: `scripts/certainty_vigor.py`, reads an eval json + re-runs consensus stats, fits and
evaluates the calibrator, writes reliability and coverage plots), then 3 as an option in `eval_vigor.py`, then 4 as a
training run once the four-city Task 04 checkpoint exists. Gate for "means something": AUROC ≥ 0.85 for error < 5 m
and a coverage curve whose 90 % point has ≤ 5 % gross misses.

## Results (26 Sep 2026, Task 04 matcher at 2 m cells, `checkpoints/vigor_chicago_same_30k_erp_depth_cell0125_last.pt`)

Calibration frames: 1000 held-out Chicago training frames after the 200 used for checkpoint selection (`eval_vigor.py
--calib`); test frames: the 3000-sample Chicago draw every table uses. Files: `experiments/10_loc2_matcher/certainty_
chicago_same_30k_erp_depth_cell0125_last_se2.{json,png}`; jobs 8794472 (test), 8794473 (calibration), 8794474 (fit).

| Score | AUROC < 2 m | AUROC < 5 m | AUROC < 10 m | AP < 5 m | ECE | median @100 % | median @90 % | > 10 m @90 % | median @80 % | > 10 m @80 % | conformal cov. (0.90) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inlier ratio, isotonic | 0.697 | 0.804 | 0.806 | 0.934 | 0.015 | 1.76 m | 1.61 m | 7.7 % | 1.54 m | 5.7 % | 0.884 |
| logistic on the statistics | 0.797 | 0.892 | 0.887 | 0.973 | 0.031 | 1.76 m | 1.60 m | 7.5 % | 1.51 m | 4.4 % | 0.909 |
| oracle (sorted by true error) | | | | | | 1.76 m | 1.58 m | 1.7 % | 1.43 m | 0.0 % | |

Univariate AUROC for error < 5 m (values below 0.5 mean "lower is better"): ego_mass_2cells 0.888, ego_entropy 0.123,
ego_support_cells 0.123, ego_top1 0.876, vote_mass_2cells 0.873, vote_top1 0.863, vote_entropy 0.140,
vote_support_cells 0.140, inliers_peak 0.804, n_modes 0.206, peak_means_m 0.212, n_inlier_modes 0.314,
ego_peak_pose_m 0.360, n_valid_tokens 0.380, spread_hyp_m 0.431, pose_centre_m 0.459, cert_mean 0.464,
cert_median 0.520, placed_depth_m 0.499.

Reading. The ego-map statistics (each token's categorical shifted by its own offset from the vehicle, so agreeing tokens
pile up on the vehicle cell) are the strongest signals and carry what the inlier ratio misses: the diffuse-blob failure.
The calibrated logistic keeps 0 % gross misses up to 40 % coverage and under 2 % up to 60 %; at 90 % coverage it removes
a third of them (11.6 % → 7.5 %). The remaining gross misses are confidently wrong — a sharp, consistent vote at a
repeated street pattern — and no post-hoc score on the vote map separates them from correct frames (the oracle shows
what a perfect ranker would give). Gate: AUROC ≥ 0.85 passed; ≤ 5 % gross at 90 % missed (met at 80 %). The
conformal wrapper is honest (0.909 at a 0.90 target) but marginal and same-city only.

Consequences. (1) The calibrated score is ready for the particle filter's likelihood (idea 7). (2) Step 4, a head trained on
pose correctness, is the only route to the confidently-wrong half. (3) The ego-map construction should replace the
unshifted token map in the training pose loss (`pose_heatmap_nll`), see docs/decisions.md 2026-09-26.
