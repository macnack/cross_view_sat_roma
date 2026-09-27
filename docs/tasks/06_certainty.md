# Task 06 — A certainty that means something

**Status (27 Sep 2026):** steps 1–4 implemented, reviewed and run (results below). Step 4's verdict: the head does not beat the calibrated logistic; the confidently-wrong frames are not separable from the matcher's own evidence. Certainty of record: the logistic on the 19 statistics (or the token-stream head, equal within noise). Written from the measured VIGOR rows
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

## Step 4 — pose-correctness head (design agreed 27 Sep 2026, Maciej: "start from 1")

**Question.** Can a head that sees the frozen matcher's vote maps *and* per-token evidence, trained against the event
"coarse pose error < τ", beat the post-hoc logistic (AUROC 0.892, 7.5 % gross at 90 % coverage) and reach the missed
gate (≤ 5 % gross at 90 % coverage)? The confidently-wrong half is the target.

**Matcher.** Frozen: `checkpoints/vigor_samearea_4city_erp_depth_cell0125_last.pt` (Task 04, four cities, 60k,
`configs/vigor_cell0125.yaml`, `--solver se2`, the default consensus — sweep 2026-09-27 kept it). The head predicts the
correctness of the **coarse** pose (the fine window is centred on it, so a wrong coarse pose is unrecoverable).

**Data.** Labels and inputs come from one pass of the frozen matcher over frames it never trained on:
- train/val of the head: the checkpoint's held-out 20 % of the four-city training list after the 400 selection frames
  (`eval_vigor.py --calib` rule, `--assume-train-split --val-samples 400 --train-cities Chicago NewYork SanFrancisco
  Seattle` since this checkpoint's train dict predates the split record), up to 8000 frames, all four cities; the last
  1000 of them are the head's validation frames (early stopping / temperature); nothing else is tuned on them;
- test: the 3000-sample Chicago draw and the 12000-sample all-cities draw every table uses (`VigorPairs(limit, seed 0)`),
  disjoint from the training list by construction.
Per frame the cache stores (float16, ~120 KB): ego vote map (56×56, `vote_stats.ego_vote_map`), reference vote map
(56×56, `vote_stats.vote_map`), the pose footprint (56×56: query tokens sent through the coarse H, rasterised as inlier
= +1 / outlier = −1 by the consensus's inlier set), the reference-cell validity (56×56), per-token rows (T ≤ 1568 × 8:
p at the cell H sends the token to, top-1 mass, entropy of the token's categorical, certainty logit, distance of the
token's peak from the pose-consistent cell in cells, placed depth m, azimuth column / 56, row / 28), the 19 frame
statistics `Match.stats` already gives (`certainty_vigor.py` feature names), the coarse error in m, the frame id and
city. Script: `scripts/certainty_cache_vigor.py` (`make vigor-cert-cache`), built on `eval_vigor.py`'s loaders and
`consensus_for_query`; the cache is the sweep cache's pattern (`experiments/10_loc2_matcher/cert_cache/<tag>_<draw>.pkl`),
with the same reproduction gate: the cached error must equal `eval_vigor.py`'s for the default consensus.

**Head** (`src/bevloc/model/certainty_head.py`, `CertaintyHead`, < 1 M parameters):
- map stream: 4 channels × 56×56 → conv(32, s2) → conv(64, s2) → conv(64, s2) → GAP → 64;
- token stream: per-token MLP 8 → 64 → 64, masked attention pooling (learned query) → 64; plus mean and max over
  tokens of the pose-consistency p (2);
- frame stream: the 19 statistics, standardised on the training frames → 32;
- fusion: concat → 128 → 3 logits, one per τ ∈ {2, 5, 10} m (`certainty.targets_m`), BCE with logits, the τ = 5 m
  logit ranks the abstention curve (`certainty.rank_target_m`). Ablation rows by construction: `--streams map`,
  `--streams tokens`, `--streams frame`, all (the frame stream alone must reproduce the logistic within noise — the
  sanity check that the pipeline is right).
- training: AdamW 1e-3, weight decay 1e-4, batch 256, 60 epochs over ≤ 7000 frames, early stop on validation NLL at
  τ = 5 m, temperature scaling on the validation frames, seed in config, augmentation: random 90° rotations + flips of
  the four maps with the token azimuth column shifted consistently (the label is rotation invariant). Runs in minutes on
  a GPU. Script: `scripts/train_certainty_head.py` (`make vigor-cert-head`), config block `certainty_head:` in
  `configs/default.yaml`.

**Evaluation** (`scripts/certainty_vigor.py` extended with `--head <ckpt>`: the head's P(τ) enters the same tables as
the isotonic and logistic rows): AUROC / AP at 2, 5, 10 m, ECE and reliability, abstention curve (median and gross
fraction at 90 / 80 % coverage), split conformal at 0.90 — on the Chicago 3000 draw and on the all-cities 12000 draw
(per-city AUROC too: the head trains on four cities, the logistic was Chicago-only). Comparison rows: inlier ratio
isotonic, logistic on the 19 statistics (fit on the same 7000 frames), head with each stream, head full. The
`--hyp 8` bootstrap spread is not in the cache (8 solver runs per frame); dropped from this comparison.

**Gates.** Pipeline: frame-stream-only head within ±0.01 AUROC of the logistic on the Chicago draw. Result: full head
AUROC(< 5 m) ≥ 0.92 **and** ≤ 5 % gross at 90 % coverage on Chicago; all-cities AUROC ≥ 0.88. If the full head does not
beat the logistic by more than the bootstrap interval, the conclusion is that the confidently-wrong frames are not
separable from the matcher's own evidence, and the certainty work stops at step 1–3 (the calibrated logistic feeds the
filter).

**Not in scope.** Training the head jointly with the matcher (moving labels); the fine pass; Poznań routes.

### Step 4 results (27 Sep 2026, four-city Task 04 matcher, Chicago 3000 draw and all-cities 12000 draw)

Caches: 8000 held-out four-city training frames (`cert_cache/erpd4city_calib.pkl`, gate 0 mismatches, median 1.79 m,
5.2 % > 10 m), Chicago test 3000 (`erpd4city_chi3000_test.pkl`, 0 mismatches, 1.77 m / 9.1 % — the reported row). Heads
trained on the first 7000, validated on the last 1000 (early stop, temperature); the logistic and isotonic rows are fitted
on the same 7000. Files: `certainty_erpd4city_head_chi3000.{json,png}`, `certainty_head_erpd4city_*.{pt,json}`; jobs
8809944–8809951.

| Score (Chicago 3000) | AUROC < 2 m | AUROC < 5 m | AUROC < 10 m | AP < 5 m | ECE | median @90 % | > 10 m @90 % | median @80 % | > 10 m @80 % | conformal cov. |
|---|---|---|---|---|---|---|---|---|---|---|
| inlier ratio, isotonic | 0.676 | 0.814 | 0.840 | 0.947 | 0.022 | 1.65 m | 5.4 % | 1.59 m | 3.5 % | 0.882 |
| logistic, 19 statistics | 0.786 | 0.918 | 0.927 | 0.984 | 0.023 | 1.63 m | 4.7 % | 1.53 m | 2.4 % | 0.885 |
| head, map + tokens + frame (87 K) | 0.786 | 0.919 | 0.927 | 0.984 | 0.024 | 1.64 m | 4.5 % | 1.52 m | 2.1 % | 0.895 |
| head, map stream only (65 K) | 0.771 | 0.914 | 0.924 | 0.983 | 0.024 | 1.64 m | 4.5 % | 1.54 m | 2.0 % | 0.895 |
| head, token stream only (18 K) | 0.787 | 0.919 | 0.927 | 0.984 | 0.021 | 1.63 m | 4.3 % | 1.52 m | 2.2 % | 0.892 |
| head, frame stream only (5 K) | 0.785 | 0.913 | 0.915 | 0.982 | 0.028 | 1.64 m | 4.6 % | 1.53 m | 2.2 % | 0.890 |
| oracle | | | | | | 1.61 m | 0.0 % | 1.44 m | 0.0 % | |

Gates. Pipeline: the frame-only head is within 0.005 AUROC of the logistic — passed. Result: AUROC(< 5 m) 0.919 vs the
0.92 line and 4.5 % gross at 90 % coverage (≤ 5 %): formally at the line, but **every row including the logistic passes
the same gates**, and the head's gain over the logistic is 0.001 AUROC and 0.2 % gross — far inside the sampling noise of
a 3000-frame draw with 273 gross misses (AUROC standard error ≈ 0.01). Rows differ from the step 1–3 table because the
matcher and the calibration set changed: the four-city Task 04 checkpoint has a lighter tail (9.1 % vs 11.6 % gross) and
the calibrators see 7000 four-city frames instead of 1000 Chicago ones; the univariate AUROCs moved accordingly
(ego_mass_2cells 0.888 → 0.918, inlier ratio 0.804 → 0.814).

Reading. Three heads with different inputs (vote-map images, per-token evidence, 19 scalars) converge on the same
ranking as a logistic on the scalars: the information about pose correctness that the matcher's own output carries is
already captured by the vote-map statistics, and the remaining 4.5 % gross misses at 90 % coverage are frames whose
evidence looks like a correct frame's — sharp, consistent, well-supported votes at a repeated street pattern. No head on
the matcher's output can separate them; that would need evidence the matcher does not produce (a second reference year,
a route prior, or a verification pass at a finer scale). Decision: the certainty of record is the calibrated logistic
(cheapest, no training, same numbers); the token-stream head is kept as the trained alternative for the filter. The
certainty line stops here.

All-cities draw (12000 frames, 8809952, `certainty_erpd4city_head_all12000.{json,png}`): logistic AUROC(< 5 m) 0.874, 3.9 % gross at 90 % coverage (median 2.05 → 1.90 m); full head 0.870 / 3.9 %, tokens 0.871, map 0.868, frame 0.868; inlier ratio 0.747 / 4.4 %. The 0.88 all-cities gate is missed by every row (0.874 at best) and the head is again ≤ the logistic. Per city (logistic / full head): Chicago 0.913 / 0.909, San Francisco 0.909 / 0.901, Seattle 0.917 / 0.912, **New York 0.763 / 0.756** at a base rate of 0.72 correct — New York is where both the matcher and its certainty are weakest (facade-smeared queries, repeated grid), and it alone pulls the all-cities AUROC under the gate.
