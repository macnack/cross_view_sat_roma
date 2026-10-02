# Task 04 — Loc² geometry with the Sat-RoMa decoder as the matcher

**Status (25 Sep 2026):** agreed with Maciej (docs/decisions.md, 2026-09-25). Baseline row in progress
(`scripts/eval_loc2_vigor.py`). Step 2 implemented, unit-tested (no data), under review; not yet run on VIGOR.

**Step 2 as built** (for review; implementation choices below are not in decisions.md yet):
- `bevloc.model.depth_query.ErpDepthQuery` (`--query erp_depth`): frozen `sat493m` ERP tokens (896×448 → 56×28,
  `cfg.erp_depth.erp_size`), optional `ProjectionHead` (1×1 down to 256, two 3×3 convs with circular azimuth padding,
  one pre-norm self-attention block, zero-initialised 1×1 up, residual: identity at init; 2.2 M params;
  `cfg.erp_depth.head`, off by default, `train_vigor.py --head` turns it on). Placement: depth at the token-centre
  pixel, point = depth × token-centre ray (depth is along the ray, as `loc2_depth_vigor.py` writes it), horizontal
  components in x forward / y left; invalid when depth ≥ 35 m, ≤ 0 or non-finite. Returned as virtual BEV pixels, so
  `coarse_targets` and `SatRoMa.consensus_from_gm` apply unchanged. Tested: token at azimuth 0 on the horizon →
  (x = d, y = 0); 90° right (east) → (0, −d); the whole grid against Loc²'s x = d sinθ cosφ, y = −d sinθ sinφ; and a
  synthetic VIGOR layout where the placed landmark + `H` hits the landmark's canvas pixel computed from the label
  convention and the tile resize alone (< 1 px), with `en` checked on the same sample.
- `VigorPairs` loads `depth` (1, h, w, metres) when the query mode is erp_depth; `keep_with_depth()` drops labels
  without a PNG (train: after the train/val split; eval: after the `--limit` draw, count reported in the JSON).
- `bevloc.model.coarse.vce_pose_loss`: correspondences drawn Loc²-style from p(token)·p(cell | token), p(token) ∝
  valid · sigmoid(certainty) (1024 pairs, two-stage multinomial), weights = pair probabilities; closed-form weighted
  2-D Procrustes (rotation from atan2 of the weighted cross/dot sums, scale fixed to 1: reference and virtual BEV
  share `cfg.grid.cell_m`); Eq. 6 on 10×10 virtual points over 5 m around the camera. `vce_mode: expect`
  (soft-argmax cell per token) is the alternative. Wired into `train_lift_splat.step` behind `cfg.train.vce_weight`
  (auto = 1 for erp_depth, 0 otherwise; `--vce-weight`); CE with pose-derived targets, certainty, neighbour hinge and
  pose NLL unchanged. With VCE on, `train_vigor.py` selects the checkpoint on the validation Procrustes pose error at
  the camera (`vce_pose_m`, fixed draw) instead of the heat-map proxy.
- `eval_vigor.py` / `viz_vigor.py`: queries with `placement` go through `consensus_for_query` →
  `SatRoMa.consensus_from_gm` (this also makes the `erp` mode evaluable there: the 14×14 BEV patch mask they used does not fit a
  28×56 token grid). `scale_factor` = √(h·16·w·16)/560 (1.13 for 56×28) as in `step`; in the decoder it only
  feeds the frozen conv refiner's displacement embedding, which runs after `gm_cls` / `gm_certainty` are produced.
- After review (25 Sep): VCE pairs and expectations are restricted to reference cells with content (the coarse CE's
  `ref_cell_validity` mask); samples with no token weight or no valid cell are left out; degenerate Procrustes
  returns the identity. Pose NLL pushes every token's mass toward the *camera's* cell, which is not a placed token's
  target, so `train_vigor.py` defaults it to 0 for erp_depth (0.5 for the other modes; `--pose-nll-weight`
  overrides; the effective value is stored in the checkpoint's `train` dict and logged by `eval_vigor.py`).
  `train_vigor.py` also writes `vigor_<tag>_last.pt` next to `_best.pt`. Also: the `srt` solver is an 8-DoF homography with the published refine=False (see the note in
  `SatRoMa.__init__`), not "Loc²'s scale-aware Procrustes" as the Design paragraph says; `sim` (4-DoF) and `se2`
  (3-DoF, fixed scale: the metric-depth case) are the Procrustes-like ones (`eval_vigor.py --solver`).

## Why

On identical VIGOR samples our camera-only method (flat-ground IPM picture through the frozen `sat493m` encoder,
Sat-RoMa decoder classifying 4 m reference cells, multi-hypothesis RANSAC) sits at 2.90 m median on Chicago
same-area after the best decoder fine-tune, FG² at 1.06 m and Loc² (published) at 1.59 m. Two facts fix where the
gap is (experiments/09_vigor, decisions 2026-09-25): the Poznań warm start contributes nothing, so the plateau is
the architecture's; and Loc²'s own ablation (Tab. 12 of 2509.09792v3) puts BEV-plane matching — warp the panorama
first, match the same way — at 8.20 m median against 1.75 m for matching in the image plane and lifting only the
matched points. Our picture is a BEV-plane method.

Loc²'s correspondence loss (App. H, Eq. 22–23) is a cross-entropy over aerial points with the pose-projected point as
positive: the same functional form as RoMa's regression-by-classification. What RoMa adds on top is not the loss but
the *match decoder* that produces the distribution (RoMa Tab. 2: the Transformer decoder is the largest gain, the
loss form ≈ 10 % relative), plus certainty and multimodality that our RANSAC consumes. Hence the design below:
Loc²'s geometry and supervision, our decoder as the matcher.

## Design (fixed)

Ground tokens are the panorama's own frozen `sat493m` patch tokens (the `erp` query mode of task 03, Task 6),
optionally through a trained light projection head (Loc² §3.1: a few convs + one self-attention block — OPEN item 6).
Reference tokens: the tile canvas through the same frozen encoder, as now. Matcher: the Sat-RoMa decoder, producing
per ground token a categorical over reference cells (`gm_cls`) with certainty and GMM modes — non-square query
grid (ERP tokens), which the decoder must accept. Placement: every matched ground token gets an ego-metric 2-D point
from a monocular depth map along its ray (Loc² §3.2: x = d·sinθ·cosφ, y = −d·sinθ·sinφ, tokens beyond 35 m masked),
depth = UniK3D metric depth precomputed as uint16-mm PNGs (`scripts/loc2_depth_vigor.py`, Loc²'s own layout).
Solver: scale-aware weighted Procrustes over sampled (ground point, reference point) pairs for training (differentiable,
Loc² Eq. 2–5, scale fixed to 1 when metric depth is used) and our multi-hypothesis RANSAC over the retained modes at
test time (the `srt`/`sim` solvers in `bevloc.match.satroma` are Loc²'s scale-aware Procrustes plus consensus).
Supervision: Loc²'s VCE pose loss (Eq. 6, virtual points on a 5 m grid) + β·coarse cross-entropy with pose-derived
positives (= our existing `coarse_targets` from the placed query points) + the certainty term; β = 1.

Read-only: third_party/Loc2 (AGPL-3.0) and third_party/FG2 (GPL-3.0) are baselines only; nothing is copied from
them into `src/`. Our module is written from the equations.

## Steps

1. **Baseline row.** Loc²'s released checkpoints on our 3000 Chicago same-area and 6000 SF+Chicago cross-area draws,
   both solvers (`make loc2-depth`, `make loc2-vigor`). Gate: same-area median within 0.2 m of the paper's 1.59 m,
   which validates the depth preprocessing and the mmcv shim on this checkpoint.
2. **The matcher.** `bevloc.model.loc2_query` (or an extension of `erp_query`): depth placement of ERP tokens,
   optional projection head, training loop in `scripts/train_vigor.py` behind `--query erp_depth` (or similar), the
   VCE loss in `bevloc.model.coarse`, evaluation through the existing placed-query path (`SatRoMa.match_placed`).
   Unit tests: ray geometry (a token at azimuth 0 / depth d lands at (x=d forward, y=0) — read "x=0" in the first draft as a typo — consistent with VIGOR's
   north-at-centre panoramas and our tile canvas: verify against `VigorPairs`' H and `en`), depth-PNG decoding, the
   VCE loss on a known pose (zero at GT, grows with translation), and a forward pass of the decoder on a non-square
   token grid.
3. **Fast check.** Chicago same-area, 10k steps, batch 4, decoder (+ head) trained, everything else as the 30k IPM run;
   evaluate on the 3000 draw. Gate: better than 2.90 m median. Then the four-city run with FG²'s protocol.
4. **Ablations, in this order:** (a) coarse CE on cosine logits instead of the decoder — the "RoMa loss on Loc²"
   control, expected ≈ Loc²; (b) decoder without the projection head; (c) a fine sub-cell offset head with a robust
   loss (RoMa §3.4). (d) Under unknown orientation, VCE alone vs VCE + CE (Loc² Tab. 15 warns CE can hurt there).

## Deliverables

experiments/10_loc2_matcher/: config snapshots, eval JSONs on the same draws as 09_vigor, a REPORT.md with one
table (Loc², FG², ours-IPM, ours-Loc²-geometry, per split), the overlay sheet of the worst 20, and a decisions.md
entry per gate.

## Sub-cell stage (ablation 4c): the decoder's own conv refiner — as built, 26 Sep 2026 (for review)

**What the decoder has (read from sat-roma-infer `build.py` / `matcher.py`, checked on the released `0t1q66hy`
model):** one refinement stage, not four. `decoder_scales = ["16"]`, `conv_refiner = {"16": ConvRefiner}` (17.4 M
params, 8 depthwise 5×5 blocks), and the DINOv3 encoder emits only `{16: tokens}`: no stride-8/4/2/1 refiners and no
fine features exist. The refiner runs at the coarse token grid on the same `proj["16"]` features (512 ch) the GP and
Transformer see. Its input warp is the package's "ToWarp", `cls_to_flow_refine(gm_cls)`: a 5-neighbour soft-argmax
around each token's argmax cell, decorated `@torch.no_grad()` in the package (so W_in carries no gradient). It computes `grid_sample` of the reference features at
that warp, a displacement embedding of (warp − own coords)·40/32·`scale_factor`, and a 15×15 local correlation around
the warp. It outputs Δ and Δc, giving `flow = W_in + 16·(Δx/(4w), Δy/(4h))` and `certainty = gm_certainty + Δc`. The
decoder exposes `flow_pre_delta` (W_in), `delta_flow`, `flow` and `certainty` under `exposed_intermediates`, and it
always ran: we just never read it. Two caveats for the frozen refiner. (a) Our checkpoints fine-tuned `proj["16"]` and
the Transformer while the refiner stayed at the released weights, so it now sees drifted features. (b) For an ERP
token grid (28×56) the displacement is scaled by 1/w, 1/h of that grid (4× / 2× smaller steps than on the 14×14
grid it was trained on), and its correlation window spans ±0.25 / ±0.5 of the reference instead of ±1. On noise
inputs the released refiner moves matches by a median of about 2.6 cells, so "sub-cell" is not guaranteed.

**Evaluation (`eval_vigor.py --refine S --refine-init … [--refine-gate C] [--refine-min-cert P]`).**
`bevloc.model.refine.RefinerTap` hooks the refiner (the package is not edited) and records its inputs, so it can be
re-run on another warp. `bevloc.match.satroma.refined_for_query` turns the refined warp into correspondences on a
query grid of stride S px. At S = 16 there is one per token, read exactly. At S < 16 the warp is sampled bilinearly
between token centres (points outside the token-centre hull are dropped); this adds no information for the picture
modes, because the decoder has no finer refiner. Validity for the picture modes: at S = 16 the token set of the coarse
rows (`query_patches` of the valid fraction ≥ 0.05, exactly as `consensus_for_query`), so a null refiner covers the
same tokens as the peak row; at S < 16 each sample's own `bev_valid` pixel. For `erp` / `erp_depth` the query pixels are ERP pixels: each is placed through its own ray with its own
depth (`ErpDepthQuery.placement_at` / `ErpQuery.placement_at`; identical to the token placement at token centres,
tested). The same solver (`--solver`), inlier threshold and seed then run on these correspondences. Initialisation:
- `none`: the package path (refiner on the ToWarp warp of `gm_cls`), RANSAC from scratch.

Every init drops non-finite warps, certainties and placements (counted per sample as `nonfinite_*`). It falls back
to the coarse peak pose (counted as `fallback_*`) when fewer than `--refine-min-corr` correspondences remain, or when
the solver returns no model or raises (`LinAlgError` / `cv2.error` are caught and counted as `error_*`). A refined
row therefore never crashes the run, and a diverged refiner shows up as the coarse numbers with every sample counted
as a fallback.
- `coarse`: the same refined warp, with correspondences gated to within `--refine-gate` cells (default
  `reproj_cells` = 3) of the coarse peak pose before the RANSAC. cv2's RANSAC takes no initial model, so seeding
  means this guided-matching gate. If fewer than `--refine-min-corr` (default 8) correspondences survive the gate,
  or the gated set yields no model, the row falls back to the coarse pose (no 2- or 3-point fits); the fallbacks
  are counted.
- `ransac`: the refiner re-run on the RANSAC-consistent coarse warp W_in(token) = H_coarse(token's query point:
  token centre, or placed point), then gated like `coarse`.

Rows: `pose_refined_m` / `inliers_refined` (one init) or `pose_refined_<init>_m` (several), plus `ncorr_*`,
`nused_*`, `fallback_*`, next to the unchanged peak / means rows of the same run. `--refine 0` (default) is
bit-identical to the previous evaluator (tested against a copy of the old loop, and the solver refactor was checked
on 36 random cases). `inliers_refined` is the inlier fraction over dense, spatially correlated correspondences
(one per token at S = 16, many per token at S < 16 carrying the same interpolated information); it is not comparable
to `inliers_peak`, which counts GMM modes, and least of all at S < 16.

**Training (`train_vigor.py --refine-weight W`, `cfg.train.refine_weight`, default 0 = frozen and not saved as
before).** The refiner is unfrozen and saved in the checkpoint's decoder dict (`eval_vigor` loads it and says so).
Loss per RoMa §3.4 (2305.15404v2 Eq. 16–18, written from the paper):
- Regression: a generalised Charbonnier, α = 0.5 and s = c·16 with c = 1e-4 in normalised reference coordinates
  (RoMa's code units), of |W − W_gt| over the tokens the coarse CE supervises. The loss is shifted to be 0 at the
  ground truth.
- Certainty: `certainty_weight` (0.01) × BCE on the refined certainty, target 1 when the ground truth lies within
  `refine_cert_cells` (0.5) reference cells (Chebyshev) of the refiner's input warp, taken over valid query tokens.
  This target is **our choice** and differs from RoMa's: RoMa trains the certainty toward covisibility at every
  scale (gated by the previous scale's error being local only at the finer scales), not toward "the input warp is
  within half a cell".

W_gt = H(token centre) for the picture modes and H(placed point) for `erp_depth`. The refiner's inputs (features and
W_in) are detached, so the fine loss trains the refiner only. Checked on the real model: the fine loss alone gives
zero gradient to every non-refiner parameter, including the head. That is RoMa's coarse/fine cut: in the package (as
in RoMa) the warp half is already cut by `@torch.no_grad()` on `cls_to_flow_refine`; detaching the shared projected
features as well is our addition (in RoMa the fine features come from a separate encoder). A warm start whose
checkpoint carries a trained refiner keeps saving it even with `--refine-weight 0` (frozen at those weights). Train and val log `fine_epe_px` (refined) against
`fine_epe_in_px` (input warp) in reference px.

**Numerics (fixed after the first `--refine-weight 1` run diverged, 26 Sep).** That run trained the refiner under
the package's float16 autocast with no GradScaler. `fine_epe_px` fell from about 113 to about 21 px (input warp
about 30 px), then went NaN at step 23,425 and stayed NaN. Checkpoint selection ignored it, so `_best.pt` and
`_last.pt` carry NaN refiner weights, and the evaluation crashed on an SVD of NaN correspondences. Since the fix:
- The refiner trains in float32 (`cfg.train.refine_precision`). `RefinerTap` sets the refiner's autocast dtype and
  casts its inputs to float32; checked on the GPU that its convolutions run in float32. At eval, a checkpoint-trained
  refiner runs at the precision stored in the checkpoint; the released refiner keeps float16.
- The fine loss is computed in float32 on the squared error (no sqrt, smooth at 0).
- A batch whose refined warp, certainty or fine loss is non-finite trains the coarse terms only (`fine_skipped`,
  in the CSV).
- A step with a non-finite total loss or refiner gradient does not update. The refiner's gradient norm is clipped
  to `refine_grad_clip` (1.0).
- More than `refine_max_nonfinite` (20) such steps in a row stop the run with an error.
- A validation with any non-finite logged value, or with a skipped fine batch, is never saved as `_best.pt`;
  `_last_finite.pt` is the last checkpoint whose validation was fully finite.

Open for agreement: the gate radius, the certainty-target radius,
detaching the features as well as the warp, and whether S < 16 is worth keeping.

## Coarse-to-fine second pass — as built, 26 Sep 2026 (for review; not yet run on VIGOR)

**Why.** Halving the reference grid spacing 0.25 → 0.125 m/px (4 m → 2 m cells) took the Chicago median from 2.90 m
to 1.89 m (picture) and gives 1.76 m for erp_depth. Another halving cannot be global: at 0.0625 m/px the 896 px canvas
is 56 m and no longer holds the ~71 m tile. So the second pass matches again, at 0.0625 m/px (1 m cells), inside a
56 m window of the tile around the first pass's pose.

**Reference window (`VigorPairs`, `configs/vigor_cell00625_fine.yaml`).** `vigor.ref_window_m` (null = whole tile,
unchanged bit for bit) turns the reference into a window of the tile at `cfg.grid.cell_m`. Conventions:
- Tile frame = the frame of the sample's `en`: east, north metres from the tile centre; the tile centre is pixel
  ((w0 − 1)/2, (h0 − 1)/2) with pixel centres at integer coordinates, metres per tile px = `CITY_RES[city] · 640 / w0`.
- The window's centre is the canvas centre pixel ((S − 1)/2), placed at `ref_centre_en` (a new sample key; zeros for
  the whole tile). The window is one `cv2.warpAffine` of the tile with the exact canvas → tile affine
  (`window_affine`; area pre-filter when downsampling), black off the tile and outside `ref_window_m`.
- H (BEV px → window px) is a translation whose camera pixel is `en_to_canvas(en, ref_centre_en)`; any pose on any
  canvas maps back to the tile frame with `pose_en(H, ref_centre_en, n, cell_m, S)` (the BEV centre is the camera).
  Verified on a synthetic tile with a Gaussian marker painted at the label's pixel: the marker centroid in the window
  is within 0.013 px of H's camera pixel at 0.0625 and 0.25 m/px, for centres offset up to 25 m, including windows
  that run off the tile (tests/test_vigor_window.py). The whole-tile path's own content sits within ≈ 0.5 px of its H
  (integer placement of the resized tile, unchanged, ≈ 6 cm at 0.125 m/px; it is the path the coarse checkpoints use).
- Queries are unchanged: their placement is in metres. The picture is `ipm_erp` at the config's cell, so at
  0.0625 m/px it is 224 px = 14 m (near field only; the 1.2 m blind disc covers 4× the pixels, tested); erp_depth tokens
  are placed in virtual BEV px of 0.0625 m (2× the pixel offsets of 0.125 m, tested), so their targets reach up to the
  35 m depth cap but only those landing inside the 56 m window (and on tile content) are supervised / vote.

**How the window is chosen.** Training / validation: true camera position + a jitter uniform over the disc of radius
`vigor.ref_jitter_m` (6 m, comparable to the coarse error: median 1.8 m, R@5 0.74–0.82). A training draw is fresh
each time (seeded from the torch RNG, so runs repeat); the `--val-frac` validation copy has `jitter_seed = 0`: the same
distribution with one fixed draw per sample, so its curve is comparable across steps. Evaluation: the coarse RANSAC
pose (`peak` row), or the tile centre when the coarse pass returned nothing (counted: `no_coarse_fine`).

**Training.** `train_vigor.py --config configs/vigor_cell00625_fine.yaml`, otherwise unchanged (the jittered window is
the only difference; a warm start from the matching 0.125 m checkpoint).

**Evaluation (`eval_vigor.py --fine-config configs/vigor_cell00625_fine.yaml --fine-ckpt <pt> [--fine-gate 6]`).**
The coarse rows are computed exactly as before (tested: identical keys and values with and without the fine pass).
Then per sample: the fine window centred on the coarse `peak` pose, the query rebuilt at the fine GSD from the same
panorama, the fine decoder, and the same consensus (`--solver`, `reproj_cells` threshold in cells, i.e. 3 m → 3 × 1 m,
seed; the fine config's matcher block is replaced by the coarse run's). The fine pose is mapped back to the tile frame
and scored there. Rows: `pose_fine_m`, `yaw_fine_deg`, `inliers_fine`, `pose_fine_gated_m` (the coarse pose when the
fine pose is missing or more than `--fine-gate` m from the coarse pose: `fallback_fine`, summary
`fallback_fine_gated`), plus `en_gt`, `en_coarse`, `en_fine`, `fine_centre_en`, `fine_shift_m`; the summary adds
`fine` / `fine_gated` rows, `nopose_fine` and `no_coarse_fine`. An erp_depth fine checkpoint behind a picture coarse
checkpoint drops the panoramas without depth from both passes (counted in `meta.fine`). Two decoders are loaded
(two encoder copies; fine on an H100). A fine pass that raises keeps the sample: coarse columns intact, fine columns
None, `fine_error` = exception class and message, counted as `fine_errors` (CUDA out-of-memory is re-raised).
`train_vigor.py` records `train.grid` = {cell_m, ref_window_m, ref_jitter_m} in every checkpoint, and `--fine-ckpt`
refuses a checkpoint whose recorded cell_m differs from the fine config's (older checkpoints without the record: a
warning). `make vigor-report` lists the `fine` / `fine_gated` (and `refined*` / `hyp*`) rows after each file's peak row.

**LoFTR sanity check (`make loftr-fine`, `scripts/loftr_fine_vigor.py`).** The same coarse pass and fine window, then
kornia's pretrained LoFTR (outdoor) between the greyscale picture at 0.0625 m/px (invalid pixels black; matches on them
dropped) and the greyscale window; the pose from the same consensus on LoFTR's matches (`SatRoMa.refined_consensus`),
the same rows plus `nmatch_fine` / `nused_fine`. kornia is not in the laptop env; the script imports it only when run.
LoFTR's weights come from `$TORCH_HOME/hub/checkpoints/loftr_outdoor.ckpt` (or `--loftr-weights`): Eagle compute
nodes have no internet, so that file must be placed first.

Tests (tests/test_vigor_window.py, tests/test_two_pass.py; CPU, synthetic tiles, planted toy decoders): window H
exactness for five centres × two GSDs incl. off-tile windows (black padding checked column by column), the jitter
distribution (uniform on the disc, seeded repeatability, fixed validation draw), the fine → tile back-mapping of an
injected pose with rotation, the two-pass evaluator returning the injected fine pose (se2 and homography), the gate
(far fine pose, missing fine pose, wider gate), `run()` end to end writing the JSON, and the LoFTR plumbing with a
fake matcher (a sub-pixel translation with 1 in 7 outliers recovered exactly).
