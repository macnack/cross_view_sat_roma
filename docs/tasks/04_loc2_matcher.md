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
- Open for review: pose NLL (weight 0.5 in train_vigor) pushes every token's mass toward the *camera's* cell, which is
  not the target of a placed token; kept for comparability with the 30k IPM run, `--pose-nll-weight 0` is the
  obvious ablation. Also: the `srt` solver is an 8-DoF homography with the published refine=False (see the note in
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
