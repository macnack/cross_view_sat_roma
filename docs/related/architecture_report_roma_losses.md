# Which lifting architecture to extend with RoMa losses and the Sat-RoMa matcher

Report, 2026-09-23. Sources: the PDFs in `docs/related/` (FG² 2503.18725v1, BevSplat 2502.09080v4,
BEV-Patch-PF 2512.15111v2, RoMa 2305.15404v2), the read-only code under `third_party/`, and today's
evaluation in `experiments/05_lift_splat/REPORT.md`. Page numbers are PDF pages. Code lines were
checked against the checked-out third_party trees.

## 1. The question and the short answer

The hypothesis to test: "similarity matching (one global correlation peak per frame) is easily beaten
by RoMa-style per-patch classification losses", and the decision to prepare: which published
camera-to-BEV front-end to put in front of our Sat-RoMa decoder (frozen `sat493m` ViT-L on both
sides, RoMa regression-by-classification head over 56×56 reference cells, GMM modes per patch,
multi-hypothesis RANSAC).

Short answer. The loss claim is right in principle and already partly proven by the literature, but
it is not where our number is stuck. We have been training with RoMa's coarse loss all along, and
four training variants of it landed at 11.5–13.5 m median on 200 validation frames, all inside one
another's confidence intervals, against a centre-guess chance level of 16.9 m. The bottleneck is
the lift, not the loss. Of the three candidates, FG² is the architecture to extend: it is the only
one whose lift is designed for a panorama without depth, whose matching step is already a per-point
classification (dual-softmax over aerial points, trained with InfoNCE) and whose pose solver is a
rigid Procrustes with fixed scale, so replacing its matching head with our decoder, mixture modes
and consensus is a like-for-like swap. BevSplat's front-end is the second choice and its
weakly-supervised similarity head is exactly the thing the hypothesis says we can beat.
BEV-Patch-PF is not a lift candidate for this project because it requires measured depth and a
pinhole camera; its particle-filter observation model is what Task 5 already borrows.

## 2. The four front-ends side by side

| | FG² (CVPR 2025) | BevSplat (NeurIPS 2025) | BEV-Patch-PF (ICRA 2026) | Ours today (05_lift_splat / hybrid) |
|---|---|---|---|---|
| Camera | panorama (VIGOR) and pinhole (KITTI) | panorama and pinhole | pinhole RGB-D only | 360° ERP |
| Lift | fixed pillar of M=11 heights (±10 m in the released code, ±20 m in the paper) under each of 41×41 BEV points; features pulled by deformable cross-attention; learned softmax over height per cell ("feature selection along height", §3.2 Eq. 1–2, Fig. 2) | pre-trained monocular depth (DepthAnythingV2 / UniK3D) → 3 Gaussians per pixel with learned offset, scale, rotation, opacity → α-blended top-down render (§3.1–3.2.1, Eq. 1–3) | measured depth back-projects features; z dropped; MLP-weighted column average (§III-C) | learned depth softmax over 16 bins per ERP token, splatted (lift); hybrid adds exact IPM ground via grid_sample |
| Needs at inference | heading prior (north-aligned or two-step yaw estimate), no depth, no height, no intrinsics for panoramas | depth model, orientation predictor from G2SWeakly | depth image, intrinsics, odometry, initial pose | compass proxy, camera height (IPM only) |
| BEV grid | 41×41 points over the aerial extent (~1.8 m/cell on VIGOR; code lifts heights to ±10 m, paper says ±20 m, §4.2 vs `models/modules.py:182`) | 128×128 cells, 32 channels (§4 impl. details) | 224×224 cells at the map GSD (0.3 m), 32 channels | 224×224 at 0.25 m, 14×14 tokens of 1024 ch after the BEV head |
| Matching | cosine/τ between 128-d descriptors, dual-softmax with dustbin, N_S=1024 sampled correspondences (§3.3 Eq. 3–4) | cosine similarity of confidence-weighted BEV feature against satellite features in a sliding window → one location probability map (§3.2.2 Eq. 4) | per-particle rotated patch, distinctiveness-weighted cosine (§III-B Eq. 3) | Sat-RoMa: per-patch categorical over 56×56 cells, GMM modes, RANSAC |
| Pose solver | weighted Kabsch/Procrustes, rigid, scale fixed by GSD; RANSAC optional (§3.3, Eq. 5) | peak of the similarity map (position); yaw from a separate predictor | the particle filter is the estimator | sRT or fixed-scale SE(2) RANSAC (Task 4) |
| Supervision | pose only; L_VCE (virtual-point error) + InfoNCE on pose-derived pseudo-correspondences (§3.4 Eq. 6–9) | weak: peak of positive map above peaks of negatives (Eq. 5), optional noisy-GPS peak loss (Eq. 6) | InfoNCE over 63 negative poses + distinctiveness BCE (Eq. 7–11) | RoMa coarse CE + certainty + neighbour hinge + pose-heatmap NLL |
| Headline (mean / median m) | VIGOR known ori: same 1.95 / 1.08, cross 2.41 / 1.37; unknown ori two-step: 3.78 / 1.70, 5.95 / 2.40 (Tab. 1). KITTI same 0.75 / 0.52, cross 7.45 / 4.03 (Tab. 2) | VIGOR aligned: same 3.15 / 1.45, cross 3.03 / 1.41 (λ₁=0); with noisy GPS 2.87 / 1.58, 2.84 / 1.36 (Tab. 2). KITTI same 5.82 / 2.85, cross 7.05 / 3.22 (λ₁=0) (Tab. 1) | ATE RMSE: seen routes ~1.7 m (Tab. I, six routes), unseen 3.61 m; text says 3.10 m seen | validation manifest: 11.3–13.5 m median, R@5 0.12–0.19, R@10 0.37–0.47; chance 16.9 m |
| Practical | GPL-3; mmcv `MultiScaleDeformableAttention` CUDA op (torch 2.2 pin); Drive checkpoints | README says MIT but no LICENSE file; the two CUDA rasterizers are Inria 3DGS forks (research/non-commercial licence); depth precomputed offline; DINOv2 trunk frozen, only the DPT head trains; OneDrive checkpoints (interactive login) | MIT declared; manifpy; HF checkpoint; off-road datasets only | ours |

Two ablations in these papers matter for us more than the headlines.

- BevSplat, Tab. 3 (KITTI, same backbone, no GPS): IPM 9.02 / 5.54 m, Lift-Splat-Shoot 16.14 / 13.94 m,
  OrienterNet-style lift 15.59 / 13.80 m, direct projection of the depth point cloud 7.59 / 4.25 m,
  BevSplat 5.82 / 2.85 m. Under a fixed matcher, the learned depth-bin lift (LSS, our current
  query) is the worst front-end tested, and plain IPM beats it by a wide margin. Our Task 2 run is
  the same comparison on our data, and its early validation cross-entropy already points the same
  way (3.77 at step 600 for IPM versus 4.2 for every lift checkpoint).
- FG², Tab. 3 (VIGOR validation): learned height selection 2.17 / 1.18 m against sum over height
  2.34 / 1.27 m and max over height 2.24 / 1.25 m; using all correspondences instead of a sampled
  sparse subset is worse (2.55 / 1.45 m). Sparse, selected correspondences with a rigid solver beat
  dense pooling. That is the regime Sat-RoMa's mode selection plus consensus lives in.

## 3. Does a RoMa loss beat similarity matching? What the evidence says

Where the classification-plus-consensus formulation should win, and why:

- Multimodality. RoMa §3.4 argues from a scale-space model that the coarse conditional match
  distribution is multimodal near structure boundaries, so the coarse stage must be a classifier,
  not a regressor; the ablation (Tab. 2, setup V→VI) shows regression-by-classification lowering
  the 5 px failure rate (100 − PCK) from 3.2 % to 2.8 % on MegaDepth with everything else fixed.
  RoMa v2 (2511.15706v3, p. 6) replaces the 64×64 anchor classification by a dense softmax
  cross-entropy over all reference patches plus a robust warp term: the head changed, but the
  coarse objective is still a categorical over reference locations, the same family as FG²'s
  dual-softmax and as our `gm_cls`. Neither RoMa paper measures repeated structure directly;
  the multimodality argument is theoretical (Fig. 3) and the evidence is the ablation above. A single correlation peak
  (BevSplat Eq. 4, BEV-Patch-PF Eq. 3) cannot represent "this facade could be either of those two
  buildings"; a per-patch categorical can, and RANSAC over the mode set decides late. Sat-RoMa's own
  result (ACIVS 2026: keeping secondary modes cuts corner error from 30.5 m to 4.12 m on cross-season
  overhead pairs) is the same mechanism measured on stale maps, which is our deployment case.
- Partial validity. A ground BEV is valid on a fraction of the grid (the ego disc, occlusions,
  range). A per-patch loss simply drops invalid patches (our `matchable` mask); a global similarity
  must pool over them or learn a distinctiveness map to down-weight them, which BEV-Patch-PF adds
  as a second loss (Eq. 9–10) and BevSplat as a confidence head.
- Published head-to-head. FG²'s sparse matching + Procrustes (a per-point dual-softmax with an
  InfoNCE loss, structurally a RoMa-type classification over aerial points) beats the correlation
  and dense-flow methods in its tables (DenseFlow, HC-Net, GGCVT) on VIGOR mean error by 26–28 %.
  BevSplat, with a similarity head, reaches comparable medians on VIGOR (1.41–1.58 m) but with a
  much stronger lift; the two papers are not a clean loss ablation against each other.

Where it does not help:

- When every patch votes the same neighbourhood. Today's layer sheet (`viz/layers_seq.jpg`) shows
  the lift's BEV as a radial star of 1568 rays × 16 bins, the 14×14 query tokens as a smooth blob,
  and the certainty map as the same band in every frame. With that input the decoder learns the
  layout prior; a sharper loss on the same tokens moves nothing, and the four checkpoints in
  REPORT.md are the measurement of that. The pose-heatmap NLL and the neighbour hinge were added on
  top of the coarse CE and did not change the median.
- Low-texture ground. Where the BEV is uniform asphalt, no per-patch distribution is peaked; both
  formulations fall back on layout, and the filter (Task 5) has to carry it.

So the honest reading of "RoMa losses beat similarity" is: yes for the matcher, provided the front-end
puts different content into different patches. The choice below is therefore about lifting.

## 4. Ranking: what to extend with RoMa losses and the Sat-RoMa decoder

1. FG²'s lift, our matcher. Reuse: the pillar grid, deformable cross-attention retrieval and the
   learned height softmax (`models/modules.py:170-225`), iterated as in `model_vigor.py:109-167`;
   run it on the ERP through our frozen `sat493m` instead of DINOv2 (FG² keeps its backbone frozen
   too, `modules.py:36-81`). Replace: the 128-d projector, dual-softmax and Procrustes with our
   decoder (`f_q_pyramid[16]`, `(B, 1024, 14, 14)`), mixture modes and RANSAC, trained with our
   coarse CE; keep FG²'s sparse-sampling lesson by letting RANSAC pick modes. Interface: FG²'s
   `grd_query` reshaped to `(B, 1024, 41, 41)` (`model_vigor.py:169`) becomes a 224×224 / 0.25 m grid
   with 14×14 tokens after our BEV head, or the pillar grid is defined directly at 14×14 with 16 px
   patches. Why first: it is the only lift built for a panorama without depth, its height selection
   is the learned answer to "which height explains this cell" that our hybrid approximates with a
   flat ground plane, and its solver is already rigid with fixed scale, so H7 transfers. Risks: the
   mmcv deformable-attention op against torch 2.13/cu130 (rewrite the retrieval as bilinear
   `grid_sample` at the projected pillar points, which is what the op does with learned offsets
   removed), GPL-3 if code is copied rather than re-implemented, and the heading prior (FG² needs
   near-north-aligned panoramas; we have a compass proxy within a few degrees). Effort: 3–4 days for
   a grid_sample re-implementation of the pillar lift inside `bevloc/model`, 1 day to train on Eagle,
   1 day to evaluate on both manifests.

2. Our hybrid query, kept as the control. Already trained tonight on Eagle. If it matches or beats
   the lift, the dense-ground argument is confirmed and (1) only has to beat the hybrid, not the
   lift. Effort: 0 days beyond the queued runs.

3. BevSplat's front-end. Reuse: a pre-trained panoramic depth model (UniK3D) and the per-pixel
   Gaussian primitives with learned opacity, which BevSplat shows are what beat direct projection
   (Tab. 3, w/o OPT 7.42 / 4.16 m against w/ OPT 5.82 / 2.85 m). Replace: the similarity head and
   weak loss with our decoder and coarse CE, which is the cleanest test of the hypothesis in this
   list. Risks: the CUDA rasterizer build on Eagle, a depth model at inference (the kick-off's
   "no dense depth network" decision would be overridden for this track), and OneDrive-only
   checkpoints, and the rasterizer's Inria research licence if its code is reused rather than
   re-implemented. Effort: 4–6 days, mostly the rasterizer and depth plumbing.

4. BEV-Patch-PF. Not a lift candidate: pinhole RGB-D, forward half-plane only, measured depth at
   inference. What to take: the per-particle observation model (one aerial encode, rotated
   `grid_sample` per particle, uncertainty-flattened likelihood, Eq. 4–6) and the learned
   distinctiveness map. Our Task 5 filter already reads the vote heatmap; its σ-flattening (Eq. 5)
   is the one addition worth copying once the per-frame numbers justify a filter.

## 5. What to run first

1. Let tonight's Eagle batch finish and score `ipm`, `hybrid`, `hybrid_warm` and `pose_nll` on the
   validation manifest with `make eval-pose`; the IPM row against the lift rows is the BevSplat Tab. 3
   comparison on our data.
2. If IPM ≥ lift (overlapping or better intervals), the learned depth-bin lift is retired and the
   hybrid becomes "IPM + above-horizon", which is FG²'s question in a cheaper form.
3. Implement the FG² pillar lift with `grid_sample` retrieval and a height softmax as `--query fg2`
   behind `bevloc.model.query.build_query`, train with the same recipe, score on both manifests.
4. Re-score the best of 1–3 with `--solver se2` (H7) and run `track_route` on the test route
   (Task 5), reporting median, p95 and the ≤5 m fraction against dead reckoning and per-frame RANSAC.
5. Only if 3 lands within a few metres of BevSplat's VIGOR medians on our data, consider its
   Gaussian front-end as an ablation row; otherwise leave it as related work.
