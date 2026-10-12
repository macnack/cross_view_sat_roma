# Task 07 — PanoRoMa-BEV: pillar feature lift (FG² geometry, depth-taught height), Sat-RoMa decoder as matcher

**Status (13 Oct 2026):** agreed with Maciej (docs/decisions.md, 2026-10-13). Code merged on the task03 branch
(`--query pillar`, `bevloc.model.pillar_query`, tests/test_pillar_query.py). Smoke on the laptop, gate runs on
Eagle (experiments/22_pillar_lift/). Both supervision rows are run ("i have compute u can do both").

## Why

PanoRoMa (task 04) matches the panorama's own `sat493m` tokens in the image plane and places each matched token by
UniK3D depth afterwards. On VIGOR same-area it is level with FG² (1.07 vs 0.98 m), but it (a) needs a monocular
depth model at train *and* inference, (b) is behind on cross-area (coarse 2.27–2.50 m vs FG² 1.41 m) and Poznań
(4.5 vs FG² 4.15 / Loc² 3.50 m), and (c) collapses on band input: KITScenes 4.70 vs FG² 2.34 m, VIGOR cropped to
±27.6° 1.06 → 6.09 m "even with exact depth" (experiments/20, 21).

The kick-off's method (docs/kickoff.html Fig. 2–4, H2 + H3) was a *feature-level* BEV: DINOv3 features projected
into a gravity-aligned top-down grid by geometry, LiDAR/depth used only to teach that projection, matched in latent
space against `encoder(map)` with Sat-RoMa modes + SE(2) consensus. It was only ever trained on four Fixtor routes
(`hybrid`, 14.9–17.5 m, decision 2026-09-24) and the BEV route was then dropped on a misreading of Loc² Tab. 12:
that ablation is an HC-Net *pixel* homography warp under unknown heading (an IPM-picture ablation), not a feature
lift. FG² (CVPR 2025) is a learned feature BEV with per-cell height selection at 0.98 m on the same test: the
counter-example. The decoder went 2.90 → 1.07 m on VIGOR only with 52k panoramas × 100 epochs; the feature lift
was never trained at that scale.

## Design (fixed)

- **Query `pillar`** (`bevloc.model.pillar_query.PillarQuery`; `fg2` is the baseline wrapper). BEV grid of
  `pillar.cell_m` = 2 m cells over ±`pillar.range_m` = 35 m (35 × 35), in the ego axes of `depth_query.ego_axes`
  (forward = the camera's z flattened, left, up) — on VIGOR with `R_NORTH` that is north-up, FG²'s aerial-frame
  ground grid under known orientation and the nadir regime Sat-RoMa was trained in. Per cell, M = 11 slices at
  `z_above_ground_m` = [0, 0.5, 1, 2, 3, 4, 6, 8, 11, 15, 20] above the ground plane z = −`cam_height_m`
  (slice 0 = the flat-ground IPM point). Each 3-D point → camera frame via `R_w2c` → ERP (u, v) by `project_erp`,
  the exact inverse of `erp_query.rays_at` (band-aware) → bilinear `grid_sample` of the frozen scale-16 token map
  (sub-patch; one token column circularly padded on both sides so the azimuth seam interpolates across). Height
  selection: softmax over M of `MLP([slice feature ; learned height embedding]) → 1`, weighted sum (FG² Eq. 2).
  BEV head: `ProjectionHead(circular=False)` (1×1 down to 256, two 3×3 convs, one self-attention block, zero-init
  1×1 up: the identity at init) with a fixed sinusoidal embedding of the cell's metric position added inside the
  256-d branch. Output `(B, 1024, 35, 35)` into the decoder at `f_q_pyramid[16]`.
- **Placement** = the cell centres through `metric_to_bev_px` (deterministic): the placed-token paths
  (`coarse_targets(query_xy=…)`, VCE, `consensus_from_gm`, `se2`, the two-pass fine decoder) apply unchanged.
  Valid cells: at least one slice inside the panorama's elevation band and above the ego cut
  (`max_depression_deg`: 90 = off on VIGOR — a 20° cut would delete the ground within 6.9 m; 20 on Poznań), and
  outside `blind_radius_m`.
- **Row A** (pose only, pure FG² supervision): coarse CE + VCE + pose-NLL 0.5, `--height-weight 0`.
- **Row B** (depth-taught height, the kick-off's H3): + `height_weight` · CE on the per-cell height softmax,
  target = the slice whose camera range agrees with the UniK3D v2 range at its projected pixel within
  max(`height_tol_m` 1 m, `height_tol_frac` 0.1 · range); cells with no agreeing slice get no height loss.
  Depth is loaded **only** when training with `--height-weight > 0`; inference is image-only in both rows
  (`eval_vigor.py` never loads depth for the pillar mode; band datasets need no depth files).
- **Fine pass**: `configs/vigor_cell00625_fine.yaml` with `pillar.cell_m 1.0, range_m 28` (56 × 56 over the 56 m
  window). The lift's parameters do not depend on the grid, so the coarse checkpoint warm-starts the fine pass;
  the checkpoint stores the architecture keys only (`PILLAR_ARCH_KEYS`), the grid keys always come from the config
  (`eval_vigor.check_fine_grid` compares the recorded `pillar_cell_m / pillar_range_m`).
- **Encoder rows**: default both sides `sat493m`, decoder warm-start `0t1q66hy`. Ablation: DINOv3-LVD (web) on both
  sides with the decoder warm-started from Eagle `sat_roma_shear/workspace/checkpoints/train_roma_sat_tpupzyfr_latest.pth`
  (Sat-RoMa trained with `vit_large_patch16_dinov3_lvd`, ImageNet stats, shear-30 recipe, 24 M samples):
  `matcher.checkpoint` may now be a local export directory (`bevloc.match.satroma.load_matcher`).
- Not in v1 (ablation ladder if the gate fails): FG² deformable offsets → six lift iterations → LVD encoder →
  the kick-off's `v·d` geometry descriptor. H5 contrastive stays out.
- `--refine` (sub-cell refiner) needs ERP-pixel `placement_at` and is refused for the pillar query
  (`check_refinable`); the two-pass fine decoder is the method of record anyway.
- FG² is GPL-3: `pillar_query.py` is written from the paper (§3.2 Eq. 1–2) and this document; nothing is taken
  from `third_party/FG2`.

## Gate and success (fixed before any run)

Gate: Chicago same-area, `configs/vigor_cell0125.yaml`, regularisation D (label smoothing 0.1, weight decay 0.3 /
0.05, photometric 0.5, rot90 + ±10° + flip), 20 epochs, held-out 20 % validation, coarse pass on the standard
3000-sample draw, `se2`. **Row A must beat PanoRoMa's own first pass (1.33 m median)**, else the task reports a
negative and the ablation ladder follows. Success for the paper: two-pass ≤ FG² on cross-area (1.41 m) and on
KITScenes / band input, with no depth model at inference; same-area expected near FG² (0.98 m) rather than below it.
If the gate is marginal, Row A is rerun with `--pose-nll-weight 0` (the vote-heatmap window around the camera cell
penalises correct far matches more for a ±35 m grid than for the 14 m picture).

## Runs (experiments/22_pillar_lift/)

1. Smoke on the laptop (`data/vigor_smoke`: 160 train / 40 test Chicago frames pulled from Eagle): 60 steps,
   batch 2, Row B — loss finite, height loss supervises > 50 % of the valid cells, a checkpoint round-trips.
2. Gate on Eagle (4× H100, `make eagle-train`): Chicago only, 20 epochs, D, tags `pillar_A_chi20`
   (`--height-weight 0`) and `pillar_B_chi20` (`--height-weight 1`); `eval_vigor.py --limit 3000 --solver se2`.
3. If a row passes: the experiment-18 recipe (four-city same-area 100 epochs coarse → fine; cross-area NY + Seattle
   → SF + Chicago), evaluated on the full Chicago test (offset calibration), cross-area 6000, KITScenes Frankfurt,
   Poznań three-way `prior` protocol.
4. LVD row: export `tpupzyfr` with `export_public_weights.py` (sat_roma, public-inference-release worktree), patch
   `pretrain_model` to `vit_large_patch16_dinov3_lvd` in the JSON, place under `checkpoints/satroma_tpupzyfr/`;
   copy the timm `vit_large_patch16_dinov3.lvd1689m` weights into Eagle's `$P/hf_cache/hub` (jobs run offline);
   `configs/vigor_cell0125_lvd.yaml` = the gate config with `matcher.checkpoint: checkpoints/satroma_tpupzyfr`.

## Commands

```bash
# smoke (laptop)
PYTHONPATH=src:.pydeps python scripts/train_vigor.py --config configs/vigor_cell0125.yaml --query pillar \
  --root data/vigor_smoke --split samearea --cities Chicago --val-frac 0.2 --val-samples 16 --val-every 20 \
  --batch 2 --steps 60 --height-weight 1 --tf32 --encoder-dtype bfloat16 --decoder-dtype bfloat16 \
  --tag pillar_smoke_B --out experiments/22_pillar_lift
# gate (Eagle, 4x H100)
make eagle-train TRAIN_ARGS="--config configs/vigor_cell0125.yaml --query pillar --split samearea --cities Chicago \
  --val-frac 0.2 --val-samples 400 --batch 32 --epochs 20 --save-every-epochs 5 --resume auto \
  --out experiments/22_pillar_lift --label-smoothing 0.1 --weight-decay-decoder 0.3 --weight-decay-head 0.05 \
  --aug-photometric 0.5 --aug-geometric rot90,flip --aug-rot-deg 10 --select-by pose --height-weight 0 --tag pillar_A_chi20"
```
