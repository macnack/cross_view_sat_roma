# Task 02 — FG² and BevSplat transfer to Fixtor × Poznań

**Status (23 Sep 2026):** design approved; scaffolding started. Step 2 (manifest +
coordinate tests) done. Step 1 (native FG²/BevSplat reproduction) **blocked** —
no VIGOR imagery on disk; BevSplat OneDrive checkpoints not downloadable
non-interactively; local env missing `mmcv`. See
`experiments/06_fg2_bevsplat/STATUS.md`. Do not proceed to Fixtor adapters until
§1 clears.

## Question

Do the released FG² and BevSplat methods localize our real Mapillary 360 panoramas
against Poznań public orthophotos, first zero-shot and then after minimal
fine-tuning on exactly the training routes available to our Lift-Splat track?

This task tests method transfer. It does not attempt to improve either baseline,
reimplement its architecture, or make our method look better through mismatched
inputs.

## Decision rule

Evaluate each method in two stages:

1. **Released checkpoint:** use the authors' pretrained VIGOR checkpoint without
   changing learned weights.
2. **Minimal fine-tuning:** only if zero-shot is not competitive, freeze the
   foundation backbone and train the method's projection/matching heads for
   2,000 updates on our four Fixtor training routes. Do not change the method's
   core architecture.

Classify the final result:

- **Competitive:** median position error ≤ 10 m and recall@5 m ≥ 25%.
- **Works but weaker:** median position error ≤ 20 m and recall@10 m ≥ 30%.
- **Does not transfer:** neither condition is met after minimal fine-tuning.

These are local-localization gates, not global-retrieval claims.

## Methods under test

### FG²

- Source: `third_party/FG2`, kept read-only.
- Released setting: VIGOR panoramic model.
- Native representation: frozen DINOv2 features; learned selection among
  candidate heights for every BEV point; local ground/aerial descriptors;
  dual-softmax correspondences with a dustbin; weighted Procrustes or RANSAC.
- Diagnostic modes:
  - **oracle heading:** isolates whether the learned BEV and descriptors transfer;
  - **native unknown heading:** the headline result, using FG²'s released
    orientation procedure.
- Important native assumptions to preserve and report:
  `grd_bev_res=41`, `grd_height_res=11`, approximately 71 m metric support, and
  the checkpoint's expected VIGOR panorama/aerial normalization.

### BevSplat

- Source: official `wangqww/BevSplat` repository, pinned as a read-only
  submodule under `third_party/BevSplat`; record the commit and checkpoint hashes.
- Released setting: VIGOR panoramic model.
- Native representation: pretrained monocular depth, learned feature-bearing
  3D Gaussians, BEV rendering, aerial feature correlation and native orientation
  estimation.
- Diagnostic modes:
  - **oracle heading:** representation and translation transfer;
  - **native orientation:** the headline result.
- Run single-frame inference only in the primary comparison. BevSplat
  multi-frame fusion is a separate optional ablation after the single-frame
  result is known.

## Common data protocol

Reuse the current `05_lift_splat` split:

- Train:
  - `iHfmEq03Tc6752Y4Ke8wlC`
  - `NWVA14Y83pMRsijaGFkmQS`
  - `gXabFhpwk2dcl0i4518mDQ`
  - `doQ3OhJBKe56c8UxjAFmat`
- Held-out route: `IcRzj0wTLZX874qitxVsQa`.
- Keep `irAsBUKtGCfhPHuMbmOcLd` reserved and untouched.
- Pose labels: Mapillary `computed_geometry` as the position proxy and
  `computed_compass_angle` for heading. The footprint overlay remains
  approximate and must not be called survey GT.
- Primary map: Poznań 2025 orthophoto.
- Cross-year check: Poznań 2024 orthophoto with the same held-out frames.
- Primary evaluation size: at least 200 held-out frames spread uniformly along
  the route, after spatial filtering for complete map support.
- Seeds: `0, 1, 2` for fine-tuned runs; zero-shot is deterministic where the
  authors' implementation permits it.

Create one immutable manifest containing frame IDs, panorama paths, pose proxy,
map year, crop centre perturbation and crop rotation. Every method consumes the
same manifest.

## Fairness rules

1. **Same geographic prior:** crop centres are displaced by the same sampled
   offset, bounded by 10% of the 224 m reference edge (approximately ±22 m),
   and rotated within ±10°.
2. **Same source data:** identical ERP source image, orthophoto source, route
   split and pose proxy.
3. **Method-native rasterization:** each baseline receives its checkpoint's
   expected image size, physical BEV support, GSD and normalization. Do not force
   FG² or BevSplat through Sat-RoMa's 224/896 tensor sizes.
4. **Same metric evaluation:** convert every output to map-frame `(x, y, yaw)`
   before scoring.
5. **No hidden pose help:** oracle-heading runs are diagnostics only. Headline
   metrics use each method's own orientation path.
6. **No architecture tuning:** resizing, coordinate conversion, dataset
   adapters and checkpoint loading fixes are allowed. New layers, loss terms or
   solver changes are not part of this task.
7. **Comparable adaptation budget:** 2,000 updates, frozen foundation backbone,
   the authors' native objective and the same four training routes. Log wall
   time, peak GPU memory and trainable parameter count.

Because the methods have different native footprints, fairness means identical
pose uncertainty and map content—not identical input pixels.

## Outputs

Use:

```text
experiments/06_fg2_bevsplat/
├── manifest.json
├── fg2_zero/
├── fg2_finetune_seed{0,1,2}/
├── bevsplat_zero/
├── bevsplat_finetune_seed{0,1,2}/
├── baseline_lss/
└── REPORT.md
```

Every run directory contains:

- resolved `config.yaml`;
- checkpoint source and SHA-256;
- `metrics.json`;
- `frames.csv` with position error, yaw error, success/rejection and runtime;
- `viz.jpg`;
- training log when applicable.

The visual sheet must show, for the same frame IDs:

1. ERP input;
2. method-native depth/height/BEV representation;
3. orthophoto crop;
4. correspondence or pose heatmap;
5. approximate Mapillary footprint and predicted footprint;
6. position and yaw errors.

## Metrics

Report separately for 2025 and 2024:

- median and mean position error in metres;
- recall@1/5/10/20 m;
- median and mean absolute yaw error;
- recall@1/5/10°;
- finite-pose / match rate;
- 90th and 95th percentile position error;
- inference time per frame and peak GPU memory;
- zero-shot versus fine-tuned delta;
- mean ± standard deviation across fine-tuning seeds.

Place our `fixtor_seq` result in the same report as a reference, but rerun it on
the same ≥200-frame manifest. The existing n=48 result (8.7 m median,
R@5=27.1%, R@10=58.3%) is historical and is not a fair final comparison.

## Steps and checkpoints

### 1. Reproducibility check on native data

- Fetch the released FG² and BevSplat VIGOR checkpoints.
- Verify hashes and licenses.
- Run each authors' smallest native VIGOR evaluation or supplied example.
- Record one published metric reproduced within the tolerance expected from the
  released code.
- **Stop and report** if a native checkpoint cannot be reproduced; do not debug
  our adapter before the source method works.

Output: one native example image and one reproduced number per method.

### 2. Shared manifest and coordinate tests

- Add a Fixtor manifest builder using the existing Mapillary/Geoportal loaders.
- Unit-test pixel↔metric↔map transforms with synthetic translations and rotations.
- Verify that an injected `(5 m, -3 m, 7°)` pose is recovered by each adapter's
  output conversion to numerical tolerance.

Output: a map overview of all selected held-out frames and one transform-check
number.

### 3. FG² zero-shot

- Wrap FG² without editing `third_party/FG2`.
- Render method-native panorama and aerial inputs from the shared manifest.
- Run a 20-frame smoke set in oracle-heading and native-orientation modes.
- Save BEV height-selection, correspondences and pose overlays.
- If geometry is sane, run all ≥200 held-out frames for 2025 and 2024.

Output: `fg2_zero/metrics.json` and `viz.jpg`.

### 4. BevSplat zero-shot

- Pin the official repository and released VIGOR checkpoint.
- Wrap its preprocessing, Gaussian BEV renderer, score map and pose output.
- Run the same 20-frame smoke set in oracle-heading and native-orientation modes.
- Check that learned depths/Gaussians land inside the physical BEV support.
- If geometry is sane, run all ≥200 held-out frames for 2025 and 2024.

Output: `bevsplat_zero/metrics.json` and `viz.jpg`.

### 5. Four-frame adaptation checks

For each method that is not competitive zero-shot:

- overfit four training frames with all stochastic augmentation disabled;
- require the native training loss to fall and median training position error to
  reach ≤2 m;
- **stop that method** if it cannot overfit after 500 updates. Treat this as an
  adapter/training failure, not evidence that the scientific method fails.

Output: one loss curve and four pose overlays per method.

### 6. Minimal fine-tuning

- Freeze the foundation encoder.
- Train only the authors' projection, BEV, matching and orientation modules.
- Use 2,000 updates, seeds 0/1/2, the same route list and cross-year sampling.
- Select checkpoints using a validation subset from the training routes; never
  inspect the held-out route during selection.
- Preserve each method's native losses and solver.

Output: three checkpoints and logs per method.

### 7. Final evaluation and report

- Evaluate every selected checkpoint on the immutable ≥200-frame held-out
  manifest.
- Rerun our LSS baseline on that manifest.
- Produce aligned good/mid/fail visual sheets using the same frame IDs.
- Write `experiments/06_fg2_bevsplat/REPORT.md` with:
  - zero-shot and fine-tuned results;
  - oracle-heading diagnostic versus native orientation;
  - 2025 versus 2024;
  - success classification from the decision rule;
  - observed failure modes;
  - whether the novelty framing in the LSS audit must change.

Output: one comparison figure and the final classification for each method.

## Planned repository interface

Implementation should add thin project-owned adapters and leave third-party
trees unchanged:

```text
src/bevloc/baselines/fg2.py
src/bevloc/baselines/bevsplat.py
src/bevloc/baselines/common.py
scripts/build_baseline_manifest.py
scripts/eval_fg2.py
scripts/eval_bevsplat.py
scripts/train_fg2.py
scripts/train_bevsplat.py
scripts/report_baselines.py
tests/test_baseline_coordinates.py
```

Add Makefile targets:

- `baselines-manifest`
- `fg2-smoke`, `fg2-eval`, `fg2-train`
- `bevsplat-smoke`, `bevsplat-eval`, `bevsplat-train`
- `baselines-report`

## Compute

- Local 12 GB GPU: native reproduction, 20-frame smoke tests, coordinate tests
  and four-frame overfit.
- Eagle H100: 2,000-update fine-tuning and ≥200-frame evaluations.
- Do not change the Eagle Torch/CUDA pin without first testing each authors'
  dependencies in the container.

## Out of scope

- Durham transfer;
- global city-scale retrieval;
- particle filtering;
- multi-frame comparison before single-frame transfer is established;
- hyperparameter sweeps;
- architecture changes to rescue a failing baseline;
- replacing the methods' native solvers with Sat-RoMa RANSAC.
