# CLAUDE.md — camera-only BEV-to-orthophoto localization

## What this project is
GNSS-denied localization of a ground vehicle by matching a bird's-eye-view (BEV) built from a
single 360° spherical camera against public orthophotos. LiDAR is used ONLY at training time to
teach geometry (surface normals, planarity, wall–ground contact line). At inference: camera, IMU
roll/pitch, known camera height. Matcher: our Sat-RoMa multi-hypothesis matcher (docs/satroma_acivs2026.pdf).
Full plan and hypotheses H1–H7: docs/kickoff.html. Decisions already taken: docs/decisions.md.

## Current task
docs/tasks/03_eval_density_solver_pf.md (branch `task03/eval-density-solver-pf`): manifest evaluation with
bootstrap intervals, query modes (lift | ipm | hybrid | erp), solver ablation, particle filter. Results and
verdicts: experiments/05_lift_splat/REPORT.md + VERDICT.md, experiments/07_track/REPORT.md. Task 01 is
reported (experiments/01_kickoff/REPORT.md, H1 falsified); task 02 (FG²/BevSplat) is blocked on data/env.
Method of record since 2026-09-24: camera-only IPM picture + Sat-RoMa (`--query ipm`); learned lifts are
ablation rows only. VIGOR benchmark rows (ours, FG², Loc², like-for-like draws): experiments/09_vigor/.
Next task (agreed 2026-09-25): docs/tasks/04_loc2_matcher.md — Loc² geometry with the Sat-RoMa decoder as matcher.
Open design decisions are listed at the end of docs/decisions.md.

## Repo layout
- third_party/   read-only: Dur360BEV, RoMa, RoMaV2, FG2, VIGOR, bev-patch-pf, lift-splat-shoot. Wrap, never edit.
                 Sat-RoMa inference code is used from ~/Github/sat-roma-infer (or $SATROMA_INFER_DIR).
- configs/       default.yaml = every tunable parameter; scripts snapshot it into their output folder
- src/bevloc/    data/ (frames, calib, ortho, mapillary), bev/ (lens, projection, ground/contact, variants, mask,
                 mosaic, ipm_sphere), model/ (coarse loss, query modes: lift_splat, hybrid_query, erp_query, query.py),
                 match/ (Sat-RoMa wrapper, se2 solver), eval/ (metrics, bootstrap report), track/ (particle filter),
                 viz.py — see README.md
- scripts/       thin CLIs, one per Makefile target (`make help`); scripts/attic = rejected approaches
- tests/         `make test`
- data/          see data/README.md
- experiments/   one folder per run: config.yaml snapshot, metrics, overlays
- docs/          kick_off.html, related/ papers, decisions.md, tasks/

## Coordinate frames (fill in as verified; mark UNVERIFIED until checked on data)
- LiDAR (Ouster OS1-128): x forward, y left, z up. .bin scans: 9 fields/point — see Dur360BEV code for field order.
- Camera (Ricoh Theta S): raw frames are 1280×720 dual fisheye → ERP 1280×640 built with Dur360BEV's 196°/203° piecewise lens
  model (`bevloc.bev.fisheye`), NOT fisheye_tools' 203° equidistant remap (~6 px off). Axis convention VERIFIED: forward = ERP
  centre, azimuth to the right, z up. Camera←LiDAR: R = I, t = (0, 0, −0.27) m (provisional, see `bevloc.data.calib`).
  Car body hides everything below ≈ −25° elevation.
- INS (OxTS RT3000v3): lat/lon/alt, roll/pitch/yaw. Stored yaw VERIFIED as radians CCW from east (bearing
  cw from north = 90° − yaw°; residual σ 2.5° vs direction of travel, `make oxts`) — NOT the NED device
  convention. Roll and pitch are identically zero in every file: no IMU attitude in this dataset.
  INS→LiDAR lever arm unknown; estimate from data.
- Map: British National Grid EPSG:27700; orthophoto GeoTIFFs under data/ortho/durham/<year>/.
- BEV grid: metric, ego-centred, x forward / y left, gravity-aligned; 224×224 cells at 0.25 m (56 m extent, radius 28 m),
  inside the Sat-RoMa checkpoint's 35–70 m query band. Reference crop: 896 px at the same GSD (224 m).

## Compute
- Local: 12 GB GPU → smoke tests, ≤ batch 2, ≤ 20 frames.
- Eagle cluster: all real runs. `eagle.man.poznan.pl`, SLURM, proxima partition, H100 GPUs, account `pl1269-01`
  (or `pl0467-01`). Submit: `sbatch slurm/train_fusion.sbatch` (see `slurm/README.md` — container-based, UNTESTED,
  built from ~/Github/sat_roma's verified pattern; no Eagle access from this session to confirm it). GPU time is free;
  optimize for correctness.
- Environment: environment.yml is for Eagle only (pinned; do not upgrade torch/CUDA) — but its 2.4/cu121 pin predates
  knowing Eagle's real target is H100 (sm_90); see slurm/README.md's "Torch pin" note before relying on it. Locally
  use the existing conda env `bev-patch-pf` (torch 2.13+cu130; the laptop's Blackwell GPU cannot run torch 2.4/cu121).

## Working rules
- Run things through the Makefile (`make help`); new steps get a script + target + config entry, not a one-off snippet.
- Small verifiable steps: after each step show one image or one number before continuing.
- Ask before any design decision not in docs/decisions.md; append the agreed decision there.
- Reproducible: seeds and configs in experiments/<run>/config.yaml. Never commit data or checkpoints.
- If code or papers contradict this file or the task, say so explicitly.
- Unit sanity checks on every projection: a known LiDAR wall must land on the orthophoto building edge.
