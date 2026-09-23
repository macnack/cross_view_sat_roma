# Task 01 (kick-off) — every step is one target. `make help` lists them.
# Local env: the existing conda env `bev-patch-pf` (environment.yml is for Eagle only);
# pure-python extras (einops, pytest) live in ./.pydeps so that env is not modified.
PY      ?= $(HOME)/miniconda3/envs/bev-patch-pf/bin/python
PY_KISS ?= $(HOME)/miniconda3/envs/kissslam/bin/python
CONFIG  ?= configs/default.yaml
FRAMES  ?= 0 1000 1600 2000
REF     ?= 200
RUN      = PYTHONPATH=src:.pydeps $(PY)

.PHONY: help deps test fetch stats mask lens viewer oxts odometry bevs mosaic smoke h1 targets overfit train calib all roma-viz mapillary mapillary-scan mapillary-sample ipm-smoke lift-overfit lift-splat lift-eval lift-splat-years lift-aug-viz lift-layers-viz lift-splat-aug lift-splat-pose-nll lift-splat-seq baselines-manifest fg2-smoke fg2-eval fg2-train bevsplat-smoke bevsplat-eval bevsplat-train baselines-report
help:
	@grep -E '^[a-z0-9_-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | expand -t 12

deps:      ## install pure-python extras into ./.pydeps (does not touch the conda env)
	$(PY) -m pip install -q --target .pydeps --no-deps einops
	$(PY) -m pip install -q --target .pydeps pytest

test:      ## unit tests (no data needed)
	$(RUN) -m pytest tests -q -p no:cacheprovider

IMAGE ?= 735205558899799
# OSM city boundary of Poznań (west,south,east,north).
BBOX  ?= 16.7315878,52.2919238,17.0717065,52.5093282
SCAN_NAME ?= poznan

mapillary: ## one Mapillary 360 sequence (env MAPILLARY_TOKEN, IMAGE=<id>)
	MAPILLARY_TOKEN="$$MAPILLARY_TOKEN" $(RUN) -m mapillary_dl --image $(IMAGE)

mapillary-scan: ## list 360 sequences in BBOX=west,south,east,north (default: Poznań)
	MAPILLARY_TOKEN="$$MAPILLARY_TOKEN" $(RUN) -m mapillary_dl --bbox $(BBOX) --name $(SCAN_NAME)

FRAMES_PER_USER ?= 12

mapillary-sample: ## 12 frames from the longest in-box sequence of each user in the Poznań scan
	MAPILLARY_TOKEN="$$MAPILLARY_TOKEN" $(RUN) -m mapillary_dl --sample-scan data/mapillary/scans/$(SCAN_NAME).json --frames $(FRAMES_PER_USER)

fetch:     ## stream the Dur360BEV subset from Hugging Face (~170 GB through the pipe, ~35 GB kept, ~2 h)
	setsid nohup scripts/fetch_dur360bev_subset.sh data/dur360bev > data/fetch.log 2>&1 < /dev/null &
	@echo "running detached; follow with: tail -f data/fetch.log"

stats:     ## LiDAR return statistics, BEV occupancy, road-plane height
	$(RUN) scripts/calib_lidar_stats.py --config $(CONFIG)

mask:      ## ERP validity mask (ego vehicle + seams) -> experiments/00_calib/erp_valid_*.npy
	$(RUN) scripts/calib_erp_mask.py --config $(CONFIG)

lens:      ## static lens-model and camera-height check images
	$(RUN) scripts/calib_lens_check.py --config $(CONFIG)

viewer:    ## interactive LiDAR<->camera overlay (HTML); FRAMES="0 1600"
	$(RUN) scripts/calib_overlay_viewer.py --config $(CONFIG) --frames $(FRAMES)

oxts:      ## OxTS conventions on data: yaw zero/handedness, roll/pitch, modality sync
	$(RUN) scripts/oxts_check.py --config $(CONFIG)

odometry:  ## KISS-ICP poses for the dense clip (kissslam env)
	$(PY_KISS) scripts/lidar_odometry.py

bevs:      ## BEV variant panels; FRAMES="1000 2000" or ranges "500:1400:100"
	$(RUN) scripts/make_bevs.py --config $(CONFIG) --frames $(FRAMES)

mosaic:    ## BEV mosaic at 0/1/5/10 m from REF (needs `make odometry`)
	$(RUN) scripts/make_mosaic.py --config $(CONFIG) --ref $(REF)

SOLVER ?=

smoke:     ## Sat-RoMa wrapper end-to-end on a synthetic pair; SOLVER=se2 for the fixed-scale solver
	$(RUN) scripts/smoke_satroma.py --config $(CONFIG) $(if $(SOLVER),--solver $(SOLVER),) --out experiments/00_smoke$(if $(SOLVER),_$(SOLVER),)

h1:        ## zero-shot Sat-RoMa on EA-covered frames, plus the token cosine probe
	$(RUN) scripts/h1_ea.py --config $(CONFIG)

roma-probe: ## zero-shot RoMa posterior (top-k of the GT cell) on RGB BEV vs EA
	$(RUN) scripts/train_roma_rgb.py --config $(CONFIG) --probe 40 --variant oracle_b

roma-rgb:  ## fine-tune Sat-RoMa's coarse head on RGB BEV vs EA (fine refiner frozen)
	$(RUN) scripts/train_roma_rgb.py --config $(CONFIG)

roma-cosine: ## same pairs, but a learned cosine instead of the Sat-RoMa decoder
	$(RUN) scripts/train_roma_rgb.py --config $(CONFIG) --head cosine --out experiments/03_roma_rgb/oracle_b_cosine

roma-viz:  ## sheet of local vs global search windows and a few matched frames
	$(RUN) scripts/viz_roma_rgb.py --config $(CONFIG)

ipm-smoke: ## one Mapillary frame, flat-ground IPM over the 2025 Poznań orthophoto
	$(RUN) scripts/ipm_mapillary.py

lift-overfit: ## spherical Lift-Splat plumbing: 4 Fixtor frames, local RoMa CE must fall
	$(RUN) scripts/train_lift_splat.py --overfit 4 --steps 40 --out experiments/05_lift_splat/overfit4

lift-splat: ## multi-seq Lift-Splat + RoMa CE; hold out IcRzj; 2000 steps
	$(RUN) scripts/train_lift_splat.py --out experiments/05_lift_splat/fixtor_multi --steps 2000

lift-eval: ## RANSAC pose (peak + GMM means) on the held-out route
	$(RUN) scripts/eval_lift_splat.py --ckpt checkpoints/05_lift_splat_fixtor_multi_best.pt \
		--out experiments/05_lift_splat/fixtor_multi --n 48

lift-splat-years: ## resume best ckpt; cross-year refs 2025/2024/2021 + neighbour hinge
	$(RUN) scripts/train_lift_splat.py --out experiments/05_lift_splat/fixtor_years \
		--ckpt checkpoints/05_lift_splat_fixtor_multi_best.pt \
		--years 2025,2024,2021 --val-year 2025 --steps 3000 \
		--neighbour-radius 4 --neighbour-weight 0.5

lift-aug-viz: ## preview random poses + ref scales used as training aug
	$(RUN) scripts/viz_lift_aug.py

lift-layers-viz: ## ERP→depth→BEV→f_q→match→pose layer sheet (seq ckpt)
	$(RUN) scripts/viz_lift_layers.py --ckpt checkpoints/05_lift_splat_fixtor_seq_best.pt --tag seq

lift-splat-aug: ## cross-year + multi-scale ref + pose/colour/attitude aug; resume years best
	$(RUN) scripts/train_lift_splat.py --out experiments/05_lift_splat/fixtor_aug \
		--ckpt checkpoints/05_lift_splat_fixtor_years_best.pt \
		--years 2025,2024,2021 --val-year 2025 --steps 3000 \
		--neighbour-radius 4 --neighbour-weight 0.5

lift-splat-pose-nll: ## tonight deep-dive: CE + pose-heatmap NLL from aug best
	$(RUN) scripts/train_lift_splat.py --out experiments/05_lift_splat/fixtor_pose_nll \
		--ckpt checkpoints/05_lift_splat_fixtor_aug_best.pt \
		--years 2025,2024,2021 --val-year 2025 --steps 2500 \
		--neighbour-radius 4 --neighbour-weight 0.5 \
		--pose-nll-weight 0.5

lift-splat-seq: ## multi-frame Mapillary splat (0/2/5 m) + pose NLL; prefers pose_nll then aug best
	@CKPT=checkpoints/05_lift_splat_fixtor_pose_nll_best.pt; \
	[ -f $$CKPT ] || CKPT=checkpoints/05_lift_splat_fixtor_aug_best.pt; \
	$(RUN) scripts/train_lift_splat.py --out experiments/05_lift_splat/fixtor_seq \
		--ckpt $$CKPT \
		--years 2025,2024,2021 --val-year 2025 --steps 2000 \
		--neighbour-radius 4 --neighbour-weight 0.5 \
		--pose-nll-weight 0.5 \
		--seq-dists 0,2,5

# --- Eagle (docs/tasks/03_eval_density_solver_pf.md Task 0; slurm/README.md) ---
EAGLE_DIR ?= /mnt/storage_6/project_data/pl1269-01/krupka_maciej/cross_view_sat_roma

eagle-sync: ## push Fixtor panoramas, manifests and best checkpoints to Eagle scratch (13 GB first time)
	rsync -avP data/mapillary/Fixtor eagle:$(EAGLE_DIR)/data/mapillary/
	rsync -avP experiments/06_fg2_bevsplat/manifest*.json eagle:$(EAGLE_DIR)/experiments/06_fg2_bevsplat/
	rsync -avP checkpoints/05_lift_splat_fixtor_*_best.pt eagle:$(EAGLE_DIR)/checkpoints/

mapillary-seq: ## download one Mapillary sequence by id: SEQ=<sequence id> (env MAPILLARY_TOKEN)
	MAPILLARY_TOKEN="$$MAPILLARY_TOKEN" $(RUN) -m mapillary_dl --sequence $(SEQ) --out data/mapillary

manifest-val:  ## 200-frame validation manifest on IcRzj (2025+2024)
	$(RUN) scripts/build_baseline_manifest.py --config $(CONFIG) --seq Fixtor/IcRzj0wTLZX874qitxVsQa \
		--out experiments/06_fg2_bevsplat/manifest.json --n 200

manifest-test: ## 200-frame TEST manifest on the reserved route irAsBUK (make mapillary-seq SEQ=irAsBUKtGCfhPHuMbmOcLd first)
	$(RUN) scripts/build_baseline_manifest.py --config $(CONFIG) --seq Fixtor/irAsBUKtGCfhPHuMbmOcLd \
		--out experiments/06_fg2_bevsplat/manifest_test.json --n 200

eval-pose: ## score CKPT on MANIFEST with bootstrap CIs and a centre-guess chance row; TAG names the json
	$(RUN) scripts/eval_pose.py --config $(CONFIG) --ckpt $(CKPT) --manifest $(MANIFEST) --tag $(TAG) $(EVAL_ARGS)

pose-report: ## experiments/05_lift_splat/REPORT.md from every eval_*.json
	$(RUN) scripts/report_pose.py

eagle-submit: ## submit CMD="scripts/x.py ..." JOB=name as one H100 job (run on Eagle, repo root)
	mkdir -p slurm/logs   # SLURM does not create the --output directory itself
	@# CMD goes through the environment, NOT --export=ALL,CMD=...: sbatch splits --export on commas,
	@# which silently truncated "--years 2025,2024,2021 ..." and "--seq-dists 0,2,5" (2026-09-23).
	CMD="$(CMD)" sbatch --job-name=$(JOB) --export=ALL slurm/run.sbatch

ipm-train: ## camera-only RGB-IPM query (frozen encoder, decoder fine-tune) with the Lift-Splat recipe; task 03 Task 2
	$(RUN) scripts/train_lift_splat.py --config $(CONFIG) --query ipm --out experiments/05_lift_splat/fixtor_ipm \
		--years 2025,2024,2021 --val-year 2025 --steps 3000 \
		--neighbour-radius 4 --neighbour-weight 0.5 --pose-nll-weight 0.5

lift-splat-pose-nll-only: ## A alone: pose-heatmap NLL, single frame, from aug best (separates A from B); task 03 Task 1
	$(RUN) scripts/train_lift_splat.py --config $(CONFIG) --out experiments/05_lift_splat/fixtor_pose_nll \
		--ckpt checkpoints/05_lift_splat_fixtor_aug_best.pt \
		--years 2025,2024,2021 --val-year 2025 --steps 2000 \
		--neighbour-radius 4 --neighbour-weight 0.5 --pose-nll-weight 0.5

hybrid-viz: ## ERP | IPM ground mask | dense ground PCA | f_q PCA | pose boxes for a hybrid checkpoint (CKPT=, TAG=)
	$(RUN) scripts/viz_hybrid.py --config $(CONFIG) --ckpt $(CKPT) --tag $(TAG)

ROUTE ?= Fixtor/IcRzj0wTLZX874qitxVsQa
YEAR ?= 2025

track-route: ## particle filter along ROUTE with CKPT's heatmap as observation (TAG=, YEAR=, TRACK_ARGS=)
	$(RUN) scripts/track_route.py --config $(CONFIG) --ckpt $(CKPT) --route $(ROUTE) --year $(YEAR) --tag $(TAG) $(TRACK_ARGS)

# --- Task 02: FG² / BevSplat transfer (docs/tasks/02_fg2_bevsplat.md) ---
baselines-manifest: ## immutable ≥200-frame Fixtor held-out manifest (2025+2024)
	$(RUN) scripts/build_baseline_manifest.py --out experiments/06_fg2_bevsplat/manifest.json --n 200

fg2-smoke: ## FG² 20-frame smoke on the shared manifest (oracle + native heading)
	$(RUN) scripts/eval_fg2.py --manifest experiments/06_fg2_bevsplat/manifest.json \
		--out experiments/06_fg2_bevsplat/fg2_zero --n 20 --smoke

fg2-eval: ## FG² zero-shot on the full held-out manifest
	$(RUN) scripts/eval_fg2.py --manifest experiments/06_fg2_bevsplat/manifest.json \
		--out experiments/06_fg2_bevsplat/fg2_zero

fg2-train: ## FG² minimal fine-tune (frozen DINO, 2000 updates); SEED=0/1/2
	$(RUN) scripts/train_fg2.py --manifest experiments/06_fg2_bevsplat/manifest.json \
		--out experiments/06_fg2_bevsplat/fg2_finetune_seed$${SEED:-0} --seed $${SEED:-0} --steps 2000

bevsplat-smoke: ## BevSplat 20-frame smoke on the shared manifest
	$(RUN) scripts/eval_bevsplat.py --manifest experiments/06_fg2_bevsplat/manifest.json \
		--out experiments/06_fg2_bevsplat/bevsplat_zero --n 20 --smoke

bevsplat-eval: ## BevSplat zero-shot on the full held-out manifest
	$(RUN) scripts/eval_bevsplat.py --manifest experiments/06_fg2_bevsplat/manifest.json \
		--out experiments/06_fg2_bevsplat/bevsplat_zero

bevsplat-train: ## BevSplat minimal fine-tune; SEED=0/1/2
	$(RUN) scripts/train_bevsplat.py --manifest experiments/06_fg2_bevsplat/manifest.json \
		--out experiments/06_fg2_bevsplat/bevsplat_finetune_seed$${SEED:-0} --seed $${SEED:-0} --steps 2000

baselines-report: ## write experiments/06_fg2_bevsplat/REPORT.md from run metrics
	$(RUN) scripts/report_baselines.py --root experiments/06_fg2_bevsplat

targets:   ## fusion training: check the coarse-loss class layout against the released checkpoint
	$(RUN) scripts/train_fusion.py --config $(CONFIG) --check-targets

overfit:   ## fusion training plumbing test: 4 frames, SYNTHETIC reference (no orthophoto needed)
	$(RUN) scripts/train_fusion.py --config $(CONFIG) --overfit 4 --steps 500 --out experiments/02_fusion/overfit4

train:     ## fusion BEV + Sat-RoMa decoder on the real orthophoto (needs data.ortho in the config)
	$(RUN) scripts/train_fusion.py --config $(CONFIG)

calib: stats mask lens oxts viewer   ## all calibration artefacts
all: test calib odometry bevs mosaic smoke   ## everything that needs no orthophoto
