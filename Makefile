# Task 01 (kick-off) — every step is one target. `make help` lists them.
# Local env: the existing conda env `bev-patch-pf` (environment.yml is for Eagle only);
# pure-python extras (einops, pytest) live in ./.pydeps so that env is not modified.
PY      ?= $(HOME)/miniconda3/envs/bev-patch-pf/bin/python
PY_KISS ?= $(HOME)/miniconda3/envs/kissslam/bin/python
CONFIG  ?= configs/default.yaml
FRAMES  ?= 0 1000 1600 2000
REF     ?= 200
RUN      = PYTHONPATH=src:.pydeps $(PY)

.PHONY: help deps test fetch stats mask lens viewer odometry bevs mosaic smoke calib all
help:
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | expand -t 12

deps:      ## install pure-python extras into ./.pydeps (does not touch the conda env)
	$(PY) -m pip install -q --target .pydeps --no-deps einops
	$(PY) -m pip install -q --target .pydeps pytest

test:      ## unit tests (no data needed)
	$(RUN) -m pytest tests -q -p no:cacheprovider

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

odometry:  ## KISS-ICP poses for the dense clip (kissslam env)
	$(PY_KISS) scripts/lidar_odometry.py

bevs:      ## BEV variant panels; FRAMES="1000 2000" or ranges "500:1400:100"
	$(RUN) scripts/make_bevs.py --config $(CONFIG) --frames $(FRAMES)

mosaic:    ## BEV mosaic at 0/1/5/10 m from REF (needs `make odometry`)
	$(RUN) scripts/make_mosaic.py --config $(CONFIG) --ref $(REF)

smoke:     ## Sat-RoMa wrapper end-to-end on a synthetic pair
	$(RUN) scripts/smoke_satroma.py --config $(CONFIG)

calib: stats mask lens viewer   ## all calibration artefacts
all: test calib odometry bevs mosaic smoke   ## everything that needs no orthophoto
