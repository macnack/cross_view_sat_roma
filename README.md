# cross_view_sat_roma — camera-only BEV-to-orthophoto localization

Project context, hypotheses and rules: `CLAUDE.md`, `docs/kick_off.html`, `docs/decisions.md`.
Current task: `docs/tasks/01_kickoff.md`. This README is only about **running the code**.

## Quick start
```bash
git clone --recurse-submodules git@github.com:macnack/cross_view_sat_roma.git   # or: git submodule update --init
make deps      # einops + pytest into ./.pydeps (the conda env is not modified)
make test      # 19 unit tests, no data needed
make fetch     # Dur360BEV subset (~2 h, ~35 GB kept); see data/README.md
make all       # calibration artefacts, odometry, BEV panels, mosaic, matcher smoke test
make help      # list of targets
```
Each target is one script in `scripts/`; all read `configs/default.yaml` (`CONFIG=...` to override)
and write a `config.yaml` snapshot (resolved config, command line, git revision) next to their outputs.
Common overrides: `make bevs FRAMES="500:1400:100"`, `make mosaic REF=250`, `make viewer FRAMES="0 1600"`.

Environments: locally the existing conda env `bev-patch-pf` (`PY=` to change) and `kissslam` for
`make odometry` (`PY_KISS=`). `environment.yml` is for the Eagle cluster only.
Sat-RoMa is loaded from `~/Github/sat-roma-infer` (or `SATROMA_INFER_DIR`); weights come from the HF hub.

## Layout
```
configs/default.yaml      every tunable parameter (calibration, grid, ground/edge/mask settings, matcher)
src/bevloc/
  config.py  run.py       YAML config + run snapshots; shared CLI set-up
  data/dur360.py          frames: dual fisheye -> ERP, LiDAR scan parsing (verified byte layout), OxTS
  data/calib.py           camera<-LiDAR extrinsic, heights
  data/ortho.py           EPSG:27700 reference crops + GT query->reference homography
  bev/fisheye.py          Dur360BEV lens model: LiDAR -> dual fisheye, ERP remap maps
  bev/spherical.py        LiDAR -> ERP
  bev/grid.py             BEV grid, gravity alignment, IPM, LiDAR splatting
  bev/ground.py           ground segmentation, contact line (wall/obstacle feet), wall edges
  bev/variants.py         the kick-off query variants (ipm_raw, ipm_cl, ipm_edge, oracle_a, oracle_b, ipm_fp)
  bev/mask.py             ego-vehicle + seam validity mask
  bev/mosaic.py           multi-frame mosaic from relative poses
  match/satroma.py        Sat-RoMa wrapper (+ mode / inlier statistics, patch-validity threshold)
  eval/metrics.py         position / corner / yaw error, recall
  viz.py                  display only
scripts/                  thin CLIs (see `make help`); scripts/attic = rejected approaches, with reasons
tests/                    geometry, ortho, scan parsing, mosaic
experiments/<run>/        outputs + config.yaml snapshot
```

## Status (2026-09-17)
Done: loader, lens model + camera<-LiDAR check, ego/seam mask, BEV variants, mosaic, ortho crop + GT code
(tested on a synthetic GeoTIFF), matcher wrapper smoke-tested.
Blocked: OxTS poses (end of the download) -> yaw convention, roll/pitch, timestamp pairing;
Durham orthophoto -> wall-on-ortho overlay and the actual H1 matching run.
Open decisions: see the end of `docs/decisions.md`.
