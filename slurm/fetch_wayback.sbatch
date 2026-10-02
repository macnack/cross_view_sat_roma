#!/bin/bash
# Esri World Imagery Wayback windows for the VIGOR tiles (task 05): the `make wayback-fetch` / `make wayback-calib`
# commands on a CPU node with internet (no GPU needed; the sat_roma container has cv2 / numpy / yaml / torch for the
# draw). The scripts are called directly, like slurm/fetch_vigor.sbatch (the container has no `make`).
# Submit from the repo root on Eagle:
#   make eagle-fetch-wayback WAYBACK_TARGET=wayback-calib CITY=Chicago
#   make eagle-fetch-wayback WAYBACK_TARGET=wayback-fetch SPLIT=samearea CITIES=Chicago LIMIT=3000 YEARS="2025 2021 2019"
#   make eagle-fetch-wayback WAYBACK_TARGET=wayback-calib CITY=Chicago WAYBACK_ARGS="--pairs-dir /mnt/storage_6/project_data/pl1269-01/krupka_maciej/vigor"   (offline re-measure)
#   ... WAYBACK_ARGS="--dry-run" first to see the releases chosen and the volume.
#SBATCH --job-name=fetch_wayback
#SBATCH --account=pl1269-01
#SBATCH --partition=proxima-cpu
#SBATCH --cpus-per-task=8                 # wayback.workers (6) threads decode / warp / encode; the global rate_hz (10/s), not the CPUs, bounds the fetch
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
mkdir -p slurm/logs
P=/mnt/storage_6/project_data/pl1269-01/krupka_maciej
export VIGOR_DIR="${VIGOR_DIR:-$P/vigor}"
export PYTHONPATH="src:.pydeps"
export PYTHONUNBUFFERED=1
SIF="${SIF:-/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif}"
TARGET="${WAYBACK_TARGET:-wayback-fetch}"
CONFIG="${CONFIG:-configs/default.yaml}"
echo "[wayback] host=$(hostname) date=$(date -Is) target=$TARGET root=$VIGOR_DIR"
case "$TARGET" in
  wayback-calib)
    CMD="scripts/calibrate_wayback_vigor.py --config $CONFIG --root $VIGOR_DIR --city ${CITY:?CITY= is required} ${WAYBACK_ARGS:-}" ;;
  wayback-fetch)
    CMD="scripts/fetch_wayback_vigor.py --config $CONFIG --root $VIGOR_DIR --split ${SPLIT:-samearea} --limit ${LIMIT:-0}"
    [ -n "${CITIES:-}" ] && CMD="$CMD --cities $CITIES"
    [ -n "${YEARS:-}" ] && CMD="$CMD --years $YEARS"
    CMD="$CMD ${WAYBACK_ARGS:-}" ;;
  *) echo "unknown WAYBACK_TARGET $TARGET (wayback-fetch | wayback-calib)"; exit 2 ;;
esac
echo "[wayback] python $CMD"
singularity exec --bind /mnt/storage_5,/mnt/storage_6 "$SIF" python $CMD
echo "[wayback] exit=$? date=$(date -Is)"
du -sh "$VIGOR_DIR"/*/wayback_* "$VIGOR_DIR"/wayback_tiles 2>/dev/null || true
