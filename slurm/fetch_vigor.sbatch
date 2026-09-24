#!/bin/bash
# Download + extract VIGOR from the shared Google Drive folder (scripts/fetch_vigor.py). CPU only.
# Submit from the repo root on Eagle:  make eagle-fetch-vigor     (resubmit after a quota error: it resumes)
#SBATCH --job-name=fetch_vigor
#SBATCH --account=pl1269-01
#SBATCH --partition=proxima-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
mkdir -p slurm/logs
P=/mnt/storage_6/project_data/pl1269-01/krupka_maciej
export VIGOR_DIR="${VIGOR_DIR:-$P/vigor}"
export PYTHONPATH="src:.pydeps"
export PYTHONUNBUFFERED=1
SIF="${SIF:-/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif}"
echo "[fetch] host=$(hostname) date=$(date -Is) out=$VIGOR_DIR"
singularity exec --bind /mnt/storage_5,/mnt/storage_6 "$SIF" python scripts/fetch_vigor.py --out "$VIGOR_DIR"
echo "[fetch] exit=$? date=$(date -Is)"
du -sh "$VIGOR_DIR"/* 2>/dev/null || true
