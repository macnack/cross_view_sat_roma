#!/bin/bash
# Generic single-GPU job for this repo on Eagle (proxima H100), reusing sat_roma's container
# plus a pip overlay in ./.pydeps for rasterio/pyproj (no container of our own is needed).
# Submit from the repo root on the login node:
#   sbatch --job-name=<name> --export=ALL,CMD="scripts/x.py --args" slurm/run.sbatch
# or:  make eagle-submit JOB=<name> CMD="scripts/x.py --args"
#SBATCH --job-name=cvsr
#SBATCH --account=pl1269-01
#SBATCH --partition=proxima
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
mkdir -p slurm/logs
S=/mnt/storage_5/scratch/pl0467-01/mackop
export SAT_DATA_DIR="${SAT_DATA_DIR:-$S/sat_data}"
export HF_HOME="${HF_HOME:-$S/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$S/hf_cache_global/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="src:.pydeps"
export SATROMA_INFER_DIR="$PWD/third_party/sat_roma_infer"
export PYTHONUNBUFFERED=1
SIF="${SIF:-/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif}"
echo "[run] host=$(hostname) date=$(date -Is) CMD=$CMD"
singularity exec --nv --bind /mnt/storage_5,/mnt/storage_6 "$SIF" python $CMD
echo "[run] exit=$? date=$(date -Is)"
