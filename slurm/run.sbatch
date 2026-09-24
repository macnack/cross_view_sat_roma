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
# storage_6 (NFS project_data) on purpose: the Lustre scratch on storage_5 gave I/O errors and
# "transport endpoint shutdown" on 2026-09-23; project_data is mounted on login and compute nodes.
P=/mnt/storage_6/project_data/pl1269-01/krupka_maciej
export SAT_DATA_DIR="${SAT_DATA_DIR:-$P/sat_data}"
export VIGOR_DIR="${VIGOR_DIR:-$P/vigor}"
export HF_HOME="${HF_HOME:-$P/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$P/hf_cache/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-$P/torch_home}"        # hub/checkpoints/dinov2_vitl14_pretrain.pth for FG²
export PYTHONPATH="src:.pydeps"
export SATROMA_INFER_DIR="$PWD/third_party/sat_roma_infer"
export PYTHONUNBUFFERED=1
SIF="${SIF:-/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif}"
echo "[run] host=$(hostname) date=$(date -Is) CMD=$CMD"
singularity exec --nv --bind /mnt/storage_5,/mnt/storage_6 "$SIF" python $CMD
echo "[run] exit=$? date=$(date -Is)"
