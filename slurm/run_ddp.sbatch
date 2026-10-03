#!/bin/bash
# One proxima node, 4 H100: torchrun (single node, one process per GPU) of a DDP-aware script inside sat_roma's
# container (scripts/train_vigor.py; bevloc.model.ddp). --batch in CMD is the GLOBAL batch (split over the ranks).
# Submit from the repo root on the login node:
#   make eagle-submit-ddp JOB=<name> CMD="scripts/train_vigor.py ... --batch 32 --workers 6 --pin-memory"
# CMD goes through the environment (see eagle-submit in the Makefile); NPROC (default 4) = processes = GPUs.
# The whole node: 4 GPUs, 64 CPUs (16 per rank: DataLoader workers + the main process), 480 GB of the 773 GB.
#SBATCH --job-name=cvsr-ddp
#SBATCH --account=pl1269-01
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm/logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
mkdir -p slurm/logs
P=/mnt/storage_6/project_data/pl1269-01/krupka_maciej
export SAT_DATA_DIR="${SAT_DATA_DIR:-$P/sat_data}"
export VIGOR_DIR="${VIGOR_DIR:-$P/vigor_corrected}"   # SliceMatch corrected labels only (raw labels raise RawLabelsError)
export HF_HOME="${HF_HOME:-$P/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$P/hf_cache/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-$P/torch_home}"
export PYTHONPATH="src:.pydeps"
export SATROMA_INFER_DIR="$PWD/third_party/sat_roma_infer"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"          # torchrun would force 1; the main processes do little CPU work
NPROC="${NPROC:-4}"
SIF="${SIF:-/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif}"
echo "[ddp] host=$(hostname) date=$(date -Is) nproc=$NPROC cpus=${SLURM_CPUS_PER_TASK:-?} git=$(git log --oneline -1) CMD=$CMD"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
singularity exec --nv --bind /mnt/storage_5,/mnt/storage_6 "$SIF" \
  python -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node "$NPROC" $CMD
echo "[ddp] exit=$? date=$(date -Is)"
