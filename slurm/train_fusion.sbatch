#!/bin/bash
# Eagle SLURM job: scripts/train_fusion.py inside the container built by
# build_container.sh. Single GPU, proxima H100 -- matches sat_roma's
# slurm/profiles/hw/1gpu_proxima.env sizing (docs/eagle_hpc_slurm_ssh_guide.md
# in ~/Github/sat_roma), scaled down for this repo's much smaller model
# (7.6M trainable BEV params + a fine-tuned 84M-param decoder vs sat_roma's
# full DINOv3 training jobs) -- MEM/TIME are a starting guess, not measured.
#
# Submit:  sbatch slurm/train_fusion.sbatch [extra train_fusion.py args...]
# Dry-run the resource lines only: sbatch --test-only slurm/train_fusion.sbatch
#
#SBATCH --job-name=fusion_train
#SBATCH --account=pl1269-01
#SBATCH --partition=proxima
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.log

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p slurm/logs

SIF="${SIF:-$HOME/.singularity/cross_view_sat_roma.sif}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the Dur360BEV copy under project_data/scratch}"
ORTHO="${ORTHO:?set ORTHO to the reference GeoTIFF/VRT under project_data/scratch}"

singularity exec --nv \
  --bind "$DATA_ROOT":/data/dur360bev:ro \
  --bind "$ORTHO":/data/ortho.tif:ro \
  --bind "$PWD"/checkpoints:/app/checkpoints \
  --bind "$PWD"/experiments:/app/experiments \
  "$SIF" \
  python scripts/train_fusion.py --config configs/eagle.yaml --ortho /data/ortho.tif "$@"
