#!/bin/bash
# Short multi-GPU probes of scripts/train_vigor.py on one proxima node (4 H100): runs every entry of PROBE_RUNS
# (separated by ';', each "NPROC args...": NPROC processes via torchrun, a plain single-GPU python when NPROC=1, or
# "py script.py args" = that script alone),
# with PROBE_ARGS prepended to every entry's args, one after the other. A --profile entry prints PROFILE json lines
# (one per rank + one aggregate); a --check-encoder entry prints one CHECK json line.
#   make eagle-probe-ddp PROBE_ARGS="--config ... --tag probe --out /tmp/probe" \
#        PROBE_RUNS="4 --batch 32 --profile 10; 4 --batch 32 --tf32 --profile 10"
#   grep -E 'PROFILE|CHECK' slurm/logs/<job>_<id>.log
#SBATCH --job-name=probe-ddp
#SBATCH --account=pl1269-01
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=0:30:00
#SBATCH --output=slurm/logs/%x_%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
P=/mnt/storage_6/project_data/pl1269-01/krupka_maciej
export SAT_DATA_DIR="${SAT_DATA_DIR:-$P/sat_data}"
export VIGOR_DIR="${VIGOR_DIR:-$P/vigor}"
export HF_HOME="${HF_HOME:-$P/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$P/hf_cache/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-$P/torch_home}"
export PYTHONPATH="src:.pydeps"
export SATROMA_INFER_DIR="$PWD/third_party/sat_roma_infer"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
SIF="${SIF:-/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif}"
echo "[probe-ddp] host=$(hostname) date=$(date -Is) cpus=${SLURM_CPUS_PER_TASK:-?} git=$(git log --oneline -1)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
singularity exec --nv "$SIF" python -c "import torch, timm; print('[probe-ddp] torch', torch.__version__, 'cuda', torch.version.cuda, 'nccl', torch.cuda.nccl.version(), 'timm', timm.__version__)" || true
IFS=';' read -ra RUNS <<< "${PROBE_RUNS:?PROBE_RUNS is required}"
for run in "${RUNS[@]}"; do
  read -r n args <<< "$run"
  echo "[probe-ddp] nproc=$n args=$args $(date -Is)"
  if [ "$n" = py ]; then                                  # a helper script (e.g. a checkpoint key check), no PROBE_ARGS
    singularity exec --nv --bind /mnt/storage_5,/mnt/storage_6 "$SIF" python $args
  elif [ "$n" = 1 ]; then
    CUDA_VISIBLE_DEVICES=0 singularity exec --nv --bind /mnt/storage_5,/mnt/storage_6 "$SIF" \
      python scripts/train_vigor.py ${PROBE_ARGS:-} $args
  else
    singularity exec --nv --bind /mnt/storage_5,/mnt/storage_6 "$SIF" \
      python -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node "$n" scripts/train_vigor.py ${PROBE_ARGS:-} $args
  fi
  echo "[probe-ddp] exit=$? $(date -Is)"
done
echo "[probe-ddp] done date=$(date -Is)"
