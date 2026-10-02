#!/bin/bash
# Throughput / memory probe of scripts/train_vigor.py --profile (one H100): loops over batch sizes, DataLoader
# workers, prefetch factors and pin_memory, one python process per setting (a clean CUDA context each), and stops
# the batch loop at the first CUDA OOM (exit code 3). Every setting prints one "PROFILE {json}" line.
#   make eagle-probe PROBE_BATCHES="4 8 16 32 48 64" PROBE_WORKERS=8 PROBE_ARGS="--config configs/vigor_cell0125.yaml ..."
#   grep PROFILE slurm/logs/<job>_<id>.log
# Keep --time short (probes are <= 30 min); override on the command line: SBATCH_ARGS="--time=0:20:00".
#SBATCH --job-name=probe
#SBATCH --account=pl1269-01
#SBATCH --partition=proxima
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
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
SIF="${SIF:-/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif}"
echo "[probe] host=$(hostname) date=$(date -Is) cpus=${SLURM_CPUS_PER_TASK:-?} git=$(git log --oneline -1)"
echo "[probe] batches='${PROBE_BATCHES:-4}' workers='${PROBE_WORKERS:-8}' prefetch='${PROBE_PREFETCH:-2}' pin='${PROBE_PIN:-0}' args=${PROBE_ARGS:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
for b in ${PROBE_BATCHES:-4}; do
  oom=0
  for w in ${PROBE_WORKERS:-8}; do
    for pf in ${PROBE_PREFETCH:-2}; do
      for pin in ${PROBE_PIN:-0}; do
        echo "[probe] batch=$b workers=$w prefetch=$pf pin=$pin $(date -Is)"
        singularity exec --nv --bind /mnt/storage_5,/mnt/storage_6 "$SIF" python scripts/train_vigor.py \
          ${PROBE_ARGS:-} --batch "$b" --workers "$w" --prefetch "$pf" $([ "$pin" = 1 ] && echo --pin-memory) \
          --profile "${PROBE_STEPS:-20}" --tag probe --out "${PROBE_OUT:-/tmp/probe_vigor}"
        rc=$?
        echo "[probe] exit=$rc"
        [ $rc -eq 3 ] && oom=1
      done
    done
  done
  [ $oom -eq 1 ] && { echo "[probe] OOM at batch $b: stopping the batch loop"; break; }
done
echo "[probe] done date=$(date -Is)"
