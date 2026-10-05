#!/bin/bash
# KITScenes scene -> VIGOR-layout dataset, built ON Eagle from the sources (no copy from a laptop): downloads one
# scene from HuggingFace (gated: needs the HF token of an account that accepted the terms), extracts it, and converts it
# with scripts/kitscenes_to_vigor.py (the free 20 cm DOP20 tiles come from the Hessen / LGL-BW WMS). CPU node with
# internet (proxima-cpu, like slurm/fetch_vigor.sbatch); no GPU.
#
#   make eagle-fetch-kitscenes SCENE=142f1419-b6f2-4215-4055-6eb161f63043 KIT_SPLIT=val NAME=frankfurt_142f1419
#
# One-time, once per account: put the HF token in a private file the job reads (never printed, never on the command line):
#   ssh -t eagle 'umask 077; read -rs -p "HF token: " T; echo "$T" > $HF_TOKEN_FILE'      (default path below)
# Env: SCENE, SPLIT (val|train|test|overlap_train_val), NAME (output folder under kit_vigor/), STRIDE, PRIOR_M, SEED, WMS,
# HF_TOKEN_FILE, CONVERT_ARGS. KITScenes data is CC BY-NC 4.0 (kitscenes.com): it stays under $P, never in git.
#SBATCH --job-name=fetch_kit
#SBATCH --account=pl1269-01
#SBATCH --partition=proxima-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run sbatch from the repo root}"
mkdir -p slurm/logs
P=/mnt/storage_6/project_data/pl1269-01/krupka_maciej
SCENE="${SCENE:?SCENE=<uuid>}"
SPLIT="${SPLIT:-val}"
NAME="${NAME:?NAME=<output folder name>}"
TOKEN_FILE="${HF_TOKEN_FILE:-$P/.hf_token_kit}"
KS="$P/kitscenes/data"
SIF="${SIF:-/mnt/storage_6/project_data/pl0467-01/container_mackop/sat_roma_2026-09-02.sif}"
export PYTHONPATH="src:.pydeps"
export PYTHONUNBUFFERED=1
export KITSCENES_DEVKIT_DIR="$PWD/third_party/kitscenes"      # git submodule update --init third_party/kitscenes (login node)
[ -f "$KITSCENES_DEVKIT_DIR/kitscenes/sensors.py" ] || { echo "devkit missing: git submodule update --init third_party/kitscenes"; exit 2; }
echo "[kit] host=$(hostname) date=$(date -Is) scene=$SCENE split=$SPLIT name=$NAME"

if [ ! -f "$KS/$SPLIT/$SCENE/poses.txt" ]; then
  [ -s "$TOKEN_FILE" ] || { echo "no HF token in $TOKEN_FILE (see the header of this script)"; exit 3; }
  mkdir -p "$KS/$SPLIT"
  CFG="$(mktemp)"; chmod 600 "$CFG"; trap 'rm -f "$CFG"' EXIT
  printf 'header = "Authorization: Bearer %s"\n' "$(tr -d '\n\r ' < "$TOKEN_FILE")" > "$CFG"   # the token never appears in argv / the log
  URL="https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal/resolve/main/data/$SPLIT/$SCENE.tar"
  TAR="$KS/$SPLIT/$SCENE.tar"
  curl -L --fail --retry 5 --retry-delay 10 -C - -K "$CFG" -o "$TAR" "$URL"
  tar -xf "$TAR" -C "$KS/$SPLIT" && rm -f "$TAR"
fi
[ -f "$KS/$SPLIT/$SCENE/poses.txt" ] || { echo "scene not extracted at $KS/$SPLIT/$SCENE"; exit 4; }
du -sh "$KS/$SPLIT/$SCENE"

singularity exec --bind /mnt/storage_5,/mnt/storage_6 "$SIF" python scripts/kitscenes_to_vigor.py \
  --scene-dir "$KS/$SPLIT/$SCENE" --out "$P/kit_vigor/$NAME" \
  ${STRIDE:+--stride $STRIDE} ${PRIOR_M:+--prior-m $PRIOR_M} ${SEED:+--seed $SEED} ${WMS:+--wms $WMS} ${CONVERT_ARGS:-}
echo "[kit] exit=$? date=$(date -Is)"
ls "$P/kit_vigor/$NAME/Chicago/panorama" | wc -l
