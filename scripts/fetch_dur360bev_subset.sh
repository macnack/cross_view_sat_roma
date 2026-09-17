#!/usr/bin/env bash
# Stream the Dur360BEV tarball from Hugging Face and keep only a subset.
#
# The dataset is one gzip'd tar split into 2 GB parts, ordered by modality
# (all images, then labels, LiDAR, OxTS ...), so no part is usable alone and
# gzip cannot be seeked. We therefore stream every part once, in order, and
# let tar keep only matching members. At most one part sits on disk at a time.
#
# Kept:
#   - every 10th frame (name ends in 0) of every modality, whole drive
#   - dense clip: frames 0000000000-0000000299, every modality
#   - all oxts/, all timestamps.txt, dataformat.txt, metadata/, md5sums.txt
#
# Usage: scripts/fetch_dur360bev_subset.sh [OUT_DIR]   (default data/dur360bev)
set -euo pipefail

OUT_DIR="${1:-data/dur360bev}"
REPO="https://huggingface.co/datasets/TomEeee/Dur360BEV/resolve/main"
PREFIX="Dur360BEV_dataset.tar"
TMP_DIR="${OUT_DIR}/.parts"
mkdir -p "$OUT_DIR" "$TMP_DIR"

# part suffixes aa..db (80 parts), as listed on the hub on 2026-09-17
parts=()
for a in a b c d; do
  for b in {a..z}; do
    parts+=("${a}${b}")
    [[ "${a}${b}" == "db" ]] && break 2
  done
done

DECOMP="gzip -dc"
command -v pigz >/dev/null && DECOMP="pigz -dc"

stream_parts() {
  for s in "${parts[@]}"; do
    f="${TMP_DIR}/${PREFIX}${s}"
    echo "[fetch] part ${s}" >&2
    # resume-safe download to disk, then emit; never retry into the pipe
    until curl -sSL --fail -C - -o "$f" "${REPO}/${PREFIX}${s}"; do
      echo "[fetch] retry ${s}" >&2; sleep 10
    done
    cat "$f"
    rm -f "$f"
  done
}

stream_parts | $DECOMP | tar -xf - -C "$OUT_DIR" --strip-components=1 \
  --wildcards \
  '*/data/*0.*' \
  '*/data/00000000??.*' '*/data/00000001??.*' '*/data/00000002??.*' \
  '*/oxts/data/*' \
  '*/timestamps.txt' '*/dataformat.txt' '*/metadata/*' '*/md5sums.txt'

rmdir "$TMP_DIR" 2>/dev/null || true
echo "[fetch] done -> $OUT_DIR" >&2
