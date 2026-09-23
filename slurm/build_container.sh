#!/bin/bash
# Build the .sif from the pushed Docker image. Run INSIDE an interactive job
# (singularity-build only works on a worker node, not the login node) --
# no GPU needed, so proxima-cpu is enough (mirrors sat_roma's build_container.sh):
#
#   srun --account=<grant> -p proxima-cpu --pty bash
#   ./slurm/build_container.sh
#
# Prereq (one-time, before first use -- home has a 1 GB quota):
#   mkdir -p ~/<grant>/project_data/containers_$USER
#   ln -s ~/<grant>/project_data/containers_$USER ~/.singularity
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${1:-$HOME/.singularity/cross_view_sat_roma.sif}"
sudo singularity build "$OUT" slurm/cross_view_sat_roma.def
echo "built $OUT"
