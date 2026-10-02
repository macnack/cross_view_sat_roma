#!/bin/bash
# Print one line per FINISHED job among the given "<name>_<jobid>" log stems, then the number still queued.
# Runs on the Eagle login node from the repo root:  bash slurm/watch.sh lift_A_8738202 ev_ipm_8738471 ...
# A job is finished when its id is absent from squeue; its line carries the exit status, any Traceback,
# and the evaluation summary lines. Meant to be polled by a monitor loop; prints nothing else.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
queued=$(squeue -u "$USER" -h -o "%i")
left=0
for stem in "$@"; do
  id=${stem##*_}
  if echo "$queued" | grep -qx "$id"; then left=$((left + 1)); continue; fi
  f="slurm/logs/${stem}.log"
  if [ -f "$f" ]; then
    echo "DONE ${stem%_*}: $(grep -E 'exit=|Traceback|Error|^(2025|2024):' "$f" | tail -3 | tr '\n' '|')"
  else
    echo "DONE ${stem%_*}: (no log)"
  fi
done
echo "QUEUED $left"
