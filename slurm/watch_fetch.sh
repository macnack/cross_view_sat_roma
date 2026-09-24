#!/bin/bash
# Poll a fetch job from the laptop and print new progress lines and the final state.
#   bash slurm/watch_fetch.sh <jobid>    (uses ssh eagle; exits when the job has left the queue)
set -u
JOB="$1"
LOG="/mnt/storage_6/project_data/pl1269-01/krupka_maciej/cross_view_sat_roma/slurm/logs/fetch_vigor_${JOB}.log"
seen=""
while true; do
  state=$(ssh -o BatchMode=yes eagle "squeue -h -j $JOB -o %t 2>/dev/null" 2>/dev/null) || { sleep 60; continue; }
  lines=$(ssh -o BatchMode=yes eagle "tr '\r' '\n' < $LOG | grep -E '^(fetch|skip|  ok|  error|extract|done|FAILED|\[fetch\])'" 2>/dev/null)
  while IFS= read -r l; do
    case "$seen" in *"|$l|"*) ;; *) echo "$l"; seen="$seen|$l|";; esac
  done <<< "$lines"
  [ -z "$state" ] && { echo "JOB_FINISHED"; ssh -o BatchMode=yes eagle "du -sh /mnt/storage_6/project_data/pl1269-01/krupka_maciej/vigor/* 2>/dev/null"; exit 0; }
  sleep 240
done
