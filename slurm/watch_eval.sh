#!/bin/bash
# Follow one Eagle job from the laptop: print new progress / result lines from its log until it leaves the queue.
#   bash slurm/watch_eval.sh <jobid> <jobname> [grep-pattern]
set -u
JOB="$1"
NAME="$2"
PAT="${3:-^(  [0-9]+ done|[A-Za-z]+ +n +[0-9]+|wrote |[0-9]+ samples|Traceback|.*Error|\[run\] exit)}"
LOG="/mnt/storage_6/project_data/pl1269-01/krupka_maciej/cross_view_sat_roma/slurm/logs/${NAME}_${JOB}.log"
seen=""
while true; do
  state=$(ssh -o BatchMode=yes eagle "squeue -h -j $JOB -o %t 2>/dev/null" 2>/dev/null) || { sleep 60; continue; }
  lines=$(ssh -o BatchMode=yes eagle "tr '\r' '\n' < $LOG 2>/dev/null | grep -E '$PAT'" 2>/dev/null)
  while IFS= read -r l; do
    [ -z "$l" ] && continue
    case "$seen" in *"|$l|"*) ;; *) echo "$l"; seen="$seen|$l|";; esac
  done <<< "$lines"
  [ -z "$state" ] && { echo "JOB_FINISHED $NAME"; exit 0; }
  sleep 120
done
