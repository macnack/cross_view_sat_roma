#!/bin/bash
# Follow every job listed in slurm/watch_list.txt from the laptop (one watch_eval.sh per job, in parallel,
# lines prefixed with the job name). Prints the queue first, then new log lines as they appear, and exits
# when every listed job has left the queue. Re-run any time: finished jobs print their final lines once.
#   make eagle-watch            (or: bash slurm/watch_all.sh [list-file])
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LIST="${1:-$HERE/watch_list.txt}"
echo "== queue $(date +%H:%M) =="
ssh -o BatchMode=yes eagle 'squeue -u $USER -o "%i %j %t %M %R" | grep -v -E "cnxL|dinov3L"' 2>/dev/null || echo "(ssh eagle failed)"
echo "== following $(grep -c -v -E '^\s*(#|$)' "$LIST") job(s) from $LIST =="
pids=()
while read -r job name pat; do
  case "$job" in ''|'#'*) continue ;; esac
  ( bash "$HERE/watch_eval.sh" "$job" "$name" "$pat" | sed -u "s/^/[$name] /" ) &
  pids+=($!)
done < <(grep -v -E '^\s*(#|$)' "$LIST")
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
wait
echo "== all listed jobs finished $(date +%H:%M) =="
