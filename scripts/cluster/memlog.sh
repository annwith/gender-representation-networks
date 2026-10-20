#!/bin/bash
# Every 30 s, while run_cpu_stages.sh runs: total RSS of the user's gender-networks processes and
# the system's available memory, appended to outputs/logs/memlog.txt.
cd "${REPO_DIR:-$(dirname "$0")/../..}" || exit 1
user="${RUN_USER:-$USER}"
while pgrep -u "$user" -f run_cpu_stages.sh >/dev/null; do
  rss=$(ps -u "$user" -o rss=,cmd= | grep -E "gender-networks|multiprocessing" | grep -v grep | awk '{s+=$1} END {printf "%.1f", s/1048576}')
  avail=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
  echo "$(date '+%F %T') rss_gib=$rss avail_gib=$avail" >> outputs/logs/memlog.txt
  sleep 30
done
