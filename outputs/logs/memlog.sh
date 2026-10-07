#!/bin/bash
# Every 30 s: total RSS of this user's gender-networks processes and system available memory.
while pgrep -u 200208 -f run_cpu_stages.sh >/dev/null; do
  rss=$(ps -u 200208 -o rss=,cmd= | grep -E "gender-networks|multiprocessing" | grep -v grep | awk '{s+=$1} END {printf "%.1f", s/1048576}')
  avail=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
  echo "$(date '+%F %T') rss_gib=$rss avail_gib=$avail" >> outputs/logs/memlog.txt
  sleep 30
done
