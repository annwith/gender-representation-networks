#!/bin/bash
# CPU stages of the main run, detached from the session; no GPU visible.
# Usage (on the cluster): nohup scripts/cluster/run_cpu_stages.sh &
# Logs go to outputs/logs/<stage>_main.log; start and end times to outputs/logs/cpu_stages.status.
cd "${REPO_DIR:-$(dirname "$0")/../..}" || exit 1
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
mkdir -p outputs/logs
for s in metrics analyze report; do
  echo "$(date '+%F %T') start $s" >> outputs/logs/cpu_stages.status
  uv run gender-networks $s --config configs/experiment.yaml > outputs/logs/${s}_main.log 2>&1
  rc=$?
  echo "$(date '+%F %T') end $s rc=$rc" >> outputs/logs/cpu_stages.status
  [ $rc -ne 0 ] && exit $rc
done
