#!/bin/bash
# CPU stages of the main run, detached from the session; no GPU visible.
cd /home/unicamp/200208/gender-representation-networks
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
for s in metrics analyze report; do
  echo "$(date '+%F %T') start $s" >> outputs/logs/cpu_stages.status
  uv run gender-networks $s --config configs/experiment.yaml > outputs/logs/${s}_main.log 2>&1
  rc=$?
  echo "$(date '+%F %T') end $s rc=$rc" >> outputs/logs/cpu_stages.status
  [ $rc -ne 0 ] && exit $rc
done
