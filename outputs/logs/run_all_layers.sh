#!/usr/bin/env bash
# Runs sample → extract (GPU 3 only if idle) → knn → metrics → analyze → lens for configs/all-layers.yaml.
set -euo pipefail
cd /home/unicamp/200208/gender-representation-networks
CFG=configs/all-layers.yaml
LOG=outputs/logs
GPU=3
status() { echo "$(date '+%F %T') $*" >> $LOG/pipeline_all-layers.log; }
status "início"
for stage in sample extract knn metrics analyze lens; do
  if [ "$stage" = extract ]; then
    busy=$(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "$busy" -ne 0 ]; then status "GPU $GPU ocupada ($busy processos); parei antes do extract"; exit 2; fi
    status "GPU $GPU livre; extract"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU uv run gender-networks extract --config $CFG > $LOG/extract_all-layers.log 2>&1 \
      || { status "extract falhou"; exit 1; }
  else
    status "$stage"
    CUDA_VISIBLE_DEVICES="" uv run gender-networks $stage --config $CFG > $LOG/${stage}_all-layers.log 2>&1 \
      || { status "$stage falhou"; exit 1; }
  fi
done
status "fim"
