#!/usr/bin/env bash
# Reruns analyze + report for main and all-layers after adding the community × sense NMI to P4.
set -uo pipefail
cd /home/unicamp/200208/gender-representation-networks
run() {
  local name=$1 cfg=$2
  uv run gender-networks analyze --config "$cfg" --force > "outputs/logs/analyze_${name}_nmi_sense.log" 2>&1 \
    && uv run gender-networks report --config "$cfg" --force > "outputs/logs/report_${name}_nmi_sense.log" 2>&1
  echo "$name exit $?"
}
run main configs/experiment.yaml &
run all-layers configs/all-layers.yaml &
wait
