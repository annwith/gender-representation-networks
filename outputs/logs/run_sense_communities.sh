#!/usr/bin/env bash
# Reruns analyze + report for main and all-layers after adding p4_sense_communities and the
# figure p4_comunidades_sentidos.
set -uo pipefail
cd /home/unicamp/200208/gender-representation-networks
run() {
  local name=$1 cfg=$2
  uv run gender-networks analyze --config "$cfg" --force > "outputs/logs/analyze_${name}_sense_communities.log" 2>&1 \
    && uv run gender-networks report --config "$cfg" --force > "outputs/logs/report_${name}_sense_communities.log" 2>&1
  echo "$name exit $?"
}
run main configs/experiment.yaml &
run all-layers configs/all-layers.yaml &
wait
