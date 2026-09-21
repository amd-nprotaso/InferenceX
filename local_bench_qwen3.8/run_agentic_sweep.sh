#!/usr/bin/env bash
set -uo pipefail
# Paired A/B: for each concurrency, run baseline then conv_decode back to back so
# the two arms sit as close together in time as possible. Each point is a fresh
# server; the recipe handles its own teardown.
cd "$(dirname "$0")"
export INFMAX_CONTAINER_WORKSPACE=/var/home/my_InferenceX/qwen3.8/InferenceX
export TP=${TP:-4}
export DURATION=${DURATION:-1800}
CONC_LIST=${CONC_LIST:-"1 4 8 12 16"}
ARMS=${ARMS:-"baseline conv_decode"}
mkdir -p agentic_results/logs
for CONC in $CONC_LIST; do
  for ARM in $ARMS; do
    echo "### $(date --iso-8601=seconds)  arm=$ARM conc=$CONC"
    pkill -f "[p]ython3 -m sglang.launch_server" 2>/dev/null && sleep 20
    ARM="$ARM" CONC="$CONC" ./run_agentic_arm.sh \
        > "agentic_results/logs/${ARM}_conc${CONC}.log" 2>&1
    echo "###   exit=$? at $(date --iso-8601=seconds)"
  done
done
echo "SWEEP COMPLETE $(date --iso-8601=seconds)"
