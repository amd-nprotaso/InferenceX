#!/usr/bin/env bash
# A/B for aiter topk_gating router (PR #40708 + Flash-Next hidden size):
# baseline = SGLANG_ROCM_USE_AITER_TOPK_GATING=0, patch = 1. Interleaved reps.
set -uo pipefail

BENCH_DIR=/var/home/my_InferenceX/InferenceX/local_bench_qwen3.8
OUT=$(cd "$(dirname "$0")" && pwd)
PORT=8888
REPS=${REPS:-2}
ARMS=${ARMS:-"baseline patch"}

cd "$BENCH_DIR"

run_arm() {
    local arm=$1 rep=$2 flag
    [[ "$arm" == "patch" ]] && flag=1 || flag=0
    local tag="${arm}_r${rep}"
    echo "=== [$(date +%T)] $tag (SGLANG_ROCM_USE_AITER_TOPK_GATING=$flag)"

    SGLANG_ROCM_USE_AITER_TOPK_GATING=$flag TP=1 EP_SIZE=1 CONC=1 PORT=$PORT \
        RESULT_DIR="$OUT/$tag" setsid ./run_server.sh > "$OUT/server_$tag.log" 2>&1 &
    local spid=$!

    local ok=0
    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
        if ! kill -0 $spid 2>/dev/null; then break; fi
        sleep 10
    done
    if [[ $ok != 1 ]]; then
        echo "!!! server for $tag failed to become healthy; see server_$tag.log"
        kill -- -$spid 2>/dev/null; wait $spid 2>/dev/null
        return 1
    fi
    echo "=== [$(date +%T)] server healthy"

    ARM=$tag ISL=1024 OSL=1024 CONC_LIST=1 NUM_PROMPTS_MULT=8 WARMUP_MULT=2 \
        RESULT_DIR="$OUT/$tag" ./run_client.sh > "$OUT/client_$tag.log" 2>&1
    echo "=== [$(date +%T)] client exit=$?"

    kill -- -$spid 2>/dev/null
    for _ in $(seq 1 30); do kill -0 $spid 2>/dev/null || break; sleep 2; done
    kill -9 -- -$spid 2>/dev/null
    wait $spid 2>/dev/null
    sleep 10
}

for rep in $(seq 1 "$REPS"); do
    for arm in $ARMS; do
        run_arm "$arm" "$rep"
    done
done
echo "=== [$(date +%T)] all done"
