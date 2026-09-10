#!/usr/bin/env bash
set -euo pipefail

# One-shot: bring the server up, wait for readiness, run the concurrency sweep,
# tear everything down. This is the single-script shape of
# benchmarks/single_node/agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh, but for
# fixed-seq-len random-dataset benchmarking instead of AgentX trace replay.
#
# Use qwen3.5_fp4_sglang_server.sh + qwen3.5_fp4_sglang_bench.sh separately when
# you want to keep one server warm across many sweeps -- model load is minutes.
#
# Usage:
#   ./qwen3.5_fp4_sglang_run.sh
#   MODE=quick ./qwen3.5_fp4_sglang_run.sh
#   TP=2 CONC_LIST="4 8 16" ./qwen3.5_fp4_sglang_run.sh
#
# Env: everything understood by the server and bench scripts, plus
#      RESULT_DIR GPU_MONITOR SERVER_READY_TIMEOUT

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

# Reused for wait_for_server_ready / stop_background_process_tree / GPU monitoring.
source "$REPO_ROOT/benchmarks/benchmark_lib.sh"

export PORT=${PORT:-8888}
export HOST=${HOST:-127.0.0.1}
export MODE=${MODE:-full}
export TP=${TP:-4}
export CONC_LIST=${CONC_LIST:-"4 8 16"}
GPU_MONITOR=${GPU_MONITOR:-1}

RESULT_DIR=${RESULT_DIR:-"$HERE/bench_results/$(date +%Y%m%d_%H%M%S)"}
export RESULT_DIR
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

# Size the engine for the largest sweep point, not the first one.
PEAK_CONC=0
for c in $CONC_LIST; do
    [ "$c" -gt "$PEAK_CONC" ] && PEAK_CONC=$c
done
export CONC=${CONC:-$PEAK_CONC}

SERVER_PID=""
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    [[ "$GPU_MONITOR" == "1" ]] && stop_gpu_monitor
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    echo "=== results in $RESULT_DIR (exit $exit_code)"
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "=== launching server (tp=$TP, peak conc=$CONC) -> $SERVER_LOG"
"$HERE/qwen3.5_fp4_sglang_server.sh" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Model load is minutes on a 397B checkpoint; poll slowly and let the helper
# fail fast if the server process dies during load.
wait_for_server_ready \
    --port "$PORT" \
    --server-log "$SERVER_LOG" \
    --server-pid "$SERVER_PID" \
    --sleep-interval "${SERVER_READY_TIMEOUT:-60}"

if [[ "$GPU_MONITOR" == "1" ]]; then
    start_gpu_monitor --output "$RESULT_DIR/gpu_metrics.csv"
fi

"$HERE/qwen3.5_fp4_sglang_bench.sh"
