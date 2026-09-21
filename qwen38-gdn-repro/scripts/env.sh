#!/usr/bin/env bash
# Shared configuration. Sourced by every other script; not executable on its own.
#
# Override anything from the caller's environment, e.g.
#   SGLANG_DIR=/my/sglang AITER_DIR=/my/aiter ./run_all.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Checkouts the patches are applied to.
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang}
AITER_DIR=${AITER_DIR:-/sgl-workspace/aiter}

MODEL=${MODEL:-amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8888}

# Benchmark shape. ISL is deliberately 131072: the GDN decode recurrence is a
# larger share of the step at long context, which is where these kernels pay.
ISL=${ISL:-131072}
OSL=${OSL:-1024}
CONC_LIST=${CONC_LIST:-"1 4 16"}
NUM_PROMPTS_MULT=${NUM_PROMPTS_MULT:-4}
WARMUP_MULT=${WARMUP_MULT:-1}

# Server sizing. The sweep runs every concurrency point against ONE server, so
# the server is sized for the largest point and shared across the whole sweep.
TP=${TP:-1}
SERVER_CONC=${SERVER_CONC:-16}

RESULTS_DIR=${RESULTS_DIR:-$REPO_ROOT/results}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}

# Seconds to wait for /health before giving up. Cold start loads the checkpoint,
# JITs AITER/FlyDSL kernels and captures CUDA graphs; 4 min is typical on MI355X.
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}

# The three arms. Everything the experiment varies lives in this one mapping:
# an arm is nothing but a set of kernel gates on an otherwise identical server.
#   baseline    : stock Triton GDN decode + Triton causal-conv1d prefill
#   conv_only   : + AITER FlyDSL prefill conv        (patches 0002/0004)
#   conv_decode : + AITER FlyDSL GDN decode as well  (patches 0001/0003/0005)
arm_env() {
    case "$1" in
        baseline)    echo "" ;;
        conv_only)   echo "SGLANG_AITER_GDN_CONV=1" ;;
        conv_decode) echo "SGLANG_AITER_GDN_CONV=1 SGLANG_AITER_GDN_DECODE=1" ;;
        *) echo "unknown arm '$1' (expected baseline|conv_only|conv_decode)" >&2; return 2 ;;
    esac
}

ARMS=${ARMS:-"baseline conv_only conv_decode"}

wait_for_server() {
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
    while (( SECONDS < deadline )); do
        if curl -sf -m 5 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        # A server that died during load will never answer; fail fast instead of
        # burning the full timeout.
        if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server process $SERVER_PID exited before becoming ready" >&2
            return 1
        fi
        sleep 5
    done
    echo "ERROR: server not ready after ${SERVER_READY_TIMEOUT}s" >&2
    return 1
}

stop_server() {
    # Match on the module path, and bracket the first character so the pattern
    # cannot match the calling shell's own command line.
    pkill -f "[p]ython3 -m sglang.launch_server" 2>/dev/null || true
    for _ in $(seq 1 30); do
        pgrep -f "[p]ython3 -m sglang.launch_server" >/dev/null 2>&1 || return 0
        sleep 2
    done
    pkill -9 -f "[p]ython3 -m sglang.launch_server" 2>/dev/null || true
    sleep 3
}
