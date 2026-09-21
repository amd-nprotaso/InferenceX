#!/bin/bash

set -euo pipefail
set -x

MODEL=${MODEL:-amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8}
export MODEL_PATH="${MODEL_PATH:-$MODEL}"

PORT=${PORT:-8888}

TP=${TP:-1}
EP_SIZE=${EP_SIZE:-1}
CONC=${CONC:-64}

DRY_RUN=${DRY_RUN:-0}

RESULT_DIR=${RESULT_DIR:-/tmp/results}

export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"

export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export AITER_FLYDSL_FORCE=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=2.32
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp-size "$TP"
    --ep-size "$EP_SIZE"
    --attention-backend aiter
    --moe-runner-backend aiter
    --mamba-ssm-dtype bfloat16
    --page-size 32
    --kv-cache-dtype auto
    --chunked-prefill-size 16384
    --watchdog-timeout 1200
    # MTP: leave non-static headroom for the NEXTN draft head's verification
    # batch and AITER spec-decode workspaces. The cookbook STP cell runs 0.9;
    # the B300 NVFP4 MTP sibling runs 0.80 and the H200 FP8 one 0.85. The
    # ~126 GiB checkpoint is ~16 GB/GPU across TP8 on 288 GB parts, so 0.85
    # still leaves a ~229 GB/GPU static share for weights plus KV.
    --mem-fraction-static 0.85
    --model-loader-extra-config '{"enable_multithread_load": true}'
    # NEXTN silently resets --max-running-requests to 48 when it is unset, so
    # this must stay explicit and sized to the AgentX concurrency.
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    # Decode-specific spelling as the MI355X Qwen3.5/DeepSeek-V4/GLM-5.2 SGLang
    # MTP arms use; recent SGLang splits --cuda-graph-max-bs into
    # decode/prefill variants and rejects the old prefix as ambiguous.
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --stream-interval 50
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL_PATH"
    --enable-metrics
    --enable-cache-report
)

if [[ -n "${RESULT_DIR:-}" ]]; then
    mkdir -p "$RESULT_DIR"
    printf '%q ' "${SGLANG_CMD[@]}" > "$RESULT_DIR/sglang_command.txt"
    printf '\n' >> "$RESULT_DIR/sglang_command.txt"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    set +x
    printf '%q ' "${SGLANG_CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${SGLANG_CMD[@]}"