#!/usr/bin/env bash
set -euo pipefail

# Launch the Qwen3.8-Flash-Next SGLang server for one arm.
#
#   ARM=baseline ./run_server.sh          # foreground
#   ARM=conv_decode ./run_server.sh
#
# The ONLY difference between arms is the kernel gate env vars (see arm_env in
# env.sh). Every server flag below is identical across arms, so a delta between
# two arms is attributable to the kernels and nothing else.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ARM=${ARM:-baseline}
GATES="$(arm_env "$ARM")"

export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export AITER_FLYDSL_FORCE=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

# Per-arm kernel gates. Unset first so a stale value in the caller's environment
# cannot silently turn a kernel on in the baseline arm -- exactly the failure
# this whole package is built to make impossible.
unset SGLANG_AITER_GDN_CONV SGLANG_AITER_GDN_DECODE
for kv in $GATES; do export "${kv?}"; done

echo "=== arm    : $ARM"
echo "=== gates  : ${GATES:-<none, stock Triton kernels>}"
echo "=== model  : $MODEL"

MAX_RUNNING_REQUESTS=$((2 * SERVER_CONC))
CUDA_GRAPH_MAX_BS="$SERVER_CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

# Fixed MTP acceptance so the speculative decode path is comparable run to run
# rather than drifting with sampled content.
export SGLANG_SIMULATE_ACC_LEN=2.32
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp-size "$TP"
    --ep-size 1
    --attention-backend aiter
    --moe-runner-backend aiter
    --mamba-ssm-dtype bfloat16
    --page-size 32
    --kv-cache-dtype auto
    --chunked-prefill-size 16384
    --watchdog-timeout 1200
    --mem-fraction-static 0.85
    --model-loader-extra-config '{"enable_multithread_load": true}'
    # NEXTN silently resets --max-running-requests to 48 when unset, so this
    # must stay explicit and sized to the sweep's top concurrency.
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    # topk=1 means a linear draft chain. The AITER decode kernel only handles
    # that shape (a tree verify needs a parent-state reload it does not
    # implement), so raising topk silently drops it back to Triton.
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --stream-interval 50
    --scheduler-recv-interval 30
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL"
    --enable-metrics
    --enable-cache-report
)

# Escape hatch for diagnostics only; empty by default, so a measured arm is
# byte-identical to what it was before this existed. Used to profile decode:
# the decode forward is a CUDA-graph replay, and ROCm does not attribute
# graph-internal kernels, so a torch-profiler decode trace shows only the
# NEXTN draft passes. EXTRA_SERVER_ARGS="--disable-decode-cuda-graph" makes
# the 48-layer target forward eager and therefore visible -- at the cost of
# decode timings that no longer match production.
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    SGLANG_CMD+=($EXTRA_SERVER_ARGS)
    echo "=== extra  : $EXTRA_SERVER_ARGS"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${SGLANG_CMD[@]}"; printf '\n'; exit 0
fi

exec "${SGLANG_CMD[@]}"
