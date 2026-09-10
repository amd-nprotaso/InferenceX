#!/usr/bin/env bash
set -euo pipefail
set -x

# SGLang server for Qwen3.5-397B-A17B MXFP4 with native EAGLE MTP.
#
# Standalone twin of benchmarks/single_node/agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh:
# same engine flags and env, but no CI harness, no trace replay, no eval --
# it only brings the server up and holds it in the FOREGROUND so you can bench
# against it repeatedly. Ctrl-C stops it.
#
# Usage:
#   ./qwen3.5_fp4_sglang_server.sh                    # tp=4, peak conc 32
#   TP=2 CONC=16 PORT=8888 ./qwen3.5_fp4_sglang_server.sh
#   SIMULATE_ACC=0 ./qwen3.5_fp4_sglang_server.sh     # real MTP acceptance
#
# Env: MODEL MODEL_PATH PORT TP EP_SIZE CONC MEM_FRAC_STATIC PREFILL_TOKENS
#      MAX_RUNNING_REQUESTS CUDA_GRAPH_MAX_BS SIMULATE_ACC DISABLE_RADIX_CACHE
#      KV_OFFLOADING(none|dram) SCHEDULER_RECV_INTERVAL SKIP_DOWNLOAD
#      SGLANG_GDN_CHUNK_H_{BV,NUM_WARPS,NUM_STAGES} QR_QUANT QR_MAX_MB
#
# HF_TOKEN / HF_HUB_CACHE come from the environment -- export them, do not
# hardcode a token in this file.

MODEL=${MODEL:-amd/Qwen3.5-397B-A17B-MXFP4}
PORT=${PORT:-8888}
TP=${TP:-4}
EP_SIZE=${EP_SIZE:-1}
CONC=${CONC:-32}                     # peak concurrency you intend to benchmark
MEM_FRAC_STATIC=${MEM_FRAC_STATIC:-0.80}
SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}
SIMULATE_ACC=${SIMULATE_ACC:-1}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-0}
KV_OFFLOADING=${KV_OFFLOADING:-none}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
DRY_RUN=${DRY_RUN:-0}                # print the resolved launch command, exit
# QuickReduce all-reduce path -- the #1 prefill target in
# optimization_suggestion.txt (+114.2 ms, 2.28x vs NV).
#
# /etc/environment:34 sets ROCM_QUICK_REDUCE_QUANTIZATION=INT8 CONTAINER-WIDE,
# so the ambient value is INT8 whether you meant it or not. These runs are
# specified against NONE, so we drive the real vars from dedicated knobs and
# ignore the ambient value entirely -- otherwise a ${VAR:-NONE} default would
# silently resolve to INT8 and the "off" arm would never actually run.
#
#   QR_QUANT=NONE|INT8|FP8   quantization (default NONE = force RCCL ring)
#   QR_MAX_MB=<n>            QuickReduce size cutoff; 8 keeps QR for decode-size
#                            messages and routes the 268 MB prefill all-reduce
#                            to RCCL. Set high (e.g. 2048) to keep QR for both.
QR_QUANT=${QR_QUANT:-NONE}
QR_MAX_MB=${QR_MAX_MB:-8}
export ROCM_QUICK_REDUCE_QUANTIZATION="$QR_QUANT"
export ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB="$QR_MAX_MB"

# GDN chunk-state Triton kernel geometry. Defaults here are the tuned winner
# from kernel_tuning/chunk_gated_delta_rule (1.42x on the kernel vs sglang's
# BV=32/w=4/s=2). Read by chunk_delta_h.py:24-26 at IMPORT time, so these must
# be exported before launch_server starts -- a plain assignment does not reach
# the TP workers.
export SGLANG_GDN_CHUNK_H_BV=${SGLANG_GDN_CHUNK_H_BV:-16}
export SGLANG_GDN_CHUNK_H_NUM_WARPS=${SGLANG_GDN_CHUNK_H_NUM_WARPS:-4}
export SGLANG_GDN_CHUNK_H_NUM_STAGES=${SGLANG_GDN_CHUNK_H_NUM_STAGES:-3}

# Must be >= the longest ISL you plan to benchmark. 32768 covers the
# long-prefill sweeps; the agentic recipe uses 16384 for its trace shape.
PREFILL_TOKENS=${PREFILL_TOKENS:-32768}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$SKIP_DOWNLOAD" != "1" && "$DRY_RUN" != "1" ]]; then
    if [[ -n "${MODEL_PATH:-}" ]]; then
        if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
            hf download "$MODEL" --local-dir "$MODEL_PATH"
        fi
    else
        hf download "$MODEL"
    fi
fi
export MODEL_PATH="${MODEL_PATH:-$MODEL}"

if [[ "$DRY_RUN" != "1" ]]; then
    rocm-smi || true
    amd-smi || true
fi

# Batch sizing follows the agentic recipe: headroom of 2x the benchmark
# concurrency, with CUDA graphs covering the whole running set (capped at 128
# so graph capture stays bounded). Under EAGLE the effective decode batch is
# ~num-draft-tokens x running requests, so undersizing this silently drops
# decode off graphs and inflates ITL.
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-$((2 * CONC))}
if [[ -z "${CUDA_GRAPH_MAX_BS:-}" ]]; then
    CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
    [ "$CUDA_GRAPH_MAX_BS" -gt 128 ] && CUDA_GRAPH_MAX_BS=128
fi

PARALLEL_ARGS=(
    --tp "$TP"
    --dp 1
    --ep-size "$EP_SIZE"
)

TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

CACHE_ARGS=()
if [[ "$DISABLE_RADIX_CACHE" == "1" ]]; then
    # Cleanest fixed-seq-len numbers: no prefix reuse across requests at all.
    # Leave it on (default) if you want --flush-cache between sweep points instead.
    CACHE_ARGS+=(--disable-radix-cache)
elif [[ "$KV_OFFLOADING" == "dram" ]]; then
    CACHE_ARGS+=(
        --enable-hierarchical-cache
        --hicache-ratio "${HICACHE_RATIO:-1.5}"
        --hicache-write-policy "${HICACHE_WRITE_POLICY:-write_through}"
        --hicache-io-backend "${HICACHE_IO_BACKEND:-direct}"
        --hicache-mem-layout "${HICACHE_MEM_LAYOUT:-page_first_direct}"
    )
fi

export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export AITER_FLYDSL_FORCE=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
#export ROCM_QUICK_REDUCE_QUANTIZATION=INT8
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

# Synthetic acceptance length from the committed golden distribution. Throughput
# numbers produced under this are a PROJECTION -- the draft model is not really
# being verified at AL 3.39. Set SIMULATE_ACC=0 to measure real acceptance.
if [[ "$SIMULATE_ACC" == "1" ]]; then
    export SGLANG_SIMULATE_ACC_LEN=3.39
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
    "${PARALLEL_ARGS[@]}"
    --attention-backend aiter
    --mem-fraction-static "$MEM_FRAC_STATIC"
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --watchdog-timeout 1200
    --page-size 16
    --kv-cache-dtype fp8_e4m3
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens "$PREFILL_TOKENS"
    --chunked-prefill-size "$PREFILL_TOKENS"
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --stream-interval 50
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --enable-metrics
    --enable-cache-report
    "${CACHE_ARGS[@]}"
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
