#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/data/hf_cache/
export TP=1 CONC=1 ISL=32678 OSL=32678
export SGLANG_AITER_GDN_DECODE=1 SGLANG_AITER_GDN_CONV=1 SGLANG_AITER_HC_MIX=1 AITER_GDR_FLYDSL=1 SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1
export PYTHONNOUSERSITE=1 SGLANG_USE_AITER=1 SGLANG_USE_AITER_UNIFIED_ATTN=1 AITER_FLYDSL_FORCE=1 SGLANG_MAMBA_SSM_DTYPE=bfloat16 SGLANG_TIMEOUT_KEEP_ALIVE=1800
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
MODEL=amd/Qwen3.8-Flash-Next-Quark-MXFP4

exec python3 -m sglang.launch_server \
 --model-path "$MODEL" --served-model-name "$MODEL" \
 --host 0.0.0.0 --port 30000 --trust-remote-code --tp-size 1 --ep-size 1 \
 --context-length 65540 --attention-backend aiter --page-size 32 --kv-cache-dtype auto \
 --chunked-prefill-size 16384 --watchdog-timeout 1200 --mem-fraction-static 0.85 \
 --model-loader-extra-config '{"enable_multithread_load": true}' \
 --max-running-requests 2 --cuda-graph-max-bs-decode 1 \
 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
 --stream-interval 50 --scheduler-recv-interval 30 --tokenizer-path "$MODEL" --enable-metrics --enable-cache-report
