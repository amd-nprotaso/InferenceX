#!/usr/bin/env bash
set -euo pipefail

# Torch-profiler trace collection against a running Qwen3.8-Flash-Next server
# (qwen38-gdn-repro/scripts/run_server.sh, ARM=conv_decode).
#
# This is NOT a benchmark: the printed numbers are throwaway. It is
# local_bench/qwen3.5_fp4_sglang_profile.sh retargeted at Qwen3.8, keeping the
# two client settings run_client.sh adds for this model:
#   --tokenize-prompt    random prompts land exactly at ISL rather than a few
#                        tokens short after detokenize/re-tokenize
#   --random-range-ratio 1   exact ISL, no jitter (FlyDSL kernels are
#                        shape-specialised; chunk sizes want T % 64 == 0)
#
# Usage:
#   ./run_profile.sh
#   PROFILE_NUM_STEPS=50 ./run_profile.sh
#   BY_STAGE=0 PROFILE_NUM_STEPS=8 ./run_profile.sh   # one merged trace
#
# Env: HOST PORT MODEL ISL OSL CONC_LIST PROFILE_DIR ACTIVITIES BY_STAGE
#      PROFILE_NUM_STEPS NUM_PROMPTS_MULT WARMUP_MULT RANDOM_RANGE_RATIO DRY_RUN

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8888}
MODEL=${MODEL:-amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8}

ISL=${ISL:-32768}
OSL=${OSL:-32768}
CONC_LIST=${CONC_LIST:-"1"}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1}
TOKENIZE_PROMPT=${TOKENIZE_PROMPT:-1}

# One wave: num-prompts == concurrency, so every request is in flight together
# and the profiled batch really is size CONC.
NUM_PROMPTS_MULT=${NUM_PROMPTS_MULT:-1}
# Warmup is capped at 32 output tokens by bench_serving regardless of OSL, so
# it costs one prefill. It must happen before the profiler starts, or the trace
# is dominated by AITER/FlyDSL JIT and CUDA-graph capture. bench_serving orders
# warmup -> flush_cache -> start profiler -> requests.
WARMUP_MULT=${WARMUP_MULT:-1}

PROFILE_DIR=${PROFILE_DIR:-/var/home/sglang_profiling}
LOG_DIR=${LOG_DIR:-"$PROFILE_DIR/logs"}
ACTIVITIES=${ACTIVITIES:-"CPU GPU"}

# MANDATORY at this shape. OSL=32768 under MTP is ~32768/2.32 ~= 14k verify
# forward passes plus their draft passes; an unbounded capture over 48 layers
# would be hundreds of GB and would not serialise. BY_STAGE splits the capture
# into one prefill trace and one decode trace, each capped at
# PROFILE_NUM_STEPS forward passes, and the run continues untraced after that.
BY_STAGE=${BY_STAGE:-1}
PROFILE_NUM_STEPS=${PROFILE_NUM_STEPS:-20}
PROFILE_STAGES=${PROFILE_STAGES:-}
DRY_RUN=${DRY_RUN:-0}

if python3 -c "import sglang.benchmark.serving" >/dev/null 2>&1; then
    BENCH_MODULE=sglang.benchmark.serving
else
    BENCH_MODULE=sglang.bench_serving
fi

mkdir -p "$PROFILE_DIR" "$LOG_DIR"

echo "=== profile target : http://$HOST:$PORT"
echo "=== model          : $MODEL"
echo "=== module         : $BENCH_MODULE"
echo "=== shape          : ISL=$ISL OSL=$OSL range-ratio=$RANDOM_RANGE_RATIO"
echo "=== concurrency    : $CONC_LIST"
echo "=== activities     : $ACTIVITIES  by-stage=$BY_STAGE num-steps=${PROFILE_NUM_STEPS:-<unbounded>}"
echo "=== traces         : $PROFILE_DIR"

if [[ "$DRY_RUN" != "1" ]] && ! curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: no healthy SGLang server at http://$HOST:$PORT" >&2
    exit 1
fi

# A server whose prefill budget is below ISL chunks the prompt, so the prefill
# trace shows N chunks rather than one prefill. The GDN repro server runs
# --chunked-prefill-size 16384 deliberately, so at ISL=32768 expect 2 chunks.
SERVER_PREFILL_BUDGET=$(curl -sf "http://$HOST:$PORT/server_info" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("max_prefill_tokens") or d.get("server_args",{}).get("max_prefill_tokens",""))' 2>/dev/null || true)
if [[ "$SERVER_PREFILL_BUDGET" =~ ^[0-9]+$ && "$SERVER_PREFILL_BUDGET" -lt "$ISL" ]]; then
    echo "NOTE: server max-prefill-tokens=$SERVER_PREFILL_BUDGET < ISL=$ISL;" >&2
    echo "      prefill is chunked into $(( (ISL + SERVER_PREFILL_BUDGET - 1) / SERVER_PREFILL_BUDGET )) forward passes." >&2
fi

for CONC in $CONC_LIST; do
    NUM_PROMPTS=$((CONC * NUM_PROMPTS_MULT))
    WARMUPS=$((CONC * WARMUP_MULT))
    [ "$WARMUPS" -lt 1 ] && WARMUPS=1
    TAG="isl${ISL}_osl${OSL}_c${CONC}"
    OUT_DIR="$PROFILE_DIR/$TAG"
    mkdir -p "$OUT_DIR"

    echo
    echo "=== conc=$CONC prompts=$NUM_PROMPTS warmups=$WARMUPS -> $OUT_DIR"

    # bench_serving appends its own <unix-timestamp>/ under --profile-output-dir,
    # so repeat runs of the same shape never overwrite each other.
    profile_cmd=(
        python3 -m "$BENCH_MODULE"
        --backend sglang
        --host "$HOST"
        --port "$PORT"
        --model "$MODEL"
        --tokenizer "$MODEL"
        --dataset-name random
        --random-input-len "$ISL"
        --random-output-len "$OSL"
        --random-range-ratio "$RANDOM_RANGE_RATIO"
        --num-prompts "$NUM_PROMPTS"
        --max-concurrency "$CONC"
        --warmup-requests "$WARMUPS"
        --request-rate inf
        --flush-cache
        --cache-report
        --profile
        --profile-output-dir "$OUT_DIR"
        --profile-activities $ACTIVITIES
        --output-file "$OUT_DIR/$TAG.jsonl"
    )

    [[ "$TOKENIZE_PROMPT" == "1" ]] && profile_cmd+=(--tokenize-prompt)
    [[ "$BY_STAGE" == "1" ]] && profile_cmd+=(--profile-by-stage)
    [[ -n "$PROFILE_STAGES" ]] && profile_cmd+=(--profile-stages $PROFILE_STAGES)
    [[ -n "$PROFILE_NUM_STEPS" ]] && profile_cmd+=(--profile-num-steps "$PROFILE_NUM_STEPS")

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "${profile_cmd[@]}"
        printf '\n'
        continue
    fi

    "${profile_cmd[@]}" 2>&1 | tee "$LOG_DIR/$TAG.log"
done

[[ "$DRY_RUN" == "1" ]] && exit 0

echo
echo "=== traces collected:"
find "$PROFILE_DIR" \( -name '*.trace.json*' -o -name '*.gz' \) 2>/dev/null \
    | sort | while read -r f; do
        printf '  %10s  %s\n' "$(du -h "$f" | cut -f1)" "$f"
    done
echo "=== view with chrome://tracing, https://ui.perfetto.dev, or tensorboard"
