#!/usr/bin/env bash
set -euo pipefail

# Torch-profiler trace collection for Qwen3.5-397B-A17B MXFP4 + EAGLE MTP.
#
# This is NOT a benchmark: the numbers it prints are throwaway. It drives the
# smallest request set that still produces a representative trace at each batch
# size -- long prefill (ISL 32K), near-zero decode (OSL 4), one wave of requests
# per concurrency point.
#
# Requires a running server (qwen3.5_fp4_sglang_server.sh). Its
# --max-prefill-tokens/--chunked-prefill-size must be >= ISL, else the prefill
# is split across forward passes and the trace shows chunks, not one prefill.
#
# Usage:
#   ./qwen3.5_fp4_sglang_profile.sh                       # conc 1 2 4 8 16, ISL 32768, OSL 4
#   CONC_LIST="8" ./qwen3.5_fp4_sglang_profile.sh         # single point
#   BY_STAGE=1 ./qwen3.5_fp4_sglang_profile.sh            # separate prefill/decode traces
#   PROFILE_NUM_STEPS=10 ./qwen3.5_fp4_sglang_profile.sh  # bound trace size
#   ACTIVITIES="CPU GPU MEM" ./qwen3.5_fp4_sglang_profile.sh
#
# Env: HOST PORT MODEL ISL OSL CONC_LIST PROFILE_DIR NUM_PROMPTS_MULT
#      WARMUP_MULT ACTIVITIES BY_STAGE PROFILE_NUM_STEPS PROFILE_STAGES
#      RANDOM_RANGE_RATIO LOG_DIR

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8888}
MODEL=${MODEL:-amd/Qwen3.5-397B-A17B-MXFP4}

ISL=${ISL:-32768}
OSL=${OSL:-4}
CONC_LIST=${CONC_LIST:-"1 2 4 8 16"}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1}   # exact ISL, no length jitter in the trace

# One wave: num-prompts == concurrency, so every request is in flight together
# and the profiled batch really is size CONC. More prompts would only add
# repeated identical-shaped steps to the trace.
NUM_PROMPTS_MULT=${NUM_PROMPTS_MULT:-1}
# Warmup matters more here than in a benchmark: AITER/FlyDSL JIT and CUDA-graph
# capture must be done before the profiler starts, or the trace is dominated by
# one-time compile work. bench_serving runs warmup -> flush_cache -> start
# profiler -> requests, so warmup never leaks into the captured window.
WARMUP_MULT=${WARMUP_MULT:-1}

PROFILE_DIR=${PROFILE_DIR:-/var/home/sglang_profiling5}
LOG_DIR=${LOG_DIR:-"$PROFILE_DIR/logs"}
ACTIVITIES=${ACTIVITIES:-"CPU GPU"}
BY_STAGE=${BY_STAGE:-0}
PROFILE_NUM_STEPS=${PROFILE_NUM_STEPS:-}
PROFILE_STAGES=${PROFILE_STAGES:-}
DRY_RUN=${DRY_RUN:-0}                         # print resolved commands, run nothing

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
echo "=== activities     : $ACTIVITIES  by-stage=$BY_STAGE"
echo "=== traces         : $PROFILE_DIR"

if [[ "$DRY_RUN" != "1" ]] && ! curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: no healthy SGLang server at http://$HOST:$PORT" >&2
    echo "       start one with ./qwen3.5_fp4_sglang_server.sh" >&2
    exit 1
fi

# Warn early rather than after a long run: a server whose prefill budget is
# below ISL will chunk the prompt and the trace will not show a single prefill.
SERVER_PREFILL_BUDGET=$(curl -sf "http://$HOST:$PORT/server_info" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("max_prefill_tokens") or d.get("server_args",{}).get("max_prefill_tokens",""))' 2>/dev/null || true)
if [[ "$SERVER_PREFILL_BUDGET" =~ ^[0-9]+$ && "$SERVER_PREFILL_BUDGET" -lt "$ISL" ]]; then
    echo "WARNING: server max-prefill-tokens=$SERVER_PREFILL_BUDGET < ISL=$ISL;" >&2
    echo "         prefill will be chunked across forward passes." >&2
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
        --profile
        --profile-output-dir "$OUT_DIR"
        --profile-activities $ACTIVITIES
    )

    # Splits capture into separate prefill and decode traces. The server
    # auto-stops after num_steps (defaults to 5) when this is set.
    [[ "$BY_STAGE" == "1" ]] && profile_cmd+=(--profile-by-stage)
    [[ -n "$PROFILE_STAGES" ]] && profile_cmd+=(--profile-stages $PROFILE_STAGES)
    # Caps how many forward steps land in the trace. Usually unnecessary at
    # OSL=4 -- the whole run is only ~CONC prefill steps plus a couple of
    # decode steps -- but needed if you raise OSL.
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
find "$PROFILE_DIR" -name '*.trace.json*' -o -name '*.pt.trace.json*' -o -name '*.gz' 2>/dev/null \
    | sort | while read -r f; do
        printf '  %10s  %s\n' "$(du -h "$f" | cut -f1)" "$f"
    done
echo "=== view with chrome://tracing, https://ui.perfetto.dev, or tensorboard"
