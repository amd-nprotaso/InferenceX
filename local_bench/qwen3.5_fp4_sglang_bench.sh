#!/usr/bin/env bash
set -euo pipefail

# Fixed-seq-len throughput sweep against an already-running SGLang server
# (see qwen3.5_fp4_sglang_server.sh). Drives sglang.benchmark.serving directly
# rather than the repo's vLLM-flavoured utils/bench_serving/benchmark_serving.py,
# so absolute numbers are NOT 1:1 comparable with CI results.
#
# Usage:
#   ./qwen3.5_fp4_sglang_bench.sh                 # MODE=full  : ISL 8192 / OSL 1024
#   MODE=quick   ./qwen3.5_fp4_sglang_bench.sh    # smoke test : ISL 256  / OSL 32
#   MODE=prefill ./qwen3.5_fp4_sglang_bench.sh    # prefill-heavy: ISL 32768 / OSL 4
#   CONC_LIST="4 8" ISL=4096 OSL=512 ./qwen3.5_fp4_sglang_bench.sh
#   PROFILE=1 CONC_LIST=8 ./qwen3.5_fp4_sglang_bench.sh
#
# For trace collection use ./qwen3.5_fp4_sglang_profile.sh instead.
#
# Env: HOST PORT MODEL MODE ISL OSL CONC_LIST RANDOM_RANGE_RATIO
#      NUM_PROMPTS_MULT WARMUP_MULT RESULT_DIR FLUSH_CACHE TOKENIZE_PROMPT
#      PROFILE PROFILE_DIR

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8888}
MODEL=${MODEL:-amd/Qwen3.5-397B-A17B-MXFP4}
MODE=${MODE:-full}

case "$MODE" in
    quick)
        ISL=${ISL:-256};   OSL=${OSL:-32}
        CONC_LIST=${CONC_LIST:-16}
        RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1}
        ;;
    prefill)
        # Prefill-dominant: OSL 4 means TTFT and input throughput are the signal;
        # decode/TPOT and MTP acceptance are effectively unmeasured here.
        ISL=${ISL:-32768}; OSL=${OSL:-4}
        CONC_LIST=${CONC_LIST:-"4 8 16"}
        RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1}
        ;;
    full)
        # Mirrors configs/amd-master.yaml -> qwen3.5-fp4-mi355x-sglang-mtp
        # (fixed-seq-len isl 8192 / osl 1024, tp 4, conc 4..16) with the CI-wide
        # RANDOM_RANGE_RATIO=0.8 from .github/workflows/benchmark-tmpl.yml.
        ISL=${ISL:-32768};  OSL=${OSL:-1024}
        CONC_LIST=${CONC_LIST:-"4 8 16"}
        RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.8}
        ;;
    *)
        echo "ERROR: unknown MODE='$MODE' (expected quick|full|prefill)" >&2
        exit 2
        ;;
esac

NUM_PROMPTS_MULT=${NUM_PROMPTS_MULT:-10}   # num-prompts = mult * concurrency
WARMUP_MULT=${WARMUP_MULT:-2}              # warmup reqs = mult * concurrency
RESULT_DIR=${RESULT_DIR:-./bench_results}
FLUSH_CACHE=${FLUSH_CACHE:-1}
TOKENIZE_PROMPT=${TOKENIZE_PROMPT:-0}
PROFILE=${PROFILE:-0}
PROFILE_DIR=${PROFILE_DIR:-/var/home/sglang_profiling}

# python -m sglang.bench_serving still works but is a deprecation shim in
# recent builds; prefer the new module path and fall back when absent.
if python3 -c "import sglang.benchmark.serving" >/dev/null 2>&1; then
    BENCH_MODULE=sglang.benchmark.serving
else
    BENCH_MODULE=sglang.bench_serving
fi

mkdir -p "$RESULT_DIR"

echo "=== target      : http://$HOST:$PORT"
echo "=== model       : $MODEL"
echo "=== module      : $BENCH_MODULE"
echo "=== mode        : $MODE  (ISL=$ISL OSL=$OSL range-ratio=$RANDOM_RANGE_RATIO)"
echo "=== concurrency : $CONC_LIST"
echo "=== results     : $RESULT_DIR"

if ! curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: no healthy SGLang server at http://$HOST:$PORT" >&2
    echo "       start one with ./qwen3.5_fp4_sglang_server.sh" >&2
    exit 1
fi

for CONC in $CONC_LIST; do
    NUM_PROMPTS=$((CONC * NUM_PROMPTS_MULT))
    WARMUPS=$((CONC * WARMUP_MULT))
    TAG="qwen35_mxfp4_mtp_isl${ISL}_osl${OSL}_c${CONC}"

    echo
    echo "=== conc=$CONC prompts=$NUM_PROMPTS warmups=$WARMUPS -> $RESULT_DIR/$TAG.jsonl"

    bench_cmd=(
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
        --cache-report
        --output-file "$RESULT_DIR/$TAG.jsonl"
    )

    # ignore_eos is ON by default (opt out with --disable-ignore-eos), so OSL is
    # honored exactly -- required for comparable decode/TPOT numbers under MTP.
    [[ "$FLUSH_CACHE" == "1" ]] && bench_cmd+=(--flush-cache)

    # Send token ids instead of text. Without this the random dataset detokenizes
    # then re-tokenizes, and prompts land a few tokens short of ISL even at
    # range-ratio 1 (observed 31858 / 32359 / 32462 for ISL=32768). Set this when
    # a kernel needs the prefill token count to be an exact multiple of its chunk
    # size -- e.g. the FlyDSL GDN chunk_h specialization, which requires T%64==0.
    [[ "$TOKENIZE_PROMPT" == "1" ]] && bench_cmd+=(--tokenize-prompt)

    # For trace collection prefer ./qwen3.5_fp4_sglang_profile.sh -- it sizes the
    # request set for profiling instead of for throughput. This stays as an
    # escape hatch for profiling a real benchmark shape.
    if [[ "$PROFILE" == "1" ]]; then
        mkdir -p "$PROFILE_DIR"
        bench_cmd+=(--profile --profile-output-dir "$PROFILE_DIR")
    fi

    "${bench_cmd[@]}" 2>&1 | tee "$RESULT_DIR/$TAG.log"
done

echo
echo "=== done. results in $RESULT_DIR/"
