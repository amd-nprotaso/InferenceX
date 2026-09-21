#!/usr/bin/env bash
set -euo pipefail

# Fixed-seq-len sweep against an already-running Qwen3.8 SGLang server
# (see run_server.sh). Same driver as local_bench/qwen3.5_fp4_sglang_bench.sh,
# retargeted at Qwen3.8-Flash-Next and parameterised by ARM so the baseline and
# the AITER/FlyDSL GDN arms land in separate result dirs.
#
# Usage:
#   ARM=baseline ./run_client.sh
#   ARM=aiter_gdn CONC_LIST="1 4 16" ./run_client.sh
#
# Env: HOST PORT MODEL ISL OSL CONC_LIST RANDOM_RANGE_RATIO NUM_PROMPTS_MULT
#      WARMUP_MULT RESULT_DIR ARM FLUSH_CACHE TOKENIZE_PROMPT

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8888}
MODEL=${MODEL:-amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8}

ISL=${ISL:-131072}
OSL=${OSL:-1024}
CONC_LIST=${CONC_LIST:-"1 4 16"}

# Exact ISL, no length jitter: the two arms must see identical shapes, and the
# FlyDSL kernels are shape-specialised (chunk sizes want T % 64 == 0).
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1}
TOKENIZE_PROMPT=${TOKENIZE_PROMPT:-1}

NUM_PROMPTS_MULT=${NUM_PROMPTS_MULT:-4}   # num-prompts = mult * concurrency
WARMUP_MULT=${WARMUP_MULT:-1}             # warmup reqs = mult * concurrency
FLUSH_CACHE=${FLUSH_CACHE:-1}

ARM=${ARM:-baseline}
RESULT_DIR=${RESULT_DIR:-./bench_results/$ARM}

if python3 -c "import sglang.benchmark.serving" >/dev/null 2>&1; then
    BENCH_MODULE=sglang.benchmark.serving
else
    BENCH_MODULE=sglang.bench_serving
fi

mkdir -p "$RESULT_DIR"

echo "=== arm         : $ARM"
echo "=== target      : http://$HOST:$PORT"
echo "=== model       : $MODEL"
echo "=== module      : $BENCH_MODULE"
echo "=== shape       : ISL=$ISL OSL=$OSL range-ratio=$RANDOM_RANGE_RATIO"
echo "=== concurrency : $CONC_LIST"
echo "=== results     : $RESULT_DIR"

if ! curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: no healthy SGLang server at http://$HOST:$PORT" >&2
    exit 1
fi

# Record what the server actually resolved, so a result dir is self-describing:
# which linear-attention kernel backends were picked and how big the KV pool is.
curl -sf "http://$HOST:$PORT/server_info" -o "$RESULT_DIR/server_info.json" 2>/dev/null || true

for CONC in $CONC_LIST; do
    NUM_PROMPTS=$((CONC * NUM_PROMPTS_MULT))
    WARMUPS=$((CONC * WARMUP_MULT))
    [ "$WARMUPS" -lt 1 ] && WARMUPS=1
    TAG="qwen38_${ARM}_isl${ISL}_osl${OSL}_c${CONC}"

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

    # ignore_eos is on by default, so OSL is honored exactly -- required for
    # comparable decode/TPOT numbers under MTP.
    [[ "$FLUSH_CACHE" == "1" ]] && bench_cmd+=(--flush-cache)
    # Send token ids: without this the random dataset detokenizes then
    # re-tokenizes and prompts land a few tokens short of ISL.
    [[ "$TOKENIZE_PROMPT" == "1" ]] && bench_cmd+=(--tokenize-prompt)

    "${bench_cmd[@]}" 2>&1 | tee "$RESULT_DIR/$TAG.log"
done

echo
echo "=== done. results in $RESULT_DIR/"
