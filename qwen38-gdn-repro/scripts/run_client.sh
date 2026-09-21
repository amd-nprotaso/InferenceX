#!/usr/bin/env bash
set -euo pipefail

# Fixed-shape concurrency sweep against an already-running server.
#
#   ARM=baseline ./run_client.sh
#   ARM=conv_decode CONC_LIST="1 4" ./run_client.sh
#
# Results land in $RESULTS_DIR/$ARM/ as one jsonl + one log per concurrency point.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ARM=${ARM:-baseline}
OUT_DIR="$RESULTS_DIR/$ARM"

# Exact ISL, no length jitter: both arms must see byte-identical request shapes,
# and the FlyDSL kernels are shape-specialised (chunk sizes want T % 64 == 0).
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1}
# Send token ids. Without this the random dataset detokenizes then re-tokenizes
# and prompts land a few tokens short of ISL.
TOKENIZE_PROMPT=${TOKENIZE_PROMPT:-1}
FLUSH_CACHE=${FLUSH_CACHE:-1}

if python3 -c "import sglang.benchmark.serving" >/dev/null 2>&1; then
    BENCH_MODULE=sglang.benchmark.serving
else
    BENCH_MODULE=sglang.bench_serving   # deprecation shim on older builds
fi

mkdir -p "$OUT_DIR"

echo "=== arm         : $ARM"
echo "=== target      : http://$HOST:$PORT"
echo "=== shape       : ISL=$ISL OSL=$OSL range-ratio=$RANDOM_RANGE_RATIO"
echo "=== concurrency : $CONC_LIST"
echo "=== results     : $OUT_DIR"

if ! curl -sf -m 10 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: no healthy server at http://$HOST:$PORT" >&2
    exit 1
fi

# Snapshot what the server actually resolved, so a result dir is self-describing
# (which linear-attention kernel backends were picked, how big the KV pool is).
curl -sf -m 10 "http://$HOST:$PORT/server_info" -o "$OUT_DIR/server_info.json" || true

for CONC in $CONC_LIST; do
    NUM_PROMPTS=$((CONC * NUM_PROMPTS_MULT))
    WARMUPS=$((CONC * WARMUP_MULT))
    [ "$WARMUPS" -lt 1 ] && WARMUPS=1
    TAG="qwen38_${ARM}_isl${ISL}_osl${OSL}_c${CONC}"

    echo
    echo "=== conc=$CONC prompts=$NUM_PROMPTS warmups=$WARMUPS -> $OUT_DIR/$TAG.jsonl"

    bench_cmd=(
        python3 -m "$BENCH_MODULE"
        --backend sglang --host "$HOST" --port "$PORT"
        --model "$MODEL" --tokenizer "$MODEL"
        --dataset-name random
        --random-input-len "$ISL"
        --random-output-len "$OSL"
        --random-range-ratio "$RANDOM_RANGE_RATIO"
        --num-prompts "$NUM_PROMPTS"
        --max-concurrency "$CONC"
        --warmup-requests "$WARMUPS"
        --request-rate inf
        --cache-report
        --output-file "$OUT_DIR/$TAG.jsonl"
    )
    # ignore_eos is on by default, so OSL is honored exactly -- required for
    # comparable decode/TPOT numbers under MTP.
    [[ "$FLUSH_CACHE" == "1" ]] && bench_cmd+=(--flush-cache)
    [[ "$TOKENIZE_PROMPT" == "1" ]] && bench_cmd+=(--tokenize-prompt)

    "${bench_cmd[@]}" 2>&1 | tee "$OUT_DIR/$TAG.log"
done

echo
echo "=== done: $OUT_DIR"
