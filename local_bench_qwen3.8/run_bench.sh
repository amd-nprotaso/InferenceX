#/bin/bash

export HF_HOME=/data/hf_cache/
export HF_HUB_CACHE="${HF_HOME}hub"

BENCH_DIR=/var/home/my_sglang/chang_sglang/sglang/agentx_benchmark
MODEL_PATH=/data/hf_cache/hub/models--amd--Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8/snapshots/1ad36fa866c05631c14adc6a74d7e9908c4e5888
OUTPUT_DIR="${BENCH_DIR}/logs/compare_conc8_$(date -u +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"
set -o pipefail

"${BENCH_DIR}/.venv-agentx/bin/aiperf" profile \
--scenario agentx \
--url http://127.0.0.1:18888 \
--endpoint /v1/chat/completions \
--model qwen3.8-flash-next \
--tokenizer "$MODEL_PATH" \
--tokenizer-trust-remote-code \
--concurrency 8 \
--benchmark-duration 3600 \
--public-dataset semianalysis_cc_traces_weka_062126_256k \
--warmup-requests-per-lane 10 \
--warmup-grace-period 1800 \
--failed-request-threshold 0.10 \
--server-metrics http://127.0.0.1:18888/metrics \
--output-artifact-dir "${OUTPUT_DIR}/aiperf_artifacts" \
2>&1 | tee "${OUTPUT_DIR}/client.log"