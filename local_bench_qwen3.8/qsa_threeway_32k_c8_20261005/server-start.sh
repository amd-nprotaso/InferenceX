#/bin/bash

export HF_HOME=/data/hf_cache/
export HF_HUB_CACHE="${HF_HOME}hub"

export SGLANG_AITER_GDN_DECODE=1
export SGLANG_AITER_GDN_CONV=1
export SGLANG_AITER_HC_MIX=1
#export QSA_PA_DECODE=1
export SGLANG_AITER_QSA_PA_DECODE="${QSA_AB_FLAG:?}"
export AITER_GDR_FLYDSL=1

export SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1
export SGLANG_EXACT_CHUNK_FILL=0

export SGLANG_SIMULATE_ACC_LEN=2.32
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token

MODEL_PATH=/data/hf_cache/hub/models--amd--Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8/snapshots/1ad36fa866c05631c14adc6a74d7e9908c4e5888

/opt/venv/bin/python /var/home/my_InferenceX/InferenceX/local_bench_qwen3.8/qsa_threeway_32k_c8_20261005/server_entry.py \
--model-path "$MODEL_PATH" \
--served-model-name qwen3.8-flash-next \
--host 127.0.0.1 \
--port 18888 \
--trust-remote-code \
--quantization quark \
--attention-backend aiter \
--random-seed 42 \
--tp-size 1 \
--mem-fraction-static 0.90 \
--chunked-prefill-size 16384 \
--page-size 64 \
--speculative-algorithm NEXTN \
--speculative-num-steps 3 \
--speculative-num-draft-tokens 4 \
--speculative-eagle-topk 1 \
--max-running-requests 16 \
--cuda-graph-max-bs-decode 8 \
--scheduler-recv-interval 10 \
--stream-interval 50 \
--enable-metrics \
