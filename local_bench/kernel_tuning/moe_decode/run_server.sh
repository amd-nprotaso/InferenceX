#!/usr/bin/env bash
set -euo pipefail

# Launch the Qwen3.5-397B-A17B MXFP4 SGLang server with the FlyDSL fused decode
# MoE front-end (kernel_tuning/moe_decode) enabled.
#
# Two independent changes, each with its own switch, so all four arms of the A/B
# come from this one script:
#
#   MOE_FUSED_SORT=1     change #2: FlyDSL fused router+sort in one launch,
#                        replacing aiter's topk_softmax + the two
#                        opus_moe_sorting_entry kernels.
#   MOE_ATOMIC_EPILOGUE=1
#                        change #1: tuned-CSV row token=16/inter=256/E=512/
#                        topk=10 switched from the `reduce` stage-2 epilogue to
#                        `atomic`, which deletes moe_reduction_kernel_0.
#                        Decode-only by construction: it is one row of the token
#                        tier table, and the prefill rows are untouched (the
#                        atomic epilogue is 1.6-2.8x SLOWER at prefill scale).
#
# Usage:
#   bash kernel_tuning/moe_decode/run_server.sh                 # both changes on
#   MOE_ATOMIC_EPILOGUE=0 bash .../run_server.sh                # change #2 only
#   MOE_FUSED_SORT=0 bash .../run_server.sh                     # change #1 only
#   MOE_FUSED_SORT=0 MOE_ATOMIC_EPILOGUE=0 bash .../run_server.sh   # baseline
#
#   TP=4 CONC=4 SIMULATE_ACC=0 bash kernel_tuning/moe_decode/run_server.sh
#
# Every flag of qwen3.5_fp4_sglang_server.sh still applies and is passed
# through (MODEL TP CONC PORT MEM_FRAC_STATIC SIMULATE_ACC ...).
#
# Env specific to this script:
#   MOE_FUSED_SORT        1 (default) / 0
#   MOE_ATOMIC_EPILOGUE   1 (default) / 0
#   MOE_MAX_TOKENS        token capacity the fused kernel is compiled for
#                         (default 16 = CONC 4 x 4 EAGLE draft tokens). Larger
#                         decode batches fall back to the stock path rather
#                         than being routed through a kernel that cannot hold
#                         them. 32 is the LDS ceiling at E=512; see README.
#   MOE_COUNT_COMPILES    1 to log one line per FlyDSL JIT compile

moe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
kernel_root=$(dirname -- "$moe_dir")
bench_root=$(dirname -- "$kernel_root")

MOE_FUSED_SORT=${MOE_FUSED_SORT:-1}
MOE_ATOMIC_EPILOGUE=${MOE_ATOMIC_EPILOGUE:-1}
MOE_MAX_TOKENS=${MOE_MAX_TOKENS:-16}
MOE_COUNT_COMPILES=${MOE_COUNT_COMPILES:-0}

# One sitecustomize for every FlyDSL kernel hook. Python imports that name
# exactly once, so each kernel directory must NOT ship its own on the path --
# kernel_tuning/bootstrap_both is the single entry point and loads the
# per-kernel hook files by path. The kernel source dirs still go on PYTHONPATH
# because each hook does a bare `import <module>` when its target is imported.
export PYTHONPATH="$kernel_root/bootstrap_both:$moe_dir${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$MOE_FUSED_SORT" == "1" ]]; then
    export SGLANG_FLYDSL_MOE_DECODE=1
    export SGLANG_FLYDSL_MOE_DECODE_MAX_TOKENS="$MOE_MAX_TOKENS"
    if [[ "$MOE_MAX_TOKENS" -gt 32 ]]; then
        echo "ERROR: MOE_MAX_TOKENS=$MOE_MAX_TOKENS exceeds the LDS ceiling (32) at E=512." >&2
        echo "       The fused kernel stages the whole [tokens, E+1] routing mesh in LDS." >&2
        exit 1
    fi
else
    unset SGLANG_FLYDSL_MOE_DECODE || true
fi

if [[ "$MOE_COUNT_COMPILES" == "1" ]]; then
    export SGLANG_FLYDSL_COUNT_COMPILES=1
fi

# aiter merges every configs/model_configs/*tuned_fmoe*.csv into one table and
# raises -- after rewriting the source files in place -- if two rows share a
# shape key. So the patched CSV cannot simply be prepended: fmoe_chain.py emits
# a chain with the stock qwen3.5 file dropped and ours in its place.
if [[ "$MOE_ATOMIC_EPILOGUE" == "1" ]]; then
    atomic_csv="$moe_dir/qwen3_5_397b_fp4_tuned_fmoe.atomic.csv"
    if [[ ! -f "$atomic_csv" ]]; then
        echo "ERROR: missing $atomic_csv (regenerate it, see README)" >&2
        exit 1
    fi
    AITER_CONFIG_FMOE=$(python3 "$moe_dir/fmoe_chain.py" "$atomic_csv")
    export AITER_CONFIG_FMOE
fi

echo "=== decode MoE arms"
echo "===   fused router+sort : $MOE_FUSED_SORT (max_tokens=$MOE_MAX_TOKENS)"
echo "===   atomic stage-2    : $MOE_ATOMIC_EPILOGUE"
echo "===   PYTHONPATH        : $PYTHONPATH"
echo "=== the server prints '[FlyDSL-moe] fused router+sort enabled' once per TP"
echo "=== worker when the hook lands; its absence means the arm is NOT active."

exec bash "$bench_root/qwen3.5_fp4_sglang_server.sh" "$@"
