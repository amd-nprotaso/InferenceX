#!/usr/bin/env bash
set -euo pipefail

# One launcher for every arm of a FlyDSL kernel A/B. Which kernels are active is
# decided by the env flags, each read by its own hook:
#
#   FLYDSL_CONV=1 FLYDSL_GDN=1 FLYDSL_MOE=1  ./run_server_both.sh   # everything
#   FLYDSL_CONV=1 FLYDSL_GDN=0 FLYDSL_MOE=0  ./run_server_both.sh   # conv only
#   FLYDSL_CONV=0 FLYDSL_GDN=1 FLYDSL_MOE=0  ./run_server_both.sh   # gdn only
#   FLYDSL_CONV=0 FLYDSL_GDN=0 FLYDSL_MOE=1  ./run_server_both.sh   # decode moe only
#   FLYDSL_CONV=0 FLYDSL_GDN=0 FLYDSL_MOE=0  ./run_server_both.sh   # stock
#
# Using one launcher for all arms keeps PYTHONPATH, sitecustomize and import
# order identical across arms, so the only thing that varies is the kernels.
#
# Do NOT instead concatenate the per-kernel run_server.sh PYTHONPATHs: every
# bootstrap dir contains a file named sitecustomize.py and Python imports that
# name only once, so all but the first kernel would be silently dropped.
# kernel_tuning/bootstrap_both is the single sitecustomize; it loads each
# per-kernel hook file by path.
#
# FLYDSL_CONV / FLYDSL_GDN default to 1 (historical behaviour of this script).
# FLYDSL_MOE defaults to 0 so that experiments written before the decode-MoE
# work keep measuring what they were written to measure.
#
# FLYDSL_COUNT_COMPILES=1 logs one line per FlyDSL JIT compile, so a benchmark
# window can be checked for "still compiling" rather than assumed warm.

kt_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$kt_dir/bootstrap_both:$kt_dir/conv:$kt_dir/chunk_gated_delta_rule:$kt_dir/moe_decode${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_FLYDSL_CAUSAL_CONV="${FLYDSL_CONV:-1}"
export SGLANG_FLYDSL_GDN_CHUNK_H="${FLYDSL_GDN:-1}"
export SGLANG_FLYDSL_COUNT_COMPILES="${FLYDSL_COUNT_COMPILES:-0}"

# Decode MoE: the fused FlyDSL router+sort (change #2) and the atomic stage-2
# epilogue (change #1, a tuned-CSV row). MOE_ATOMIC_EPILOGUE rides with
# FLYDSL_MOE unless set explicitly, so one flag selects the whole arm.
FLYDSL_MOE="${FLYDSL_MOE:-0}"
MOE_ATOMIC_EPILOGUE="${MOE_ATOMIC_EPILOGUE:-$FLYDSL_MOE}"
MOE_MAX_TOKENS="${MOE_MAX_TOKENS:-16}"

if [[ "$FLYDSL_MOE" == "1" ]]; then
    export SGLANG_FLYDSL_MOE_DECODE=1
    export SGLANG_FLYDSL_MOE_DECODE_MAX_TOKENS="$MOE_MAX_TOKENS"
else
    unset SGLANG_FLYDSL_MOE_DECODE || true
fi

if [[ "$MOE_ATOMIC_EPILOGUE" == "1" ]]; then
    atomic_csv="$kt_dir/moe_decode/qwen3_5_397b_fp4_tuned_fmoe.atomic.csv"
    if [[ ! -f "$atomic_csv" ]]; then
        echo "ERROR: missing $atomic_csv (see kernel_tuning/moe_decode/README.md)" >&2
        exit 1
    fi
    # Cannot simply prepend: aiter merges every model_configs/*tuned_fmoe*.csv
    # and, on a duplicate shape key, rewrites the SOURCE files in place and
    # raises. fmoe_chain.py drops the stock qwen3.5 file from the chain.
    AITER_CONFIG_FMOE=$(python3 "$kt_dir/moe_decode/fmoe_chain.py" "$atomic_csv")
    export AITER_CONFIG_FMOE
else
    unset AITER_CONFIG_FMOE || true
fi

exec bash "$kt_dir/../qwen3.5_fp4_sglang_server.sh" "$@"
