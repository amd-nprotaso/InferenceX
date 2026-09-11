#!/usr/bin/env bash
set -euo pipefail

# One launcher for every arm of a FlyDSL kernel A/B. Which kernels are active is
# decided by the two env flags, each read by its own hook:
#
#   FLYDSL_CONV=1 FLYDSL_GDN=1  ./run_server_both.sh   # both
#   FLYDSL_CONV=1 FLYDSL_GDN=0  ./run_server_both.sh   # conv only
#   FLYDSL_CONV=0 FLYDSL_GDN=1  ./run_server_both.sh   # gdn only
#   FLYDSL_CONV=0 FLYDSL_GDN=0  ./run_server_both.sh   # stock
#
# Using one launcher for all four keeps PYTHONPATH, sitecustomize and import
# order identical across arms, so the only thing that varies is the kernels.
#
# Do NOT instead concatenate the two per-kernel run_server.sh PYTHONPATHs: both
# bootstrap dirs contain a file named sitecustomize.py and Python imports that
# name only once, so one kernel would be silently dropped.
#
# FLYDSL_COUNT_COMPILES=1 logs one line per FlyDSL JIT compile, so a benchmark
# window can be checked for "still compiling" rather than assumed warm.

kt_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$kt_dir/bootstrap_both:$kt_dir/conv:$kt_dir/chunk_gated_delta_rule${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_FLYDSL_CAUSAL_CONV="${FLYDSL_CONV:-1}"
export SGLANG_FLYDSL_GDN_CHUNK_H="${FLYDSL_GDN:-1}"
export SGLANG_FLYDSL_COUNT_COMPILES="${FLYDSL_COUNT_COMPILES:-0}"
exec bash "$kt_dir/../qwen3.5_fp4_sglang_server.sh" "$@"
