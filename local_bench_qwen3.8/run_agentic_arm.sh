#!/usr/bin/env bash
set -euo pipefail

# Drive the upstream AgentX recipe once per (arm, concurrency) point.
#
#   ARM=conv_decode CONC=4 ./run_agentic_arm.sh
#
# The recipe itself is used UNMODIFIED -- it already exports SGLANG_USE_AITER=1 /
# AITER_FLYDSL_FORCE=1 and sets no --linear-attn-*-backend flags, so the linear
# attention kernels resolve to triton and the AITER hooks are on the path. The
# only thing that varies between arms is the two kernel gates below, which the
# recipe inherits from this environment.
#
# Each invocation is self-contained: the recipe starts its own server, runs the
# replay, and kills the server via its EXIT trap.

RECIPE=${RECIPE:-/var/home/my_InferenceX/qwen3.8/InferenceX/benchmarks/single_node/agentic/qwen3.8next_fp4_mi355x_sglang_mtp.sh}

ARM=${ARM:-baseline}
CONC=${CONC:-1}

# Pinned to the locally cached checkpoint, which is the one every other
# measurement in this study used. The master config names the non-PLEFP8
# variant; using that here would mean comparing against different weights.
export MODEL=${MODEL:-amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8}
export TP=${TP:-4}
export EP_SIZE=${EP_SIZE:-1}
export KV_OFFLOADING=${KV_OFFLOADING:-none}
export TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB:-2048}
export DURATION=${DURATION:-1800}
export CONC
export PORT=${PORT:-8888}

BASE_RESULT_DIR=${BASE_RESULT_DIR:-/var/home/my_InferenceX/InferenceX/local_bench_qwen3.8/agentic_results}
export RESULT_DIR="$BASE_RESULT_DIR/${ARM}/conc${CONC}"
mkdir -p "$RESULT_DIR"

# Names the aggregated InferenceX result JSON. CI derives this from experiment
# metadata; locally any stable per-point string works, and encoding the arm keeps
# the two arms' aggregates from colliding.
export RESULT_FILENAME=${RESULT_FILENAME:-qwen38next_fp4_sglang_tp${TP}_${ARM}_conc${CONC}}
export AGENTIC_OUTPUT_DIR=${AGENTIC_OUTPUT_DIR:-$RESULT_DIR}

# Per-arm kernel gates. Unset first: a stale value in the caller's environment
# turning a kernel on in the baseline arm is the one failure that would silently
# invalidate the whole comparison.
unset SGLANG_AITER_GDN_CONV SGLANG_AITER_GDN_DECODE
case "$ARM" in
    baseline)    ;;
    conv_only)   export SGLANG_AITER_GDN_CONV=1 ;;
    decode_only) export SGLANG_AITER_GDN_DECODE=1 ;;
    conv_decode) export SGLANG_AITER_GDN_CONV=1; export SGLANG_AITER_GDN_DECODE=1 ;;
    *) echo "unknown ARM '$ARM' (baseline|conv_only|decode_only|conv_decode)" >&2; exit 2 ;;
esac

echo "=============================================================="
echo "arm=$ARM conc=$CONC tp=$TP duration=${DURATION}s"
echo "gates: CONV=${SGLANG_AITER_GDN_CONV:-<off>} DECODE=${SGLANG_AITER_GDN_DECODE:-<off>}"
echo "results: $RESULT_DIR"
echo "=============================================================="

# Record the gates next to the results so an arm's directory is self-describing.
cat > "$RESULT_DIR/arm_env.json" <<EOF
{"arm": "$ARM", "conc": $CONC, "tp": $TP, "duration": $DURATION, "model": "$MODEL",
 "SGLANG_AITER_GDN_CONV": "${SGLANG_AITER_GDN_CONV:-}",
 "SGLANG_AITER_GDN_DECODE": "${SGLANG_AITER_GDN_DECODE:-}"}
EOF

"$RECIPE" 2>&1 | tee "$RESULT_DIR/recipe.log"
