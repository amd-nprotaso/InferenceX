#!/usr/bin/env bash
set -uo pipefail

# TP=1 paired A/B of the upstream AgentX CI recipe: stock Triton GDN kernels vs
# the AITER/FlyDSL ones, over the CI concurrency points that fit a single GPU.
#
# The recipe itself is run UNMODIFIED. The only thing that differs between the
# two arms is the pair of kernel gates run_agentic_arm.sh exports, so a delta is
# attributable to the kernels.
#
# Arms run back to back within a concurrency point rather than grouped by arm,
# so the two measurements being compared sit as close together in time as
# possible and share the same machine state.

cd "$(dirname "$0")"

export INFMAX_CONTAINER_WORKSPACE=${INFMAX_CONTAINER_WORKSPACE:-/var/home/my_InferenceX/qwen3.8/InferenceX}
# The 123 GB checkpoint and the cc-traces dataset live here, not in the default
# ~/.cache/huggingface. Without this every point re-downloads the model.
export HF_HOME=${HF_HOME:-/data/hf_cache}

# Patch 0006 makes each server report once per process whether each GDN kernel
# was taken or declined. Both arms run identical code -- the flag only decides
# whether a single already-resolved global gets logged -- so this keeps the A/B
# fair while making a silent Triton fallback impossible to miss. Note it reports
# the FIRST outcome per process: it proves a kernel was reached, not that every
# later batch was also accepted.
#
# Applied here and reverted on exit rather than left on the tree, because
# qwen38-gdn-repro/scripts/verify_kernels.sh git-applies 0006 itself and fails
# if it is already present.
export SGLANG_AITER_GDN_DEBUG=1
SGLANG_DIR=${SGLANG_DIR:-/sgl-workspace/sglang}
DEBUG_PATCH=${DEBUG_PATCH:-/var/home/my_InferenceX/InferenceX/qwen38-gdn-repro/patches/optional/0006-sglang-aiter-gdn-debug-logging.patch}
DEBUG_PATCH_APPLIED=0
if git -C "$SGLANG_DIR" apply "$DEBUG_PATCH" 2>/dev/null; then
    DEBUG_PATCH_APPLIED=1
    echo "diagnostics patch 0006 applied"
else
    echo "WARNING: 0006 did not apply (already present?); kernel verdicts may be missing" >&2
fi
cleanup_debug_patch() {
    if [[ "$DEBUG_PATCH_APPLIED" == "1" ]]; then
        git -C "$SGLANG_DIR" apply --reverse "$DEBUG_PATCH" \
            && echo "diagnostics patch 0006 reverted"
        DEBUG_PATCH_APPLIED=0
    fi
}
trap cleanup_debug_patch EXIT

export TP=${TP:-1}
export DURATION=${DURATION:-1800}
CONC_LIST=${CONC_LIST:-"1 4 8"}
ARMS=${ARMS:-"baseline conv_decode"}

export BASE_RESULT_DIR=${BASE_RESULT_DIR:-$PWD/agentic_results_tp1_ci}
LOG_DIR="$BASE_RESULT_DIR/logs"
mkdir -p "$LOG_DIR"

stop_server() {
    # Bracket the first character so the pattern cannot match this script's own
    # command line.
    pkill -f "[p]ython3 -m sglang.launch_server" 2>/dev/null || return 0
    for _ in $(seq 1 45); do
        pgrep -f "[p]ython3 -m sglang.launch_server" >/dev/null 2>&1 || break
        sleep 2
    done
    pkill -9 -f "[p]ython3 -m sglang.launch_server" 2>/dev/null || true
    # Let the GPU actually release the ~80 GB before the next server claims it.
    sleep 20
}

echo "sweep start $(date --iso-8601=seconds)"
echo "  tp=$TP duration=${DURATION}s conc=[$CONC_LIST] arms=[$ARMS]"
echo "  results -> $BASE_RESULT_DIR"

for CONC in $CONC_LIST; do
  for ARM in $ARMS; do
    echo "### $(date --iso-8601=seconds)  arm=$ARM conc=$CONC"
    stop_server
    ARM="$ARM" CONC="$CONC" ./run_agentic_arm.sh \
        > "$LOG_DIR/${ARM}_conc${CONC}.log" 2>&1
    echo "###   exit=$? at $(date --iso-8601=seconds)"
  done
done
stop_server
echo "SWEEP COMPLETE $(date --iso-8601=seconds)"
