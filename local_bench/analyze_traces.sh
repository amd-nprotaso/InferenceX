#!/usr/bin/env bash
set -euo pipefail

# Kernel-time analysis for the traces qwen3.5_fp4_sglang_profile.sh collected.
#
# Reads every <PROFILE_DIR>/<run>/<ts>/<ts>-TP-<rank>.trace.json.gz, attributes
# each GPU kernel to a forward-pass phase (EXTEND / TARGET_VERIFY / draft / ...)
# and an aten op, and writes per-kernel timing tables plus a run-vs-run
# comparison across the concurrency sweep.
#
# Usage:
#   ./analyze_traces.sh                                  # everything under PROFILE_DIR
#   RANKS=0 ./analyze_traces.sh                          # TP rank 0 only (4x faster)
#   PHASE=EXTEND TOP=40 ./analyze_traces.sh              # prefill kernels only
#   ROOTS=/var/home/sglang_profiling/isl32768_osl4_c8 ./analyze_traces.sh
#   FORCE=1 ./analyze_traces.sh                          # ignore the parse cache
#
# Env: PROFILE_DIR ROOTS OUT_DIR JOBS TOP RANKS PHASE METRIC FORCE NO_CPU_OP
#
# Re-runs are cheap: parsed traces are cached as parquet under OUT_DIR/cache and
# only re-read when the source trace is newer.

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

PROFILE_DIR=${PROFILE_DIR:-/var/home/sglang_profiling}
ROOTS=${ROOTS:-$PROFILE_DIR}
OUT_DIR=${OUT_DIR:-./trace_analysis}
JOBS=${JOBS:-16}
TOP=${TOP:-25}
RANKS=${RANKS:-}          # e.g. "0" or "0,1"; empty = all TP ranks
PHASE=${PHASE:-}          # e.g. "EXTEND" or "EXTEND,TARGET_VERIFY"; empty = all
METRIC=${METRIC:-total_us_per_rank}
FORCE=${FORCE:-0}
NO_CPU_OP=${NO_CPU_OP:-0}

echo "=== traces  : $ROOTS"
echo "=== output  : $OUT_DIR"
echo "=== jobs    : $JOBS   ranks: ${RANKS:-all}   phase: ${PHASE:-all}"

analyze_cmd=(python3 "$HERE/analyze_traces.py" $ROOTS -o "$OUT_DIR" -j "$JOBS" --top "$TOP")
[[ -n "$RANKS" ]]        && analyze_cmd+=(--ranks "$RANKS")
[[ -n "$PHASE" ]]        && analyze_cmd+=(--phase "$PHASE")
[[ "$FORCE" == "1" ]]    && analyze_cmd+=(--force)
[[ "$NO_CPU_OP" == "1" ]] && analyze_cmd+=(--no-cpu-op)

# exit 2 means "tables are complete for the traces we could read, but some were
# skipped" -- worth continuing to the comparison, and worth flagging at the end.
partial=0
"${analyze_cmd[@]}" || { rc=$?; [[ $rc == 2 ]] && partial=1 || exit $rc; }

compare_cmd=(python3 "$HERE/compare_traces.py" "$OUT_DIR" --metric "$METRIC" --top "$TOP")
[[ -n "$PHASE" ]] && compare_cmd+=(--phase "$PHASE")

echo
"${compare_cmd[@]}"

echo
echo "=== artifacts in $OUT_DIR/"
ls -1 "$OUT_DIR" | sed 's/^/  /'

if [[ $partial == 1 ]]; then
    echo
    echo "WARNING: some traces were skipped (see above); results cover the rest" >&2
    exit 2
fi
