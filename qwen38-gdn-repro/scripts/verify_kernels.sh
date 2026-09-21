#!/usr/bin/env bash
set -euo pipefail

# Prove which AITER kernels actually ran. RUN THIS BEFORE TRUSTING ANY NUMBER.
#
# Both kernels are dispatched behind a try/except that falls back to Triton when
# the AITER wrapper declines a batch. That fallback is silent at default log
# level, so a fully-gated-on server can quietly produce baseline numbers. That
# is not hypothetical: on Qwen3.8 the GDN decode kernel declined 100% of calls
# (fused-QKV views fail an is_contiguous() check) and the "patched" arm was
# measuring stock Triton. Patch 0005 fixes it; this script confirms it.
#
#   ./verify_kernels.sh                 # check the conv_decode arm
#   ARM=conv_only ./verify_kernels.sh
#
# Applies the optional diagnostics patch, starts a server with
# SGLANG_AITER_GDN_DEBUG=1, sends one short request, reads the verdict, then
# reverts the patch and stops the server. Leaves the tree exactly as it found it.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ARM=${ARM:-conv_decode}
DEBUG_PATCH="$REPO_ROOT/patches/optional/0006-sglang-aiter-gdn-debug-logging.patch"
LOG="$LOG_DIR/verify_${ARM}.log"
mkdir -p "$LOG_DIR"

PATCH_APPLIED=0
cleanup() {
    stop_server
    if [[ "$PATCH_APPLIED" == "1" ]]; then
        git -C "$SGLANG_DIR" apply --reverse "$DEBUG_PATCH" && PATCH_APPLIED=0
        echo "-- diagnostics patch reverted"
    fi
}
trap cleanup EXIT

stop_server
echo "== applying diagnostics patch =="
git -C "$SGLANG_DIR" apply "$DEBUG_PATCH"
PATCH_APPLIED=1

echo "== starting '$ARM' server with SGLANG_AITER_GDN_DEBUG=1 =="
SGLANG_AITER_GDN_DEBUG=1 ARM="$ARM" "$REPO_ROOT/scripts/run_server.sh" > "$LOG" 2>&1 &
SERVER_PID=$!
disown $SERVER_PID 2>/dev/null || true
wait_for_server

# A short prompt is enough: both kernels are on the very first forward pass.
# Long enough to force a real prefill, and a few output tokens to reach decode.
echo "== sending probe request =="
curl -sf -m 300 "http://$HOST:$PORT/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"text\": \"$(head -c 4000 /dev/urandom | base64 | tr -d '\n' | head -c 3000)\",
         \"sampling_params\": {\"max_new_tokens\": 32, \"temperature\": 0}}" \
    >/dev/null || { echo "ERROR: probe request failed" >&2; exit 1; }

sleep 3
echo
echo "================ VERDICT ================"
if ! grep -h "\[aiter-gdn\]" "$LOG" | sort -u | sed 's/^/  /' | grep .; then
    echo "  NO REPORT -- the dispatch sites were never reached."
    echo "  Check that the model really uses GDN linear attention and that"
    echo "  patches 0003/0004 are applied."
    exit 1
fi
echo "========================================="
echo
echo "Expected for ARM=$ARM:"
case "$ARM" in
    baseline)    echo "  both lines OFF" ;;
    conv_only)   echo "  prefill conv ACTIVE; decode OFF" ;;
    conv_decode) echo "  both ACTIVE" ;;
esac
echo
echo "Anything reading 'FELL BACK to Triton' means that kernel is NOT being"
echo "measured -- the printed reason is the shape/dtype the wrapper rejected."
echo "(full log: $LOG)"
