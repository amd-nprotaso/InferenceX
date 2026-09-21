#!/usr/bin/env bash
set -euo pipefail

# Full experiment: for each arm, start a server, sweep concurrency, stop it.
# Then print the comparison table.
#
#   ./run_all.sh                                   # all three arms
#   ARMS="baseline conv_decode" ./run_all.sh       # subset
#   CONC_LIST="1 4" NUM_PROMPTS_MULT=2 ./run_all.sh   # quicker
#
# Each arm gets a FRESH server: the kernel gates are read at import time, so
# they cannot be flipped on a live process. Budget ~4 min of startup per arm
# plus the sweep itself (~15 min for the default 1/4/16 sweep on MI355X).

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
trap 'stop_server' EXIT

for ARM in $ARMS; do
    arm_env "$ARM" >/dev/null   # reject a typo'd arm before spending 4 minutes
done

for ARM in $ARMS; do
    echo
    echo "############################################################"
    echo "# arm: $ARM   gates: $(arm_env "$ARM" | sed 's/^$/<none>/')"
    echo "############################################################"

    stop_server
    SERVER_LOG="$LOG_DIR/server_${ARM}.log"
    ARM="$ARM" "$REPO_ROOT/scripts/run_server.sh" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    disown $SERVER_PID 2>/dev/null || true
    if ! wait_for_server; then
        echo "ERROR: '$ARM' server failed to start; see $SERVER_LOG" >&2
        tail -30 "$SERVER_LOG" >&2
        exit 1
    fi

    # Record the resolved kernel backends next to the results. If this does not
    # say triton for decode/prefill/verify, the AITER hooks are not even on the
    # path -- they wrap the Triton kernel class, not the CuteDSL/FlashInfer ones.
    grep -m1 "Linear attention kernel backend" "$SERVER_LOG" || true
    grep -m1 "max_total_num_tokens" "$SERVER_LOG" || true

    ARM="$ARM" "$REPO_ROOT/scripts/run_client.sh"
    stop_server
done

echo
echo "############################################################"
echo "# comparison"
echo "############################################################"
for ARM in $ARMS; do
    [[ "$ARM" == "baseline" ]] && continue
    echo
    echo ">>> baseline vs $ARM"
    python3 "$REPO_ROOT/scripts/compare_arms.py" \
        --root "$RESULTS_DIR" --baseline baseline --patched "$ARM" || true
done

echo
echo "Reference numbers from the original MI355X run: $REPO_ROOT/RESULTS.md"
echo "Compare your own against them with:"
echo "  python3 scripts/compare_arms.py --root reference_results --baseline baseline --patched conv_decode"
