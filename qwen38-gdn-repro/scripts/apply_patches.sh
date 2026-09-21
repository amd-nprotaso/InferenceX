#!/usr/bin/env bash
set -euo pipefail

# Apply (or check, or revert) the kernel patches against the SGLang and AITER
# checkouts named in env.sh.
#
#   ./apply_patches.sh            # apply 0001-0005
#   ./apply_patches.sh --debug    # apply 0001-0005 + optional 0006 diagnostics
#   ./apply_patches.sh --check    # dry run, leaves the tree unchanged
#   ./apply_patches.sh --revert   # undo them
#
# The patches are plain `git apply` diffs, not commits. Both checkouts may carry
# unrelated local edits -- this touches only the GDN files.
#
# The series is ORDER-DEPENDENT: 0005 edits a function 0001 introduces, and 0006
# edits dispatch helpers 0003/0004 introduce. So each patch is checked against
# the tree as the previous one left it, not against the pristine tree -- which is
# also why --check has to apply and then roll back rather than dry-run all five.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

MODE=${1:-apply}

# "<checkout>|<patch file>", in dependency order.
SERIES=(
    "$AITER_DIR|$REPO_ROOT/patches/0001-aiter-flydsl-gdn-decode-varlen.patch"
    "$AITER_DIR|$REPO_ROOT/patches/0002-aiter-flydsl-causal-conv1d-prefill.patch"
    "$AITER_DIR|$REPO_ROOT/patches/0005-aiter-gdn-decode-accept-fused-qkv-views.patch"
    "$SGLANG_DIR|$REPO_ROOT/patches/0003-sglang-route-gdn-decode-to-aiter-flydsl.patch"
    "$SGLANG_DIR|$REPO_ROOT/patches/0004-sglang-route-gdn-prefill-conv-to-aiter-flydsl.patch"
)
if [[ "$MODE" == "--debug" ]]; then
    SERIES+=("$SGLANG_DIR|$REPO_ROOT/patches/optional/0006-sglang-aiter-gdn-debug-logging.patch")
fi

for d in "$AITER_DIR" "$SGLANG_DIR"; do
    [[ -d "$d/.git" ]] || { echo "ERROR: $d is not a git checkout" >&2; exit 1; }
done

APPLIED=()   # entries already on the tree, for rollback

unapply_all() {
    for ((i=${#APPLIED[@]}-1; i>=0; i--)); do
        local dir="${APPLIED[$i]%%|*}" p="${APPLIED[$i]#*|}"
        echo "  [$(basename "$dir")] revert $(basename "$p")"
        git -C "$dir" apply --reverse "$p" || echo "    WARNING: could not revert $(basename "$p")" >&2
    done
    APPLIED=()
}

apply_series() {
    local entry dir p
    for entry in "${SERIES[@]}"; do
        dir="${entry%%|*}"; p="${entry#*|}"
        echo "  [$(basename "$dir")] apply $(basename "$p")"
        if ! git -C "$dir" apply "$p"; then
            echo >&2
            echo "ERROR: $(basename "$p") did not apply. Rolling back." >&2
            unapply_all
            exit 1
        fi
        APPLIED+=("$entry")
    done
}

case "$MODE" in
    --check)
        echo "== dry run: applying the series, then rolling it back =="
        apply_series
        echo "== rolling back =="
        unapply_all
        echo "OK: the whole series applies cleanly; tree left unchanged"
        ;;
    --revert)
        echo "== reverting =="
        APPLIED=("${SERIES[@]}")
        unapply_all
        echo "OK: reverted"
        ;;
    apply|--debug)
        echo "== applying =="
        apply_series
        echo
        echo "OK: applied."
        [[ "$MODE" == "--debug" ]] && echo "     diagnostics included (SGLANG_AITER_GDN_DEBUG=1 to enable)."
        echo "     SGLang and AITER are imported from source here -- no rebuild needed,"
        echo "     but restart any running server for the change to take effect."
        echo "     Next: ./scripts/verify_kernels.sh"
        ;;
    *)
        echo "usage: $0 [apply|--debug|--check|--revert]" >&2; exit 2 ;;
esac
