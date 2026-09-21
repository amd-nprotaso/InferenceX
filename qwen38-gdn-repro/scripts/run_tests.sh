#!/usr/bin/env bash
set -euo pipefail

# Kernel accuracy tests. Needs a GPU and the patches applied, but NOT a server.
# Takes about 10 seconds.
#
#   ./run_tests.sh              # everything
#   ./run_tests.sh -k decode    # just the decode kernel
#   ./run_tests.sh -m "not slow"  # skip production-sized shapes
#
# Any extra arguments are passed straight through to pytest.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "$REPO_ROOT/tests"

if ! python3 -c "import aiter.ops.flydsl.linear_attention_kernels as m; m.flydsl_gdn_decode_varlen" 2>/dev/null; then
    echo "ERROR: the AITER GDN entry points are missing." >&2
    echo "       Run ./scripts/apply_patches.sh first." >&2
    exit 1
fi

exec python3 -m pytest -v --no-header -p no:cacheprovider "$@"
