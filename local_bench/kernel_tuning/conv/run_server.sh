#!/usr/bin/env bash
set -euo pipefail

# Apply the prefill override in every spawned Python worker.
conv_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$conv_dir/bootstrap:$conv_dir${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_FLYDSL_CAUSAL_CONV=1
exec bash "$conv_dir/../../qwen3.5_fp4_sglang_server.sh" "$@"
