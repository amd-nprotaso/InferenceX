#!/usr/bin/env bash
# Twin of run_server.sh that loads the instrumented hook in bootstrap_debug/
# instead of bootstrap/. Same env, same dispatch predicate; adds hit/miss
# counters written to $SGLANG_FLYDSL_GDN_STATS_DIR.
set -euo pipefail

gdn_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$gdn_dir/bootstrap_debug:$gdn_dir${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_FLYDSL_GDN_CHUNK_H=1
export SGLANG_FLYDSL_GDN_STATS_DIR="${SGLANG_FLYDSL_GDN_STATS_DIR:-/tmp/flydsl_gdn_stats}"
export SGLANG_FLYDSL_GDN_LOG_EVERY="${SGLANG_FLYDSL_GDN_LOG_EVERY:-200}"
exec bash "$gdn_dir/../../qwen3.5_fp4_sglang_server.sh" "$@"
