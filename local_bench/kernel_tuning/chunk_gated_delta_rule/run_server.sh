#!/usr/bin/env bash
set -euo pipefail

gdn_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$gdn_dir/bootstrap:$gdn_dir${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_FLYDSL_GDN_CHUNK_H=1
exec bash "$gdn_dir/../../qwen3.5_fp4_sglang_server.sh" "$@"
