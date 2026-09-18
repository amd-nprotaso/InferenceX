#!/usr/bin/env bash
set -euo pipefail

# Enable both FlyDSL prefill overrides in the server and its Python workers.
# Usage: TP=4 CONC=4 SIMULATE_ACC=0 bash kernel_tuning/run_both_server.sh
kernel_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
combined_bootstrap=$(mktemp -d /tmp/sglang-both-kernels.XXXXXX)
trap 'rm -f -- "$combined_bootstrap/sitecustomize.py"; rmdir -- "$combined_bootstrap"' EXIT

cat > "$combined_bootstrap/sitecustomize.py" <<'PY'
import os
import runpy
from pathlib import Path

root = Path(os.environ["SGLANG_LOCAL_KERNEL_ROOT"])
for name in ("chunk_gated_delta_rule", "conv"):
    runpy.run_path(str(root / name / "bootstrap" / "sitecustomize.py"))
PY

export SGLANG_LOCAL_KERNEL_ROOT="$kernel_root"
export PYTHONPATH="$combined_bootstrap:$kernel_root/chunk_gated_delta_rule:$kernel_root/conv${PYTHONPATH:+:$PYTHONPATH}"
# Keep the temporary bootstrap free of bytecode so it can be removed on exit.
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_FLYDSL_GDN_CHUNK_H=1
export SGLANG_FLYDSL_CAUSAL_CONV=1

bash "$kernel_root/../qwen3.5_fp4_sglang_server.sh" "$@"
