#!/usr/bin/env bash
set -euo pipefail

# FlyDSL GDN *decode* recurrence, end to end.
#
#   baseline   : FlyDSL causal conv (prefill) only          -- today's shipping arm
#   gdn_decode : the same, plus the FlyDSL decode recurrence
#
# So the only difference between the arms is
# fused_sigmoid_gating_delta_rule_update. The GDN prefill chunk_h kernel is OFF
# in both, deliberately: it is a separate change with its own A/B, and leaving
# it out keeps this measurement attributable to the decode kernel alone.
#
# Three output lengths, because the decode kernel's share of the run grows with
# OSL: at ISL 32768 the GDN decode block was 2.7% of GPU time at OSL 100 and
# 6.1% at OSL 1024.
#
# One launcher for both arms, so PYTHONPATH, sitecustomize and import order are
# identical and only the env flag varies.

cd /var/home/my_InferenceX/InferenceX/local_bench
source ../benchmarks/benchmark_lib.sh
comparison_dir="$PWD/kernel_tuning/serving_gdn_decode_20260917"
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 SKIP_DOWNLOAD=1
export TP=4 CONC=4 CONC_LIST=4 ISL=32768
export RANDOM_RANGE_RATIO=0.8 NUM_PROMPTS_MULT=10 WARMUP_MULT=2
export SIMULATE_ACC=1
# Conv on in both arms; GDN prefill chunk_h off in both.
export FLYDSL_CONV=1 FLYDSL_GDN=0 FLYDSL_MOE=0

OSL_LIST=${OSL_LIST:-"100 1024 2048"}
TRIALS=${TRIALS:-3}

server_pid=""
cleanup() { stop_background_process_tree "$server_pid" "comparison server" 60; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for backend in baseline gdn_decode; do
    export FLYDSL_GDN_DECODE=0
    if [[ "$backend" == gdn_decode ]]; then
        export FLYDSL_GDN_DECODE=1
    fi
    arm_dir="$comparison_dir/$backend"
    mkdir -p "$arm_dir"
    export RESULT_DIR="$arm_dir"
    # Proves the arm was live: dumped per pid at exit, audited by summarize.py.
    export SGLANG_FLYDSL_GDN_DECODE_STATS_DIR="$arm_dir/decode_stats"
    export SGLANG_FLYDSL_GDN_DECODE_LOG_EVERY=5000

    echo "START $backend $(date -u +%FT%TZ)"
    bash kernel_tuning/run_server_both.sh > "$arm_dir/server.log" 2>&1 &
    server_pid=$!
    wait_for_ready --endpoint http://127.0.0.1:8888/health \
        --log "$arm_dir/server.log" --pid "$server_pid" \
        --sleep-interval 5 --timeout 1800 > "$arm_dir/startup.log" 2>&1

    for osl in $OSL_LIST; do
        export OSL="$osl"
        for trial in $(seq 1 "$TRIALS"); do
            export RESULT_DIR="$arm_dir/osl${osl}/trial_${trial}"
            echo "TRIAL $backend osl=$osl $trial $(date -u +%FT%TZ)"
            printf '\n[cmp] %s osl=%s trial %s begin\n' "$backend" "$osl" "$trial" >> "$arm_dir/server.log"
            bash qwen3.5_fp4_sglang_bench.sh > "$arm_dir/osl${osl}-trial_${trial}-driver.log" 2>&1
            printf '\n[cmp] %s osl=%s trial %s end\n' "$backend" "$osl" "$trial" >> "$arm_dir/server.log"
        done
    done

    # SIGTERM, not SIGKILL: the usage counters are dumped from an atexit hook.
    cleanup
    server_pid=""
    echo "DONE $backend $(date -u +%FT%TZ)"
done
