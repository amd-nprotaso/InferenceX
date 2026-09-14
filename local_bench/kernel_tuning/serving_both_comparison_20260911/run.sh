#!/usr/bin/env bash
set -euo pipefail
cd /var/home/my_InferenceX/InferenceX/local_bench
source ../benchmarks/benchmark_lib.sh
comparison_dir="$PWD/kernel_tuning/serving_both_comparison_20260911"
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 SKIP_DOWNLOAD=1
export TP=4 CONC=4 CONC_LIST=4 ISL=32768 OSL=1024
export RANDOM_RANGE_RATIO=0.8 NUM_PROMPTS_MULT=10 WARMUP_MULT=2
export SIMULATE_ACC=1 FLYDSL_COUNT_COMPILES=1
export SGLANG_FLYDSL_GDN_STATIC=0 SGLANG_FLYDSL_GDN_LOG_EVERY=500
server_pid=""
cleanup() {
    stop_background_process_tree "$server_pid" "comparison server" 60
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for backend in baseline both; do
    export FLYDSL_CONV=0 FLYDSL_GDN=0
    if [[ "$backend" == both ]]; then
        export FLYDSL_CONV=1 FLYDSL_GDN=1
    fi
    arm_dir="$comparison_dir/$backend"
    mkdir -p "$arm_dir"
    export RESULT_DIR="$arm_dir"
    export SGLANG_FLYDSL_GDN_STATS_DIR="$arm_dir/gdn_stats"
    echo "START $backend $(date -u +%FT%TZ)"
    bash kernel_tuning/run_server_both.sh > "$arm_dir/server.log" 2>&1 &
    server_pid=$!
    wait_for_ready --endpoint http://127.0.0.1:8888/health \
        --log "$arm_dir/server.log" --pid "$server_pid" \
        --sleep-interval 5 --timeout 1800 > "$arm_dir/startup.log" 2>&1
    for trial in 1 2 3; do
        export RESULT_DIR="$arm_dir/trial_$trial"
        echo "TRIAL $backend $trial $(date -u +%FT%TZ)"
        printf '\n[comparison] trial %s begin\n' "$trial" >> "$arm_dir/server.log"
        bash qwen3.5_fp4_sglang_bench.sh > "$arm_dir/trial_$trial-driver.log" 2>&1
        printf '\n[comparison] trial %s end\n' "$trial" >> "$arm_dir/server.log"
    done
    cleanup
    server_pid=""
    echo "DONE $backend $(date -u +%FT%TZ)"
done
