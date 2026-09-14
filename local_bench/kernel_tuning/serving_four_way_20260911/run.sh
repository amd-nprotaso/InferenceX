#!/usr/bin/env bash
set -euo pipefail
cd /var/home/my_InferenceX/InferenceX/local_bench
bench_root="$PWD"
source ../benchmarks/benchmark_lib.sh
experiment_dir="$PWD/kernel_tuning/serving_four_way_20260911"
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 SKIP_DOWNLOAD=1
export TP=4 CONC=4 CONC_LIST=4 ISL=32768 OSL=1024
export RANDOM_RANGE_RATIO=0.8 NUM_PROMPTS_MULT=10 WARMUP_MULT=2
export SIMULATE_ACC=1 FLYDSL_COUNT_COMPILES=1
export SGLANG_FLYDSL_GDN_STATIC=0 SGLANG_FLYDSL_GDN_LOG_EVERY=500
export PORT=8888 HOST=127.0.0.1
server_pid=""
cleanup() { stop_background_process_tree "$server_pid" "four-way server" 60; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for round in 1 2; do
    order='baseline conv gdn both'
    if [[ "$round" == 2 ]]; then order='both gdn conv baseline'; fi
    for backend in $order; do
        export FLYDSL_CONV=0 FLYDSL_GDN=0
        case "$backend" in
            conv) export FLYDSL_CONV=1 ;;
            gdn) export FLYDSL_GDN=1 ;;
            both) export FLYDSL_CONV=1 FLYDSL_GDN=1 ;;
        esac
        arm_dir="$experiment_dir/round_$round/$backend"
        mkdir -p "$arm_dir"
        export RESULT_DIR="$arm_dir"
        export SGLANG_FLYDSL_GDN_STATS_DIR="$arm_dir/gdn_stats"
        echo "START round=$round backend=$backend $(date -u +%FT%TZ)"
        bash kernel_tuning/run_server_both.sh >> "$arm_dir/server.log" 2>&1 &
        server_pid=$!
        wait_for_ready --endpoint http://127.0.0.1:8888/health \
            --log "$arm_dir/server.log" --pid "$server_pid" \
            --sleep-interval 5 --timeout 1800 > "$arm_dir/startup.log" 2>&1
        for trial in 1 2; do
            export RESULT_DIR="$arm_dir/trial_$trial"
            echo "TRIAL round=$round backend=$backend trial=$trial $(date -u +%FT%TZ)"
            printf '\n[experiment] trial %s begin\n' "$trial" >> "$arm_dir/server.log"
            bash qwen3.5_fp4_sglang_bench.sh > "$arm_dir/trial_$trial-driver.log" 2>&1
            printf '\n[experiment] trial %s end\n' "$trial" >> "$arm_dir/server.log"
        done
        if [[ "$round" == 2 ]]; then
            echo "PROFILE backend=$backend $(date -u +%FT%TZ)"
            printf '\n[experiment] profile begin\n' >> "$arm_dir/server.log"
            (
                cd "$arm_dir"
                NUM_PROMPTS_MULT=1 WARMUP_MULT=1 ACTIVITIES=GPU \
                    PROFILE_NUM_STEPS=12 PROFILE_DIR="$arm_dir/profile" \
                    bash "$bench_root/qwen3.5_fp4_sglang_profile.sh"
            ) > "$arm_dir/profile-driver.log" 2>&1
            printf '\n[experiment] profile end\n' >> "$arm_dir/server.log"
        fi
        cleanup
        server_pid=""
        echo "DONE round=$round backend=$backend $(date -u +%FT%TZ)"
    done
done
