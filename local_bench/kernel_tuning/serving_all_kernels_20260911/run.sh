#!/usr/bin/env bash
set -euo pipefail

# End-to-end serving A/B: stock SGLang vs all three FlyDSL kernels together.
#
#   baseline : stock SGLang/aiter, nothing patched
#   all      : FlyDSL causal conv (prefill) + FlyDSL GDN chunk_h (prefill)
#              + FlyDSL fused decode-MoE router/sort and the atomic stage-2
#                epilogue (decode)
#
# Shape is the one every previous study on this box used: ISL 32768 / OSL 1024,
# TP=4, concurrency 4, range ratio 0.8, 40 requests per trial after 8 warmups.
#
# Arms alternate order every round (baseline-first, all-first, baseline-first)
# so that GPU clock drift over the ~35 min run cannot be charged to one arm.
# Both arms come from the same launcher (kernel_tuning/run_server_both.sh) so
# PYTHONPATH, sitecustomize and import order are identical and only the kernel
# env flags differ.

cd /var/home/my_InferenceX/InferenceX/local_bench
bench_root="$PWD"
source ../benchmarks/benchmark_lib.sh
experiment_dir="$PWD/kernel_tuning/serving_all_kernels_20260911"

export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 SKIP_DOWNLOAD=1
export TP=4 CONC=4 CONC_LIST=4 ISL=32768 OSL=1024
export RANDOM_RANGE_RATIO=0.8 NUM_PROMPTS_MULT=10 WARMUP_MULT=2
export SIMULATE_ACC=1 FLYDSL_COUNT_COMPILES=1
export SGLANG_FLYDSL_GDN_STATIC=0 SGLANG_FLYDSL_GDN_LOG_EVERY=500
# Proves the decode-MoE fused kernel is actually being taken inside the
# benchmark window, not merely that the import hook landed.
export SGLANG_FLYDSL_MOE_LOG_EVERY=2000
export PORT=8888 HOST=127.0.0.1

ROUNDS=${ROUNDS:-3}
TRIALS=${TRIALS:-2}

server_pid=""
cleanup() { stop_background_process_tree "$server_pid" "all-kernels server" 60; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((round = 1; round <= ROUNDS; round++)); do
    order='baseline all'
    if (( round % 2 == 0 )); then order='all baseline'; fi
    for backend in $order; do
        export FLYDSL_CONV=0 FLYDSL_GDN=0 FLYDSL_MOE=0
        if [[ "$backend" == all ]]; then
            export FLYDSL_CONV=1 FLYDSL_GDN=1 FLYDSL_MOE=1
        fi

        arm_dir="$experiment_dir/round_$round/$backend"
        mkdir -p "$arm_dir"
        export RESULT_DIR="$arm_dir"
        export SGLANG_FLYDSL_GDN_STATS_DIR="$arm_dir/gdn_stats"

        echo "START round=$round backend=$backend $(date -u +%FT%TZ)"
        bash kernel_tuning/run_server_both.sh >> "$arm_dir/server.log" 2>&1 &
        server_pid=$!
        wait_for_ready --endpoint "http://$HOST:$PORT/health" \
            --log "$arm_dir/server.log" --pid "$server_pid" \
            --sleep-interval 5 --timeout 1800 > "$arm_dir/startup.log" 2>&1

        for ((trial = 1; trial <= TRIALS; trial++)); do
            export RESULT_DIR="$arm_dir/trial_$trial"
            echo "TRIAL round=$round backend=$backend trial=$trial $(date -u +%FT%TZ)"
            printf '\n[experiment] trial %s begin\n' "$trial" >> "$arm_dir/server.log"
            bash qwen3.5_fp4_sglang_bench.sh > "$arm_dir/trial_$trial-driver.log" 2>&1
            printf '\n[experiment] trial %s end\n' "$trial" >> "$arm_dir/server.log"
        done

        cleanup
        server_pid=""
        echo "DONE round=$round backend=$backend $(date -u +%FT%TZ)"
    done
done

echo "ALL DONE $(date -u +%FT%TZ)"
