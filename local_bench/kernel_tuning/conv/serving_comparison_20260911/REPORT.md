# TP4 serving comparison: cached FlyDSL convolution

**English** | [中文](REPORT_zh.md)

Measured on 2026-09-11. Median output throughput increased **0.43%**, from **578.32** to **580.83 tok/s**. The trial ranges overlap; this small difference does not establish a statistically reliable speedup. The earlier large slowdown is absent in this comparison.

| Metric | Triton baseline | FlyDSL convolution | Change |
|---|---:|---:|---:|
| Output throughput (tok/s) | 578.324 | 580.826 | +0.433% |
| Duration (s) | 63.489 | 63.215 | -0.431% |
| Mean TTFT (ms) | 1421.255 | 1422.779 | +0.107% |
| Mean TPOT (ms) | 5.275 | 5.222 | -1.006% |

Values are medians across three trials; latency rows use the median of each trial's mean.

| Trial | Triton (tok/s) | FlyDSL (tok/s) |
|---|---:|---:|
| 1 | 576.753 | 580.826 |
| 2 | 578.324 | 580.553 |
| 3 | 581.691 | 581.150 |

All three baseline trials ran first, then all three FlyDSL trials on a fresh server. These are repeated runs within one server session per backend, not independent interleaved restarts; startup/order effects are not controlled.

TP=4, concurrency=4, nominal 32768 input / 1024 output tokens, random range ratio 0.8, seed 42. Each trial used 40 measured requests and 8 warmup requests. All 240 measured requests succeeded. Every trial consumed 1,184,722 input tokens and produced 36,717 output tokens; the range ratio varies actual request lengths. Cache hit rate was zero. Model loading and warmup are excluded from duration.

Both arms used simulated MTP acceptance length 3.39, the server script's default. These are projected throughput measurements, not real-acceptance or model-quality results. GDN chunk_h FlyDSL override was disabled in both arms. The only override difference was SGLANG_FLYDSL_CAUSAL_CONV=0/1. Four worker enablement messages appeared for FlyDSL and none for baseline. The installed prefill path calls the patched module function; no GPU profile was captured in this experiment.

Both run helpers exited successfully and released GPU memory. Server teardown logs contain cancellation/SystemExit tracebacks after benchmarking; these are not measured-request failures.

[Metadata and source hashes](metadata.json), [structured summary](summary.json), and backend subdirectories retain raw JSONL, server logs, and resolved server commands. Client .log files are overwritten between same-concurrency trials by the existing script; driver logs retain all three trials and JSONL appends all three records.

Reproduction commands (use a new RESULT_DIR to avoid appending to existing measurements):

```bash
# Run from local_bench. Use backend=baseline or backend=flydsl.
backend=baseline
override=0
if [ "$backend" = flydsl ]; then override=1; fi
env HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 SKIP_DOWNLOAD=1 \
  TP=4 CONC=4 CONC_LIST='4 4 4' ISL=32768 OSL=1024 \
  RANDOM_RANGE_RATIO=0.8 NUM_PROMPTS_MULT=10 WARMUP_MULT=2 \
  SIMULATE_ACC=1 GPU_MONITOR=0 SERVER_READY_TIMEOUT=5 \
  SGLANG_FLYDSL_GDN_CHUNK_H=0 SGLANG_FLYDSL_CAUSAL_CONV="$override" \
  PYTHONPATH="$PWD/kernel_tuning/conv/bootstrap:$PWD/kernel_tuning/conv:${PYTHONPATH:-}" \
  RESULT_DIR="$PWD/kernel_tuning/conv/serving_comparison_20260911/$backend" \
  bash qwen3.5_fp4_sglang_run.sh
```
