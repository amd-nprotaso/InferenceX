# gfx950 QSA compiler flag tuning

[中文](report_zh.md)

No repeatable E2E gain demonstrated: tuned throughput was58.98/154.36 tok/s versus baseline148.49/151.08. The faster repeat also had higher speculative acceptance. Keep the installed baseline. Both servers have been stopped; all four output hashes differ despite equal seeds.

Selected `-enable-post-misched=1` after a six-configuration microbenchmark sweep. It reduced verification latency by3.8%, with decode latency essentially unchanged. Kernel source and M128 N64 tile arithmetic were unchanged. Higher occupancy hints and reduced argument preload were slower. This experiment excludes fast math and tile changes.

## Microbenchmark

MI355X/gfx950; installed serving AITER stack; BF16, Hq24, Hkv2, D256, length-one varlen queries, selected KV2048–2051, decode/verify rows1/4. Twelve cases per variant, 20 GPU-event samples around HIP graph replay. Values below are medians across context positions. All variants passed independent FP32 checks (atol0.003, rtol0.03), max absolute error0.000937313, and their output hashes matched baseline on identical seeded synthetic inputs. This finite input check is not a guarantee for every model activation.

`post_sched` changes LLVM post-machine scheduling from0 to1. `occupancy2/3` set CK's block-per-CU hint (automatic is1 for D256); actual occupancy remains resource-dependent. `preload0/16` change `--amdgpu-kernarg-preload-count` from32. FAST_EXP2 stays1. Candidate selection uses3×decode+12×verification latency.

| Variant | Decode µs | Verification µs | Bitwise equal to baseline |
|---|---:|---:|---|
| baseline | 125.672 | 147.665 | True |
| post_sched | 125.282 | 142.026 | True |
| occupancy2 | 164.714 | 171.109 | True |
| occupancy3 | 388.166 | 397.097 | True |
| preload0 | 128.498 | 148.141 | True |
| preload16 | 126.652 | 147.982 | True |

## E2E comparison

Model amd/Qwen3.8-Flash-Next-Quark-MXFP4, TP1, concurrency1, ISL32768, OSL32768; profiling disabled. Original launcher copied locally with equal --random-seed20260928 added for both arms; HF_HOME=/data/hf_cache/. Two sequential requests per arm, baseline first. Each request follows a32768/32 warmup and cache flush. Microbenchmark execution ended before measured E2E requests.

| Trial | E2E seconds | TTFT seconds | TPOT ms | Output tok/s | Acceptance length | Verification count |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 220.674 | 1.035 | 6.703 | 148.491 | 3.1502 | 10402 |
| baseline_repeat | 216.895 | 1.032 | 6.588 | 151.078 | 3.2069 | 10218 |
| tuned | 555.619 | 1.033 | 16.925 | 58.976 | 1.2464 | 26291 |
| tuned_repeat | 212.279 | 1.030 | 6.447 | 154.363 | 3.2628 | 10043 |

Baseline mean: 149.784 output tok/s; tuned mean: 106.669; change: -28.78%.

These are descriptive means of two trials, not a confidence interval or a causal estimate of the flag's performance effect. Check acceptance and individual request values before interpreting the average. All measured requests used identical payloads, actual32768/32768 token counts, cached_tokens0, no retractions, and matching launch arguments excluding dynamic telemetry. Loaded-library mappings are saved per trial. Output hashes and serving verification-time proxies are in comparison.json; the latter include all serving work, not just the QSA kernel. Fixed seed does not enforce deterministic serving.

## Artifacts

[Compiler flag diff](compiler_flags.patch), [micro sweep](sweep.py), [micro results](micro_summary.json), [E2E harness](e2e.py), [comparison](comparison.json), [CSV](trials.csv). Builds and compiler commands are under micro/<variant>/build.json. Only the selected translation unit is rebuilt and linked into a private cache. Original installed libraries and launcher are preserved.

Run sweep.py, e2e.py, compare.py, report.py in that order. Preserve this completed directory and use a fresh artifact directory for another run; the E2E harness does not overwrite repeat symlinks. Results apply to concurrency1 only.
