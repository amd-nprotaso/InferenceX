# Four-way serving experiment at 32K/1K

**English** | [中文](REPORT_zh.md)

| Version | Median tok/s | Change | Trial range (tok/s) | TTFT (ms) | TPOT (ms) | Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 578.29 | +0.00% | 575.22–581.66 | 1456.67 | 5.236 | 3.3968 |
| conv | 578.89 | +0.10% | 577.27–582.14 | 1521.65 | 5.153 | 3.3933 |
| gdn | 575.82 | -0.43% | 571.48–578.03 | 1399.38 | 5.312 | 3.3892 |
| both | 579.47 | +0.20% | 578.36–583.66 | 1415.12 | 5.258 | 3.3788 |

Each entry summarizes four throughput trials from two fresh server sessions. Latency and acceptance columns are medians of the per-trial mean metrics. Trial ranges describe the observed samples, not confidence intervals.

| Version | Round 1 vs default | Round 2 vs default |
|---|---:|---:|
| baseline | +0.00% | +0.00% |
| conv | +0.20% | +0.12% |
| gdn | -0.21% | -0.85% |
| both | +0.73% | -0.08% |

Round comparisons use the mean throughput of each version's two trials divided by that round's default mean.

TP=4, concurrency=4, ISL=32768, OSL=1024, random range ratio 0.8, client seed 42. Each throughput trial contains 40 measured requests and 8 warmup requests, with cache flushing enabled. All 640 measured requests succeeded and had identical aggregate input/output token counts and zero cache hits. Model loading and warmup are excluded from measured duration.

Round 1 runs default, convolution only, GDN only, then both. Round 2 reverses the order. Each session starts a fresh server through `run_server_both.sh`, setting FLYDSL_CONV and FLYDSL_GDN for the selected arm. The default keeps the original server's existing tuning and disables the two custom overrides. Mirroring the order reduces a linear time trend but does not eliminate all scheduling, clock, or acceptance variation. Four trials within two sessions are not four independent server replications.

The scripts' simulated MTP acceptance target remains 3.39. Actual sampled acceptance is reported above; these are simulated-acceptance throughput results, not a quality evaluation. Neither TTFT nor TPOT is a direct kernel timer.

After each round-2 session's throughput trials, a separate GPU-only profile runs four measured requests plus four warmup requests, still at ISL=32768 and OSL=1024. Capture is bounded to 12 forward steps. Profile timings must not be treated as throughput measurements.

See [summary.json](summary.json), [metadata.json](metadata.json), and [run.sh](run.sh). Raw results and logs are organized by round, version, and trial. Compilation counts covering a trial include its warmup; startup and profiler compilations are separate. Do not append new measurements to these result directories when reproducing.

## Findings and profile evidence

Both kernels together measured +0.20% median throughput versus default, convolution-only +0.10%, and GDN-only -0.43%. The ranges overlap and the combined result changes from +0.73% in round 1 to -0.08% in round 2 when comparing round means. This is not a robust end-to-end win. GDN-only was slightly lower in both rounds, despite its faster profiled GPU kernel; these data do not isolate the remaining host, scheduling, and acceptance effects.

| Profiled kernel | Default median (µs) | Both kernels median (µs) | Ratio |
|---|---:|---:|---:|
| Causal prefill convolution | 240.49 | 103.12 | 2.33× |
| GDN chunk_h | 849.74 | 783.23 | 1.08× |

The table uses the median across the four TP ranks' per-call medians. Every trace contains 180 convolution calls and 180 GDN chunk_h calls. All profiles use the same nominal request lengths, but packing differs slightly between versions, so these are sampled serving profiles, not an exact-shape microbenchmark. Custom kernels both appear as `kernel_0`; convolution is identified by block [128,1,1] and channel-grid extent 12 or 24, GDN by block [256,1,1] and grid prefix [16,8]. Original kernels are `_causal_conv1d_fwd_kernel` and `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`.

There are isolated approximately 236 ms convolution events in the GDN-only TP3 and both-kernels TP0 traces. Their cause was not determined. They remain in [profile_summary.json](profile_summary.json); medians are used without deleting samples, and summed durations are not presented as throughput savings. Other large operations in the traces include NCCL collectives and matrix/MoE kernels. Faster execution of these two prefill operations therefore need not produce a large improvement in the full 1K-output workload. Sampled acceptance differs between versions, as shown in the throughput table. No causal claim about host overhead can be made from these GPU-focused captures alone.

All 640 throughput requests succeeded, every trial had 1,184,722 input tokens and 36,717 output tokens, and cache hits were zero. GDN-enabled sessions recorded 68,220 calls in total (including startup, warmup, throughput, and profiling), all using FlyDSL with zero fallbacks. All eight sessions had zero compilation calls after the post-warmup cache flush in either timed trial. The counter includes existing AITER FlyDSL code, not only the custom kernels.

The initial GPU snapshot was idle. Some immediate server-handoff snapshots still show residual allocations while no GPU-using KFD process is listed. After model loading, all eight sessions report the same 225.26–225.38 GB available after draft loading, consistent with those allocations having cleared before measurement. Final GPU allocation returned to approximately 298 MB per device. Source hashes remained unchanged. No competing MoE workload was observed in this rerun; continuous exclusive-access enforcement was not used.

See [validation.json](validation.json) and [profile_summary.json](profile_summary.json). All servers have stopped. The interrupted experiment remains separate and is excluded from these results.
