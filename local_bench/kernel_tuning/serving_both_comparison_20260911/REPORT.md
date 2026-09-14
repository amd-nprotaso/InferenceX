# Default versus both FlyDSL kernels

**English** | [中文](REPORT_zh.md)

Median output throughput changed **-0.18%**, from **579.46** to **578.42 tok/s**.

| Metric | Default | Both FlyDSL kernels | Change |
|---|---:|---:|---:|
| Output throughput (tok/s) | 579.459 | 578.416 | -0.180% |
| Duration (s) | 63.364 | 63.479 | +0.180% |
| Mean TTFT (ms) | 1509.163 | 1411.652 | -6.461% |
| Mean TPOT (ms) | 5.166 | 5.277 | +2.149% |
| Observed acceptance length | 3.401 | 3.386 | -0.435% |

Values are medians across three trials; latency rows use the median of each trial's mean.

| Trial | Default (tok/s) | Both kernels (tok/s) |
|---|---:|---:|
| 1 | 579.006 | 580.259 |
| 2 | 579.780 | 578.416 |
| 3 | 579.459 | 578.272 |

Measured on 2026-09-11 using `kernel_tuning/run_server_both.sh` for both arms: FLYDSL_CONV=FLYDSL_GDN=0 for default and both flags 1 for the two-kernel version. TP=4, concurrency=4, nominal input/output lengths 32768/1024, random range ratio 0.8, seed 42. Each trial used 40 measured and 8 warmup requests. All 240 measured requests succeeded, with exactly 1,184,722 input and 36,717 output tokens per trial and zero cache hits.

Both arms use simulated MTP acceptance length 3.39, so these are projected throughput measurements, not real-acceptance or quality results. The table also records observed acceptance, which can fluctuate under simulation. All three baseline trials ran first, then three trials with both kernels in a fresh server. They are repeated trials within one server session per backend, not interleaved independent restarts; small differences must be interpreted with that limitation. Loading and warmup are excluded from timed duration.

Compilation counting was enabled for both arms. GDN usage was logged every 500 calls. [summary.json](summary.json) retains compile counts, hook enablement counts, and saved GDN usage/fallback counters. [metadata.json](metadata.json) records source hashes and package versions. Each backend/trial directory contains its raw JSONL and client log; each backend also has the server log and resolved command.

[run.sh](run.sh) records the exact experiment. Use a new output directory before rerunning to avoid appending to existing JSONL results.

The throughput trial ranges overlap: this is effectively a tie, with no demonstrated end-to-end gain from enabling both kernels. TTFT improved while TPOT worsened; scheduling and simulated acceptance also varied.

All four TP workers recorded 6,030 GDN calls each, with 100% FlyDSL execution and zero fallbacks (24,120 total, including startup and warmup). Four convolution enablement messages were present. The compile count reached 132 during warmup and stayed there through all three timed trials. The default run recorded 112 calls: this counter wraps all FlyDSL compilation, including existing library kernels, not only the custom overrides. Both servers exited successfully and released GPU memory.
