# gfx950 larger-KV FMHA experiment

[中文](report_zh.md)

The larger-KV configuration improved isolated kernel latency but regressed E2E throughput in both trials. Keep the baseline serving configuration.

Tested on MI355X/gfx950 using amd/Qwen3.8-Flash-Next-Quark-MXFP4, TP=1, concurrency=1, ISL=32768, OSL=32768. The local launcher copies /var/home/qwen3_8_profiling/server-start.sh and adds the same --random-seed 20260928 to both arms; HF_HOME=/data/hf_cache/. E2E profiling was disabled. Each arm received two sequential measured requests, each preceded by a 32768/32 warmup and cache flush. Baseline ran first, then tuned.

## Kernel configuration

Baseline block tile: sequence<128,64,32,256,32,256>; tuned: sequence<64,256,32,256,32,256>. Both retain four waves; both MFMA shapes change from 32x32x16 to 16x16x16. This is the gfx950 BlockFmhaPipelineQRKSVS BF16 D256 specialization. Only one translation unit was rebuilt and relinked into an isolated AITER library. The dispatch trait is intentionally retained for this experiment; this is not a general production dispatch change.

The microbenchmark uses Hq=24, Hkv=2, D=256, query length1 per varlen row, selected KV2048–2051, decode/verification rows1/4 at concurrency1. Seven variants × twelve cases passed the independent FP32 reference check (atol0.003, rtol0.03); maximum absolute error was 0.000937313. Timing uses HIP graphs, 20 samples, with medians across context positions shown below. Micro tests finished before E2E timing.

| Variant | Decode (µs) | Verification (µs) |
|---|---:|---:|
| baseline | 128.184 | 147.982 |
| n128 | 107.058 | 113.885 |
| n256 | 158.820 | 171.083 |
| n512 | 187.702 | 238.600 |
| m64_n256 | 111.823 | 119.617 |
| m32_n256 | 226.080 | 246.720 |
| m16_n256 | 127.297 | 134.874 |

M64 N256 reduced decode latency by 12.8% and verification latency by 19.2% against baseline. Increasing only N to256 or512 was slower. M128 N128 remains faster than the selected larger-KV candidate in this microbenchmark.

## E2E results

| Trial | E2E (s) | TTFT (s) | TPOT (ms) | Output tok/s | Acceptance length | Verification steps |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 201.372 | 1.037 | 6.114 | 162.724 | 3.4863 | 9399 |
| baseline_repeat | 209.080 | 1.033 | 6.349 | 156.724 | 3.3450 | 9796 |
| tuned | 567.928 | 1.032 | 17.301 | 57.697 | 1.1939 | 27446 |
| tuned_repeat | 233.952 | 1.029 | 7.108 | 140.063 | 2.9208 | 11219 |

Arithmetic mean per-request output throughput: baseline159.724 tok/s, tuned98.880 tok/s (-38.09%). Mean latency:205.226s versus400.940s (+95.37%). These are descriptive two-trial means, not a reliable estimate of a repeatable kernel-caused effect; the large spread is visible above.

All four request payloads match, actual token counts are32768/32768, cached_tokens=0, num_retractions=0, and launch arguments match after excluding dynamic startup/internal-state telemetry. Library mappings verify original baseline and rebuilt m64_n256 for both respective trials. Output text hashes differ in all four trials, including within baseline despite fixed seeds. Model accuracy was not evaluated.

The tuned first trial had acceptance length1.194 and27446 verification steps versus baseline3.345–3.486 and9399–9796 steps. Decode elapsed time divided by verification count was20.65–20.76ms tuned versus21.24–21.31ms baseline; this is a serving proxy, not a kernel timer. Additional verification steps outweighed faster kernel execution. These observations do not establish why the generated trajectories diverged or prove that the tile change alone caused the acceptance collapse.

Both experiment servers were stopped; original installed libraries and launcher were preserved. No E2E gain demonstrated. This experiment covers concurrency1 only, not4/8/16.

## Artifacts and reproduction

- [Kernel patch](micro/m64_n256/kernel.patch), [kernel source](micro/m64_n256/kernel.cpp)
- [Micro sweep](sweep.py), [micro results](micro_summary.json)
- [E2E harness](e2e.py), [launcher patch](launcher.patch)
- [Comparison script](compare.py), [full comparison](comparison.json), [trial CSV](trials.csv)

Run `python3 qsa_largekv/sweep.py`, then `python3 qsa_largekv/e2e.py`, then `python3 qsa_largekv/compare.py` from the parent experiment workspace. Preserve this run directory first and use fresh artifact directories for another experiment; existing repeat symlinks are not overwritten by the harness.
