# Baseline QSA: HC combine split E2E comparison

[中文](report_zh.md)

The installed SGLang defines `SGLANG_HC_COMBINE_SPLIT = EnvBool(True)`: unset and `=1` enable the same setting. The separate /var/home/my_sglang/sglang checkout defaults to disabled on HIP, but it is not the serving stack used by the supplied launcher. An explicit `=0` arm is included to measure disabled versus enabled.

The primary enabled trials averaged154.048 output tok/s versus137.940 disabled (disabled10.46% lower). Decode elapsed time per verification was21.068–21.073ms enabled versus22.119–22.164ms disabled, about5.1% higher when disabled. Keep split combine enabled on this installed stack. These two-trial E2E means are affected by generated trajectories and acceptance; they do not establish a precise repeatable speedup. All six output hashes differ. All servers are stopped.

## Setup

MI355X/gfx950, model amd/Qwen3.8-Flash-Next-Quark-MXFP4, TP1, concurrency1, ISL32768, OSL32768. All arms use the installed baseline QSA module under /sgl-workspace/aiter/aiter/jit; AITER_JIT_DIR is removed from the launch environment, so none of the tuned experimental libraries are used. SGLang imports from /workspace/sglang-qwen-next/python/sglang. The original launcher is copied with equal --random-seed20260928 added, HF_HOME=/data/hf_cache/; no profiling. All other launch settings remain equal.

Two sequential requests per setting, in order unset, enabled, disabled; each setting starts a fresh server. Each measured request follows a32768/32 warmup and cache flush. The flag is explicitly removed for unset, set to1 for enabled, and set to0 for disabled. Runtime probes and scheduler environment checks confirm these values; per-trial library mappings verify the baseline QSA library.

The installed hyperconnection.py reads this flag at construction. hc_count4 and hidden_size2560 satisfy its split-shape checks. Eligible BF16/FP16 batches with at most32 rows select hc_combine_split when enabled; disabled selects hc_combine. Larger batches use the unsplit path regardless. This source-based routing explanation is not a kernel trace; measured E2E requests were unprofiled.

## Results

| Setting | Trial | E2E seconds | TTFT seconds | TPOT ms | Output tok/s | Acceptance length | Verification count |
|---|---:|---:|---:|---:|---:|---:|---:|
| unset | 1 | 255.215 | 1.030 | 7.757 | 128.394 | 2.7241 | 12029 |
| unset | 2 | 196.990 | 1.032 | 5.980 | 166.343 | 3.5410 | 9254 |
| enabled | 1 | 200.648 | 1.032 | 6.092 | 163.311 | 3.4584 | 9475 |
| enabled | 2 | 226.322 | 1.029 | 6.876 | 144.785 | 3.0650 | 10691 |
| disabled | 1 | 219.527 | 1.035 | 6.668 | 149.266 | 3.3173 | 9878 |
| disabled | 2 | 258.804 | 1.033 | 7.867 | 126.613 | 2.8175 | 11630 |

| Setting | Mean per-request output tok/s |
|---|---:|
| unset | 147.368 |
| enabled | 154.048 |
| disabled | 137.940 |

Enabled versus unset: +4.53%. Disabled versus enabled: -10.46%.

Means are arithmetic means of two per-request throughput values. The unset-versus-enabled difference cannot be attributed to enabling this flag because both settings resolve to True. Two trials per setting and differing speculative acceptance limit causal interpretation of enabled-versus-disabled differences. Individual acceptance lengths, verification counts, output hashes, and decode-time-per-verification proxies are preserved in comparison.json; the proxy includes all serving work and is not an HC kernel timer. A fixed seed does not guarantee deterministic serving outputs.

Validation checks exact32768/32768 actual token counts, successful completion, cached_tokens0, no retractions, matching request payloads, equal server arguments excluding dynamic telemetry, flag values, and baseline library mappings. No model accuracy evaluation is included. This comparison covers concurrency1 only.

## Harness audit recovery

The first enabled request completed in244.079s at134.251 tok/s, but a post-run harness assertion failed: setproctitle erased the flag from the OS environment snapshot while Python still read its value correctly. A CPU-only reproduction confirmed this. The run is preserved in enabled_audit_interrupted and excluded from the primary two-trial table. The enabled pair was restarted; enabled/disabled runs set SPT_NOENV=1 to preserve environment snapshots during process renaming. This changes process-title bookkeeping, not the HC flag or model computation. Initial unset trials are retained. Both original and resumed orchestration logs are preserved.

## Artifacts and reproduction

[Launcher](server-start.sh), [launcher patch](launcher.patch), [harness](e2e.py), [comparison script](compare.py), [full comparison](comparison.json), [CSV](trials.csv). Each setting directory contains server/client logs, flag_probe.json, server_info.json, warmup, request, output text, results, and library mappings. Repeat directories share their setting's server log.

Run e2e.py, compare.py, report.py in order from this directory or via their paths. Use a fresh artifact directory for another run and preserve these results; existing repeat symlinks are not overwritten. Original launcher and installed QSA libraries are preserved.
