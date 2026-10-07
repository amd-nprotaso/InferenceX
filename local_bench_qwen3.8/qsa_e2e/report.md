# Qwen3.8 unprofiled end-to-end QSA comparison

**English** | [中文](report_zh.md)

The N=128 CK QSA specialization did **not improve measured end-to-end throughput in this single A/B pair**: output throughput decreased 2.06%, from 152.55 to 149.40 tokens/s. The generated outputs and speculative acceptance differed, so this is not a clean estimate of the kernel change's causal effect.

## Configuration and procedure

Both arms used the supplied `/var/home/qwen3_8_profiling/server-start.sh` with `HF_HOME=/data/hf_cache/`. Model: `amd/Qwen3.8-Flash-Next-Quark-MXFP4`, snapshot `b2091c3809eb51902f46a163973362d8176f9e58`. Hardware: one MI355X, TP=1, concurrency=1. ISL=32768 and OSL=32768, NEXTN three steps/four draft tokens, chunked prefill=16384, stream interval=50. The exact original request payload was reproduced, with `temperature=0` and `ignore_eos=True`.

The launcher uses SGLang at `/workspace/sglang-qwen-next/python/sglang` and AITER at `/sgl-workspace/aiter`. The tuned library was rebuilt against this serving stack, not substituted from the earlier local-checkout benchmark. The baseline kernel source was verified byte-identical between the server build and isolated build.

Each arm started a fresh server, completed startup and graph capture, ran a 32768-input/32-output warmup, flushed the prefix cache, and measured one full request. Neither measured request used profiling. Both reported exactly 32768 input and output tokens, zero cached prompt tokens, no retractions, and a length-limit finish. Model loading, compilation and warmup time are excluded.

Baseline used the original installed QSA library. Tuned used `AITER_JIT_DIR` pointing to an isolated overlay: the QSA library was replaced, while the other available installed libraries were reused via symlinks. `/proc` mappings confirmed the intended QSA library in each scheduler. The original launcher and installed source were not edited.

The QSA change is the HIP template tile `sequence<128,64,32,256,32,256>` → `sequence<128,128,32,256,32,256>`. Query tile, wave layout, datatype, scale and masking are unchanged. N=128 was chosen because it won the earlier microbenchmark at the launcher's C=1 shape. Both baseline and tuned libraries passed a separate numerical check against FP32 attention before the serving comparison.

## Results

| Metric | Baseline | Tuned N=128 | Change |
| --- | ---: | ---: | ---: |
| Client-observed TTFT | 1.0326 s | 1.0304 s | −0.22% |
| End-to-end latency | 214.805 s | 219.332 s | +2.11% |
| TPOT | 6.5240 ms | 6.6622 ms | +2.12% |
| Output throughput | 152.547 tok/s | 149.399 tok/s | −2.06% |
| Decode interactivity | 153.280 tok/s | 150.100 tok/s | −2.07% |
| Mean speculative acceptance length | 3.2511 | 3.1348 | −3.58% |
| Speculative acceptance rate | 75.04% | 71.16% | −3.88 percentage points |
| Verification steps | 10079 | 10453 | +374 |

`(e2e − TTFT) / spec_verify_ct` is 21.210 ms for baseline and 20.884 ms for tuned, a 1.54% decrease. This is an aggregate proxy for time per verification cycle, including draft work and overhead, not a direct measurement of the QSA kernel. Different outputs/context trajectories limit its interpretation.

## Comparability and limitations

The short warmup outputs matched, but the full generated text did not. Both used greedy decoding; nevertheless, the launcher leaves the server random seed unspecified and picked 933737487 for baseline versus 995034812 for tuned. Other saved serving settings matched; recorded startup durations naturally differed. Neither the seed difference nor the kernel change has been established as the cause of output divergence. Numerical rounding and other nondeterministic execution may also matter.

This is one full request per arm. The lower tuned throughput coincides with more verification work due to reduced acceptance. These results do not justify a claim of an end-to-end speedup or a general performance regression from changing N alone. A repeated experiment with fixed server seeds and an appropriate workload suite is needed to separate kernel savings from generation variability. The long forced-output synthetic prompt is not a model-quality evaluation.

TTFT and TPOT follow the supplied client's streaming convention; stream interval=50 remains unchanged. All raw stream timestamps and final server metadata are retained.

The first baseline startup attempt reached readiness but faulted on the one-token generation triggered by `/health`; no measured request had started. Its log is preserved as `baseline/server_attempt1_health_fault.log`. The harness switched to checking the ready log marker and read-only `/get_server_info`, then retried the unchanged launcher. Both reported measurements are from successful runs using that same corrected readiness procedure.

## Artifacts and reproduction

- [Comparison CSV](comparison.csv), [comparison checks](comparison_checks.json), and [full comparison JSON](comparison.json).
- [Baseline result](baseline/client-exact-result.json) and [tuned result](tuned/client-exact-result.json).
- [Baseline loaded QSA](baseline/loaded_qsa.json) and [tuned loaded QSA](tuned/loaded_qsa.json).
- [Kernel patch](compiled/n128/kernel.patch), [build commands](compiled/n128/build.json), and [source match](source_match.json).
- [Client driver](run_client.py), [controlled switch driver](run_tuned.py), and [comparison script](compare.py).

For another pair, preserve existing output folders first. Start the original launcher for baseline, then run the client driver from the workspace. For tuned, set the absolute overlay path before invoking the same launcher:

```bash
export HF_HOME=/data/hf_cache/
export AITER_JIT_DIR=/var/home/my_InferenceX/InferenceX/local_bench_qwen3.8/qsa_e2e/tuned_jit
bash /var/home/qwen3_8_profiling/server-start.sh
```

The overlay is an experimental local build scoped to this QSA workload, not an installed production change. The benchmark-owned servers were stopped after the comparison.
