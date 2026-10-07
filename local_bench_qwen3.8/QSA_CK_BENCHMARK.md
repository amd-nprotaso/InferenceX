# Qwen3.8 CK QSA microbenchmark

**English** | [中文](QSA_CK_BENCHMARK_zh.md)

`bench_qsa_ck.py` benchmarks the AITER attention call after SGLang compacts selected KV. It uses the local checkout `/var/home/my_aiter/aiter` and builds needed modules in `.qsa_aiter_jit` against the active ROCm PyTorch. The existing attention binary in that checkout failed to load with PyTorch 2.11 because of an undefined `c10::hip::getCurrentHIPStream` symbol; the separate cache resolves this without replacing it.

```bash
python3 bench_qsa_ck.py --isl 32678 --osl 32678 --conc 1,4,8,16 \
  --profile --output qsa_ck_results
```

Run from this directory. `--dry-run` prints the shape matrix without loading PyTorch or using a GPU. `--device` selects a visible GPU. `--modes decode` or `--modes verify` restricts the sweep. `--aiter-path`, `--jit-dir`, and `--trace-dir` can override local paths. Use distinct output directories to preserve older measurements.

## Evidence and shapes

The supplied `/var/home/qwen_trace_analysis/report.md` identifies TP=1, concurrency=1, ISL=32768, OSL=32768, three NEXTN steps and four verification tokens. The script preserves the requested **32678**, not the trace's **32768**. The report derives model dimensions from the cached model configuration; AMD kernel events do not contain input tensor shapes. Concurrencies 4, 8 and 16 are source-based extrapolations, not captured trace cases.

The local SGLang sources are authoritative for reconstruction:

- `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`, `_forward_paged_attention`: compacts KV and calls `flash_attn_varlen_func` with `max_seqlen_q=1`, `max_seqlen_k=topk`, `causal=True` and the layer's scale.
- `python/sglang/srt/layers/attention/qsa/qsa_indexer.py`: token budget 2048 and compression ratio 4; expanded index width is 2051.
- `python/sglang/srt/layers/attention/qsa/kernel.py`: appends the incomplete compression group after selected complete groups.
- AITER `aiter/ops/mha.py`: BF16 D=256 falls through to CK varlen rather than the assembly v3 path.

For TP=1 the tensors are BF16, with 24 query heads, two KV heads, D=256 and scale=1/16. Decode uses `R=CONC`; four-token verification uses `R=4*CONC`. Each verification token is a separate length-one query sequence, with its own packed KV segment. This is not one length-four query against a shared KV segment.

| Tensor/parameter | Value |
| --- | --- |
| Q | `[R,24,256]` |
| K and V capacity | `[R*2051,2,256]` each |
| `cu_seqlens_q` | `[0,1,...,R]`, int32 |
| `cu_seqlens_k` | prefix sum of valid selected lengths, int32 |
| Valid selected length at long context | `2048 + visible_tokens % 4` |
| `max_seqlen_q`, `max_seqlen_k` | 1, 2051 |
| Causal | True; one query can attend to all its compacted keys |

The matrix samples the first four decode context lengths, midpoint and endpoint. Verification rows advance from each sampled context through four positions; these are synthetic verification windows, including possible speculative positions beyond the requested final output. Once the sparse budget is saturated, full context growth changes the tail residue but does not make this kernel read 32K–65K KV tokens. OSL selects context samples; it does not run that many decode steps. Prefill uses different sparse kernels and is outside this benchmark.

## Validation and measurements

Every case checks AITER output against independently computed FP32 softmax attention before timing. Random packed Q/K/V values are used; the original selected indices and tensors were not recorded. Inputs are already compacted, so indexer, top-k, KV extraction, model layers and request scheduling are excluded.

Timing uses GPU events around a graph containing 20 attention calls, repeated 30 times after warmup. It reports median, minimum and p95 of per-call graph timing. These timings include kernels issued by the AITER API and graph execution costs; they are not isolated profiler kernel durations or end-to-end serving throughput. Inputs are reused, representing a warm-cache microbenchmark.

`--profile` separately records one call per case, requires a CK `FmhaFwdKernel` event, stores its duration, and records whether the complete symbol matches the supplied trace. Profiler timing is separate from the graph timing. `metadata.json` records environment, arguments and original trace phase statistics; `results.jsonl` stores shapes, samples, numerical error and symbol checks; `profile_*.json` files can be opened in Perfetto.

The original trace's target-verification kernel median was 149.279 microseconds, with 12 calls per cycle. Draft and draft-extension medians were 147.360 and 151.2795 microseconds. These are profiled C=1 measurements under a full model workload, and should not be treated as directly interchangeable with isolated graph timing.

## Local results (2026-09-28)

All 48 cases passed the FP32 reference check and matched the complete trace kernel symbol. Ranges below span the six context samples and report per-call graph medians in microseconds.

| CONC | Decode | Verify (4 tokens/request) |
| --- | --- | --- |
| 1 | 126.2–130.9 | 147.8–148.0 |
| 4 | 143.4–148.0 | 293.5–294.1 |
| 8 | 144.2–148.9 | 441.6–443.0 |
| 16 | 285.0–294.4 | 879.2–883.1 |

[Raw results](qsa_ck_results/results.jsonl) · [Summary CSV](qsa_ck_results/summary.csv) · [Environment](qsa_ck_results/metadata.json)
