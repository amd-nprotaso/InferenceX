# Stock SGLang vs all three FlyDSL kernels — ISL 32768 / OSL 1024

Qwen3.5-397B-A17B-MXFP4, MI355X (gfx950), TP=4, concurrency 4, EAGLE MTP with
4 draft tokens, `SIMULATE_ACC=1`. 3 rounds × 2 trials per arm (6 trials each),
40 requests + 8 warmups per trial, `RANDOM_RANGE_RATIO=0.8`. Arm order
alternates every round.

| arm | what is patched |
|---|---|
| `baseline` | nothing — stock SGLang/aiter |
| `all` | FlyDSL causal conv (prefill) + FlyDSL GDN chunk_h (prefill) + FlyDSL fused decode-MoE router/sort + atomic stage-2 epilogue (decode) |

Both arms boot from the same launcher (`kernel_tuning/run_server_both.sh`), so
`PYTHONPATH`, `sitecustomize` and import order are identical and only the kernel
env flags differ.

## Result

**+0.78 % output throughput.** Small, but consistent in sign and size across all
three rounds and roughly 2× the trial-to-trial spread.

| metric | baseline | all | delta | % |
|---|---:|---:|---:|---:|
| output tok/s | 577.22 | 581.70 | +4.49 | **+0.78 %** |
| total tok/s | 19201.9 | 19351.2 | +149.2 | +0.78 % |
| duration s | 63.61 | 63.12 | −0.49 | −0.77 % |
| TPOT ms | 5.253 | 5.213 | −0.040 | −0.76 % |
| median ITL ms | 3.741 | 3.585 | −0.156 | **−4.17 %** |
| TTFT ms | 1442.6 | 1437.9 | −4.7 | −0.32 % |
| E2E ms | 6267.1 | 6228.9 | −38.2 | −0.61 % |
| accept length | 3.393 | 3.391 | −0.001 | −0.04 % |

Paired per round (each round runs both arms, order alternating):

| metric | round 1 | round 2 | round 3 | mean |
|---|---:|---:|---:|---:|
| output tok/s | +0.90 % | +0.43 % | +1.00 % | +0.78 % |
| TPOT ms | −0.95 % | −0.24 % | −1.09 % | −0.76 % |
| median ITL ms | −4.01 % | −4.59 % | −3.91 % | −4.17 % |
| TTFT ms | −0.81 % | +0.43 % | −0.56 % | −0.31 % |

Trial-to-trial spread (stdev/mean, pooled over 6 trials):
output tok/s 0.36 % / 0.33 %, TPOT 1.35 % / 2.03 %, TTFT 5.23 % / 5.68 %.

### Reading it

- The win is **decode-side**. TPOT −0.76 % and output throughput +0.78 % are the
  same effect (at fixed concurrency throughput ≈ 1/TPOT). Median ITL moves much
  further, −4.2 %, and does so identically in all three rounds.
- **TTFT is flat** (−0.3 %, against 5.2 % spread). The two prefill kernels (conv,
  GDN chunk_h) contribute nothing measurable here. That matches the earlier
  finding that GDN chunk_h wins on the kernel and vanishes end-to-end.
- **Accept length is pinned** at 3.393 vs 3.391, so the number of decode passes
  for a fixed 1024 output tokens is not a confounder.
- The MoE kernel-level win is −7.9 µs/layer × 60 layers ≈ 0.47 ms/step against a
  ~15.7 ms decode step, i.e. ~3 % of decode and ~1.8 % end-to-end if decode is
  ~60 % of wall. We measured +0.78 %. **The end-to-end result is about half the
  kernel-level projection** — reported as measured, not reconciled.

## Verification that the arm was actually live

The banner alone only proves the import hook landed. Under CUDA-graph replay the
patch's Python never runs again, so the fused kernel had to be confirmed *inside
the replayed graph*. A separate boot of the `all` arm with a decode trace
(`verify_trace/`) gives, per TP rank:

| launches | total ms | kernel |
|---:|---:|---|
| 480 | 6.3 | `moe_sorting_oneshot_fused_kernel_0` |
| 480 | 2.6 | `fused_mx_quant_moe_sort_kernel` |
| 480 | 13.2 | `mfma_moe1_..._t32x64x256_...` (stage 1) |
| 480 | 6.8 | `gemm2_a4w4_..._bm32_..._atomic_...` (stage 2) |

480 = 8 decode steps × 60 layers. So the **decode MoE runs in 4 launches per
layer**, with the fused router+sort present and the atomic stage-2 epilogue
selected — 7 → 4 confirmed end-to-end, not just in the microbenchmark.

`topkGatingSoftmax` (268), `moe_reduction_kernel_0` (240) and the 4-kernel
`opus_moe_sorting_entry` multiphase chain (240–244 each) are still in the trace,
all at **prefill** counts (240 = 4 prefill steps × 60 layers) plus 24–28 for the
MTP layer. The shape gate held: prefill kept the `reduce` epilogue and the stock
sort, exactly as intended (the atomic epilogue is 1.6–2.8× slower at prefill
scale).

Patch counters at the end of that run: `fused=800, fallback=61, miss=854`.
The 854 misses are prefill and MTP calls correctly taking the stock path. The
61 fallbacks are the MTP layer (expert 513 / topk 11): the gate passes on the
router's `[M, 512]` logits but the sort then arrives with a different
`num_experts`/`topk`, so the router is recomputed and the stock sorter runs.
Correct, but it costs one extra launch on those calls — worth a cheap MTP-layer
guard later.

JIT compiles: 4 per TP worker, all during the pre-capture warmup phase
(`FLYDSL_COUNT_COMPILES=1` lines land after "Capture target verify CUDA graph
begin" but before any capture completes). The four are the bs=1..4 variants —
`n_grid_blocks` is a `Constexpr` and depends on `moe_buf` size, so each decode
batch size compiles its own kernel. Nothing compiles inside a capture region.

## Caveats

- `MOE_MAX_TOKENS=16` means only decode batches of **bs ≤ 4** take the fused
  path; graphs are captured for bs=1..8, so bs=5..8 fall back to the stock
  router+sort. At CONC=4 the steady state is bs=4, but a higher-concurrency run
  would see less of the win. Raising the cap to 32 is possible (LDS ceiling) but
  would make the dominant bs=4 case use a 2× larger LDS mesh, so it is not a
  free change.
- `SIMULATE_ACC=1`, matching every previous study on this box. The decode-MoE
  brief argues for `SIMULATE_ACC=0`; here accept length came out pinned at
  3.391/3.393 across arms, so the confounder it worries about is absent. A
  `SIMULATE_ACC=0` confirmation is still outstanding.
- One shape only (ISL 32768 / OSL 1024, CONC 4). No prefill-heavy or
  higher-concurrency point was run.
- 0.78 % is close to the resolution limit of this harness. It is believable
  because the sign and magnitude repeat in all three rounds and because the
  mechanism is confirmed in the trace — not because the confidence interval
  excludes zero.

## Files

| file | what |
|---|---|
| `run.sh` | the experiment (3 rounds × 2 arms × 2 trials, alternating order) |
| `summarize.py` / `summary.txt` | aggregation, pooled + paired-by-round |
| `metadata.json` | shapes, arm definitions, source hashes, versions |
| `round_*/<arm>/` | server log, startup log, per-trial driver logs and results |
| `verify_trace/` | separate boot of the `all` arm with a decode trace |
| `verify_trace/decode_kernels.txt` | launch counts per kernel from that trace |

## Environment

MI355X `gfx950` · sglang `0.5.19.dev20260908+g554f817948` · aiter `4ad998328` ·
FlyDSL `0.3.2` · torch `2.9.1+rocm7.2.0` · triton `3.7.0`
