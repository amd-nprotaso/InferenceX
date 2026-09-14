# Prompt: FlyDSL re-fusion of the Qwen3.5 decode MoE front-end

Copy everything below the line into a fresh agent session running in
`/var/home/my_InferenceX/InferenceX/local_bench`.

---

## Task

The decode MoE for Qwen3.5-397B-A17B MXFP4 on MI355X (gfx950) currently costs
**70.3 µs per layer across 7 kernel launches**. The build it replaced did the
same work in **62.8 µs across 2 launches**. The two GEMMs in the new path are
*faster* than the old fused kernel (41.7 µs vs 53.0 µs, and near the memory
roofline) — the regression is entirely the five satellite kernels around them,
each pinned at the ~4.2 µs gfx950 dispatch floor.

Your job is to collapse those satellites back into the GEMMs in FlyDSL, keeping
the new GEMM mainloops. **Target: ≤50 µs/layer in ≤3 launches.** Beating 62.8 µs
is the minimum bar — that number is already proven achievable on this hardware
by the ASM kernel this build dropped.

This is mostly a **port and integration job, not a from-scratch kernel design**.
Two of the three pieces already exist; one of them is in upstream FlyDSL but not
in aiter's vendored copy. Read §"The three changes" before writing anything.

## Read these first — do not invent API

FlyDSL has a large surface and a hallucinated API call costs more than the
reading does.

| What | Where |
|---|---|
| Project conventions, layout, env vars, style gate | `/var/home/main_FlyDSL/FlyDSL/CLAUDE.md` |
| `@flyc.kernel` / `@flyc.jit`, LDS, tiled copy/MMA | `docs/kernel_authoring_guide.md` |
| Layout algebra (`zipped_divide`, `slice`, `make_layout`) | `docs/layout_system_guide.md` |
| **The kernel you are porting** — fused gating+softmax+topK+sort, one launch, all in LDS | `/var/home/main_FlyDSL/FlyDSL/kernels/moe/moe_sorting_kernel.py:696` `compile_moe_sorting_oneshot_fused` |
| Its gating helpers (reusable callbacks, LDS sink) | `/var/home/main_FlyDSL/FlyDSL/kernels/moe/topk_gating_softmax_kernel.py` — `_compute_topk_gating_layout`, `_emit_topk_gating_softmax_body` |
| **Technique reference for change #3** — warp-per-route fused route-map + MX quant + scatter-copy + scale preshuffle | `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py`, host entry `moe_kernels.py:2588` |
| aiter's vendored sorting copy (the one that is missing the fused variant) | `/sgl-workspace/aiter/aiter/ops/flydsl/kernels/moe_sorting_kernel.py:192` `_compile_moe_sorting_oneshot` |
| aiter's sorting dispatch + backend env flags | `/sgl-workspace/aiter/aiter/fused_moe.py:64-66`, `_moe_sorting_impl` at `:190-410`, `flydsl_moe_sorting_fwd` at `aiter/ops/flydsl/moe_sorting.py` |
| Stage-1/stage-2 wrappers and config selection | `aiter/fused_moe.py:2749` (`AITER_FLYDSL_FORCE`), token-tier table at `:2800-2815`, `aiter/ops/flydsl/moe_kernels.py:665,748` |
| The stage-2 device body you must not break | `aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py:221` `gemm2_body_v2`, epilogue `atomic_bf16_epilog` at `:887` |
| Kernel-name grammar (how configs are selected) | `aiter/ops/flydsl/mxfp4_kname.py` — `_FLYDSL_V2_GEMM2_RE` at `:20` |
| SGLang's call site | `sglang/python/sglang/srt/layers/moe/moe_runner/aiter.py:248` |
| Working precedent: FlyDSL kernel swapped into this same server behind an env flag | `kernel_tuning/conv/causal_conv1d_flydsl.py` + `kernel_tuning/conv/README.md` |

Project-local skills are worth invoking rather than reimplementing:
`/flydsl-kernel-authoring`, `/flydsl-tile-programming`, `/lds-optimization`,
`/kernel-trace-analysis`, `/debug-flydsl-kernel`.

Full background — how the hotspot was established, every A/B already run, and
the prefill counterpart of this problem: `trace_analysis3/` and the session
notes that produced it.

## The shape

Production decode, `qwen3.5_fp4_sglang_server.sh` defaults with `CONC=4`:

- 60 layers, `hidden=4096`, `num_experts=512`, `topk=10`,
  `moe_intermediate_size=1024`, TP4 → per-rank `inter_dim=256`
- EAGLE MTP: `--speculative-num-draft-tokens 4`, so a verify step is
  **bs=4 requests × 4 draft tokens = 16 tokens**, `rows = 16 × 10 = 160`
- Decode is CUDA-graph replayed (`--cuda-graph-max-bs 8`). Verified: no
  `grid`/`cpu_op` on any TARGET_VERIFY event in the trace.
- MXFP4 weights, **bf16 activations quantized to fp4 inline** (a4w4 two-stage).
  The shared expert is bf16 (in the checkpoint's quant `exclude` list) and runs
  as separate dense GEMMs — it is *not* part of the routed path here.

## What runs today

From `/var/home/sglang_profiling3/isl32768_osl1024_c4`, 300 verify steps,
per layer per rank, profiler-startup outliers removed:

| µs | launches | kernel | what it does |
|---:|---:|---|---|
| 28.58 | 1 | `mfma_moe1_silu_mul_afp4_wfp4_fp4_t32x64x256_pm1_fp4q_sort_async_xcd4_v2out_v33` | stage 1 gate/up + SiLU, fp4 out |
| 13.13 | 1 | `gemm2_a4w4_port_hmax8192_imax256_bm32_bn128_bk256_nt_reduce_tk10_bhoist_apf_spart4x2_bf16lds` | stage 2 down-proj, `reduce` epilog |
| 9.81 | 1 | `vllm::moe::topkGatingSoftmax<bf16,32,512,2,64,true,0,SharedExpertScoringFunc 0>` | router: softmax + top-10 of 512 |
| 8.74 | 2 | `aiter::opus_moe_sorting_entry` (`P0_v2`, `P23`) | expert sort / count / prefix-sum / scatter |
| 5.46 | 1 | `aiter::fused_mx_quant_moe_sort_kernel<bf16,fp4_t,256,16>` (`csrc/kernels/quant_kernels.cu`) | activation MX-quant + scatter into sorted layout |
| 4.58 | 1 | `moe_reduction_kernel_0` | top-10 weighted sum over the `[16,10,4096]` partial |
| **70.30** | **7** | | |

Context: the whole verify step is **11.02 ms / 1464 launches** per rank. MoE is
**4217 µs = 38 %** of it. Of the 1464 launches, **781 are under 5 µs and consume
31.5 % of the step**; the p5 kernel duration is 4.16 µs and the smallest thing in
the trace (`Memcpy DtoD`) is 3.77 µs. Five of the seven MoE kernels above are at
that floor. This is a launch-count problem, not a FLOPs or bandwidth problem.

## Baselines to beat

| config | µs/layer | launches | note |
|---|---:|---:|---|
| ASM `fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` + router | **62.8** | 2 | previous build (a16w4, everything fused). **This is your bar.** |
| current 2-stage FlyDSL | 70.3 | 7 | what you are replacing |
| the two GEMMs alone | 41.7 | 2 | the floor if every satellite disappeared |
| **target** | **≤50** | **≤3** | |

At 60 layers, 70.3 → 50 µs is **1.22 ms/step, ~11 % of the verify step**. Decode
step wall is ~15.7 ms (TPOT 4.62 ms × accept-length 3.39) at ~84 % GPU busy, and
decode is ~60 % of the `ISL 32768 / OSL 1024 / conc 4` workload → **~4–5 %
end-to-end**. That is above the ~1 % noise floor of `qwen3.5_fp4_sglang_bench.sh`,
unlike the GDN chunk_h work. Do not start unless you believe you can hold that.

## Roofline — where there is and is not headroom

Assuming uniform routing, 16 tokens × topk 10 touch
`512 × (1 − (1−10/512)^16) ≈ 139` distinct experts. Per-rank weight bytes:
`w13 = 2·256·4096` fp4 + e8m0 scales = 1.114 MB/expert;
`w2 = 4096·256` fp4 + scales = 0.541 MB/expert.

| kernel | bytes | µs | achieved | % of 8 TB/s |
|---|---:|---:|---:|---:|
| stage 1 | 154.4 MB | 28.58 | 5.40 TB/s | 68 % |
| stage 2 | 75.0 MB | 13.13 | 5.71 TB/s | 71 % |
| old fused ASM (both) | 229.4 MB | 52.99 | 4.33 TB/s | 54 % |
| the five satellites | ~1.5 MB | 28.59 | 0.05 TB/s | 0.7 % |

**Do not touch the GEMM mainloops.** They are at 68–71 % of HBM peak and the
remaining headroom is ≤1.4×, hard-won. All of the available win is in the
satellites, which move essentially no data and cost as much as stage 1.

Real routing is skewed, so 139 is an upper bound on distinct experts and the
achieved-bandwidth figures are correspondingly upper bounds. Treat them as
"there is not much left here", not as exact.

## The three changes, in the order to attempt them

### 1. Atomic stage-2 epilogue — do this first, it is a CSV row

Measured, `moe_decode_bench.py --only "conc=4"`, 30 iters, GPU 0:

| stage-2 config | stage 2 | `moe_reduction` | total |
|---|---:|---:|---:|
| `..._t32x128x256_reduce_nt_sbm32` (current tuned row) | 14.2 | 3.0 | 63.1 |
| `..._t32x128x256_atomic_sbm32` | 13.6 | — | 62.4 |
| `..._t32x128x256_atomic_persist_sbm32` | 14.3 | — | 61.8 |

The atomic epilogue removes `moe_reduction_kernel_0` for free at this scale
(−4.58 µs/layer in trace terms). Change row `token=16, inter_dim=256,
expert=512, topk=10` of
`aiter/configs/model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv`, or override via
`AITER_CONFIG_FMOE`. One launch gone, twenty minutes of work.

**This must be shape-gated.** At prefill scale the same epilogue is catastrophic
— see "Known dead ends".

### 2. Port the fused gating+sort kernel from upstream FlyDSL — the main piece

`/var/home/main_FlyDSL/FlyDSL/kernels/moe/moe_sorting_kernel.py:696`
`compile_moe_sorting_oneshot_fused` reads `gating_logits[M, E]` directly and
emits `sorted_token_ids`, `sorted_weights`, `sorted_expert_ids`,
`num_valid_ids`, `moe_buf` in **one launch**, staging every per-token-K
intermediate in LDS. It replaces `topkGatingSoftmax` *and* both
`opus_moe_sorting_entry` calls.

aiter's vendored copy (`aiter/ops/flydsl/kernels/moe_sorting_kernel.py`) has
only the **unfused** `_compile_moe_sorting_oneshot` at `:192`. That is why the
existing env flag does nothing useful — measured:

| sorting backend | sort launches | µs (sort only) | total |
|---|---:|---:|---:|
| default (Opus) | 2 | 7.8 | 62.3 |
| `AITER_USE_FLYDSL_MOE_SORTING=1` | 1 (`moe_sorting_oneshot_kernel_0`) | 7.2 | 62.2 |
| `AITER_USE_CK_MOE_SORTING=1` | 2 | 8.9 | 64.8 |

FlyDSL already collapses two sorts into one; the router still runs separately,
so the win is inside noise. **Porting the fused variant is the gap to close** —
it should take `topkGatingSoftmax`'s 9.81 µs to zero.

LDS feasibility at your shape, from the function's own formula: `E=512` →
`smem_cols=513`; gfx950 `lds_capacity_bytes=163840` → `lds_capacity_ints=40960`;
`target_occupancy=2` → `r = 40960//2//513 = 39`; `r_for_sub = ((39−2)//8)*8 = 32`;
`r_token_min = 16` → `sub_tokens = 16`. It closes at `max_tokens=16`, with room
up to 32. Re-derive this yourself rather than trusting the arithmetic here, and
check what happens at `max_tokens=32` in case the graph batch grows.

Expected: **−10 µs/layer, 7 → 5 launches.**

### 3. Fold the MX quant + scatter into the fused front-end

`aiter::fused_mx_quant_moe_sort_kernel` (5.46 µs) quantizes activations to fp4
and scatters them into the sorted layout with preshuffled e8m0 scales — after
the sort has already written the route map. Those are the same warps touching
the same routes twice.

`aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py` already does exactly
this fusion — route-map + MX quant + scatter-copy + scale-preshuffle in one
warp-per-route pass — but for the *grouped/masked* (contiguous-M) layout used by
`grouped_moe_gfx1250.py`, not the sorted-token layout your path uses. Read its
docstring: the intra-warp mapping, the butterfly `shuffle_xor` amax reduction,
and the scale-preshuffle derivation all transfer; only the destination row
computation changes (`expert_row_base[e] + slot` → `sorted_token_ids` order).

Combined with #2 this gives **one kernel: logits → top-k → sort → quantized,
scattered, preshuffled stage-1 input**. Expected: **−5.5 µs/layer, 5 → 4
launches**; 3 if `moe_buf` zero-init folds in too.

Landing all three: **70.3 → ~50 µs/layer, 7 → 3 launches.**

## Correctness constraints — these will bite

1. **The router is not a plain softmax top-k.** The live instantiation is
   `topkGatingSoftmax<bf16, 32, 512, 2, 64, true, 0, SharedExpertScoringFunc 0>`.
   Establish what `SharedExpertScoringFunc` does before replacing it. Qwen3.5's
   shared expert is bf16 and runs as separate dense GEMMs in all 60 layers, but
   the MTP layer takes a different path
   (`_fused_append_shared_experts_with_weights_kernel`, expert 513 / topk 11).
   Do not break the MTP layer while fixing the other 60.
2. **Renormalization.** `compile_moe_sorting_oneshot_fused` takes
   `renormalize: bool`. Read what SGLang's topk actually does for this model
   (`norm_topk_prob` / `renormalize` in the MoE runner) — do not assume `True`.
3. **CUDA-graph capture.** Decode is graph-replayed. Any workspace must be
   pre-allocated outside capture and cached per device — `flydsl_moe_sorting_fwd`
   in `aiter/ops/flydsl/moe_sorting.py` already does this for exactly this
   reason; follow it. No allocation, no host sync, no shape-dependent branching
   inside the replayed region.
4. **The atomic epilogue makes stage 2 order-dependent.** `moe_reduction` is
   deterministic; `global_atomic_pk_add_bf16` is not. Your correctness harness
   must use tolerances, not bitwise comparison, once #1 lands — and you must say
   so in the report. (Generation is already nondeterministic under
   `SIMULATE_ACC=1`, so this is acceptable, but it should be a stated decision
   rather than a silent one.)
5. **`expert_mask` / EP must keep working** even though `--ep-size 1` today. The
   sorting path carries an EP validity mask and `DROPPED_ROUTE_ROW` sentinels;
   the fused kernel has a `has_mask` parameter. Do not drop it.
6. **Prefill shares this code path.** `fused_moe` serves both phases. Every
   change must be gated on token count or selected through the tuned-CSV row,
   and you must re-measure prefill before claiming a win. See dead ends.

## FlyDSL conventions

From `CLAUDE.md` — these prevent the common failure modes:

- `@flyc.kernel` for the device kernel, `@flyc.jit` for the launch wrapper.
- `range_constexpr` for compile-time unrolled loops; `range(..., init=[...])`
  for `scf.for` with loop-carried values.
- Prefer `fx.rocdl.make_buffer_tensor()` + layout ops + `fx.copy_atom_call`.
  Raw `buffer_ops.create_buffer_resource()` and manual byte offsets are legacy.
- Allocate LDS with `fx.SharedAllocator` over an `@fx.struct` storage layout,
  not the legacy `SmemAllocator`.
- Clear `SmemPtr._view_cache = None` after exiting `scf.for` if you recreate
  shared-memory views, or you get MLIR dominance errors.
- No early `return`, no branch-local `return`/`yield` in traced functions.
- Cache compiled launchers per concrete tensor signature — see the `_build` /
  `lru_cache` / `launch._compiled` pattern in `causal_conv1d_flydsl.py` and the
  `@functools.lru_cache(maxsize=256)` on the upstream fused compiler.
- Run `bash scripts/check_python_style.sh --fix` before finishing.

## Validation protocol

1. **Kernel level** — `moe_decode_bench.py` (already written; drive alternate
   configs with `AITER_CONFIG_FMOE=<csv>`). It agrees with the production trace
   to within ~10 % per kernel: bench 32.1/14.2/8.0/5.6/3.0 µs vs trace
   28.58/13.13/8.74/5.46/4.58. The bench is not graph-captured, which is most of
   the difference. **Report the launch count, not just the total** — that is the
   quantity you are optimizing.
2. **Correctness** against `aiter.fused_moe.fused_moe` on the unmodified path,
   at `token ∈ {1, 2, 4, 8, 16, 32, 64}` × `E=512, topk=10, inter=256`. Model the
   harness on `kernel_tuning/conv/check_conv.py`. Include an independent FP32
   reference, graph capture/replay, and the `expert_mask` path.
3. **Trace** — re-profile with
   `CONC_LIST="4" ISL=32768 OSL=1024 ./qwen3.5_fp4_sglang_profile.sh` and confirm
   launches/step drops from **1464** and MoE from **4217 µs/step**. Note that
   `analyze_traces.py` files `gemm2_a4w4_port_*` under `gemm` rather than `moe`
   and does not drop the profiler-startup outliers (4 of them, up to 673 ms, in
   the current trace) — fix both or correct for them by hand.
4. **End-to-end**, only after the kernel is green:
   `CONC_LIST="4" ./qwen3.5_fp4_sglang_bench.sh` (MODE=full, ISL 32768 /
   OSL 1024), n≥5 paired, alternating server boots. Use **`SIMULATE_ACC=0`** —
   accept-length varied 3.38–3.42 across runs in the previous study and it
   changes the number of decode passes for a fixed 1024 output tokens, which is
   a direct confounder on a decode-side change. Report accept-length alongside
   throughput either way.
5. **Do not benchmark against `bench_results/qwen35_mxfp4_mtp_isl32768_osl1024_c4.log`
   (71.91 s / 16985 tok/s).** That run predates this build and used a different
   all-reduce configuration. Re-baseline first.
6. Hardware counters do not work in this container (no `CAP_PERFMON`,
   `rocprofv3 --pmc` hangs). You cannot measure L2 hit rate or dispatch
   occupancy directly — infer from runtime and say that you are inferring.

## Known dead ends — measured, do not redo

- **Atomic stage-2 epilogue at prefill scale is 1.6–2.8× slower.** At 327 680
  rows: `reduce` 1065+615 µs vs `atomic` 2098 µs (v2 port), and 751+564 vs 2079
  (native `mfma_moe2`). 1.34 G packed-bf16 atomic adds into a 268 MB region
  serialize on shared cache lines. Change #1 is a *decode-only* win and must be
  gated.
- **LLC panel-blocking of the reduce caps at 1.29×.** Probed at constant total
  bytes: 5.13 TB/s at 2.68 GB (HBM) vs 6.61 TB/s at 168 MB (fits the 256 MB L3).
  Not the step change the cache-size argument suggests.
- **`AITER_USE_CK_MOE_SORTING=1` is slower** (64.8 vs 62.3 µs).
- **`AITER_USE_FLYDSL_MOE_SORTING=1` alone is inside noise** (62.2 vs 62.3) —
  that is the symptom that motivates change #2, not a reason to stop.
- **Do not tune the GEMM mainloops** (see roofline table).
- **Do not fold the topk sum into K** (a `K = topk × inter` down-proj with
  gathered W2 rows). It removes the partial entirely but a token tile then needs
  up to `BM × topk` distinct experts' weight panels; B traffic explodes. Only
  viable with routing locality, which topk=10-of-512 does not have.

## Deliverables

1. `kernel_tuning/moe_decode/moe_fused_route_sort_flydsl.py` — the fused
   kernel(s) plus host entries matching the aiter signatures they replace.
2. `kernel_tuning/moe_decode/check_moe_decode.py` — the correctness suite.
3. `kernel_tuning/moe_decode/bootstrap/sitecustomize.py` + `run_server.sh` —
   opt-in integration behind `SGLANG_FLYDSL_MOE_DECODE=1`, following
   `kernel_tuning/conv/`. **Merge with the existing hooks rather than adding a
   third `sitecustomize.py`** — Python imports that name once, which is why
   `bootstrap_both/` already exists. This is the right moment to consolidate to
   one hook module with per-kernel env gates.
4. The tuned-CSV delta for change #1, as a patch against
   `qwen3_5_397b_fp4_tuned_fmoe.csv`, with the shape gate.
5. `kernel_tuning/moe_decode/README.md` — what changed, measured before/after
   for each of the three changes **independently**, launches/layer and
   µs/layer for each, the re-profiled trace numbers, and an environment/versions
   table (this build is `sglang 0.5.19.dev20260908+g554f817948`, `FlyDSL 0.3.2`,
   `torch 2.9.1+rocm7.2.0`, `triton 3.7.0+amd.rocm7.2.0`, aiter
   `4ad998328` — record what you actually see, the container moves).

## Reporting

State plainly which of the three changes worked, which did not, and by how much.
Change #1 is cheap and should land within the hour; if it does not reproduce the
62.4/61.8 µs numbers above, say so and show the measurement before continuing.

Keep kernel microbenchmarks separate from end-to-end serving numbers. The
previous FlyDSL effort on this server (`kernel_tuning/chunk_gated_delta_rule/`)
produced a kernel that won on the kernel and vanished into the noise end-to-end
because it was 3 % of wall time. This target is ~11 % of the decode step and
~4–5 % of wall — resolvable, but only just. If your measured kernel-level win
comes in below ~8 µs/layer, say so and stop rather than running a serving
benchmark that cannot distinguish it from zero.

If you find the analysis above is wrong somewhere, show the measurement. The
70.3 µs breakdown, the 62.8 µs bar, the dispatch-floor histogram and the three
A/B tables are all measured on this machine; the *attribution* of the satellite
cost to launch overhead rather than to real work is an inference from the 4.16 µs
p5 duration, and it has not been confirmed with counters.
