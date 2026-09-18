# AMD Qwen3.5 optimization and fusion opportunities

[Collective-excluded proof and revised priorities](noncollective_optimization_proof.md)

**Prioritize collective behavior in prefill, and fused MoE plus full attention in target verification.** Small-kernel fusion is useful, but it is only part of the opportunity. This report catalogs **26 trace-supported fusion hypotheses and 7 kernel/system tuning opportunities**. It is a sensitivity analysis, not a claim that any optimization has been implemented or measured.

[Interactive impact calculator](optimization_impact.html) · [All candidate budgets CSV](optimization_candidates.csv) · [Exact kernel contributions](optimization_kernel_budgets.csv) · [Hotspots](optimization_hotspots.csv) · [Baseline evidence](optimization_baselines.json)

## Baselines and data quality

| Measurement | AMD value | Meaning |
|---|---:|---|
| Target prefill pass | 702.323 ms | Mean of unaffected target passes 0 and 3; approximately 32K-token chunks |
| Target verification | 11.309 ms | Mean GPU span over 31 TARGET_VERIFY passes, bs=4 |
| Decode cycle | 14.320 ms | Mean verification-start to next verification-start over 30 intervals |
| Work outside target verification per cycle | 3.011 ms | Difference of means; includes draft, acceptance, scheduling and other work, not all idle time |
| Captured TP-0 GPU window | 3391.468 ms | First GPU-kernel start to last GPU-kernel end; includes 4 prefill passes, 31 verification passes, and other work |

AMD target kernels are recorded on one stream. Verification sums agree closely with GPU spans, making a serial savings model useful for that phase. NVIDIA verification uses two streams: about 9.05 ms summed GPU work overlaps into about 7.08 ms elapsed time. A flat kernel-sum comparison misses that scheduling benefit. Its cycle is about 8.71 ms in this trace; neither number establishes a controlled hardware ratio.

**Two prefill passes have inconsistent same-stream intervals.** A permutation event reports 641.267 ms and an expert GEMM reports 278.824 ms; both overlap later kernels on the same stream. Additional neighboring intervals also overlap by more than 100 µs. The cause is not established. The original comparison retains raw values, including inflated layer 9 and layer 39 timings. Do not treat them as optimization opportunities. This analysis excludes entire affected prefill passes 1 and 2, instead of silently replacing individual durations. Clean passes have bs=1 and 32,768 / 32,464 input tokens. Prefill estimates therefore apply to that clean sample; reweighting them over all four chunks is approximate.

## How impact is calculated

For a candidate, `B` is its measured reducible-work pool per forward. For a proposed fraction `f` of that pool eliminated:

```text
Δprefill = f × Bprefill
Δverify  = f × Bverify
new prefill = 702.323 ms − Δprefill
new verify  = 11.309 ms − Δverify
new decode cycle ≈ 14.320 ms − Δverify
latency reduction = Δ / original time
speedup = original time / new time
captured-window saving ≈ 4 × Δprefill + 31 × Δverify
```

For normal fusion rows, the pool is the standalone pass that a fused producer/consumer could absorb; retained GEMM, attention, convolution and communication work is not counted as free. For research fusion rows involving two compute kernels, the pool is explicitly the combined compute time and the assumed percentage means a faster combined implementation, not deletion. The 25/50/75% values are **what-if assumptions, not confidence intervals or forecasts**. New fused kernels still perform the arithmetic and may increase register pressure, synchronization or memory traffic.

These equations assume saved GPU time lies on the critical path, all TP ranks benefit, downstream work moves earlier, and draft/acceptance behavior and workload remain fixed. No extra launch-gap savings are added: AMD verification already has almost no measured gaps between kernels. Timing improvements on TP-0 can be swallowed by waiting for slower ranks.

Actual request E2E latency cannot be reconstructed reliably from the layer reports: request boundaries, queue time, per-request prefill chunks and accepted tokens per speculative cycle are missing. For a request with `Np` comparable prefill passes and `Nv` verification cycles, use `Δrequest ≈ Np × Δprefill + Nv × Δverify`, then divide by its measured request latency. With fixed acceptance and batch behavior, cycle throughput improves by `old_cycle / new_cycle − 1`; cycle latency percentages are not throughput percentages.

## Recommended order

| Priority | Opportunity | Measured pool per forward | Example assumption | Phase latency saving | Decode-cycle saving |
|---|---|---|---|---|---|
| T01 | Collective algorithm, transport and rank-wait tuning | 309.691 ms (Prefill) | 25% pool reduction | 77.423 ms / 11.02% | — |
| T02 | Tune existing decode fused MXFP4 MoE | 3.123 ms (Decode) | 25% pool reduction | 0.781 ms / 6.90% | 5.45% |
| T03 | Tune decode full attention and split count | 1.217 ms (Decode) | 25% pool reduction | 0.304 ms / 2.69% | 2.13% |
| F01 | All-reduce + residual add + RMSNorm | 0.515 ms (Decode) | 50% pool reduction | 0.257 ms / 2.28% | 1.80% |
| F07 | Linear convolution consumes projection layout and emits split Q/K/V | 0.414 ms (Decode) | 50% pool reduction | 0.207 ms / 1.83% | 1.45% |
| F20 | MoE weighted reduction + shared-expert combination | 10.840 ms (Prefill) | 50% pool reduction | 5.420 ms / 0.77% | — |
| F16 | Shared-expert gate/up GEMM + SiLU-and-multiply | 0.272 ms (Decode) | 50% pool reduction | 0.136 ms / 1.20% | 0.95% |
| F11 | Linear attention output + gated RMSNorm | 0.193 ms (Decode) | 50% pool reduction | 0.097 ms / 0.86% | 0.68% |
| T07 | Overlap shared expert with routed MoE | 0.827 ms (Decode) | 50% pool reduction | 0.414 ms / 3.66% | 2.89% |
| F21 | Second expert GEMM + weighted expert-output reduction | 33.591 ms (Prefill) | 50% pool reduction | 16.796 ms / 2.39% | — |

F21 has a large theoretical pool but is substantially harder than F20: expert-output accumulation may require global coordination or atomics. T07 is hiding work with overlap, not fusion; a 50% hidden fraction is only an experiment target. NVIDIA auxiliary-stream execution makes it worth testing, but resource contention may prevent equivalent overlap on AMD.

**Full-attention precision matters:** AMD records `IS_Q_FP8_0_IS_KV_FP8_1`, whereas NVIDIA records `QkvE4m3`. Its faster kernel is not a drop-in speed target at the same Q precision. Prioritize AMD split-count, tile, cache and reduction tuning while retaining BF16 Q / FP8 KV; a Q-quantization change needs separate numerical and model-quality validation.

**Already fused work should not be proposed twice:** decode uses `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256`; the core expert pipeline is already fused. Likewise, gating is already incorporated in the delta-update kernel, moe1 already fuses SiLU/multiply, and `_layer_norm_fwd_1pass_kernel` is the gated output norm. Remaining proposals target their boundaries or internal efficiency.

## Complete fusion candidate list

Pools aggregate all applicable target-model layers per forward: 45 linear-attention layers, 15 full-attention layers, or 60 MoE blocks. `P/D 50%` gives the percentage reduction in prefill / verification latency if half that candidate pool is removed. Zero means not applicable in the selected phase. **Rows overlap and must not be summed.**

| ID | Fusion | Prefill pool ms | Verify pool ms | P/D 50% saving | Feasibility |
|---|---|---:|---:|---|---|
| F01 | All-reduce + residual add + RMSNorm | 21.701 | 0.515 | 1.54% / 2.28% | High-priority prototype |
| F02 | MoE combination + all-reduce + next input RMSNorm | 21.506 | 0.541 | 1.53% / 2.39% | Medium |
| F03 | Q/K normalization + M-RoPE | 1.712 | 0.069 | 0.12% / 0.31% | High-priority prototype |
| F04 | Q/K normalization + M-RoPE + KV-cache write | 2.075 | 0.132 | 0.15% / 0.58% | Medium |
| F05 | Full-attention output/segment reduction + sigmoid gate | 0.931 | 0.063 | 0.07% / 0.28% | High-priority prototype |
| F06 | Linear input GEMM epilogue emits final QKV/Z/B/A layout | 6.013 | 0.206 | 0.43% / 0.91% | Medium |
| F07 | Linear convolution consumes projection layout and emits split Q/K/V | 8.895 | 0.414 | 0.63% / 1.83% | High-priority prototype |
| F08 | Prefill conv epilogue + Q/K L2 norm + gating/split | 5.252 | 0.000 | 0.37% / 0.00% | Medium |
| F09 | Prefill gating + chunk-local cumulative sum | 0.660 | 0.000 | 0.05% / 0.00% | Medium |
| F10 | Decode delta update consumes packed conv output | 0.000 | 0.208 | 0.00% / 0.92% | Medium |
| F11 | Linear attention output + gated RMSNorm | 3.411 | 0.193 | 0.24% / 0.86% | Medium |
| F12 | KKT solve + recompute W/U | 11.779 | 0.000 | 0.84% / 0.00% | Research |
| F13 | Chunk state recurrence + output calculation | 52.254 | 0.000 | 3.72% / 0.00% | Research |
| F14 | Gated RMSNorm + attention output projection | 3.411 | 0.193 | 0.24% / 0.86% | Research |
| F15 | Shared-expert down projection + shared/routed output combination | 10.840 | 0.284 | 0.77% / 1.26% | Medium |
| F16 | Shared-expert gate/up GEMM + SiLU-and-multiply | 0.973 | 0.272 | 0.07% / 1.20% | High-priority prototype |
| F17 | TopK routing + MoE sorting/workspace passes | 3.845 | 0.000 | 0.27% / 0.00% | Medium |
| F18 | MXFP4 quantization + token permutation | 3.353 | 0.000 | 0.24% / 0.00% | Medium |
| F19 | First expert GEMM epilogue + intermediate FP4 quantization/permutation | 3.451 | 0.000 | 0.25% / 0.00% | Medium |
| F20 | MoE weighted reduction + shared-expert combination | 10.840 | 0.000 | 0.77% / 0.00% | High-priority prototype |
| F21 | Second expert GEMM + weighted expert-output reduction | 33.591 | 0.000 | 2.39% / 0.00% | Research/high upside |
| F22 | Input/post-attention normalization + GEMM or quantization consumer | 21.841 | 0.519 | 1.55% / 2.30% | Research |
| F23 | Decode router TopK + existing fused MoE | 0.000 | 0.587 | 0.00% / 2.59% | Research |
| F24 | Merge QKV/Z and B/A input projections | 50.443 | 0.968 | 3.59% / 4.28% | Medium/requires GEMM study |
| F25 | Decode convolution state update + gated delta recurrence | 0.000 | 0.688 | 0.00% / 3.04% | Research |
| F26 | Router projection epilogue + TopK/softmax | 2.198 | 0.587 | 0.16% / 2.59% | Research |

## Non-fusion optimization pools

| ID | Optimization | Prefill pool ms | Verify pool ms | P/D 25% saving |
|---|---|---:|---:|---|
| T01 | Collective algorithm, transport and rank-wait tuning | 309.691 | 1.081 | 11.02% / 2.39% |
| T02 | Tune existing decode fused MXFP4 MoE | 0.000 | 3.123 | 0.00% / 6.90% |
| T03 | Tune decode full attention and split count | 0.000 | 1.217 | 0.00% / 2.69% |
| T04 | Tune prefill GDN state kernel | 43.091 | 0.000 | 1.53% / 0.00% |
| T05 | Tune prefill MXFP4 expert GEMMs | 71.954 | 0.000 | 2.56% / 0.00% |
| T06 | Tune input/output/shared/router GEMMs | 111.681 | 2.537 | 3.98% / 5.61% |
| T07 | Overlap shared expert with routed MoE | 14.187 | 0.827 | 0.51% / 1.83% |

## Combined scenarios without duplicate kernel credit

Fusion prototype package: F01 + F07 + F04 + F05 + F11 + F16 + F20, each at 50% of its standalone-pass pool. Compute tuning adds a 25% reduction of T02 and T03. Collective tuning adds a 25% reduction of T01. These are scenario assumptions. All pools were combined at the kernel-invocation level using the maximum proposed reduction per invocation, so a kernel is never credited twice. Compatibility and critical-path realization still require runtime testing.

| Scenario | Prefill ms | Verify ms | Decode cycle ms | Cycle latency reduction | Cycle throughput gain, fixed acceptance | Captured-window reduction |
|---|---:|---:|---:|---:|---:|---:|
| Practical fusion prototypes | 677.910 | 10.514 | 13.525 | 5.55% | 5.88% | 3.61% |
| Fusion + focused compute tuning | 677.910 | 9.429 | 12.440 | 13.13% | 15.11% | 4.60% |
| Above + 25% collective reduction | 600.487 | 9.159 | 12.170 | 15.01% | 17.67% | 13.98% |

## Implementation notes and overlap constraints

### F01: All-reduce + residual add + RMSNorm

Fuse norm into the collective epilogue. Pool is only standalone norm time; no credit for removing network transfer. Requires all TP ranks and correct residual ownership.

Overlaps / alternatives: F02,F22; T01 collective algorithm gains can interact.

### F02: MoE combination + all-reduce + next input RMSNorm

Fuse shared/routed-output combination, reduction, and next-layer input norm. Pool excludes collective transfer; final-layer norm is excluded here.

Overlaps / alternatives: F01,F15,F20,F22.

### F03: Q/K normalization + M-RoPE

Extend Q/K/gate kernel to apply rotary embedding. Pool is standalone rotary pass. Preserve Gemma RMSNorm semantics.

Overlaps / alternatives: F04.

### F04: Q/K normalization + M-RoPE + KV-cache write

Alternative to F03: eliminate rotary/cache intermediates. Pool is rotary and cache kernels. Preserve current KV precision, scaling, layout, and write ordering.

Overlaps / alternatives: F03.

### F05: Full-attention output/segment reduction + sigmoid gate

Prefill: gate in attention output epilogue. Decode: gate in reduce_segments final store. Pool is gate-only kernel. Does not eliminate segment reduction.

Overlaps / alternatives: T03 touches same attention implementation.

### F06: Linear input GEMM epilogue emits final QKV/Z/B/A layout

Remove standalone layout pass using projection epilogue or a consumer reading original layout. Two separate projections must complete before combined consumers run.

Overlaps / alternatives: F07.

### F07: Linear convolution consumes projection layout and emits split Q/K/V

Fuse layout/split work around FlyDSL prefill conv or decode conv update. Pool is the two layout kernels; convolution time is retained. Preserve state update and Z/B/A branches.

Overlaps / alternatives: F06,F08,F10.

### F08: Prefill conv epilogue + Q/K L2 norm + gating/split

Normalize Q/K and emit gates from conv output; row reductions and gate inputs require a compatible tile. Pool is split/gating/L2 passes, not convolution arithmetic.

Overlaps / alternatives: F07,F09.

### F09: Prefill gating + chunk-local cumulative sum

Compute gating and chunk scan in one kernel. Retain the scan dependency; pool is separate cumsum time.

Overlaps / alternatives: F08; T04 may redesign this boundary.

### F10: Decode delta update consumes packed conv output

Alternative consumer-side split elimination: update reads conv layout directly. Preserve speculative state/rollback semantics.

Overlaps / alternatives: F07.

### F11: Linear attention output + gated RMSNorm

Prefill chunk output or decode recurrence epilogue applies existing gated RMSNorm; cross-channel reduction can raise registers and reduce occupancy. Pool is standalone norm.

Overlaps / alternatives: F12,F14.

### F12: KKT solve + recompute W/U

Pool is BOTH kernels, not removable overhead. Savings percentages model faster combined execution; triangular solve/tiling/global dependencies may prevent useful fusion.

Overlaps / alternatives: T04.

### F13: Chunk state recurrence + output calculation

Pool is BOTH compute kernels. Requires dependency-aware persistent/chunk scheduling; do not treat recurrent computation as removable.

Overlaps / alternatives: F11,T04.

### F14: Gated RMSNorm + attention output projection

Alternative to F11: projection consumes normalization inline. Norm reduction and GEMM tiling may require recomputation or synchronization.

Overlaps / alternatives: F11.

### F15: Shared-expert down projection + shared/routed output combination

Fuse combination into the last completed producer or consumer; need both expert paths and correct gate. Pool is combination pass.

Overlaps / alternatives: F02,F20; alternative epilogue placement.

### F16: Shared-expert gate/up GEMM + SiLU-and-multiply

GEMM epilogue produces activated shared-expert intermediate. Pool is separate activation; preserve both gate/up halves.

Overlaps / alternatives: Shared-expert overlap can hide some savings.

### F17: TopK routing + MoE sorting/workspace passes

Reduce routing/sorting passes and initialization through a coordinated routing backend. Pool is sorting only. Global histogram/prefix dependencies can still require multiple launches.

Overlaps / alternatives: F18,F19; do not simply delete global scans.

### F18: MXFP4 quantization + token permutation

Quantize directly to expert-consumption layout. Pool is permutation passes only; quantization retained. Reuse scales without excess token duplication.

Overlaps / alternatives: F17,F19.

### F19: First expert GEMM epilogue + intermediate FP4 quantization/permutation

Pool uses half of quantization time (one of two calls per layer) plus second permutation. Requires correct per-group scales and output packing; no extra SiLU saving because moe1 already fuses it.

Overlaps / alternatives: F18,T05.

### F20: MoE weighted reduction + shared-expert combination

Use moe_reduction_kernel_0 final stores to combine shared expert. Pool is combination only; weighted reduction retained.

Overlaps / alternatives: F02,F15.

### F21: Second expert GEMM + weighted expert-output reduction

Avoid materializing expert outputs or use tile-owned accumulation. Global expert-to-token reduction, atomics, determinism and precision can erase gains. Pool is standalone reduction.

Overlaps / alternatives: T05; F20 can follow only if dependencies allow.

### F22: Input/post-attention normalization + GEMM or quantization consumer

Alternative placement to F01. Reduction dimensions and shared consumers can force recomputation. Pool is normalization, not GEMM time.

Overlaps / alternatives: F01,F02.

### F23: Decode router TopK + existing fused MoE

Integrate routing only if expert selection can feed persistent/block scheduling without duplicate work. Decode experts already run as one fused aiter::fmoe kernel.

Overlaps / alternatives: T02; do not re-count already fused expert stages.

### F24: Merge QKV/Z and B/A input projections

Pool is BOTH projection GEMMs, not removable overhead. Concatenate compatible weights and use one GEMM or grouped launch, with custom output layout. Shape/precision compatibility and changed tiling determine whether it helps.

Overlaps / alternatives: F06,F07,T06.

### F25: Decode convolution state update + gated delta recurrence

Pool is BOTH compute kernels. Conv and recurrent states have different layouts; verify query-order dependencies, speculative state snapshots and rollback before considering a persistent fused kernel.

Overlaps / alternatives: F07,F10,F11.

### F26: Router projection epilogue + TopK/softmax

Alternative to F23: router GEMM emits top-k selections. Global expert-dimension reduction may not fit the GEMM epilogue; exact routing and scoring semantics must be preserved. Pool is TopK only.

Overlaps / alternatives: F17,F23,T06.

### T01: Collective algorithm, transport and rank-wait tuning

Pool includes all target-step collectives, including embedding/final boundaries. Separate transfer from waiting using all-rank traces and topology; TP-0 alone cannot diagnose bandwidth.

Overlaps / alternatives: F01,F02; transfer and wait improvements are not guaranteed.

### T02: Tune existing decode fused MXFP4 MoE

Tune persistent scheduling, expert tiling, weight loads and occupancy for recorded batch shape; pool is whole existing fused MoE.

Overlaps / alternatives: F23.

### T03: Tune decode full attention and split count

Tune NUM_SEGMENTS_PER_SEQ=128, query tiling, and reduction; fewer segments trades occupancy for reduction cost. Pool is attention+reduction. Keep BF16 Q and FP8 KV; NVIDIA FP8-Q path is not a precision-equivalent target.

Overlaps / alternatives: F05 touches same output stage.

### T04: Tune prefill GDN state kernel

Pool is recurrence computation. AMD is already faster than NVIDIA for this named stage in these traces. Tune only after larger bottlenecks.

Overlaps / alternatives: F12,F13.

### T05: Tune prefill MXFP4 expert GEMMs

Pool is two expert GEMMs on unaffected prefill passes. Tune tile sizes, traffic and expert imbalance; malformed duration records excluded.

Overlaps / alternatives: F19,F21.

### T06: Tune input/output/shared/router GEMMs

Pool includes all target-step GEMMs including logits. Use per-module shapes; compare alternative kernels within AMD. Improvements here are not the same as deleting GEMMs.

Overlaps / alternatives: F06,F14,F16,F22; use per-kernel union accounting.

### T07: Overlap shared expert with routed MoE

Pool is shared-expert work that may be hidden behind routed MoE, not deleted. AMD target work is on one stream; NVIDIA uses auxiliary streams. Resource contention and join dependencies limit overlap.

Overlaps / alternatives: F16 and shared-expert GEMM tuning affect same pool.

## Additional boundaries requiring more evidence

- Metadata construction, index/gather/scatter, state-slot clearing, and KV-index preparation may admit small fusions or reuse. They have different dependencies and shapes; generic utility-kernel names alone do not prove they can be combined. No combined savings are credited.
- Draft/acceptance/logits/sampling changes could reduce the roughly 3.011 ms outside target verification per cycle. This residual is not a removable overhead budget and includes useful work. The 60-layer report does not establish module/dependency attribution for all of it. Profile that path separately before proposing a numeric fusion gain.
- Attention output projection plus collective, or MoE output plus collective, can overlap communication with tiled GEMM/output production. Transfer remains necessary. Count benefits against the existing T01/epilogue pools, not as an additional copy of them.
- Reusing prefill chunk metadata across all 45 linear layers and eliminating repeated temporary initialization is worth checking in source. The first-layer sequence includes setup work that is not a normal per-layer cost.

## Validation needed before claiming E2E gains

1. Capture clean traces on all four AMD ranks with the same workload, precision, shapes and graph settings; resolve the overlapping prefill timestamps and identify rank wait versus collective transfer.
2. Prototype F01 and the decode full-attention split-count sweep first, alongside fused-MoE tile/occupancy tuning. Measure new kernel time and all-rank phase latency.
3. Verify outputs against the existing kernels, including chunk boundaries, FP4 scales, Gemma normalization, rotary positions and speculative-state rollback.
4. Benchmark request TTFT, token latency, throughput and accepted tokens per verification. Report p50/p95 over repeated runs; do not substitute kernel-sum savings for measured E2E gains.
