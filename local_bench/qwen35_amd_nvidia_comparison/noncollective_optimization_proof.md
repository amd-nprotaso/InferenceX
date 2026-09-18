# AMD vs NVIDIA: non-collective kernel evidence and optimization priorities

[CPU copy/scalar synchronization investigation](cpu_copy_analysis.md)

**Collective operations are excluded from this analysis. The claim “AMD uses more MoE kernels” is true for prefill and false for decode in these traces.** Counts are verified per layer, not extrapolated from one example. More launches establish implementation fragmentation, but do not establish that launch overhead caused a latency gap.

[Count proof CSV](moe_kernel_count_proof.csv) · [MoE substages](moe_substage_proof.csv) · [Per-layer counts/times](noncollective_per_layer.csv) · [Exact event evidence](noncollective_event_evidence.csv) · [Kernel summary](noncollective_kernel_summary.csv) · [Visual proof](noncollective_proof.html)

## Scope and comparison rules

- Prefill uses target forwards 0 and 3 in both traces: matching `bs=1, toks=32768` and `bs=1, toks=32464` labels. AMD forwards 1 and 2 have inconsistent overlapping timestamps and are excluded. Matching labels do not prove equal cached context, request history, precision or all runtime settings.
- Decode uses all 31 `TARGET_VERIFY bs=4` target forwards per vendor; draft-model work is excluded. This is speculative verification, not plain single-token decode.
- All-reduce and all-gather kernels are removed, including boundary collectives and NVIDIA fused all-reduce/normalization kernels. Consequently NVIDIA normalization embedded in those kernels cannot be timed separately; its absence is not a zero-cost normalization result.
- “Kernels” below means GPU kernel **invocations**, including repeated symbols, not unique kernel names. Durations are summed GPU execution time, not elapsed module latency. NVIDIA overlaps streams, so sums cannot be used as E2E latency.
- AMD prefill module attribution is recorded; AMD decode and NVIDIA layer/module grouping use the previously documented attention and MoE boundaries. Full MoE-block boundaries are more reliable than attributing individual NVIDIA GEMMs to the router versus shared expert.

## Proof: MoE launch counts

Entire block includes shared expert, router, routed experts and their combination. Routed core includes routing/sorting/quantization/permutation/expert computation/finalization, but excludes shared/router dense GEMMs, shared activation, copy helper and shared/routed combination.

| Phase | Scope | AMD calls/layer | NVIDIA calls/layer | AMD calls/forward | NVIDIA calls/forward |
|---|---|---:|---:|---:|---:|
| Prefill | Entire MoE block | 17 | 13 | 1020 | 780 |
| Prefill | Routed expert core incl routing | 12 | 7 | 720 | 420 |
| Decode | Entire MoE block | 7 | 13 | 420 | 780 |
| Decode | Routed expert core incl routing | 2 | 5 | 120 | 300 |

Every one of the **120 prefill layer instances per vendor** has the same counts; every one of the **1,860 decode layer instances per vendor** also has the reported count. AMD has **240 extra whole-MoE launches per prefill forward (+30.8%)**, but **360 fewer per decode forward (−46.2%)**.

### Where the extra prefill launches come from

| Routed-core stage | AMD calls/layer | NVIDIA calls/layer | AMD ms/forward | NVIDIA ms/forward |
|---|---:|---:|---:|---:|
| Routing / TopK | 1 | 3 | 2.198 | 8.030 |
| Sorting / workspace | 4 | 0 | 3.845 | 0.000 |
| Quantization | 2 | 1 | 6.000 | 3.399 |
| Permutation | 2 | 0 | 3.353 | 0.000 |
| Expert GEMM 1 | 1 | 1 | 28.926 | 28.022 |
| Expert GEMM 2 | 1 | 1 | 43.028 | 36.126 |
| Expert reduction / finalize | 1 | 1 | 33.591 | 33.563 |

AMD routed-core sequence is `TopK ×1 → sorting/workspace ×4 → quantize ×1 → permute ×1 → expert GEMM1 ×1 → quantize ×1 → permute ×1 → expert GEMM2 ×1 → reduction ×1`, totaling **12** launches. NVIDIA uses three routing kernels, one quantization kernel, two expert GEMMs and one finalize kernel: **7**. Its surrounding MoE block also includes a separate copy helper, so zero entries in the table mean no separately named kernel in that bucket, not proof that all underlying work disappeared.

**Extra launches are not the main measured MoE discrepancy by themselves.** AMD prefill MoE totals **152.872 ms** versus NVIDIA **144.732 ms** (+5.6%), despite 30.8% more launches. The second expert GEMM accounts for **43.028 vs 36.126 ms**, a **6.903 ms** gap. The separate weighted-output reductions are almost equal: **33.591 vs 33.563 ms**. AMD routing + sorting + quantization + permutation is **15.396 ms**; NVIDIA explicit routing + quantization is **11.430 ms**, with another **10.175 ms copy helper** outside the routed-core definition. Comparing only launch counts would miss these differences.

### Decode contradicts the blanket “more AMD kernels” claim

AMD routed core is `topkGatingSoftmax ×1 + aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256 ×1`. NVIDIA uses routing, quantization, expert GEMM1, expert GEMM2 and finalize: five launches. The whole MoE block is **5.136 ms AMD vs 5.147 ms NVIDIA** in summed GPU time. AMD is already more fused here; tune the fused kernel or its boundaries, rather than proposing to fuse expert stages that are already one launch. NVIDIA can still have lower elapsed latency because of stream overlap.

## Attention and other operations, excluding collectives

Times below aggregate the 60-layer target forward. Attention pipelines include layout, convolution/state update or cache handling, attention computation, gating and gated output normalization. Their dense input/output projections are listed separately.

| Category | Prefill AMD ms | Prefill NVIDIA ms | Decode AMD ms | Decode NVIDIA ms |
|---|---:|---:|---:|---:|
| MoE block including shared expert and router | 152.872 | 144.732 | 5.136 | 5.147 |
| Attention projections | 91.460 | 67.176 | 1.563 | 1.455 |
| Linear-attention pipeline | 87.135 | 97.924 | 1.295 | 0.941 |
| Full-attention pipeline | 32.645 | 38.116 | 1.546 | 0.530 |
| Standalone layer norms | 21.662 | 19.144 | 0.515 | 0.003 † |
| Outside decoder layers | 0.616 | 0.567 | 0.129 | 0.121 |

† NVIDIA decode normalization is mostly inside excluded collective-fusion kernels. Do not interpret 0.003 ms as the total NVIDIA normalization cost, or subtract it from AMD normalization to estimate a pure non-collective opportunity.

**Prefill:** the largest positive non-collective category gap is attention projections (**+24.285 ms** AMD), followed by the whole MoE block (**+8.139 ms**). Both linear and full-attention pipelines have lower summed time on AMD for the selected passes. That makes projection GEMM selection and MoE output/GEMM2 work stronger evidence-based priorities than assuming all AMD attention kernels are slower.

**Decode:** the full-attention pipeline is **1.546 vs 0.530 ms** (2.91× summed-time ratio, **+1.015 ms** AMD). Linear attention is **1.295 vs 0.941 ms** (**+0.354 ms**). MoE block sums are essentially equal. Full attention is therefore the clearest measured non-collective discrepancy in decode. The AMD symbol has BF16 Q (`IS_Q_FP8_0`) and FP8 KV, while NVIDIA uses `QkvE4m3`; the ratio is not a precision-equivalent attainable speedup.

### Decode linear attention: equal launch counts, different kernel costs

| Stage | AMD kernel | AMD ms/forward | NVIDIA ms/forward |
|---|---|---:|---:|
| Input layout | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 0.206 | 0.130 |
| Convolution update | `_causal_conv1d_update_kernel` | 0.204 | 0.183 |
| QKV split | `fused_qkv_split_gdn_prefill_kernel` | 0.208 | 0.068 |
| Gated delta update | `fused_sigmoid_gating_delta_rule_update_kernel` | 0.484 | 0.457 |
| Gated output norm | `_layer_norm_fwd_1pass_kernel` | 0.193 | 0.102 |

Both implementations use five core linear-attention launches per layer, or 225 over 45 layers. The split, layout and output-normalization kernels explain more of the gap than the delta-update arithmetic. This supports layout/split fusion and output-norm fusion even though launch counts are identical.

## Revised non-collective optimization priorities and pipeline impact

All collectives and collective-fusion proposals are held fixed. End-to-end denominators still include unchanged communication and surrounding work, because removing them from the denominator would exaggerate E2E gains. The fractions below are hypothetical reductions of measured AMD pools, not forecasts.

| Opportunity | AMD measured pool | Assumption | Saved per forward | Phase latency reduction |
|---|---:|---|---:|---:|
| Tune attention projection GEMMs | 91.460 ms (Prefill) | 25% reduction | 22.865 ms | 3.26% |
| Tune prefill expert GEMM2 | 43.028 ms (Prefill) | 25% reduction | 10.757 ms | 1.53% |
| Fuse MoE weighted reduction with shared-expert combination | 10.840 ms (Prefill) | 50% reduction | 5.420 ms | 0.77% |
| Fuse expert GEMM2 with weighted reduction (research) | 33.591 ms (Prefill) | 50% reduction | 16.796 ms | 2.39% |
| Tune decode full attention + segment reduction | 1.217 ms (Decode) | 25% reduction | 0.304 ms | 2.69% |
| Tune existing fused decode MoE | 3.123 ms (Decode) | 25% reduction | 0.781 ms | 6.90% |
| Linear convolution consumes projection layout and emits split Q/K/V | 0.414 ms (Decode) | 50% reduction | 0.207 ms | 1.83% |
| Linear attention output + gated RMSNorm | 0.193 ms (Decode) | 50% reduction | 0.097 ms | 0.86% |
| Shared-expert gate/up GEMM + SiLU-and-multiply | 0.272 ms (Decode) | 50% reduction | 0.136 ms | 1.20% |
| Q/K normalization + M-RoPE + KV-cache write | 0.132 ms (Decode) | 50% reduction | 0.066 ms | 0.58% |
| Full-attention output/segment reduction + sigmoid gate | 0.063 ms (Decode) | 50% reduction | 0.031 ms | 0.28% |

The following package uses only non-collective donor kernels: layout/split fusion, QK/RoPE/cache fusion, full-attention output gating, GDN output norm, shared-expert activation, and shared/routed combination. Each saves 50% of its standalone-pass pool. The second scenario adds 25% faster decode fused MoE and attention/reduction. Donor pools are disjoint; no collective time or collective-fused normalization is credited.

| Scenario | Prefill ms | Verification ms | Decode cycle ms | Cycle latency reduction | Captured GPU-window reduction |
|---|---:|---:|---:|---:|---:|
| Non-collective fusion package | 688.760 | 10.630 | 13.641 | 4.74% | 2.22% |
| Above + decode MoE/attention tuning | 688.760 | 9.545 | 12.556 | 12.32% | 3.21% |

Baselines remain **702.323 ms prefill**, **11.309 ms verification**, **14.320 ms decode cycle**, and **3391.468 ms captured GPU window**. These modeled savings require matching benefits on every TP rank, critical-path realization, unchanged precision/acceptance and no added synchronization. Request-level E2E improvement still requires request timing and accepted-token measurements. The captured-window model reweights two clean prefill samples over four chunks and is approximate.

## Exact verification artifacts

- `moe_kernel_count_proof.csv`: min/max and mean launches for every MoE layer instance.
- `moe_substage_proof.csv`: explains every routed-core and surrounding MoE launch.
- `noncollective_event_evidence.csv`: original exact symbol, timestamp, duration, stream and correlation ID, indexed by vendor/phase/forward/layer.
- `noncollective_excluded_events.csv`: explicit audit of removed collectives, including fused collective kernels.
- `noncollective_categories.csv`: complete category partition of the retained target-step kernels.
- `noncollective_candidates.csv`: prior candidate list with collective tuning and collective-fusion proposals removed.
