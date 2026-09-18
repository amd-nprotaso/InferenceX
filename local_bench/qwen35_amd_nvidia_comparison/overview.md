# Qwen3.5 kernel comparison: AMD MI355X vs NVIDIA B200

[CPU copy/scalar synchronization investigation](cpu_copy_analysis.md)

[Collective-excluded proof and revised priorities](noncollective_optimization_proof.md)

[PyTorch module sublists](modules.md) · [Module/kernel CSV](kernels_by_module.csv)

[Interactive per-layer report](comparison.html) · [Per-layer Markdown](comparison.md) · [Exact kernel inventory CSV](kernels_by_layer.csv) · [Kernel execution sequences](representative_sequences.csv) · [Layer timing CSV](layer_summary.csv)

The reports cover all **60 target-model layers**, with **45 linear-attention layers** and **15 full-attention layers**. Layer indices are zero-based. Full attention is at **3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59**; every other layer is linear attention. Each layer has separate AMD and NVIDIA inventories for prefill and decode, including projections, normalization, attention, MoE, and utility kernels.

**Decode here is speculative `TARGET_VERIFY bs=4`, not ordinary single-token decoding.** Each trace has four target prefill passes and 31 target-verification passes. Draft-model work is excluded. Both runs use TP=4, and these files represent TP-0 only.

The tables below compare functional stages. `…` abbreviates long symbols; full trace symbols and variants are preserved in the CSV and expandable HTML. Sharing a kernel name does not imply identical generated machine code.

## Prefill: linear_attention

Applies to layers 0, 1, 2, 4, 5, 6, …, 56, 57, 58. The per-layer report includes all variants observed across the four prefill chunks.

| Stage | AMD | NVIDIA |
|---|---|---|
| Input projections | rocBLAS/Tensile `Cijk_…` GEMMs | `nvjet_sm100_tst_…` GEMMs |
| QKV/Z/B/A layout | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | Same named kernel |
| Causal convolution | **`kernel_0`**, identified by stack as `causal_conv1d_flydsl.py: causal_conv1d_fn` | **`_causal_conv1d_fwd_kernel`** |
| Split Q/K/V | `fused_qkv_split_gdn_prefill_kernel` | Same named kernel |
| Gating | `fused_gdn_gating_kernel` | Same named kernel |
| Q/K normalization | `l2norm_fwd_kernel` | Same named kernel |
| Cumulative gating | `chunk_local_cumsum_scalar_kernel` | Same named kernel |
| KKT solve | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | Same named kernel |
| Recompute W/U | `recompute_w_u_fwd_kernel` | Same named kernel |
| State recurrence | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | Same named kernel |
| Output calculation | `chunk_fwd_kernel_o` | Same named kernel |
| Output normalization | `_layer_norm_fwd_1pass_kernel` | Same named kernel |
| Output projection | rocBLAS/Tensile GEMM | `nvjet_sm100_tst_…` GEMM |

## Prefill: full_attention

Applies to layers 3, 7, 11, …, 59.

| Stage | AMD | NVIDIA |
|---|---|---|
| QKV projection | rocBLAS/Tensile GEMM | `nvjet_sm100_tst_…` GEMM |
| Q/K normalization, rotary embedding, gate preparation | **`_fused_qk_gemma_rmsnorm_gate_kernel` + `_triton_mrope_forward_fused`** | **`_fused_qk_rmsnorm_rope_gate_kernel`** |
| Cache preparation | **`reshape_and_cache_flash`** | **`fused_fp8_qkv_kv_cache_kernel<…>`** |
| Attention | **`aiter::fmha_fwd_hd256_fp8_causal_group_gfx950`** | **`fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext`** |
| Output gating | `_fused_sigmoid_mul_kernel` | Same named kernel |
| Output projection | rocBLAS/Tensile GEMM | `nvjet_sm100_tst_…` GEMM |

## Decode: linear_attention — TARGET_VERIFY

Applies to the same 45 linear-attention layers.

| Stage | AMD | NVIDIA |
|---|---|---|
| Input projections | rocBLAS/Tensile `Cijk_…MT32x16x1024…` and `…MT16x16x1024…` | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT`, `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT`, and `cublasLt::splitKreduce_kernel<…>` |
| QKV/Z/B/A layout | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | Same named kernel |
| Convolution state update | `_causal_conv1d_update_kernel` | Same named kernel |
| Split Q/K/V | `fused_qkv_split_gdn_prefill_kernel` | Same named kernel, despite “prefill” in its name |
| Gated delta-rule update | **`fused_sigmoid_gating_delta_rule_update_kernel`** | **Same named kernel** |
| Output normalization | `_layer_norm_fwd_1pass_kernel` | Same named kernel |
| Output projection | `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` |

## Decode: full_attention — TARGET_VERIFY

Applies to the same 15 full-attention layers.

| Stage | AMD | NVIDIA |
|---|---|---|
| QKV projection | rocBLAS/Tensile `Cijk_…MT32x16x1024…` | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` + `cublasLt::splitKreduce_kernel<…>` |
| Q/K normalization, rotary embedding, gate preparation | `_fused_qk_gemma_rmsnorm_gate_kernel` + `_triton_mrope_forward_fused` | `_fused_qk_rmsnorm_rope_gate_kernel` |
| Cache preparation | `reshape_and_cache_flash` | `fused_fp8_qkv_kv_cache_kernel<…>` |
| Attention | **`kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu_2_num_stages_2_ALL_DECODE_0_SHUFFLED_KV_CACHE_0_IS_Q_FP8_0_IS_KV_FP8_1`** | **`fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen`** |
| Attention segment reduction | **`reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128`** | No separate segment-reduction kernel observed in these target-layer sequences |
| Output gating | `_fused_sigmoid_mul_kernel` | Same named kernel |
| Output projection | `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` |

## Other kernels inside each decoder layer

These are included in the detailed report for both attention types.

| Component | AMD | NVIDIA |
|---|---|---|
| Layer normalization | `_gemma_rmsnorm_kernel`, `_gemma_fused_add_rmsnorm_kernel` | FlashInfer/CUTLASS `RMSNormKernel`, `FusedAddRMSNormKernel`; decode also uses fused all-reduce/normalization |
| Prefill communication | `ncclDevKernel_Generic_1` | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` |
| Decode communication | `aiter::cross_device_reduce_1stage` | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` |
| Shared-expert activation | `sgl_hip::activation::act_and_mul_kernel` | `(anonymous namespace)::act_and_mul_kernel` |
| Prefill MoE routing | `vllm::moe::topkGatingSoftmax`, `aiter::opus_moe_sorting_entry<…>` variants | `routingIndicesHistogramScoresKernel`, `routingInitExpertCounts`, `routingIndicesCoopKernel` |
| Prefill MoE quantization | `aiter::dynamic_per_group_scaled_quant_kernel`, `aiter::mxfp4_moe_sort_kernel` | FlashInfer/CUTLASS `NVFP4QuantizeTMAKernel` |
| Prefill expert GEMMs | `mfma_moe1_silu_mul_afp4_wfp4_bf16_…`, `mfma_moe2_afp4_wfp4_bf16_…` | `bmm_E2m1_E2m1E2m1_…`, `bmm_Bfloat16_E2m1E2m1_…` |
| Decode MoE | `topkGatingSoftmax` + **`aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256`** | `routingIndicesDynBlockKernel`, `NVFP4QuantizeLinearKernel`, two `bmm_…` expert GEMMs, `finalizeKernel` |
| Shared/routed expert output combination | `_fused_gate_sigmoid_mul_add_kernel` | Same named kernel |

## Attribution and timing

GPU launches and graph replays are linked to steps by correlation ID. Each target forward must have 60 MoE end markers and exactly the expected linear/full-attention marker in every layer. All **240 AMD prefill layer markers** were also checked against CPU decoder-module intervals. NVIDIA compiled execution and decode graph replay use **inferred execution-order layer indices**, supported by the repeated 3-linear/1-full sequence.

Standalone reductions between layers are recorded in [boundary_collectives.csv](boundary_collectives.csv). A fused NVIDIA reduction plus next-layer normalization cannot be divided, so it is assigned to the incoming layer. This affects comparisons of whole-layer timing.

Timings are summed GPU kernel durations, not wall-clock latency. Prefill chunk shapes differ between the two traces, and initialization and stream overlap can affect the totals. Use these files to compare implementation and kernel inventories; they are not a controlled hardware speed comparison. [validation.json](validation.json) records input paths, step shapes, coverage, and the AMD convolution stack evidence.


## Optimization analysis and timing caveat

[AMD optimization plan](optimization_plan.md) · [Interactive impact calculator](optimization_impact.html). Two AMD prefill passes contain overlapping same-stream durations, including 641 ms and 279 ms outliers; raw timing tables retain them. Use the clean-pass budgets in the optimization plan for impact estimates.
