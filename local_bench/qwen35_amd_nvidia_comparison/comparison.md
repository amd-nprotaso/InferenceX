# Qwen3.5: AMD MI355X vs NVIDIA B200

[PyTorch module sublists](modules.md) · [Module/kernel CSV](kernels_by_module.csv)

These TP-0 traces contain a 60-layer target model (45 linear-attention, 15 full-attention) and a separate draft model. Decode means speculative TARGET_VERIFY, bs=4; neither trace contains a plain DECODE step. Only target-model forwards with 60 MoE end markers are included: four prefill and 31 verification forwards per vendor. Layer numbers are zero-based execution order. Full-attention layers are 3, 7, 11, ..., 59. GPU kernels are linked to CPU launch/graph replay calls by correlation ID, then to the enclosing step. Layer boundaries use the repeated MoE end marker and attention signatures; all 240 AMD prefill layer markers were checked against CPU decoder-module intervals. NVIDIA compiled execution and graph replay layer indices are inferred from the validated repeated 3-linear/1-full ordering, not explicit module IDs. Each row includes the layer input normalization, attention, projections, shared expert and MoE through _fused_gate_sigmoid_mul_add_kernel. Standalone output reductions between layers are in boundary_collectives.csv. NVIDIA fused reduction+normalization is assigned to the following layer because it cannot be divided. Timing is summed kernel duration, not elapsed latency; streams can overlap. Prefill averages combine different chunk shapes and may include lazy initialization, so these are descriptive trace timings, not controlled speed ratios. Exact symbols, including templates and utility kernels, are preserved in CSV and expandable HTML. Labels in overview tables are shortened. Unassigned kernels include metadata, embedding, logits, sampling, draft work, and inter-layer boundaries; they are not silently attributed to target layers.

## Inputs

- AMD: `/var/home/my_InferenceX/traces/amd/1789642053.9277868-TP-0.trace.json`; TP=4; target step shapes: `{'step[EXTEND bs=1 toks=32768]': 1, 'step[EXTEND bs=2 toks=32766]': 1, 'step[EXTEND bs=2 toks=32768]': 1, 'step[EXTEND bs=1 toks=32464]': 1, 'step[TARGET_VERIFY bs=4]': 31}`
- NVIDIA: `/var/home/my_InferenceX/traces/nvidia/1788441037.314373-TP-0.trace.json`; TP=4; target step shapes: `{'step[EXTEND bs=1 toks=32768]': 2, 'step[EXTEND bs=2 toks=32766]': 1, 'step[EXTEND bs=1 toks=32464]': 1, 'step[TARGET_VERIFY bs=4]': 31}`

AMD `kernel_0` host stack: ['causal_conv1d_flydsl.py(128): causal_conv1d_fn', 'flydsl/compiler/jit_function.py(1632): __call__', 'flydsl/compiler/jit_executor.py(210): __call__', '<flydsl-dispatch>(48): dispatch']

## Per-layer kernel inventory

For each vendor: exact-symbol entries are aggregated across all target forwards in that phase. Counts are calls per forward. CSV contains full signatures; this report uses shortened labels.


## Prefill: linear_attention

### Layer 0

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_rmsnorm_kernel` | 1 | `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign16o40961_tensorptrbf1...` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `at::native::unrolled_elementwise_kernel` | 1 | `at::native::unrolled_elementwise_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 4 | `at::native::vectorized_elementwise_kernel` | 4 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `at::native::(anonymous namespace)::CatArrayBatchedCopy_contig` | 1 | `at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 1.25 | `at::native::unrolled_elementwise_kernel` | 1.5 |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 | `at_cuda_detail::cub::detail::scan::DeviceScanInitKernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `at_cuda_detail::cub::detail::scan::DeviceScanKernel` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `chunk_fwd_kernel_o` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `_gemma_fused_add_rmsnorm_kernel` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `vllm::moe::topkGatingSoftmax` | 1 | `memcpy128` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `moe_reduction_kernel_0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 1

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 2

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 4

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 5

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 6

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 8

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 9

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1.25 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 10

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 12

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 13

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 14

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 16

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 17

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 18

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 20

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 21

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 22

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 24

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 25

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 26

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 28

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 29

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 30

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 32

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 33

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 34

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 36

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 37

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 38

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 40

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 41

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 42

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 44

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 45

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 46

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 0.75 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 48

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 49

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 50

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 52

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 53

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 54

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 56

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 57

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |
### Layer 58

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `rocBLAS/Tensile MT32x192x128` | 0.75 | `nvjet_sm100_tst_32x256_64x6_1x4_h_bz_TNN` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 1 | `at::native::vectorized_elementwise_kernel` | 1 |
| `at::native::index_elementwise_kernel` | 1.5 | `at::native::index_elementwise_kernel` | 1.75 |
| `at::native::index_elementwise_kernel` | 2 | `at::native::index_elementwise_kernel` | 2 |
| `kernel_0` | 1 | `_causal_conv1d_fwd_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `fused_gdn_gating_kernel` | 1 | `fused_gdn_gating_kernel` | 1 |
| `l2norm_fwd_kernel` | 2 | `l2norm_fwd_kernel` | 2 |
| `chunk_local_cumsum_scalar_kernel` | 1 | `chunk_local_cumsum_scalar_kernel` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `at::native::vectorized_elementwise_kernel` | 2 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | 1 |
| `recompute_w_u_fwd_kernel` | 1 | `recompute_w_u_fwd_kernel` | 1 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1 |
| `chunk_fwd_kernel_o` | 1 | `chunk_fwd_kernel_o` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.25 | `at::native::unrolled_elementwise_kernel` | 0.5 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy128` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 | `at::native::vectorized_gather_kernel` | 0.25 |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `at::native::vectorized_gather_kernel` | 0.5 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `rocBLAS/Tensile MT32x224x128` | 0.25 |  |  |

## Prefill: full_attention

### Layer 3

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 7

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 11

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 15

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 19

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 23

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 27

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 31

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 35

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 39

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 43

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 47

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 51

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 55

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |
### Layer 59

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign128...` | 2 |
| `rocBLAS/Tensile MT256x256x64` | 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 3 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `_triton_mrope_forward_fused` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `reshape_and_cache_flash` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16VarSeqQ128Kv128PersistentContext` | 1 |
| `at::native::vectorized_elementwise_kernel` | 2 | `_fused_sigmoid_mul_kernel` | 1 |
| `aiter::fmha_fwd_hd256_fp8_causal_group_gfx950` | 1 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 2 |
| `ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)` | 1 | `memcpy128` | 1 |
| `rocBLAS/Tensile MT256x256x64` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeTMAKernel_object_at__CopyAtom_ThrID10_TVLayoutSrc1819201_TVLayoutDst1819201_Val...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesHistogramScoresKernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingInitExpertCounts` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `moe::dev::routing::routingIndicesCoopKernel` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_bN_ldgsts_ldgstsSf_rg...` | 1 |
| `aiter::opus_moe_sorting_entry` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `aiter::dynamic_per_group_scaled_quant_kernel` | 2 | `moe::dev::finalize::finalizeKernelVecLoad` | 1 |
| `aiter::mxfp4_moe_sort_kernel` | 1 | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_xcd4_v32` | 1 | `at::native::vectorized_elementwise_kernel` | 0.5 |
| `aiter::mxfp4_moe_sort_kernel` | 1 |  |  |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale_fix3_fp4opt_v1_persist_cu256_sbm128_acc0` | 1 |  |  |
| `moe_reduction_kernel_0` | 1 |  |  |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 |  |  |
| `rocBLAS/Tensile MT256x256x64` | 1 |  |  |
| `at::native::unrolled_elementwise_kernel` | 0.5 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 1.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel` | 1 |  |  |
| `(anonymous namespace)::elementwise_kernel_with_index` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 0.25 |  |  |
| `compute_cuda_kernel` | 0.5 |  |  |
| `at::native::_scatter_gather_elementwise_kernel` | 0.5 |  |  |
| `at::native::index_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_elementwise_kernel` | 1.5 |  |  |
| `at::native::index_elementwise_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::reduce_kernel` | 1 |  |  |
| `at::native::vectorized_elementwise_kernel` | 0.5 |  |  |
| `at::native::vectorized_gather_kernel` | 1 |  |  |
| `at::native::elementwise_kernel_manual_unroll` | 0.5 |  |  |

## Decode (TARGET_VERIFY): linear_attention

### Layer 0

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_rmsnorm_kernel` | 1 | `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64409640961_tensorptrbf16gmemalign16o40961_tensorptrbf1...` | 1 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `_gemma_fused_add_rmsnorm_kernel` | 1 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 1

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 2

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 4

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 5

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 6

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 8

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 9

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 10

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 12

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 13

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 14

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 16

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 17

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 18

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 20

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 21

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 22

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 24

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 25

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 26

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 28

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 29

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 30

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 32

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 33

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 34

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 36

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 37

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 38

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 40

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 41

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 42

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 44

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 45

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 46

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 48

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 49

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 50

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 52

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 53

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 54

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 56

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 57

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 58

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_4x1_v_bz_TNT` | 1 |
| `rocBLAS/Tensile MT16x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_1x1_h_bz_splitK_TNT` | 1 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_causal_conv1d_update_kernel` | 1 | `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 1 |
| `fused_qkv_split_gdn_prefill_kernel` | 1 | `_causal_conv1d_update_kernel` | 1 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 1 | `fused_qkv_split_gdn_prefill_kernel` | 1 |
| `_layer_norm_fwd_1pass_kernel` | 1 | `fused_sigmoid_gating_delta_rule_update_kernel` | 1 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `_layer_norm_fwd_1pass_kernel` | 1 |
| `aiter::cross_device_reduce_1stage` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `memcpy32_post` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
|  |  | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
|  |  | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |

## Decode (TARGET_VERIFY): full_attention

### Layer 3

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 7

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 11

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 15

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 19

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 23

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 27

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 31

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 35

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 39

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 43

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 47

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 51

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 55

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |
### Layer 59

| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |
|---|---:|---|---:|
| `_gemma_fused_add_rmsnorm_kernel` | 2 | `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` | 2 |
| `rocBLAS/Tensile MT32x16x1024` | 1 | `nvjet_sm100_tst_64x16_64x16_2x1_2cta_v_bz_splitK_TNT` | 1 |
| `_fused_qk_gemma_rmsnorm_gate_kernel` | 1 | `cublasLt::splitKreduce_kernel` | 3 |
| `_triton_mrope_forward_fused` | 1 | `_fused_qk_rmsnorm_rope_gate_kernel` | 1 |
| `reshape_and_cache_flash` | 1 | `at::native::vectorized_elementwise_kernel` | 2 |
| `at::native::vectorized_elementwise_kernel` | 1 | `(anonymous namespace)::fused_fp8_qkv_kv_cache_kernel` | 1 |
| `kernel_unified_attention_3d_num_query_heads_8_num_queries_per_kv_8_BLOCK_SIZE_16_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128_num_warps_2_waves_per_eu...` | 1 | `fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausalP16MultiCtasKvVarSeqQ32Kv128StaticSwapsAbForGen` | 1 |
| `reduce_segments_num_query_heads_8_TILE_SIZE_16_HEAD_SIZE_256_NUM_SEGMENTS_PER_SEQ_128` | 1 | `_fused_sigmoid_mul_kernel` | 1 |
| `_fused_sigmoid_mul_kernel` | 1 | `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_TNT` | 2 |
| `hgemm_bf16_16x64x256x3_SPK2_W1x1x2_BLDS1_TN_AS1_0` | 1 | `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 2 |
| `aiter::cross_device_reduce_1stage` | 1 | `memcpy32_post` | 1 |
| `hgemm_bf16_16x64x64x8_SPK8_W1x2x1_BLDS1_TN_AS1_0` | 2 | `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o409640961_tensorptri8gmemalign...` | 1 |
| `sgl_hip::activation::act_and_mul_kernel` | 1 | `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 1 |
| `rocBLAS/Tensile MT16x16x128` | 1 | `(anonymous namespace)::act_and_mul_kernel` | 1 |
| `vllm::moe::topkGatingSoftmax` | 1 | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_...` | 1 |
| `aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256` | 1 | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 1 |
| `_fused_gate_sigmoid_mul_add_kernel` | 1 | `moe::dev::finalize::finalizeKernel` | 1 |
|  |  | `_fused_gate_sigmoid_mul_add_kernel` | 1 |

Tables list independent vendor inventories in first-seen order. Rows do not assert one-to-one equivalence between individual AMD and NVIDIA kernels.
