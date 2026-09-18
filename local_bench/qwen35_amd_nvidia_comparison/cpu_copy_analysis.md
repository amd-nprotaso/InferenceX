# Qwen3.5 CPU copy and scalar-read investigation

**Confirmed: the AMD trace contains substantial CPU-blocking GPU scalar reads and a recurring decode event synchronization. It does not confirm that ordinary `aten::copy_` is generally slower on AMD. Neither trace contains an event literally named `copy_id`.** The likely relevant operations are `aten::copy_`, `aten::item`, `_local_scalar_dense`, `hipMemcpyWithStream`, and the sequence-length resolver.

[Exact runtime/caller evidence](cpu_copy_runtime_evidence.csv) · [Call-site totals](cpu_copy_by_callsite.csv) · [CPU scope totals](cpu_copy_scope_summary.csv) · [AITER stages](cpu_copy_aiter_by_stage.csv) · [Timing limitations](cpu_copy_validation.json)

## What the traces prove

| Operation | AMD count | AMD CPU time | NVIDIA count | NVIDIA CPU time |
|---|---:|---:|---:|
| `aten::copy_` | 2204 | 50.265 ms | 1888 | 282.601 ms |
| `aten::_to_copy` | 1177 | 47.891 ms | 902 | 144.241 ms |
| `aten::_local_scalar_dense` | 522 | 2261.733 ms | 315 | 271.301 ms |
| `aten::item` | 522 | 2262.102 ms | 315 | 271.525 ms |

These are CPU scope durations across the full captured trace, including target, draft and scheduler work. **Do not add nested scopes**: `item` contains `_local_scalar_dense`, and `_to_copy` usually contains `copy_`. Counts and CPU totals are also workload- and synchronization-placement dependent. Median `aten::copy_` duration is **5.52 µs AMD vs 6.63 µs NVIDIA**; its inclusive total is lower on AMD. The clear AMD excess is scalar extraction and its synchronization, not ordinary copy latency.

## Confirmed AMD call sites

| Site | Blocking calls | Total CPU time | Largest CPU call | Correlated GPU-copy time, summed |
|---|---:|---:|---:|---:|
| AITER update_single_wrapper | 8 | 1155.323 ms | 579.846 ms | 0.033232 ms |
| AITER forward_extend | 160 | 1105.258 ms | 38.223 ms | 0.814600 ms |
| Mamba clear_slots | 3 | 22.197 ms | 11.299 ms | 0.014997 ms |
| FLA prepare_chunk_indices | 8 | 10.947 ms | 5.994 ms | 0.041592 ms |

### AITER attention metadata: implicit scalar indexing

The longest calls are in `aiter_backend.py: update_single_wrapper`. The trace stack is:

```text
update_single_wrapper
  aten::item
    aten::_local_scalar_dense
      hipMemcpyWithStream   # 4 bytes, runtime kind=2
```

One call holds the CPU for **553.428 ms**, but its correlated GPU transfer takes **3.719 µs**. Another holds it for **579.846 ms**. Both longest calls occur while preparing **draft prefill** metadata behind already queued target GPU work. This is not a 553–580 ms data transfer.

The local source has a matching implicit scalar-read candidate:

```python
token_num = kv_indptr[-1]
kv_indices[token_num:] = kv_indices[0]
```

Using a device scalar as a Python slice boundary requires a host scalar. Prefer an already-known equivalent host token count, or a device kernel that accepts the device boundary. Preserve the padded-tail behavior and prove the host count equals the device cumulative count before substituting it.

[Local source: aiter_backend.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py:3069). The installed source has matching function locations and behavior; a source revision/digest was not embedded in the supplied trace.

### AITER full attention: repeated host reads during prefix prefill

`AITER forward_extend` makes **160 blocking scalar-copy calls**, totaling **1105.258 ms** of CPU time, for only **0.814600 ms** of correlated GPU-copy time. These occur in the two prefix-prefill chunks, across target and draft work; none of these calls is inside a `TARGET_VERIFY` step. Of the 160 calls, **150 are target prefill** and **10 are draft prefill**. Many individual calls are about 37–38 ms.

The corresponding local context-prefill branch contains:

```python
total_k = int(seq_lens.sum().item())
seq_ids = torch.repeat_interleave(torch.arange(bs, device=q.device), seq_lens)
asm_cp_ok = bool(int(page_slot.max().item()) < kv_pages.numel())
asm_cp_ok = int(tok_idx.max().item()) < key_buffer.shape[0]
```

Explicit `.item()` calls and dynamic-shape preparation force device results onto the CPU. Candidate fixes are to build reusable metadata once per batch, derive validated sizes from existing host metadata, provide a known `output_size` to `repeat_interleave` where valid, and move bounds validation to a device-side mechanism. **Do not remove out-of-bounds guards just to remove synchronization.**

[Local source: context-prefill branch](/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py:2596). The trace attributes calls to the function; individual source-expression attribution is inferred from the matching installed source, not recorded Python line-by-line execution.

### Linear attention: chunk-index round trip

`prepare_chunk_indices` transfers chunk counts to the CPU with `.tolist()`, constructs indices, and sends the result back to the GPU. In AMD this produces **8 blocking calls across four prefill passes**: four 4/8-byte reads plus four approximately 4 KB uploads. CPU runtime totals **10.947 ms**; GPU transfers total **41.592 µs**.

This path is also present on NVIDIA: eight `cudaMemcpyAsync` calls total **7.699 ms**, with associated synchronization calls. It is a shared algorithmic round trip, not exclusively an AMD issue. The function is tensor-cached; the observed work is once per prefill pass, not once per all 45 linear-attention layers.

[Local source: FLA index.py](/sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/index.py:16). Build indices directly on-device, or from host lengths already available to the scheduler, and reuse metadata across layers while respecting cache keys and chunk changes.

### Mamba slot clearing: copying a scalar zero

`clear_slots` has three 2-byte blocking HtoD copies totaling **22.197 ms** of CPU time. The two long calls take **10.874 ms** and **11.299 ms**; the actual three transfers total **14.997 µs**. The trace stack is `clear_slots → index_put_ → to/copy_ → hipMemcpyWithStream`.

The installed fast path uses `temporal[:, indices] = 0`, consistent with a host BF16 scalar being materialized on the device. Extend the existing slot-clear device kernel to clear temporal state, or use an appropriate device-side indexed fill. Preserve ordering on the forward stream. These long calls occur in prefill passes with other known GPU timestamp anomalies, so their host timestamps are stronger evidence than a detailed GPU critical-path estimate.

[Local source: memory_pool.py](/sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py:968).

## Decode: an explicit HIP host wait in sequence-length resolution

AMD `resolve_seq_lens_cpu` has **31 `hipEventSynchronize` calls**, totaling **289.313 ms**, or **9.333 ms per call**. Median is **9.585 ms**. It also makes 31 small `hipMemcpyWithStream` calls totaling just **0.542 ms**; the wait dominates the CPU scope.

The installed source explicitly distinguishes HIP:

```python
if _is_hip:
    # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
    self.publish_ready.synchronize()
else:
    self.publish_ready.wait()
```

**This confirms an AMD-specific host-synchronization path.** It explains where CPU scheduling overlap may be constrained, but does not prove the workaround is slower end-to-end. The source itself records a TPOT regression with the alternative. Re-test with the current runtime using a controlled comparison; do not simply replace the call. Preserve event freshness and sequence-length publication/consumption ordering.

[Local source: overlap_utils.py](/sgl-workspace/sglang/python/sglang/srt/managers/overlap_utils.py:472). NVIDIA also spends time waiting elsewhere: 31 result-processing event waits total **85.284 ms**, and its prefill allocation and slot-donation paths contain long scalar synchronization. Host wait location is affected by scheduling.

## Why the CPU totals are not E2E savings

The two AMD AITER call sites accumulate **2260.581 ms** of CPU blocking, but correlated GPU copies total only **0.847832 ms**. Approximately **99.79%** of the CPU-call intervals overlap recorded GPU activity—even after excluding four grossly inconsistent AMD GPU intervals (>10 ms). Thus the CPU is mostly waiting for GPU work already in flight. Deleting a scalar read cannot delete that computation.

Only **4.753 ms** within those AITER CPU-call intervals has no recorded GPU activity. This is **not** a total removable-overhead bound: it omits post-return effects, may contain unrelated gaps, and changes in scheduling can move waits elsewhere. The corresponding aggregate gaps from matched transfers to the next operation on that same stream are about **4.453 ms**, also observational rather than causal savings. Those gap columns are especially misleading on dedicated copy streams, where long idle intervals can be normal.

Likewise, the decode resolver wait overlaps **288.000 of 289.313 ms** with GPU activity. It is invalid to subtract **9.333 ms** from the **14.320 ms** decode cycle as an expected speedup.

GPU memcpy direction labels have an additional caveat: some AMD runtime calls report HIP kind=2 (device-to-host) while their correlated GPU event is labeled `Memcpy DtoD`. The report preserves both labels. The scalar-extraction CPU stack, runtime kind and byte count identify the host-read behavior; the GPU label alone should not be used to classify it.

## What to optimize next

1. **Remove repeated prefix-prefill metadata round trips in AITER.** Reuse validated batch metadata and avoid device scalar values controlling Python allocation/indexing. This can improve CPU submission and graph compatibility.
2. **Avoid implicit scalar extraction in `update_single_wrapper`.** Use a proven-equivalent host count or keep padding/indexing on the GPU.
3. **Remove the FLA chunk-index `.tolist()` round trip and scalar slot-clear copy.** These have clear dependency boundaries and smaller scope.
4. **A/B test the HIP publish-event workaround.** Measure GPU gaps, scheduler overlap, TPOT, accepted tokens and E2E latency under identical workload and all four ranks. Keep the current workaround until the alternative wins reproducibly.

No serving code was changed. The confirmed diagnosis is host/device synchronization in metadata and scheduling; a general AMD memory-copy-bandwidth problem or a specific `copy_id` function is not established.
