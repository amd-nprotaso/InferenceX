"""Summarize confirmed CPU copy/scalar synchronization without claiming wait is savings."""
import csv,json,html
from pathlib import Path
from collections import defaultdict
P=Path('qwen35_amd_nvidia_comparison')
def read(n):return list(csv.DictReader((P/n).open()))
scopes=read('cpu_copy_scope_summary.csv');sites=read('cpu_copy_by_callsite.csv');events=read('cpu_copy_runtime_evidence.csv')
def find(v,site,op):return next(r for r in sites if r['vendor']==v and r['site']==site and r['operation']==op)
def num(r,k):return float(r[k])
aiter=[find('AMD','AITER update_single_wrapper','hipMemcpyWithStream'),find('AMD','AITER forward_extend','hipMemcpyWithStream')]
wait=sum(num(r,'cpu_total_ms') for r in aiter);busy=sum(num(r,'gpu_busy_during_call_ms') for r in aiter);copy=sum(num(r,'matched_gpu_total_ms') for r in aiter)
resolve=next(r for r in sites if r['vendor']=='AMD' and 'resolve_seq_lens_cpu' in r['site'] and r['operation']=='hipEventSynchronize')
selected=[r for r in events if r['vendor']=='AMD' and r['site'].startswith('AITER') and r['operation']=='hipMemcpyWithStream']
stage=defaultdict(lambda:[0,0.])
for r in selected:
    a=stage[r['site'],r['stage']];a[0]+=1;a[1]+=float(r['duration_us'])/1000
stage_rows=[dict(site=site,stage=s,count=n,cpu_ms=t) for (site,s),(n,t) in stage.items()]
with (P/'cpu_copy_aiter_by_stage.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(stage_rows[0]));w.writeheader();w.writerows(stage_rows)
lines=['# Qwen3.5 CPU copy and scalar-read investigation','',
'**Confirmed: the AMD trace contains substantial CPU-blocking GPU scalar reads and a recurring decode event synchronization. It does not confirm that ordinary `aten::copy_` is generally slower on AMD. Neither trace contains an event literally named `copy_id`.** The likely relevant operations are `aten::copy_`, `aten::item`, `_local_scalar_dense`, `hipMemcpyWithStream`, and the sequence-length resolver.','',
'[Exact runtime/caller evidence](cpu_copy_runtime_evidence.csv) · [Call-site totals](cpu_copy_by_callsite.csv) · [CPU scope totals](cpu_copy_scope_summary.csv) · [AITER stages](cpu_copy_aiter_by_stage.csv) · [Timing limitations](cpu_copy_validation.json)','',
'## What the traces prove','',
'| Operation | AMD count | AMD CPU time | NVIDIA count | NVIDIA CPU time |','|---|---:|---:|---:|']
for op in ('aten::copy_','aten::_to_copy','aten::_local_scalar_dense','aten::item'):
    a=next(r for r in scopes if r['vendor']=='AMD' and r['operation']==op);n=next(r for r in scopes if r['vendor']=='NVIDIA' and r['operation']==op)
    lines.append(f'| `{op}` | {a["count"]} | {num(a,"sum_ms"):.3f} ms | {n["count"]} | {num(n,"sum_ms"):.3f} ms |')
lines+=['',
'These are CPU scope durations across the full captured trace, including target, draft and scheduler work. **Do not add nested scopes**: `item` contains `_local_scalar_dense`, and `_to_copy` usually contains `copy_`. Counts and CPU totals are also workload- and synchronization-placement dependent. Median `aten::copy_` duration is **5.52 µs AMD vs 6.63 µs NVIDIA**; its inclusive total is lower on AMD. The clear AMD excess is scalar extraction and its synchronization, not ordinary copy latency.','',
'## Confirmed AMD call sites','',
'| Site | Blocking calls | Total CPU time | Largest CPU call | Correlated GPU-copy time, summed |','|---|---:|---:|---:|---:|']
for site in ('AITER update_single_wrapper','AITER forward_extend','Mamba clear_slots','FLA prepare_chunk_indices'):
    r=find('AMD',site,'hipMemcpyWithStream');lines.append(f'| {site} | {r["count"]} | {num(r,"cpu_total_ms"):.3f} ms | {num(r,"cpu_max_us")/1000:.3f} ms | {num(r,"matched_gpu_total_ms"):.6f} ms |')
lines+=['',
'### AITER attention metadata: implicit scalar indexing','',
'The longest calls are in `aiter_backend.py: update_single_wrapper`. The trace stack is:',
'',
'```text\nupdate_single_wrapper\n  aten::item\n    aten::_local_scalar_dense\n      hipMemcpyWithStream   # 4 bytes, runtime kind=2\n```','',
'One call holds the CPU for **553.428 ms**, but its correlated GPU transfer takes **3.719 µs**. Another holds it for **579.846 ms**. Both longest calls occur while preparing **draft prefill** metadata behind already queued target GPU work. This is not a 553–580 ms data transfer.','',
'The local source has a matching implicit scalar-read candidate:',
'',
'```python\ntoken_num = kv_indptr[-1]\nkv_indices[token_num:] = kv_indices[0]\n```','',
'Using a device scalar as a Python slice boundary requires a host scalar. Prefer an already-known equivalent host token count, or a device kernel that accepts the device boundary. Preserve the padded-tail behavior and prove the host count equals the device cumulative count before substituting it.','',
'[Local source: aiter_backend.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py:3069). The installed source has matching function locations and behavior; a source revision/digest was not embedded in the supplied trace.','',
'### AITER full attention: repeated host reads during prefix prefill','',
'`AITER forward_extend` makes **160 blocking scalar-copy calls**, totaling **1105.258 ms** of CPU time, for only **0.814600 ms** of correlated GPU-copy time. These occur in the two prefix-prefill chunks, across target and draft work; none of these calls is inside a `TARGET_VERIFY` step. Of the 160 calls, **150 are target prefill** and **10 are draft prefill**. Many individual calls are about 37–38 ms.','',
'The corresponding local context-prefill branch contains:',
'',
'```python\ntotal_k = int(seq_lens.sum().item())\nseq_ids = torch.repeat_interleave(torch.arange(bs, device=q.device), seq_lens)\nasm_cp_ok = bool(int(page_slot.max().item()) < kv_pages.numel())\nasm_cp_ok = int(tok_idx.max().item()) < key_buffer.shape[0]\n```','',
'Explicit `.item()` calls and dynamic-shape preparation force device results onto the CPU. Candidate fixes are to build reusable metadata once per batch, derive validated sizes from existing host metadata, provide a known `output_size` to `repeat_interleave` where valid, and move bounds validation to a device-side mechanism. **Do not remove out-of-bounds guards just to remove synchronization.**','',
'[Local source: context-prefill branch](/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py:2596). The trace attributes calls to the function; individual source-expression attribution is inferred from the matching installed source, not recorded Python line-by-line execution.','',
'### Linear attention: chunk-index round trip','',
'`prepare_chunk_indices` transfers chunk counts to the CPU with `.tolist()`, constructs indices, and sends the result back to the GPU. In AMD this produces **8 blocking calls across four prefill passes**: four 4/8-byte reads plus four approximately 4 KB uploads. CPU runtime totals **10.947 ms**; GPU transfers total **41.592 µs**.','',
'This path is also present on NVIDIA: eight `cudaMemcpyAsync` calls total **7.699 ms**, with associated synchronization calls. It is a shared algorithmic round trip, not exclusively an AMD issue. The function is tensor-cached; the observed work is once per prefill pass, not once per all 45 linear-attention layers.','',
'[Local source: FLA index.py](/sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/index.py:16). Build indices directly on-device, or from host lengths already available to the scheduler, and reuse metadata across layers while respecting cache keys and chunk changes.','',
'### Mamba slot clearing: copying a scalar zero','',
'`clear_slots` has three 2-byte blocking HtoD copies totaling **22.197 ms** of CPU time. The two long calls take **10.874 ms** and **11.299 ms**; the actual three transfers total **14.997 µs**. The trace stack is `clear_slots → index_put_ → to/copy_ → hipMemcpyWithStream`.','',
'The installed fast path uses `temporal[:, indices] = 0`, consistent with a host BF16 scalar being materialized on the device. Extend the existing slot-clear device kernel to clear temporal state, or use an appropriate device-side indexed fill. Preserve ordering on the forward stream. These long calls occur in prefill passes with other known GPU timestamp anomalies, so their host timestamps are stronger evidence than a detailed GPU critical-path estimate.','',
'[Local source: memory_pool.py](/sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py:968).','',
'## Decode: an explicit HIP host wait in sequence-length resolution','',
f'AMD `resolve_seq_lens_cpu` has **{resolve["count"]} `hipEventSynchronize` calls**, totaling **{num(resolve,"cpu_total_ms"):.3f} ms**, or **{num(resolve,"cpu_total_ms")/int(resolve["count"]):.3f} ms per call**. Median is **{num(resolve,"cpu_median_us")/1000:.3f} ms**. It also makes 31 small `hipMemcpyWithStream` calls totaling just **0.542 ms**; the wait dominates the CPU scope.','',
'The installed source explicitly distinguishes HIP:',
'',
'```python\nif _is_hip:\n    # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.\n    self.publish_ready.synchronize()\nelse:\n    self.publish_ready.wait()\n```','',
'**This confirms an AMD-specific host-synchronization path.** It explains where CPU scheduling overlap may be constrained, but does not prove the workaround is slower end-to-end. The source itself records a TPOT regression with the alternative. Re-test with the current runtime using a controlled comparison; do not simply replace the call. Preserve event freshness and sequence-length publication/consumption ordering.','',
'[Local source: overlap_utils.py](/sgl-workspace/sglang/python/sglang/srt/managers/overlap_utils.py:472). NVIDIA also spends time waiting elsewhere: 31 result-processing event waits total **85.284 ms**, and its prefill allocation and slot-donation paths contain long scalar synchronization. Host wait location is affected by scheduling.','',
'## Why the CPU totals are not E2E savings','',
f'The two AMD AITER call sites accumulate **{wait:.3f} ms** of CPU blocking, but correlated GPU copies total only **{copy:.6f} ms**. Approximately **{100*busy/wait:.2f}%** of the CPU-call intervals overlap recorded GPU activity—even after excluding four grossly inconsistent AMD GPU intervals (>10 ms). Thus the CPU is mostly waiting for GPU work already in flight. Deleting a scalar read cannot delete that computation.','',
f'Only **{wait-busy:.3f} ms** within those AITER CPU-call intervals has no recorded GPU activity. This is **not** a total removable-overhead bound: it omits post-return effects, may contain unrelated gaps, and changes in scheduling can move waits elsewhere. The corresponding aggregate gaps from matched transfers to the next operation on that same stream are about **4.453 ms**, also observational rather than causal savings. Those gap columns are especially misleading on dedicated copy streams, where long idle intervals can be normal.','',
f'Likewise, the decode resolver wait overlaps **{num(resolve,"gpu_busy_during_call_ms"):.3f} of {num(resolve,"cpu_total_ms"):.3f} ms** with GPU activity. It is invalid to subtract **9.333 ms** from the **14.320 ms** decode cycle as an expected speedup.','',
'GPU memcpy direction labels have an additional caveat: some AMD runtime calls report HIP kind=2 (device-to-host) while their correlated GPU event is labeled `Memcpy DtoD`. The report preserves both labels. The scalar-extraction CPU stack, runtime kind and byte count identify the host-read behavior; the GPU label alone should not be used to classify it.','',
'## What to optimize next','',
'1. **Remove repeated prefix-prefill metadata round trips in AITER.** Reuse validated batch metadata and avoid device scalar values controlling Python allocation/indexing. This can improve CPU submission and graph compatibility.','2. **Avoid implicit scalar extraction in `update_single_wrapper`.** Use a proven-equivalent host count or keep padding/indexing on the GPU.','3. **Remove the FLA chunk-index `.tolist()` round trip and scalar slot-clear copy.** These have clear dependency boundaries and smaller scope.','4. **A/B test the HIP publish-event workaround.** Measure GPU gaps, scheduler overlap, TPOT, accepted tokens and E2E latency under identical workload and all four ranks. Keep the current workaround until the alternative wins reproducibly.','',
'No serving code was changed. The confirmed diagnosis is host/device synchronization in metadata and scheduling; a general AMD memory-copy-bandwidth problem or a specific `copy_id` function is not established.','']
(P/'cpu_copy_analysis.md').write_text('\n'.join(lines))
# A compact view of the long copy and its GPU overlap.
e=next(r for r in events if r['vendor']=='AMD' and r['site']=='AITER update_single_wrapper' and float(r['duration_us'])>500000)
page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>CPU copy and synchronization evidence</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#17212d}p{line-height:1.6}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}.bar{height:28px;background:#3478c7;color:white;margin:12px 0;padding-left:6px}.cpu{background:#c1504c}code{overflow-wrap:anywhere}</style><h1>AMD CPU copy/scalar-read evidence</h1><p><b>Confirmed: blocking scalar reads and an explicit HIP event wait. Not confirmed: a general AMD copy-bandwidth problem.</b></p><p><a href="cpu_copy_analysis.md">Full investigation and source links</a> · <a href="cpu_copy_runtime_evidence.csv">Exact event evidence</a></p>'''
page+=f'<h2>One 4-byte read</h2><p>Correlation {e["correlation"]}; {html.escape(e["stage"])}.</p><div class="bar cpu">CPU blocked: {float(e["duration_us"])/1000:.3f} ms</div><div class="bar" style="width:{100*float(e["gpu_busy_during_cpu_call_us"])/float(e["duration_us"]):.2f}%">Overlapping GPU activity: {float(e["gpu_busy_during_cpu_call_us"])/1000:.3f} ms</div><p>Actual correlated GPU transfer: <b>{float(e["gpu_transfer_us"]):.3f} µs</b>. The long CPU scope is mostly waiting, not moving four bytes.</p><h2>AMD call sites</h2><table><tr><th>Call site</th><th>Calls</th><th>CPU blocking ms</th><th>GPU transfer ms</th></tr>'
for site in ('AITER update_single_wrapper','AITER forward_extend','Mamba clear_slots','FLA prepare_chunk_indices'):
    r=find('AMD',site,'hipMemcpyWithStream');page+=f'<tr><td>{site}</td><td>{r["count"]}</td><td>{num(r,"cpu_total_ms"):.3f}</td><td>{num(r,"matched_gpu_total_ms"):.6f}</td></tr>'
page+='</table><p>Do not add nested CPU scopes or treat CPU wait time as removable E2E latency. See the report for timestamp anomalies, direction-label discrepancies, and the existing MI355 event-wait workaround.</p></html>'
(P/'cpu_copy_evidence.html').write_text(page)
for n in ['overview.md','noncollective_optimization_proof.md']:
    p=P/n;s=p.read_text();link='[CPU copy/scalar synchronization investigation](cpu_copy_analysis.md)'
    if link not in s:
        first,rest=s.split('\n',1);p.write_text(first+'\n\n'+link+'\n'+rest)
print('AITER blocking ms',wait,'GPU copy ms',copy,'GPU overlap %',busy/wait*100)
print('Decode resolver mean ms',num(resolve,'cpu_total_ms')/31)
print(json.dumps(stage_rows,indent=2))
