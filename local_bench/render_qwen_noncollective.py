"""Render collective-excluded evidence and revised optimization priorities."""
import csv,json,html
from collections import defaultdict
from pathlib import Path
P=Path('qwen35_amd_nvidia_comparison')
def read(name):return list(csv.DictReader((P/name).open()))
proof=read('moe_kernel_count_proof.csv');cats=read('noncollective_categories.csv');stages=read('moe_substage_proof.csv');kernels=read('noncollective_kernel_summary.csv');base=json.loads((P/'optimization_baselines.json').read_text())
lookup={(r['vendor'],r['phase'],r['category']):r for r in cats}
moe={(r['vendor'],r['phase'],r['scope']):r for r in proof}
def val(v,p,c,key='sum_gpu_ms_per_forward'):return float(lookup[v,p,c][key])
lines=['# AMD vs NVIDIA: non-collective kernel evidence and optimization priorities','',
'**Collective operations are excluded from this analysis. The claim “AMD uses more MoE kernels” is true for prefill and false for decode in these traces.** Counts are verified per layer, not extrapolated from one example. More launches establish implementation fragmentation, but do not establish that launch overhead caused a latency gap.','',
'[Count proof CSV](moe_kernel_count_proof.csv) · [MoE substages](moe_substage_proof.csv) · [Per-layer counts/times](noncollective_per_layer.csv) · [Exact event evidence](noncollective_event_evidence.csv) · [Kernel summary](noncollective_kernel_summary.csv) · [Visual proof](noncollective_proof.html)','',
'## Scope and comparison rules','',
'- Prefill uses target forwards 0 and 3 in both traces: matching `bs=1, toks=32768` and `bs=1, toks=32464` labels. AMD forwards 1 and 2 have inconsistent overlapping timestamps and are excluded. Matching labels do not prove equal cached context, request history, precision or all runtime settings.','- Decode uses all 31 `TARGET_VERIFY bs=4` target forwards per vendor; draft-model work is excluded. This is speculative verification, not plain single-token decode.','- All-reduce and all-gather kernels are removed, including boundary collectives and NVIDIA fused all-reduce/normalization kernels. Consequently NVIDIA normalization embedded in those kernels cannot be timed separately; its absence is not a zero-cost normalization result.','- “Kernels” below means GPU kernel **invocations**, including repeated symbols, not unique kernel names. Durations are summed GPU execution time, not elapsed module latency. NVIDIA overlaps streams, so sums cannot be used as E2E latency.','- AMD prefill module attribution is recorded; AMD decode and NVIDIA layer/module grouping use the previously documented attention and MoE boundaries. Full MoE-block boundaries are more reliable than attributing individual NVIDIA GEMMs to the router versus shared expert.','',
'## Proof: MoE launch counts','',
'Entire block includes shared expert, router, routed experts and their combination. Routed core includes routing/sorting/quantization/permutation/expert computation/finalization, but excludes shared/router dense GEMMs, shared activation, copy helper and shared/routed combination.','',
'| Phase | Scope | AMD calls/layer | NVIDIA calls/layer | AMD calls/forward | NVIDIA calls/forward |','|---|---|---:|---:|---:|---:|']
for p in ('Prefill','Decode'):
    for scope in ('Entire MoE block','Routed expert core incl routing'):
        a=moe['AMD',p,scope];n=moe['NVIDIA',p,scope]
        lines.append(f'| {p} | {scope} | {float(a["calls_per_layer"]):g} | {float(n["calls_per_layer"]):g} | {float(a["calls_per_forward"]):g} | {float(n["calls_per_forward"]):g} |')
lines+=['',
'Every one of the **120 prefill layer instances per vendor** has the same counts; every one of the **1,860 decode layer instances per vendor** also has the reported count. AMD has **240 extra whole-MoE launches per prefill forward (+30.8%)**, but **360 fewer per decode forward (−46.2%)**.','',
'### Where the extra prefill launches come from','',
'| Routed-core stage | AMD calls/layer | NVIDIA calls/layer | AMD ms/forward | NVIDIA ms/forward |','|---|---:|---:|---:|---:|']
for stage in ['Routing / TopK','Sorting / workspace','Quantization','Permutation','Expert GEMM 1','Expert GEMM 2','Expert reduction / finalize']:
    pair=[]
    for v in ['AMD','NVIDIA']:
        r=next((r for r in stages if r['vendor']==v and r['phase']=='Prefill' and r['substage']==stage),None)
        pair.append((float(r['calls_per_layer']),float(r['gpu_ms_per_forward'])) if r else (0,0))
    a,n=pair;lines.append(f'| {stage} | {a[0]:g} | {n[0]:g} | {a[1]:.3f} | {n[1]:.3f} |')
lines+=['',
'AMD routed-core sequence is `TopK ×1 → sorting/workspace ×4 → quantize ×1 → permute ×1 → expert GEMM1 ×1 → quantize ×1 → permute ×1 → expert GEMM2 ×1 → reduction ×1`, totaling **12** launches. NVIDIA uses three routing kernels, one quantization kernel, two expert GEMMs and one finalize kernel: **7**. Its surrounding MoE block also includes a separate copy helper, so zero entries in the table mean no separately named kernel in that bucket, not proof that all underlying work disappeared.','',
'**Extra launches are not the main measured MoE discrepancy by themselves.** AMD prefill MoE totals **152.872 ms** versus NVIDIA **144.732 ms** (+5.6%), despite 30.8% more launches. The second expert GEMM accounts for **43.028 vs 36.126 ms**, a **6.903 ms** gap. The separate weighted-output reductions are almost equal: **33.591 vs 33.563 ms**. AMD routing + sorting + quantization + permutation is **15.396 ms**; NVIDIA explicit routing + quantization is **11.430 ms**, with another **10.175 ms copy helper** outside the routed-core definition. Comparing only launch counts would miss these differences.','',
'### Decode contradicts the blanket “more AMD kernels” claim','',
'AMD routed core is `topkGatingSoftmax ×1 + aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256 ×1`. NVIDIA uses routing, quantization, expert GEMM1, expert GEMM2 and finalize: five launches. The whole MoE block is **5.136 ms AMD vs 5.147 ms NVIDIA** in summed GPU time. AMD is already more fused here; tune the fused kernel or its boundaries, rather than proposing to fuse expert stages that are already one launch. NVIDIA can still have lower elapsed latency because of stream overlap.','',
'## Attention and other operations, excluding collectives','',
'Times below aggregate the 60-layer target forward. Attention pipelines include layout, convolution/state update or cache handling, attention computation, gating and gated output normalization. Their dense input/output projections are listed separately.','',
'| Category | Prefill AMD ms | Prefill NVIDIA ms | Decode AMD ms | Decode NVIDIA ms |','|---|---:|---:|---:|---:|']
for c in ['MoE block including shared expert and router','Attention projections','Linear-attention pipeline','Full-attention pipeline','Standalone layer norms','Outside decoder layers']:
    lines.append('| '+c+' | '+' | '.join(f'{val(v,p,c):.3f}'+(' †' if c=='Standalone layer norms' and v=='NVIDIA' and p=='Decode' else '') for p in ('Prefill','Decode') for v in ('AMD','NVIDIA'))+' |')
lines+=['',
'† NVIDIA decode normalization is mostly inside excluded collective-fusion kernels. Do not interpret 0.003 ms as the total NVIDIA normalization cost, or subtract it from AMD normalization to estimate a pure non-collective opportunity.','',
'**Prefill:** the largest positive non-collective category gap is attention projections (**+24.285 ms** AMD), followed by the whole MoE block (**+8.139 ms**). Both linear and full-attention pipelines have lower summed time on AMD for the selected passes. That makes projection GEMM selection and MoE output/GEMM2 work stronger evidence-based priorities than assuming all AMD attention kernels are slower.','',
'**Decode:** the full-attention pipeline is **1.546 vs 0.530 ms** (2.91× summed-time ratio, **+1.015 ms** AMD). Linear attention is **1.295 vs 0.941 ms** (**+0.354 ms**). MoE block sums are essentially equal. Full attention is therefore the clearest measured non-collective discrepancy in decode. The AMD symbol has BF16 Q (`IS_Q_FP8_0`) and FP8 KV, while NVIDIA uses `QkvE4m3`; the ratio is not a precision-equivalent attainable speedup.','',
'### Decode linear attention: equal launch counts, different kernel costs','',
'| Stage | AMD kernel | AMD ms/forward | NVIDIA ms/forward |','|---|---|---:|---:|']
for n,label in [('fused_qkvzba_split_reshape_cat_contiguous_kernel','Input layout'),('_causal_conv1d_update_kernel','Convolution update'),('fused_qkv_split_gdn_prefill_kernel','QKV split'),('fused_sigmoid_gating_delta_rule_update_kernel','Gated delta update'),('_layer_norm_fwd_1pass_kernel','Gated output norm')]:
    values=[sum(float(r['gpu_ms_per_forward']) for r in kernels if r['vendor']==v and r['phase']=='Decode' and r['kernel']==n) for v in ('AMD','NVIDIA')]
    lines.append(f'| {label} | `{n}` | {values[0]:.3f} | {values[1]:.3f} |')
lines+=['',
'Both implementations use five core linear-attention launches per layer, or 225 over 45 layers. The split, layout and output-normalization kernels explain more of the gap than the delta-update arithmetic. This supports layout/split fusion and output-norm fusion even though launch counts are identical.','',
'## Revised non-collective optimization priorities and pipeline impact','',
'All collectives and collective-fusion proposals are held fixed. End-to-end denominators still include unchanged communication and surrounding work, because removing them from the denominator would exaggerate E2E gains. The fractions below are hypothetical reductions of measured AMD pools, not forecasts.','',
'| Opportunity | AMD measured pool | Assumption | Saved per forward | Phase latency reduction |','|---|---:|---|---:|---:|']
# Distinct proposed pools; do not present baseline vendor gaps as guaranteed savings.
items=[('Tune attention projection GEMMs','Prefill',val('AMD','Prefill','Attention projections'),.25),('Tune prefill expert GEMM2','Prefill',43.0284665,.25),('Fuse MoE weighted reduction with shared-expert combination','Prefill',10.840389,.5),('Fuse expert GEMM2 with weighted reduction (research)','Prefill',33.5911995,.5),('Tune decode full attention + segment reduction','Decode',1.217362709677419,.25),('Tune existing fused decode MoE','Decode',3.12290796774194,.25)]
candidates=read('optimization_candidates.csv');candidate={r['id']:r for r in candidates}
for cid,p in [('F07','Decode'),('F11','Decode'),('F16','Decode'),('F04','Decode'),('F05','Decode')]:items.append((candidate[cid]['title'],p,float(candidate[cid][p+'_pool_ms']),.5))
for title,p,pool,f in items:
    den=base['prefill_ms'] if p=='Prefill' else base['decode_ms'];lines.append(f'| {title} | {pool:.3f} ms ({p}) | {f:.0%} reduction | {pool*f:.3f} ms | {100*pool*f/den:.2f}% |')
# Fusion pool choices have no identical donor kernel; F15 serves both phases.
ids=['F07','F04','F05','F11','F16','F15']
saves={p:sum(float(candidate[cid][p+'_pool_ms'])*.5 for cid in ids) for p in ('Prefill','Decode')}
scenarios=[]
for label,tune in [('Non-collective fusion package',False),('Above + decode MoE/attention tuning',True)]:
    ps=saves['Prefill'];ds=saves['Decode']+(sum(float(candidate[cid]['Decode_pool_ms'])*.25 for cid in ['T02','T03']) if tune else 0)
    scenarios.append(dict(scenario=label,prefill_saved_ms=ps,verify_saved_ms=ds,new_prefill_ms=base['prefill_ms']-ps,new_verify_ms=base['decode_ms']-ds,new_cycle_ms=base['cycle_ms']-ds,cycle_latency_reduction_pct=100*ds/base['cycle_ms'],captured_window_reduction_pct=100*(4*ps+31*ds)/base['captured_window_ms']))
lines+=['','The following package uses only non-collective donor kernels: layout/split fusion, QK/RoPE/cache fusion, full-attention output gating, GDN output norm, shared-expert activation, and shared/routed combination. Each saves 50% of its standalone-pass pool. The second scenario adds 25% faster decode fused MoE and attention/reduction. Donor pools are disjoint; no collective time or collective-fused normalization is credited.','',
'| Scenario | Prefill ms | Verification ms | Decode cycle ms | Cycle latency reduction | Captured GPU-window reduction |','|---|---:|---:|---:|---:|---:|']
for s in scenarios:lines.append(f'| {s["scenario"]} | {s["new_prefill_ms"]:.3f} | {s["new_verify_ms"]:.3f} | {s["new_cycle_ms"]:.3f} | {s["cycle_latency_reduction_pct"]:.2f}% | {s["captured_window_reduction_pct"]:.2f}% |')
lines+=['',
'Baselines remain **702.323 ms prefill**, **11.309 ms verification**, **14.320 ms decode cycle**, and **3391.468 ms captured GPU window**. These modeled savings require matching benefits on every TP rank, critical-path realization, unchanged precision/acceptance and no added synchronization. Request-level E2E improvement still requires request timing and accepted-token measurements. The captured-window model reweights two clean prefill samples over four chunks and is approximate.','',
'## Exact verification artifacts','',
'- `moe_kernel_count_proof.csv`: min/max and mean launches for every MoE layer instance.','- `moe_substage_proof.csv`: explains every routed-core and surrounding MoE launch.','- `noncollective_event_evidence.csv`: original exact symbol, timestamp, duration, stream and correlation ID, indexed by vendor/phase/forward/layer.','- `noncollective_excluded_events.csv`: explicit audit of removed collectives, including fused collective kernels.','- `noncollective_categories.csv`: complete category partition of the retained target-step kernels.','- `noncollective_candidates.csv`: prior candidate list with collective tuning and collective-fusion proposals removed.','']
(P/'noncollective_optimization_proof.md').write_text('\n'.join(lines))
(P/'noncollective_scenarios.json').write_text(json.dumps(scenarios,indent=2))
filtered=[r for r in candidates if r['id'] not in ('F01','F02','T01')]
with (P/'noncollective_candidates.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(filtered[0]));w.writeheader();w.writerows(filtered)
# Standalone visual with proof counts and summed-time category comparison.
page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Non-collective kernel proof</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px;background:#f8fafc;color:#17212d}p{line-height:1.6}.card{padding:20px;background:white;border:1px solid #d8e0eb;border-radius:8px;margin:20px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}.bar{height:25px;color:white;padding-left:6px;margin:8px 0;min-width:30px;background:#b94040}.nv{background:#287945}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.note{color:#556174}select{padding:8px}</style><h1>MoE, attention and other kernels — collectives excluded</h1><p><a href="noncollective_optimization_proof.md">Complete proof and revised optimization plan</a> · <a href="noncollective_event_evidence.csv">Exact trace-event evidence</a></p><p><b>AMD has more MoE launches in prefill, but fewer in decode.</b> Launch counts do not establish launch overhead or elapsed latency.</p><div class="pair">'''
for p in ('Prefill','Decode'):
    page+='<div class="card"><h2>'+p+': whole MoE block</h2>'
    for v,cl in [('AMD',''),('NVIDIA','nv')]:
        r=moe[v,p,'Entire MoE block'];n=float(r['calls_per_layer']);page+=f'<div class="bar {cl}" style="width:{n/17*95:.1f}%">{v}: {n:g} calls/layer</div>'
    page+='<p>'+('Verified across 120 layer instances per vendor.' if p=='Prefill' else 'Verified across 1,860 layer instances per vendor.')+'</p></div>'
page+='</div><div class="card"><h2>Measured non-collective GPU work</h2><select id="phase"><option>Prefill</option><option>Decode</option></select><table><thead><tr><th>Category</th><th>AMD calls/fwd</th><th>NVIDIA calls/fwd</th><th>AMD summed ms</th><th>NVIDIA summed ms</th></tr></thead><tbody id="rows"></tbody></table><p class="note">Sums are not elapsed latency. NVIDIA normalization inside excluded collective-fusion kernels is absent from the standalone-normalization row. Matching prefill token labels do not establish identical cached context or precision.</p></div>'
data=json.dumps(cats).replace('</','<\\/')
page+='''<script>const data=DATA;function render(){let p=document.getElementById('phase').value;let names=[...new Set(data.filter(r=>r.phase===p).map(r=>r.category))];document.getElementById('rows').innerHTML=names.map(c=>{let a=data.find(r=>r.phase===p&&r.vendor==='AMD'&&r.category===c),n=data.find(r=>r.phase===p&&r.vendor==='NVIDIA'&&r.category===c);return '<tr><td>'+c+'</td><td>'+(+a.calls_per_forward).toFixed(1)+'</td><td>'+(+n.calls_per_forward).toFixed(1)+'</td><td>'+(+a.sum_gpu_ms_per_forward).toFixed(3)+'</td><td>'+(+n.sum_gpu_ms_per_forward).toFixed(3)+'</td></tr>'}).join('')}document.getElementById('phase').onchange=render;render();</script></html>'''.replace('DATA',data)
(P/'noncollective_proof.html').write_text(page)
for name in ['overview.md','optimization_plan.md']:
    p=P/name;s=p.read_text();link='[Collective-excluded proof and revised priorities](noncollective_optimization_proof.md)'
    if link not in s:
        first,rest=s.split('\n',1);p.write_text(first+'\n\n'+link+'\n'+rest)
print(json.dumps(scenarios,indent=2))
print('Verified MoE counts are constant across every selected layer instance:',all(r['min_calls_per_layer']==r['max_calls_per_layer'] for r in proof))
