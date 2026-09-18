"""Prove trace-level non-collective counts and timings without equating counts with speed."""
import csv,json,statistics
from collections import Counter,defaultdict
from qwen_layer_comparison import load,PATHS,OUT,short,write_csv,collective
from qwen_module_comparison import layer_segments,inference_roles

def is_collective(n):
    return collective(n) or any(t in n for t in ('allgather_vec','_all_gather_kernel_inner','AllGather','ReduceScatter'))

def moe_stage(n):
    if 'topkGatingSoftmax' in n or 'routingIndices' in n or 'routingInitExpert' in n:return 'Routing / TopK'
    if 'opus_moe_sorting_entry' in n:return 'Sorting / workspace'
    if 'dynamic_per_group_scaled_quant' in n or 'Quantize' in n:return 'Quantization'
    if 'mxfp4_moe_sort_kernel' in n:return 'Permutation'
    if n.startswith(('mfma_moe1','bmm_E2m1')):return 'Expert GEMM 1'
    if n.startswith(('mfma_moe2','bmm_Bfloat16')):return 'Expert GEMM 2'
    if 'aiter::fmoe_bf16' in n:return 'Fused expert pipeline'
    if n.startswith('moe_reduction') or 'moe::dev::finalize' in n:return 'Expert reduction / finalize'
    if n=='_fused_gate_sigmoid_mul_add_kernel':return 'Shared/routed combination'
    if 'act_and_mul_kernel' in n:return 'Shared-expert activation'
    if n.startswith(('nvjet_','Cijk_','hgemm_')) or 'splitKreduce' in n:return 'Shared/router GEMMs and split-K reductions'
    if n.startswith('memcpy'):return 'Copy helper'
    return 'Other MoE helper'

def category(role,li):
    if role in ('input_proj','gating_proj','qkv_proj','out_proj'):return 'Attention projections'
    if role in ('input_norm','post_norm'):return 'Standalone layer norms'
    if role in ('attention_layout','qk_prepare','qk_rope_fused','rotary','radix','gated_norm','attention_gate'):return 'Full-attention pipeline' if li%4==3 else 'Linear-attention pipeline'
    return 'MoE block including shared expert and router'

records=[];denoms={};step_shapes={};excluded=[];step_stats=[]
for vendor,path in PATHS.items():
    d,steps,groups,rt=load(vendor,path);index={}
    for phase,f,l,seg in layer_segments(steps,groups):
        for e,role in zip(seg,inference_roles(seg,l%4!=3,vendor)):index[id(e)]=(l,role)
    seen=Counter()
    for si,s in enumerate(steps):
        es=groups[si]
        if sum(e['name']=='_fused_gate_sigmoid_mul_add_kernel' for e in es)!=60:continue
        p='Prefill' if 'EXTEND' in s['name'] else 'Decode';f=seen[p];seen[p]+=1
        if p=='Prefill' and f not in (0,3):continue
        denoms[vendor,p]=denoms.get((vendor,p),0)+1
        step_shapes[vendor,p,f]=s['name']
        lo=min(e['ts'] for e in es);hi=max(e['ts']+e['dur'] for e in es)
        kept=[]
        for e in es:
            l,role=index.get(id(e),(-1,'outside_layer'))
            if is_collective(e['name']):
                excluded.append(dict(vendor=vendor,phase=p,forward=f,kernel=e['name'],duration_us=e['dur']));continue
            c='Outside decoder layers' if l<0 else category(role,l)
            st=moe_stage(e['name']) if c.startswith('MoE block') else c
            r=dict(vendor=vendor,phase=p,forward=f,layer=l,category=c,substage=st,kernel=e['name'],short_label=short(e['name']),duration_us=e['dur'],timestamp_us=e['ts'],stream=e['args'].get('stream'),correlation=e['args'].get('correlation'))
            records.append(r);kept.append(e)
        step_stats.append(dict(vendor=vendor,phase=p,forward=f,shape=s['name'],full_target_gpu_span_ms=(hi-lo)/1000,noncollective_kernel_calls=len(kept),noncollective_kernel_sum_ms=sum(e['dur'] for e in kept)/1000))
    del d,steps,groups,rt,index
assert [step_shapes['AMD','Prefill',f] for f in (0,3)]==[step_shapes['NVIDIA','Prefill',f] for f in (0,3)]
assert not any(is_collective(r['kernel']) for r in records)
write_csv(OUT/'noncollective_event_evidence.csv',records)
write_csv(OUT/'noncollective_excluded_events.csv',excluded)
write_csv(OUT/'noncollective_step_evidence.csv',step_stats)
agg=defaultdict(lambda:[0,0.]);detail=defaultdict(lambda:[0,0.]);perlayer=defaultdict(lambda:[0,0.])
for r in records:
    v,p,c,st=r['vendor'],r['phase'],r['category'],r['substage']
    a=agg[v,p,c];a[0]+=1;a[1]+=r['duration_us']
    a=detail[v,p,st,r['kernel']];a[0]+=1;a[1]+=r['duration_us']
    if r['layer']>=0:
        a=perlayer[v,p,r['forward'],r['layer'],c];a[0]+=1;a[1]+=r['duration_us']
summary=[dict(vendor=v,phase=p,category=c,calls_per_forward=n/denoms[v,p],sum_gpu_ms_per_forward=t/denoms[v,p]/1000) for (v,p,c),(n,t) in agg.items()]
write_csv(OUT/'noncollective_categories.csv',summary)
write_csv(OUT/'noncollective_kernel_summary.csv',[dict(vendor=v,phase=p,substage=st,kernel=n,calls_per_forward=c/denoms[v,p],gpu_ms_per_forward=t/denoms[v,p]/1000) for (v,p,st,n),(c,t) in detail.items()])
write_csv(OUT/'noncollective_per_layer.csv',[dict(vendor=v,phase=p,forward=f,layer=l,category=c,calls=n,gpu_us=t) for (v,p,f,l,c),(n,t) in perlayer.items()])
proof=[]
core={'Routing / TopK','Sorting / workspace','Quantization','Permutation','Expert GEMM 1','Expert GEMM 2','Fused expert pipeline','Expert reduction / finalize'}
for v in PATHS:
    for p in ('Prefill','Decode'):
        for scope in ('Entire MoE block','Routed expert core incl routing'):
            selected=[r for r in records if r['vendor']==v and r['phase']==p and r['category'].startswith('MoE block') and (scope=='Entire MoE block' or r['substage'] in core)]
            counts=Counter((r['forward'],r['layer']) for r in selected)
            assert len(counts)==denoms[v,p]*60
            proof.append(dict(vendor=v,phase=p,scope=scope,min_calls_per_layer=min(counts.values()),max_calls_per_layer=max(counts.values()),calls_per_layer=len(selected)/len(counts),calls_per_forward=len(selected)/denoms[v,p],gpu_ms_per_forward=sum(r['duration_us'] for r in selected)/denoms[v,p]/1000,layer_instances_checked=len(counts)))
write_csv(OUT/'moe_kernel_count_proof.csv',proof)
stages=[]
for v in PATHS:
    for p in ('Prefill','Decode'):
        for st in sorted({r['substage'] for r in records if r['category'].startswith('MoE block')}):
            rr=[r for r in records if r['vendor']==v and r['phase']==p and r['substage']==st]
            if rr:stages.append(dict(vendor=v,phase=p,substage=st,calls_per_layer=len(rr)/denoms[v,p]/60,gpu_ms_per_forward=sum(r['duration_us'] for r in rr)/denoms[v,p]/1000))
write_csv(OUT/'moe_substage_proof.csv',stages)
print('MOE PROOF',json.dumps(proof,indent=2));print('CATEGORIES',json.dumps(summary,indent=2));print('MOE STAGES',json.dumps(stages,indent=2))
print('TOP DIFFERING NONCOLLECTIVE KERNELS')
for v in PATHS:
    for p in ('Prefill','Decode'):
        dd=[(st,n,c,t/denoms[v,p]/1000) for (vv,pp,st,n),(c,t) in detail.items() if vv==v and pp==p]
        print(v,p,[(short(n),round(t,3)) for st,n,c,t in sorted(dd,key=lambda x:x[3],reverse=True)[:18]])
