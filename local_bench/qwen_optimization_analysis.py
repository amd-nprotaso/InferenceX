"""Trace-based AMD optimization budgets; estimates are not measured speedups."""
import json,csv,statistics,re
from collections import Counter,defaultdict
from pathlib import Path
from qwen_layer_comparison import load,PATHS,OUT,short,write_csv
from qwen_module_comparison import layer_segments

def union_us(events):
    ranges=sorted((e['ts'],e['ts']+e['dur']) for e in events)
    end=-float('inf');total=0
    for a,b in ranges:
        total+=max(0,b-max(a,end));end=max(end,b)
    return total

def inspect():
    for vendor,path in PATHS.items():
        d,steps,groups,runtimes=load(vendor,path)
        targets=[];agg=defaultdict(list)
        for i,s in enumerate(steps):
            es=groups[i]
            if sum(e['name']=='_fused_gate_sigmoid_mul_add_kernel' for e in es)!=60:continue
            phase='P' if 'EXTEND' in s['name'] else 'D'
            lo=min(e['ts'] for e in es);hi=max(e['ts']+e['dur'] for e in es)
            targets.append((phase,lo,hi))
            for e in es:agg[phase,e['name']].append(e['dur'])
            if phase=='P' or len(targets)<11:
                print(vendor,phase,i,'cpu_us',round(s['dur'],2),'span_us',round(hi-lo,2),'sum_us',round(sum(e['dur'] for e in es),2),'union_us',round(union_us(es),2),'streams',Counter(e['args'].get('stream') for e in es))
                for e in es:
                    if e['dur']>10000:print('LONG',short(e['name']),e['dur'],'stream',e['args'].get('stream'),'relstart',e['ts']-lo)
        ds=[x for x in targets if x[0]=='D'];print(vendor,'decode start-to-start us',statistics.mean(b[1]-a[1] for a,b in zip(ds,ds[1:])), 'span mean',statistics.mean(x[2]-x[1] for x in ds))
        for phase in ['P','D']:
            print(vendor,phase,'TOP KERNELS')
            for (p,n),vals in sorted(agg.items(),key=lambda kv:sum(kv[1]),reverse=True):
                if p!=phase:continue
                print(short(n),'count',len(vals),'sum_ms',round(sum(vals)/1000,3),'mean_us',round(statistics.mean(vals),3),'median_us',round(statistics.median(vals),3),'max_us',round(max(vals),3))
        gpu=[e for e in d['traceEvents'] if e.get('cat')=='kernel']
        print(vendor,'GPU annotation summary',Counter(e['name'] for e in d['traceEvents'] if e.get('cat')=='gpu_user_annotation'))
        print(vendor,'all-kernel streams',Counter(e['args'].get('stream') for e in gpu))
        print(vendor,'all kernels span',max(e['ts']+e['dur'] for e in gpu)-min(e['ts'] for e in gpu))

if __name__=='__main__' and '--budgets' not in __import__('sys').argv:inspect()

def build_budgets():
    from qwen_module_comparison import inference_roles
    d,steps,groups,runtimes=load('AMD',PATHS['AMD'])
    roles={}
    for phase,forward,li,seg in layer_segments(steps,groups):
        for e,role in zip(seg,inference_roles(seg,li%4!=3,'AMD')):roles[id(e)]=(li,role)
    records=[];baselines=[];counts=Counter();bad=[]
    for i,s in enumerate(steps):
        es=groups[i]
        if sum(e['name']=='_fused_gate_sigmoid_mul_add_kernel' for e in es)!=60:continue
        phase='Prefill' if 'EXTEND' in s['name'] else 'Decode'
        forward=counts[phase];counts[phase]+=1
        lo=min(e['ts'] for e in es);hi=max(e['ts']+e['dur'] for e in es)
        invalid=[]
        for a,b in zip(es,es[1:]):
            if a['args'].get('stream')==b['args'].get('stream') and a['ts']+a['dur']>b['ts']+100:
                invalid.append(a)
        bad.extend(dict(phase=phase,forward=forward,kernel=e['name'],duration_us=e['dur']) for e in invalid)
        baselines.append(dict(phase=phase,forward=forward,label=s['name'],start_us=lo,end_us=hi,span_ms=(hi-lo)/1000,sum_ms=sum(e['dur'] for e in es)/1000,valid_for_budgets=not invalid))
        for e in es:
            layer,role=roles.get(id(e),(-1,'outside_layer'))
            if role=='outside_layer' and ('ncclDevKernel' in e['name'] or 'cross_device_reduce' in e['name']):role='boundary_reduce'
            records.append(dict(phase=phase,forward=forward,layer=layer,role=role,kernel=e['name'],duration_us=e['dur'],valid=not invalid))
    gpu=[e for e in d['traceEvents'] if e.get('cat')=='kernel']
    ds=[b for b in baselines if b['phase']=='Decode']
    baseline=dict(prefill_ms=statistics.mean(b['span_ms'] for b in baselines if b['phase']=='Prefill' and b['valid_for_budgets']),decode_ms=statistics.mean(b['span_ms'] for b in ds),cycle_ms=statistics.mean((b['start_us']-a['start_us'])/1000 for a,b in zip(ds,ds[1:])),captured_window_ms=(max(e['ts']+e['dur'] for e in gpu)-min(e['ts'] for e in gpu))/1000,prefill_passes=4,decode_passes=31,budget_prefill_passes=2,anomalies=bad,forwards=baselines)
    valid=[r for r in records if r['valid']]
    counts={'Prefill':2,'Decode':31}
    # Each selector defines the measured pool an optimization could REDUCE;
    # it does not assume that producer/consumer arithmetic disappears.
    candidates=[]
    def add(cid,title,kind,select,feasibility,mechanism,overlap):
        candidates.append(dict(id=cid,title=title,kind=kind,select=select,feasibility=feasibility,mechanism=mechanism,overlap=overlap))
    name=lambda r:r['kernel'];role=lambda r:r['role'];phase=lambda r:r['phase']
    add('F01','All-reduce + residual add + RMSNorm','fusion',lambda r: name(r)=='_gemma_fused_add_rmsnorm_kernel','High-priority prototype','Fuse norm into the collective epilogue. Pool is only standalone norm time; no credit for removing network transfer. Requires all TP ranks and correct residual ownership.','F02,F22; T01 collective algorithm gains can interact')
    add('F02','MoE combination + all-reduce + next input RMSNorm','fusion',lambda r: name(r)=='_fused_gate_sigmoid_mul_add_kernel' or (role(r)=='input_norm' and r['layer']>0),'Medium','Fuse shared/routed-output combination, reduction, and next-layer input norm. Pool excludes collective transfer; final-layer norm is excluded here.','F01,F15,F20,F22')
    add('F03','Q/K normalization + M-RoPE','fusion',lambda r:name(r)=='_triton_mrope_forward_fused','High-priority prototype','Extend Q/K/gate kernel to apply rotary embedding. Pool is standalone rotary pass. Preserve Gemma RMSNorm semantics.','F04')
    add('F04','Q/K normalization + M-RoPE + KV-cache write','fusion',lambda r:name(r) in ('_triton_mrope_forward_fused','reshape_and_cache_flash'),'Medium','Alternative to F03: eliminate rotary/cache intermediates. Pool is rotary and cache kernels. Preserve current KV precision, scaling, layout, and write ordering.','F03')
    add('F05','Full-attention output/segment reduction + sigmoid gate','fusion',lambda r:name(r)=='_fused_sigmoid_mul_kernel','High-priority prototype','Prefill: gate in attention output epilogue. Decode: gate in reduce_segments final store. Pool is gate-only kernel. Does not eliminate segment reduction.','T03 touches same attention implementation')
    add('F06','Linear input GEMM epilogue emits final QKV/Z/B/A layout','fusion',lambda r:name(r)=='fused_qkvzba_split_reshape_cat_contiguous_kernel','Medium','Remove standalone layout pass using projection epilogue or a consumer reading original layout. Two separate projections must complete before combined consumers run.','F07')
    add('F07','Linear convolution consumes projection layout and emits split Q/K/V','fusion',lambda r:name(r) in ('fused_qkvzba_split_reshape_cat_contiguous_kernel','fused_qkv_split_gdn_prefill_kernel'),'High-priority prototype','Fuse layout/split work around FlyDSL prefill conv or decode conv update. Pool is the two layout kernels; convolution time is retained. Preserve state update and Z/B/A branches.','F06,F08,F10')
    add('F08','Prefill conv epilogue + Q/K L2 norm + gating/split','fusion',lambda r:phase(r)=='Prefill' and name(r) in ('fused_qkv_split_gdn_prefill_kernel','fused_gdn_gating_kernel','l2norm_fwd_kernel'),'Medium','Normalize Q/K and emit gates from conv output; row reductions and gate inputs require a compatible tile. Pool is split/gating/L2 passes, not convolution arithmetic.','F07,F09')
    add('F09','Prefill gating + chunk-local cumulative sum','fusion',lambda r:phase(r)=='Prefill' and name(r)=='chunk_local_cumsum_scalar_kernel','Medium','Compute gating and chunk scan in one kernel. Retain the scan dependency; pool is separate cumsum time.','F08; T04 may redesign this boundary')
    add('F10','Decode delta update consumes packed conv output','fusion',lambda r:phase(r)=='Decode' and name(r)=='fused_qkv_split_gdn_prefill_kernel','Medium','Alternative consumer-side split elimination: update reads conv layout directly. Preserve speculative state/rollback semantics.','F07')
    add('F11','Linear attention output + gated RMSNorm','fusion',lambda r:name(r)=='_layer_norm_fwd_1pass_kernel','Medium','Prefill chunk output or decode recurrence epilogue applies existing gated RMSNorm; cross-channel reduction can raise registers and reduce occupancy. Pool is standalone norm.','F12,F14')
    add('F12','KKT solve + recompute W/U','fusion research',lambda r:phase(r)=='Prefill' and name(r) in ('chunk_gated_delta_rule_fwd_kkt_solve_kernel','recompute_w_u_fwd_kernel'),'Research','Pool is BOTH kernels, not removable overhead. Savings percentages model faster combined execution; triangular solve/tiling/global dependencies may prevent useful fusion.','T04')
    add('F13','Chunk state recurrence + output calculation','fusion research',lambda r:phase(r)=='Prefill' and name(r) in ('chunk_gated_delta_rule_fwd_kernel_h_blockdim64','chunk_fwd_kernel_o'),'Research','Pool is BOTH compute kernels. Requires dependency-aware persistent/chunk scheduling; do not treat recurrent computation as removable.','F11,T04')
    add('F14','Gated RMSNorm + attention output projection','fusion research',lambda r:name(r)=='_layer_norm_fwd_1pass_kernel','Research','Alternative to F11: projection consumes normalization inline. Norm reduction and GEMM tiling may require recomputation or synchronization.','F11')
    add('F15','Shared-expert down projection + shared/routed output combination','fusion',lambda r:name(r)=='_fused_gate_sigmoid_mul_add_kernel','Medium','Fuse combination into the last completed producer or consumer; need both expert paths and correct gate. Pool is combination pass.','F02,F20; alternative epilogue placement')
    add('F16','Shared-expert gate/up GEMM + SiLU-and-multiply','fusion',lambda r:'act_and_mul_kernel' in name(r),'High-priority prototype','GEMM epilogue produces activated shared-expert intermediate. Pool is separate activation; preserve both gate/up halves.','Shared-expert overlap can hide some savings')
    add('F17','TopK routing + MoE sorting/workspace passes','fusion',lambda r:phase(r)=='Prefill' and 'opus_moe_sorting_entry' in name(r),'Medium','Reduce routing/sorting passes and initialization through a coordinated routing backend. Pool is sorting only. Global histogram/prefix dependencies can still require multiple launches.','F18,F19; do not simply delete global scans')
    add('F18','MXFP4 quantization + token permutation','fusion',lambda r:phase(r)=='Prefill' and 'mxfp4_moe_sort_kernel' in name(r),'Medium','Quantize directly to expert-consumption layout. Pool is permutation passes only; quantization retained. Reuse scales without excess token duplication.','F17,F19')
    add('F19','First expert GEMM epilogue + intermediate FP4 quantization/permutation','fusion',lambda r:phase(r)=='Prefill' and (('dynamic_per_group_scaled_quant_kernel' in name(r)) or ('mxfp4_moe_sort_kernel' in name(r) and '<256, 128, 4, 32>' in name(r))),'Medium','Pool uses half of quantization time (one of two calls per layer) plus second permutation. Requires correct per-group scales and output packing; no extra SiLU saving because moe1 already fuses it.','F18,T05')
    add('F20','MoE weighted reduction + shared-expert combination','fusion',lambda r:phase(r)=='Prefill' and name(r)=='_fused_gate_sigmoid_mul_add_kernel','High-priority prototype','Use moe_reduction_kernel_0 final stores to combine shared expert. Pool is combination only; weighted reduction retained.','F02,F15')
    add('F21','Second expert GEMM + weighted expert-output reduction','fusion research',lambda r:phase(r)=='Prefill' and name(r)=='moe_reduction_kernel_0','Research/high upside','Avoid materializing expert outputs or use tile-owned accumulation. Global expert-to-token reduction, atomics, determinism and precision can erase gains. Pool is standalone reduction.','T05; F20 can follow only if dependencies allow')
    add('F22','Input/post-attention normalization + GEMM or quantization consumer','fusion research',lambda r:name(r) in ('_gemma_rmsnorm_kernel','_gemma_fused_add_rmsnorm_kernel'),'Research','Alternative placement to F01. Reduction dimensions and shared consumers can force recomputation. Pool is normalization, not GEMM time.','F01,F02')
    add('F23','Decode router TopK + existing fused MoE','fusion research',lambda r:phase(r)=='Decode' and 'topkGatingSoftmax' in name(r),'Research','Integrate routing only if expert selection can feed persistent/block scheduling without duplicate work. Decode experts already run as one fused aiter::fmoe kernel.','T02; do not re-count already fused expert stages')
    add('F24','Merge QKV/Z and B/A input projections','fusion research',lambda r:role(r) in ('input_proj','gating_proj'),'Medium/requires GEMM study','Pool is BOTH projection GEMMs, not removable overhead. Concatenate compatible weights and use one GEMM or grouped launch, with custom output layout. Shape/precision compatibility and changed tiling determine whether it helps.','F06,F07,T06')
    add('F25','Decode convolution state update + gated delta recurrence','fusion research',lambda r:phase(r)=='Decode' and name(r) in ('_causal_conv1d_update_kernel','fused_sigmoid_gating_delta_rule_update_kernel'),'Research','Pool is BOTH compute kernels. Conv and recurrent states have different layouts; verify query-order dependencies, speculative state snapshots and rollback before considering a persistent fused kernel.','F07,F10,F11')
    add('F26','Router projection epilogue + TopK/softmax','fusion research',lambda r:'topkGatingSoftmax' in name(r),'Research','Alternative to F23: router GEMM emits top-k selections. Global expert-dimension reduction may not fit the GEMM epilogue; exact routing and scoring semantics must be preserved. Pool is TopK only.','F17,F23,T06')
    add('T01','Collective algorithm, transport and rank-wait tuning','kernel/system tuning',lambda r:'ncclDevKernel' in name(r) or 'cross_device_reduce' in name(r),'Highest prefill priority','Pool includes all target-step collectives, including embedding/final boundaries. Separate transfer from waiting using all-rank traces and topology; TP-0 alone cannot diagnose bandwidth.','F01,F02; transfer and wait improvements are not guaranteed')
    add('T02','Tune existing decode fused MXFP4 MoE','kernel tuning',lambda r:phase(r)=='Decode' and 'aiter::fmoe_bf16' in name(r),'Highest decode compute priority','Tune persistent scheduling, expert tiling, weight loads and occupancy for recorded batch shape; pool is whole existing fused MoE.','F23')
    add('T03','Tune decode full attention and split count','kernel tuning',lambda r:phase(r)=='Decode' and (name(r).startswith('kernel_unified_attention_3d') or name(r).startswith('reduce_segments')),'High-priority prototype','Tune NUM_SEGMENTS_PER_SEQ=128, query tiling, and reduction; fewer segments trades occupancy for reduction cost. Pool is attention+reduction. Keep BF16 Q and FP8 KV; NVIDIA FP8-Q path is not a precision-equivalent target.','F05 touches same output stage')
    add('T04','Tune prefill GDN state kernel','kernel tuning',lambda r:phase(r)=='Prefill' and name(r)=='chunk_gated_delta_rule_fwd_kernel_h_blockdim64','Secondary','Pool is recurrence computation. AMD is already faster than NVIDIA for this named stage in these traces. Tune only after larger bottlenecks.','F12,F13')
    add('T05','Tune prefill MXFP4 expert GEMMs','kernel tuning',lambda r:phase(r)=='Prefill' and name(r).startswith(('mfma_moe1','mfma_moe2')),'High-impact but implementation-heavy','Pool is two expert GEMMs on unaffected prefill passes. Tune tile sizes, traffic and expert imbalance; malformed duration records excluded.','F19,F21')
    add('T06','Tune input/output/shared/router GEMMs','kernel tuning',lambda r:name(r).startswith(('Cijk_','hgemm_')),'High-impact broad tuning','Pool includes all target-step GEMMs including logits. Use per-module shapes; compare alternative kernels within AMD. Improvements here are not the same as deleting GEMMs.','F06,F14,F16,F22; use per-kernel union accounting')
    add('T07','Overlap shared expert with routed MoE','scheduling',lambda r:role(r) in ('shared_up','shared_activation','shared_down'),'High-value experiment','Pool is shared-expert work that may be hidden behind routed MoE, not deleted. AMD target work is on one stream; NVIDIA uses auxiliary streams. Resource contention and join dependencies limit overlap.','F16 and shared-expert GEMM tuning affect same pool')
    rows=[];details=[]
    for c in candidates:
        row={k:v for k,v in c.items() if k!='select'}
        for p in counts:
            selected=[r for r in valid if r['phase']==p and c['select'](r)]
            def weight(r):return .5 if c['id']=='F19' and 'dynamic_per_group_scaled_quant_kernel' in r['kernel'] else 1
            budget=sum(r['duration_us']*weight(r) for r in selected)/counts[p]/1000
            row[p+'_pool_ms']=budget
            row[p+'_calls_per_forward']=sum(weight(r) for r in selected)/counts[p]
            den=baseline['prefill_ms'] if p=='Prefill' else baseline['decode_ms']
            for f in [.25,.5,.75]:row[p+'_latency_reduction_pct_at_'+str(int(f*100))]=100*budget*f/den
            detail=defaultdict(lambda:[0,0.])
            for r in selected:
                a=detail[r['kernel']];a[0]+=weight(r);a[1]+=r['duration_us']*weight(r)
            for n,(calls,total) in detail.items():details.append(dict(candidate=c['id'],phase=p,kernel=n,calls_per_forward=calls/counts[p],pool_ms_per_forward=total/counts[p]/1000))
        row['cycle_reduction_pct_at_50']=100*.5*row['Decode_pool_ms']/baseline['cycle_ms']
        row['captured_window_reduction_pct_at_50']=100*.5*(4*row['Prefill_pool_ms']+31*row['Decode_pool_ms'])/baseline['captured_window_ms']
        rows.append(row)
    write_csv(OUT/'optimization_candidates.csv',rows);write_csv(OUT/'optimization_kernel_budgets.csv',details)
    (OUT/'optimization_baselines.json').write_text(json.dumps(baseline,indent=2))
    # Non-additive package: take max fraction for each event to avoid double counting.
    packages=[]
    plans=[('Practical fusion prototypes',{'F01':.5,'F07':.5,'F04':.5,'F05':.5,'F11':.5,'F16':.5,'F20':.5}),('Fusion + focused compute tuning',{'F01':.5,'F07':.5,'F04':.5,'F05':.5,'F11':.5,'F16':.5,'F20':.5,'T02':.25,'T03':.25}),('Above + 25% collective reduction',{'F01':.5,'F07':.5,'F04':.5,'F05':.5,'F11':.5,'F16':.5,'F20':.5,'T02':.25,'T03':.25,'T01':.25})]
    cmap={c['id']:c for c in candidates}
    for title,plan in plans:
        save={p:sum(r['duration_us']*max([f for cid,f in plan.items() if cmap[cid]['select'](r)]+[0]) for r in valid if r['phase']==p)/counts[p]/1000 for p in counts}
        packages.append(dict(title=title,assumptions=plan,prefill_saved_ms=save['Prefill'],decode_saved_ms=save['Decode'],new_prefill_ms=baseline['prefill_ms']-save['Prefill'],new_decode_ms=baseline['decode_ms']-save['Decode'],new_cycle_ms=baseline['cycle_ms']-save['Decode'],new_captured_window_ms=baseline['captured_window_ms']-4*save['Prefill']-31*save['Decode']))
    (OUT/'optimization_scenarios.json').write_text(json.dumps(packages,indent=2))
    hotspot=defaultdict(lambda:[0,0.])
    for r in valid:
        a=hotspot[r['phase'],r['kernel']];a[0]+=1;a[1]+=r['duration_us']
    write_csv(OUT/'optimization_hotspots.csv',[dict(phase=p,kernel=n,short_label=short(n),calls_per_forward=c/counts[p],ms_per_forward=t/counts[p]/1000) for (p,n),(c,t) in sorted(hotspot.items(),key=lambda kv:kv[1][1]/counts[kv[0][0]],reverse=True)])
    print('BASELINE',json.dumps({k:v for k,v in baseline.items() if k!='forwards'},indent=2))
    for r in rows:print(r['id'],r['title'],'P/D pools ms',round(r['Prefill_pool_ms'],4),round(r['Decode_pool_ms'],4),'50% phase gains',round(r['Prefill_latency_reduction_pct_at_50'],3),round(r['Decode_latency_reduction_pct_at_50'],3),'cycle',round(r['cycle_reduction_pct_at_50'],3))
    print('PACKAGES',json.dumps(packages,indent=2))

if __name__=='__main__' and '--budgets' in __import__('sys').argv:build_budgets()
