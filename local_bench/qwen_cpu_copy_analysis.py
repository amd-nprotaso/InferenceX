"""Inspect CPU copy scopes, runtime waits, and GPU transfers in supplied traces."""
import json,re,csv,bisect,statistics
from collections import defaultdict,Counter
from pathlib import Path
from inspect_qwen_traces import PATHS
from qwen_layer_comparison import OUT,write_csv

def stats(es):
    ds=sorted(e.get('dur',0) for e in es)
    return dict(count=len(ds),sum_ms=sum(ds)/1000,median_us=statistics.median(ds),p95_us=ds[min(len(ds)-1,int(.95*len(ds)))],max_us=max(ds))

def inspect():
    for v,path in PATHS.items():
        d=json.load(open(path));es=d['traceEvents'];g=defaultdict(list)
        for e in es:
            n=e.get('name','');cat=e.get('cat','')
            if any(t in n.lower() for t in ('copy','memcpy','synchronize','copy_id')) and cat not in ('kernel','ac2g'):
                g[cat,n].append(e)
        print(v,'literal copy_id matches',[(e.get('cat'),e.get('name')) for e in es if 'copy_id' in e.get('name','').lower()][:20])
        for (cat,n),events in sorted(g.items(),key=lambda kv:sum(e.get('dur',0) for e in kv[1]),reverse=True)[:90]:print(cat,n,stats(events))
        cpu=[e for e in es if e.get('cat')=='cpu_op' and 'copy' in e.get('name','').lower()]
        for e in sorted(cpu,key=lambda e:e.get('dur',0),reverse=True)[:12]:
            print('LONG_COPY',e)
            a=e['ts'];b=a+e['dur'];tid=e['tid'];pid=e['pid']
            parents=[x for x in es if x.get('cat') in ('python_function','user_annotation','cpu_op') and x.get('tid')==tid and x.get('pid')==pid and x.get('ts',0)<=a and x.get('ts',0)+x.get('dur',0)>=b]
            parents.sort(key=lambda x:x.get('dur',0))
            print('PARENTS',[(x['name'],x.get('dur')) for x in parents[:12]])
            print('CHILD_RUNTIME',[(x['name'],x.get('dur'),x.get('args')) for x in es if x.get('cat') in ('cuda_runtime','cuda_driver') and x.get('tid')==tid and a<=x.get('ts',0)<=b][:15])
if __name__=='__main__' and len(__import__('sys').argv)==1:inspect()

def runtime_inspect():
    for v,path in PATHS.items():
        es=json.load(open(path))['traceEvents']
        funcs=[e for e in es if e.get('cat') in ('python_function','user_annotation','cpu_op')]
        gpu=defaultdict(list)
        for e in es:
            if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset'):gpu[e.get('args',{}).get('correlation')].append(e)
        calls=[e for e in es if e.get('cat')=='cuda_runtime' and e['name'] in ('hipMemcpyWithStream','cudaStreamSynchronize','hipEventSynchronize','cudaEventSynchronize')]
        print('VENDOR',v)
        for e in sorted(calls,key=lambda x:x['dur'],reverse=True)[:16]:
            a=e['ts'];b=a+e['dur'];parents=[x for x in funcs if x.get('tid')==e['tid'] and x.get('pid')==e['pid'] and x.get('ts',0)<=a and x.get('ts',0)+x.get('dur',0)>=b]
            parents.sort(key=lambda x:x.get('dur',0))
            print('RUNTIME',e)
            print('PARENTS',[(x['cat'],x['name'],x['dur']) for x in parents[:9]])
            print('GPU',gpu.get(e['args'].get('correlation')))
if '--runtime' in __import__('sys').argv:runtime_inspect()

class ScopeIndex:
    def __init__(self,events):
        groups=defaultdict(list)
        for e in events:
            cat=e.get('cat');n=e.get('name','')
            if (cat=='python_function' and ('/sglang/' in n or n.startswith('nn.Module:'))) or cat in ('cpu_op','user_annotation'):
                groups[e['pid'],e['tid']].append(e)
        self.data={}
        for key,es in groups.items():
            es.sort(key=lambda e:(e['ts'],-e.get('dur',0)));parents=[];stack=[]
            for j,e in enumerate(es):
                end=e['ts']+e.get('dur',0)
                while stack and es[stack[-1]]['ts']+es[stack[-1]].get('dur',0)<end-.01:stack.pop()
                parents.append(stack[-1] if stack else -1);stack.append(j)
            self.data[key]=([e['ts'] for e in es],es,parents)
    def enclosing(self,e):
        starts,es,parents=self.data.get((e['pid'],e['tid']),([],[],[]))
        j=bisect.bisect_right(starts,e['ts'])-1;out=[]
        while j>=0:
            x=es[j]
            if x['ts']+x.get('dur',0)>=e['ts']+e.get('dur',0)-.1:out.append(x)
            j=parents[j]
        return out

def merged_intervals(es):
    result=[]
    for a,b in sorted((e['ts'],e['ts']+e['dur']) for e in es):
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result

def overlap(a,b,intervals):
    starts,ends=intervals
    j=bisect.bisect_left(ends,a);total=0
    while j<len(starts) and starts[j]<b:
        total+=max(0,min(b,ends[j])-max(a,starts[j]));j+=1
    return total

def report_data():
    rows=[];summaries=[];scope_rows=[];literal={};details={}
    for vendor,path in PATHS.items():
        es=json.load(open(path))['traceEvents'];idx=ScopeIndex(es)
        literal[vendor]=sum('copy_id' in e.get('name','').lower() for e in es)
        gpuby=defaultdict(list)
        gpu=[e for e in es if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
        for e in gpu:gpuby[e.get('args',{}).get('correlation')].append(e)
        # Exclude grossly inconsistent AMD GPU records from overlap evidence.
        rejected=[e for e in gpu if vendor=='AMD' and e['dur']>10000]
        validgpu=[e for e in gpu if not (vendor=='AMD' and e['dur']>10000)]
        ints=merged_intervals(validgpu);ints=([i[0] for i in ints],[i[1] for i in ints])
        bystream=defaultdict(list)
        for e in validgpu:bystream[e.get('args',{}).get('stream')].append(e)
        for stream in bystream:bystream[stream].sort(key=lambda x:x['ts'])
        for e in es:
            if e.get('cat')=='cpu_op' and e['name'] in ('aten::copy_','aten::_to_copy','aten::_local_scalar_dense','aten::item'):
                scope_rows.append(dict(vendor=vendor,operation=e['name'],duration_us=e['dur'],timestamp_us=e['ts']))
            if e.get('cat')!='cuda_runtime' or not any(t in e['name'] for t in ('Memcpy','Synchronize')):continue
            anc=idx.enclosing(e);names=[x['name'] for x in anc]
            caller=next((x['name'] for x in anc if x.get('cat')=='python_function' and '/sglang/' in x['name']),'unresolved')
            cpu=next((x['name'] for x in anc if x.get('cat')=='cpu_op'),'none')
            anns=[x['name'] for x in anc if x.get('cat')=='user_annotation']
            step=next((n for n in anns if n.startswith('step[')),'outside step')
            draft=any(n=='draft_extend' or n=='draft' for n in anns)
            stage=('draft ' if draft else 'target ')+step if step!='outside step' else (anns[0] if anns else 'outside annotations')
            relevant='other'
            if 'aiter_backend.py' in caller:relevant='AITER '+caller.split(': ')[-1]
            elif 'fla/index.py' in caller:relevant='FLA '+caller.split(': ')[-1]
            elif 'clear_slots' in caller:relevant='Mamba clear_slots'
            elif 'donate_mamba_ping_pong_slot' in caller:relevant='Mamba slot donation'
            elif 'copy_to_cpu' in caller:relevant='Result copy_to_cpu'
            elif 'process_batch_result_decode' in caller:relevant='Decode result event wait'
            elif '_eagle_prefill_tail_tokens' in caller:relevant='Eagle prefill tail tokens'
            elif 'memory_pool.py' in caller and caller.endswith(': alloc'):relevant='Request slot allocation'
            elif 'buffers.py' in caller or 'cuda_graph_buffer_registry.py' in caller:relevant='Graph input buffer copies'
            else:relevant=caller
            ge=gpuby.get(e['args'].get('correlation'),[])
            end=e['ts']+e['dur'];busy=overlap(e['ts'],end,ints)
            row=dict(vendor=vendor,operation=e['name'],site=relevant,caller=caller,cpu_parent=cpu,stage=stage,duration_us=e['dur'],timestamp_us=e['ts'],relative_ms=(e['ts']-min(x['ts'] for x in gpu))/1000,correlation=e['args'].get('correlation'),external_id=e['args'].get('External id',''),bytes=e['args'].get('bytes',sum(x['args'].get('bytes',0) for x in ge) if ge else ''),runtime_kind=e['args'].get('kind',''),gpu_event_names='; '.join(x['name'] for x in ge),gpu_transfer_us=sum(x['dur'] for x in ge),gpu_busy_during_cpu_call_us=busy,no_recorded_gpu_activity_us=max(0,e['dur']-busy),post_transfer_gap_us='',ancestors=' > '.join(names[:12]))
            # Gap on transfer stream after the matched copy to next recorded GPU op.
            if len(ge)==1 and ge[0].get('cat')=='gpu_memcpy' and ge[0]['dur']<10000:
                x=ge[0];seq=bystream[x['args']['stream']];ts=[z['ts'] for z in seq];j=bisect.bisect_right(ts,x['ts']+.01)
                if j<len(seq):row['post_transfer_gap_us']=max(0,seq[j]['ts']-x['ts']-x['dur'])
            rows.append(row)
        details[vendor]=dict(literal_copy_id_events=literal[vendor],excluded_gpu_intervals_for_overlap=[dict(name=e['name'],duration_us=e['dur'],timestamp_us=e['ts']) for e in rejected])
        del es,idx,gpu,validgpu,gpuby
    write_csv(OUT/'cpu_copy_runtime_evidence.csv',rows)
    grouped=defaultdict(list)
    for r in rows:grouped[r['vendor'],r['site'],r['operation']].append(r)
    for (v,site,op),rr in grouped.items():
        ds=[r['duration_us'] for r in rr];nonempty=[r for r in rr if r['gpu_event_names']]
        summaries.append(dict(vendor=v,site=site,operation=op,count=len(rr),cpu_total_ms=sum(ds)/1000,cpu_median_us=statistics.median(ds),cpu_max_us=max(ds),matched_gpu_total_ms=sum(r['gpu_transfer_us'] for r in nonempty)/1000,gpu_busy_during_call_ms=sum(r['gpu_busy_during_cpu_call_us'] for r in rr)/1000,no_gpu_activity_during_call_ms=sum(r['no_recorded_gpu_activity_us'] for r in rr)/1000,post_transfer_gap_total_ms=sum(r['post_transfer_gap_us'] for r in rr if r['post_transfer_gap_us']!='')/1000))
    write_csv(OUT/'cpu_copy_by_callsite.csv',summaries)
    scopes=[]
    for v in PATHS:
        for name in ('aten::copy_','aten::_to_copy','aten::_local_scalar_dense','aten::item'):
            rr=[r for r in scope_rows if r['vendor']==v and r['operation']==name];ds=[r['duration_us'] for r in rr]
            scopes.append(dict(vendor=v,operation=name,count=len(rr),sum_ms=sum(ds)/1000,median_us=statistics.median(ds),max_us=max(ds)))
    write_csv(OUT/'cpu_copy_scope_summary.csv',scopes)
    (OUT/'cpu_copy_validation.json').write_text(json.dumps(details,indent=2))
    print('CPU SCOPES',json.dumps(scopes,indent=2))
    print('SITES',json.dumps(sorted(summaries,key=lambda r:r['cpu_total_ms'],reverse=True)[:35],indent=2))
    for vendor in PATHS:
        for site in ('AITER forward_extend','AITER update_single_wrapper','FLA prepare_chunk_indices','Mamba clear_slots'):
            rr=[r for r in rows if r['vendor']==vendor and r['site']==site]
            if rr:print('DETAIL',vendor,site,Counter(r['stage'] for r in rr),Counter((r['cpu_parent'],str(r['bytes']),r['runtime_kind']) for r in rr))
    print('WAIT EXAMPLE',json.dumps(next(r for r in rows if r['vendor']=='AMD' and r['site']=='AITER update_single_wrapper' and r['duration_us']>500000),indent=2))
if '--report' in __import__('sys').argv:report_data()
