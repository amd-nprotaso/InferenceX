"""Analyze the two supplied Qwen3.5 traces without changing the inputs."""
import json, re, csv, bisect, sys
from collections import Counter, defaultdict
from pathlib import Path
from inspect_qwen_traces import PATHS
OUT = Path('qwen35_amd_nvidia_comparison')

def short(n):
    if n.startswith('Cijk_'): return 'rocBLAS/Tensile '+re.search(r'MT\d+x\d+x\d+',n)[0]
    if n.startswith('_Z'): 
        from sglang_trace_lib import shorten
        return shorten(n, 150)
    if n.startswith('void '): return n[5:].split('<')[0]
    return n if len(n)<160 else n[:157]+'...'

def load(vendor,path):
    d=json.load(open(path)); ev=d['traceEvents']
    steps=sorted([e for e in ev if e.get('cat')=='user_annotation' and e.get('name','').startswith('step[')],key=lambda e:e['ts'])
    runtimes={e['args']['correlation']:e for e in ev if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
    bystep=defaultdict(list)
    st=[e['ts'] for e in steps]
    for e in ev:
        if e.get('cat')!='kernel':continue
        rt=runtimes.get(e.get('args',{}).get('correlation'))
        if rt is None:continue
        i=bisect.bisect_right(st,rt['ts'])-1
        if i>=0 and rt['ts']<=steps[i]['ts']+steps[i]['dur'] and rt['tid']==steps[i]['tid']:
            bystep[i].append(e)
    for ks in bystep.values():ks.sort(key=lambda e:e['ts'])
    return d,steps,bystep,runtimes

if __name__=='__main__' and '--report' not in sys.argv:
    OUT.mkdir(exist_ok=True)
    for vendor,path in PATHS.items():
        d,steps,groups,runtimes=load(vendor,path)
        print(vendor, 'mapped',sum(map(len,groups.values())), 'total',sum(e.get('cat')=='kernel' for e in d['traceEvents']))
        for i,s in enumerate(steps):
            ks=groups[i]; c=Counter(e['name'] for e in ks)
            if i<10 or 'TARGET_VERIFY' in s['name'] and i<20:
                print(i,s['name'],len(ks),'linear',c['fused_qkvzba_split_reshape_cat_contiguous_kernel'],'moe_end',c['_fused_gate_sigmoid_mul_add_kernel'])
        for phase in ['EXTEND','TARGET_VERIFY']:
            i=next(i for i,s in enumerate(steps) if phase in s['name'] and sum(e['name']=='fused_qkvzba_split_reshape_cat_contiguous_kernel' for e in groups[i])==45)
            print('SEQUENCE',vendor,phase,i)
            for j,e in enumerate(groups[i][:155]):print(j,short(e['name']))
        mods=[e for e in d['traceEvents'] if e.get('name','').startswith('nn.Module: Qwen3_5') and 'DecoderLayer' in e['name']]
        print('module_counts',Counter(e['name'].split('_')[-1] for e in mods))

# Report generation intentionally preserves exact trace symbols in CSV and HTML.
def collective(n):
    return any(x in n for x in ['ncclDevKernel','cross_device_reduce','all_reduce_kernel','oneshotAllreduceFusionKernel'])

def initial_norm(n):
    return n=='_gemma_rmsnorm_kernel' or 'normkernelsrmsnormRMSNormKernel' in n

def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def report():
    import html, statistics
    OUT.mkdir(exist_ok=True)
    inventory=defaultdict(lambda: [0,0.0])
    samples=Counter(); sequences={}; metadata={}; boundary=[]; validation=[]
    for vendor,path in PATHS.items():
        d,steps,groups,runtimes=load(vendor,path)
        phases=Counter(); shapes=Counter(); attributed=0; validated=0; core_total=0
        mods=sorted([e for e in d['traceEvents'] if e.get('name','').startswith('nn.Module: Qwen3_5') and 'DecoderLayer' in e['name']],key=lambda e:e['ts'])
        for si,s in enumerate(steps):
            ks=groups[si]
            ends=[j for j,e in enumerate(ks) if e['name']=='_fused_gate_sigmoid_mul_add_kernel']
            if len(ends)!=60:continue # Draft model EXTEND has zero or one marker.
            phase='Prefill' if 'EXTEND' in s['name'] else 'Decode (TARGET_VERIFY)'
            forward=phases[phase]; phases[phase]+=1; shapes[s['name']]+=1
            begin=next(j for j,e in enumerate(ks) if initial_norm(e['name']))
            layer_mods=[m for m in mods if s['ts']<=m['ts'] and m['ts']+m['dur']<=s['ts']+s['dur']]
            if vendor=='AMD' and phase=='Prefill':assert len(layer_mods)==60
            for li,end in enumerate(ends):
                if li:begin=ends[li-1]+1
                # A standalone reduction belongs to the preceding layer output;
                # a fused reduction+norm belongs to the incoming layer boundary.
                while collective(ks[begin]['name']) and 'oneshotAllreduceFusionKernel' not in ks[begin]['name']:
                    boundary.append(dict(vendor=vendor,phase=phase,forward=forward,preceding_layer=li-1,kernel=ks[begin]['name'],duration_us=ks[begin]['dur']))
                    begin+=1
                seg=ks[begin:end+1]
                linear=sum(e['name']=='fused_qkvzba_split_reshape_cat_contiguous_kernel' for e in seg)
                full=sum(e['name'] in ('_fused_qk_gemma_rmsnorm_gate_kernel','_fused_qk_rmsnorm_rope_gate_kernel') for e in seg)
                assert (linear,full)==((0,1) if li%4==3 else (1,0)),(vendor,si,li,linear,full)
                kind='full_attention' if full else 'linear_attention'
                key=(vendor,phase,li,kind)
                samples[key]+=1
                if key not in sequences:sequences[key]=[dict(name=e['name'],duration_us=e['dur'],timestamp_us=e['ts'],stream=e['args'].get('stream')) for e in seg]
                if vendor=='AMD' and phase=='Prefill':
                    m=layer_mods[li]
                    expected='AttentionDecoderLayer' if full else 'LinearDecoderLayer'
                    assert expected in m['name']
                    marker=next(e for e in seg if e['name'] in ('fused_qkvzba_split_reshape_cat_contiguous_kernel','_fused_qk_gemma_rmsnorm_gate_kernel'))
                    rt=runtimes[marker['args']['correlation']]
                    assert m['ts']<=rt['ts']<=m['ts']+m['dur']
                    validated+=1
                for e in seg:
                    v=inventory[key+(e['name'],)];v[0]+=1;v[1]+=e['dur'];attributed+=1
                core_total+=1
            tail=ends[-1]+1
            if tail<len(ks) and collective(ks[tail]['name']) and 'oneshotAllreduceFusionKernel' not in ks[tail]['name']:
                boundary.append(dict(vendor=vendor,phase=phase,forward=forward,preceding_layer=59,kernel=ks[tail]['name'],duration_us=ks[tail]['dur']))
        metadata[vendor]=dict(path=path,device=d['deviceProperties'][0]['name'],world_size=d['distributedInfo']['world_size'],target_forwards=dict(phases),step_shapes=dict(shapes),total_kernels=sum(e.get('cat')=='kernel' for e in d['traceEvents']),kernels_linked_to_any_step=sum(map(len,groups.values())),layer_kernels=attributed,layer_instances=core_total,amd_prefill_modules_validated=validated)
        # Explain the otherwise opaque AMD prefill convolution symbol using its host stack.
        if vendor=='AMD':
            k=next(e for e in d['traceEvents'] if e.get('cat')=='kernel' and e['name']=='kernel_0')
            rt=runtimes[k['args']['correlation']]
            parents=[e for e in d['traceEvents'] if e.get('cat')=='python_function' and e.get('tid')==rt['tid'] and e['ts']<=rt['ts']<=e['ts']+e.get('dur',0) and any(t in e.get('name','').lower() for t in ('conv','flydsl'))]
            metadata[vendor]['kernel_0_host_stack']=[e['name'] for e in parents]
        del d,groups,runtimes
    rows=[]
    for key,(calls,total) in inventory.items():
        v,p,l,k,n=key; count=samples[key[:-1]]
        rows.append(dict(vendor=v,phase=p,layer=l,attention_type=k,kernel=n,short_label=short(n),forwards=count,calls=calls,calls_per_forward=calls/count,total_gpu_us=round(total,3),gpu_us_per_forward=round(total/count,3),mean_call_us=round(total/calls,3)))
    write_csv(OUT/'kernels_by_layer.csv',rows)
    write_csv(OUT/'boundary_collectives.csv',boundary)
    seqrows=[]
    for (v,p,l,k),seq in sequences.items():
        for j,e in enumerate(seq):seqrows.append(dict(vendor=v,phase=p,layer=l,attention_type=k,sequence_index=j,**e))
    write_csv(OUT/'representative_sequences.csv',seqrows)
    (OUT/'validation.json').write_text(json.dumps(metadata,indent=2))
    bykey=defaultdict(list)
    for r in rows:bykey[(r['vendor'],r['phase'],r['layer'])].append(r)
    summary=[]
    for p in ['Prefill','Decode (TARGET_VERIFY)']:
        for l in range(60):
            r=dict(phase=p,layer=l,attention_type='full_attention' if l%4==3 else 'linear_attention')
            for v in PATHS:
                rr=bykey[v,p,l];r[v+'_calls_per_forward']=round(sum(x['calls_per_forward'] for x in rr),3);r[v+'_gpu_us_per_forward']=round(sum(x['gpu_us_per_forward'] for x in rr),3)
            summary.append(r)
    write_csv(OUT/'layer_summary.csv',summary)
    note=('These TP-0 traces contain a 60-layer target model (45 linear-attention, 15 full-attention) and a separate draft model. '
          'Decode means speculative TARGET_VERIFY, bs=4; neither trace contains a plain DECODE step. '
          'Only target-model forwards with 60 MoE end markers are included: four prefill and 31 verification forwards per vendor. '
          'Layer numbers are zero-based execution order. Full-attention layers are 3, 7, 11, ..., 59. '
          'GPU kernels are linked to CPU launch/graph replay calls by correlation ID, then to the enclosing step. '
          'Layer boundaries use the repeated MoE end marker and attention signatures; all 240 AMD prefill layer markers were checked against CPU decoder-module intervals. '
          'NVIDIA compiled execution and graph replay layer indices are inferred from the validated repeated 3-linear/1-full ordering, not explicit module IDs. '
          'Each row includes the layer input normalization, attention, projections, shared expert and MoE through _fused_gate_sigmoid_mul_add_kernel. '
          'Standalone output reductions between layers are in boundary_collectives.csv. NVIDIA fused reduction+normalization is assigned to the following layer because it cannot be divided. '
          'Timing is summed kernel duration, not elapsed latency; streams can overlap. Prefill averages combine different chunk shapes and may include lazy initialization, so these are descriptive trace timings, not controlled speed ratios. '
          'Exact symbols, including templates and utility kernels, are preserved in CSV and expandable HTML. Labels in overview tables are shortened. '
          'Unassigned kernels include metadata, embedding, logits, sampling, draft work, and inter-layer boundaries; they are not silently attributed to target layers.')
    md=['# Qwen3.5: AMD MI355X vs NVIDIA B200','',note,'','## Inputs','']
    for v,m in metadata.items():md.append(f"- {v}: `{m['path']}`; TP={m['world_size']}; target step shapes: `{m['step_shapes']}`")
    md+=['','AMD `kernel_0` host stack: '+str(metadata['AMD'].get('kernel_0_host_stack')), '', '## Per-layer kernel inventory', '', 'For each vendor: exact-symbol entries are aggregated across all target forwards in that phase. Counts are calls per forward. CSV contains full signatures; this report uses shortened labels.','']
    blocks=[]
    for p in ['Prefill','Decode (TARGET_VERIFY)']:
        for kind in ['linear_attention','full_attention']:
            md+=['',f'## {p}: {kind}','']
            for l in range(60):
                if (l%4==3)!=(kind=='full_attention'):continue
                md += [f'### Layer {l}','','| AMD kernel | Calls/forward | NVIDIA kernel | Calls/forward |','|---|---:|---|---:|']
                ar=bykey['AMD',p,l]; nr=bykey['NVIDIA',p,l]
                for j in range(max(len(ar),len(nr))):
                    a=ar[j] if j<len(ar) else None;n=nr[j] if j<len(nr) else None
                    md.append('| '+(f"`{a['short_label']}` | {a['calls_per_forward']:g}" if a else ' | ')+' | '+(f"`{n['short_label']}` | {n['calls_per_forward']:g}" if n else ' | ')+' |')
                cols=[]
                for v,rr in [('AMD',ar),('NVIDIA',nr)]:
                    h=f'<h3>{v}</h3><p>{sum(r["calls_per_forward"] for r in rr):g} calls/forward; {sum(r["gpu_us_per_forward"] for r in rr):.2f} summed GPU µs/forward</p><table><tr><th>Kernel (expand exact symbol)</th><th>Calls/fwd</th><th>µs/fwd</th></tr>'
                    for r in rr:h+=f'<tr><td><details><summary>{html.escape(r["short_label"])}</summary><code>{html.escape(r["kernel"])}</code></details></td><td>{r["calls_per_forward"]:g}</td><td>{r["gpu_us_per_forward"]:.3f}</td></tr>'
                    h+='</table><details><summary>Execution sequence: first target forward in this phase</summary><ol>'
                    for e in sequences[v,p,l,kind]:h+=f'<li><code title="{html.escape(e["name"],quote=True)}">{html.escape(short(e["name"]))}</code> ({e["duration_us"]:.3f} µs)</li>'
                    h+='</ol></details>';cols.append(h)
                blocks.append(f'<details class="layer" data-phase="{p}" data-kind="{kind}" data-layer="{l}"><summary>{p}: {kind} — Layer {l}</summary><div class="columns"><section>{cols[0]}</section><section>{cols[1]}</section></div></details>')
    md+=['','Tables list independent vendor inventories in first-seen order. Rows do not assert one-to-one equivalence between individual AMD and NVIDIA kernels.']
    (OUT/'comparison.md').write_text('\n'.join(md)+'\n')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Qwen3.5 kernel comparison</title><style>body{font:15px system-ui;margin:24px;color:#17212d;background:#f8fafc}h1{font-size:26px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{min-width:0}table{border-collapse:collapse;width:100%;background:white}td,th{padding:7px;border:1px solid #ddd;text-align:left;vertical-align:top}code{overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer;overflow-wrap:anywhere}.layer{margin:10px 0;padding:12px;background:white;border:1px solid #ccd5df;border-radius:6px}.layer>summary{font-weight:650}nav{position:sticky;top:0;background:#e9eff7;padding:12px;z-index:1}input,select,button{padding:7px;margin-right:10px}p{line-height:1.6;max-width:1400px}li{margin:4px} @media(max-width:900px){.columns{grid-template-columns:1fr}}</style><h1>Qwen3.5: AMD MI355X vs NVIDIA B200</h1>'''
    page+='<p>'+html.escape(note)+'</p><p>Inventory columns are independent lists, not one-to-one kernel matches. Search matches exact symbols as well as labels. Expand a layer to compare every kernel; expand a kernel for its exact trace symbol.</p>'
    page+='''<nav><select id="phase"><option value="">All phases</option><option>Prefill</option><option>Decode (TARGET_VERIFY)</option></select><select id="kind"><option value="">All attention types</option><option>linear_attention</option><option>full_attention</option></select><input id="layer" type="number" min="0" max="59" placeholder="Layer 0–59"><input id="search" placeholder="Search kernel"><button id="expand">Expand visible layers</button></nav>'''
    page+=''.join(blocks)+'''<script>const layers=[...document.querySelectorAll('.layer')];const ids=['phase','kind','layer','search'];function filter(){const [p,k,l,s]=ids.map(id=>document.getElementById(id).value);layers.forEach(e=>e.hidden=!!((p&&e.dataset.phase!==p)||(k&&e.dataset.kind!==k)||(l!==''&&e.dataset.layer!==l)||(s&&!e.textContent.toLowerCase().includes(s.toLowerCase()))));}ids.forEach(id=>document.getElementById(id).addEventListener('input',filter));document.getElementById('expand').onclick=()=>layers.filter(e=>!e.hidden).forEach(e=>e.open=true);</script></html>'''
    (OUT/'comparison.html').write_text(page)
    print(json.dumps(metadata,indent=2))
    print('OUTPUT',OUT.resolve(),len(rows),'inventory rows',len(summary),'layer comparisons')
    for p in ['Prefill','Decode (TARGET_VERIFY)']:
        for l in [1,3]:
            print(p,'layer',l)
            for v in PATHS:print(v,[r['short_label'] for r in bykey[v,p,l] if not any(t in r['kernel'] for t in ['at::native','Cijk_','nvjet','rocprim','memcpy','cublasLt'])])

if __name__=='__main__' and '--report' in sys.argv:
    report()
