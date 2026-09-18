"""Add recorded/inferred torch module sublists to the Qwen trace comparison."""
import bisect, csv, json, re
from collections import defaultdict, Counter
from pathlib import Path
from qwen_layer_comparison import load, PATHS, OUT, short, collective, initial_norm, write_csv

def module_index(events):
    groups=defaultdict(list)
    for e in events:
        if e.get('name','').startswith('nn.Module:'):
            groups[e['pid'],e['tid']].append(e)
    result={}
    for key, mods in groups.items():
        mods.sort(key=lambda e:(e['ts'],-e['dur']))
        stack=[]; paths=[]
        for e in mods:
            while stack and stack[-1]['ts']+stack[-1]['dur'] < e['ts']+e['dur']-0.01:stack.pop()
            stack.append(e);paths.append(list(stack))
        result[key]=([e['ts'] for e in mods],paths)
    return result

def recorded_path(rt,index):
    starts,paths=index.get((rt['pid'],rt['tid']),([],[]))
    j=bisect.bisect_right(starts,rt['ts'])-1
    if j<0:return []
    return [m['name'].removeprefix('nn.Module: ') for m in paths[j] if m['ts']<=rt['ts']<=m['ts']+m['dur']]

def layer_segments(steps,groups):
    phases=Counter()
    for si,s in enumerate(steps):
        ks=groups[si]; ends=[j for j,e in enumerate(ks) if e['name']=='_fused_gate_sigmoid_mul_add_kernel']
        if len(ends)!=60:continue
        phase='Prefill' if 'EXTEND' in s['name'] else 'Decode (TARGET_VERIFY)'
        forward=phases[phase];phases[phase]+=1
        start=next(j for j,e in enumerate(ks) if initial_norm(e['name']))
        for li,end in enumerate(ends):
            if li:start=ends[li-1]+1
            while collective(ks[start]['name']) and 'oneshotAllreduceFusionKernel' not in ks[start]['name']:start+=1
            yield phase,forward,li,ks[start:end+1]

if __name__=='__main__':
    d,steps,groups,runtimes=load('AMD',PATHS['AMD']);index=module_index(d['traceEvents'])
    for phase,forward,li,seg in layer_segments(steps,groups):
        if phase!='Prefill' or forward or li not in (0,3):continue
        print('LAYER',li)
        last=None
        for e in seg:
            p=recorded_path(runtimes[e['args']['correlation']],index)
            if p!=last:print('PATH',p);last=p
            print(' ',short(e['name']))

def inference_roles(seg,linear,vendor):
    """Use attention anchors and semantic symbols; avoid guessing ambiguous GEMMs."""
    names=[e['name'] for e in seg]
    split=names.index('fused_qkvzba_split_reshape_cat_contiguous_kernel') if linear else names.index('_fused_qk_gemma_rmsnorm_gate_kernel' if vendor=='AMD' else '_fused_qk_rmsnorm_rope_gate_kernel')
    out=names.index('_layer_norm_fwd_1pass_kernel') if linear else names.index('_fused_sigmoid_mul_kernel')
    comm=next(j for j in range(out+1,len(names)) if collective(names[j]))
    moe_start=comm+1
    if 'oneshotAllreduceFusionKernel' not in names[comm]:moe_start+=1
    roles=[];proj=0;post_gemms=0;act_seen=False
    for j,n in enumerate(names):
        label=short(n)
        if j==0:role='input_norm'
        elif j<split:
            if 'splitKreduce' not in n:proj+=1
            role=('input_proj' if proj<=1 else 'gating_proj') if linear else 'qkv_proj'
        elif j==split:role='attention_layout' if linear else ('qk_rope_fused' if vendor=='NVIDIA' else 'qk_prepare')
        elif j<out:
            role='rotary' if '_triton_mrope' in n else 'radix'
        elif j==out:role='gated_norm' if linear else 'attention_gate'
        elif j<comm:role='out_proj'
        elif j==comm:role='post_norm_fused' if 'oneshotAllreduceFusionKernel' in n else 'attention_reduce'
        elif j<moe_start:role='post_norm'
        elif n=='_fused_gate_sigmoid_mul_add_kernel':role='moe_combine'
        elif 'act_and_mul_kernel' in n:role='shared_activation';act_seen=True
        elif any(s in n for s in ['topkGatingSoftmax','routingIndices','routingInitExpert']):role='topk' if vendor=='AMD' else 'routing_fused'
        elif any(s in n for s in ['moe','MoE','Quantize','quantization','dynamic_per_group_scaled_quant']):role='experts'
        elif n.startswith(('Cijk_','hgemm_','nvjet_')):
            post_gemms+=1
            if vendor=='AMD':role=['shared_up','shared_down','router'][min(post_gemms-1,2)]
            else:role='shared_down' if act_seen else 'moe_projection_ambiguous'
        elif 'splitKreduce' in n or 'memcpy' in n:role='moe_projection_ambiguous'
        else:role='moe_other'
        roles.append(role)
    return roles

def inferred_path(role,linear):
    root='Qwen3_5LinearDecoderLayer' if linear else 'Qwen3_5AttentionDecoderLayer'
    attn=[root,'Qwen3_5GatedDeltaNet'] if linear else [root]
    moe=[root,'Qwen2MoeSparseMoeBlock'];shared=moe+['Qwen2MoeMLP (shared expert)']
    paths={
        'input_norm':[root,'GemmaRMSNorm (input; may fuse preceding reduction)'],
        'input_proj':attn+['MergedColumnParallelLinear (QKV/Z projection)'],
        'gating_proj':attn+['MergedColumnParallelLinear (B/A gating projection)'],
        'qkv_proj':[root,'QKVParallelLinear'],
        'attention_layout':attn,
        'qk_prepare':[root],
        'qk_rope_fused':[root,'Fused Q/K preparation + MRotaryEmbedding (cross-module)'],
        'rotary':[root,'MRotaryEmbedding'],
        'radix':attn+['RadixLinearAttention' if linear else 'RadixAttention'],
        'gated_norm':attn+['RMSNorm (gated output)'],
        'attention_gate':[root],
        'out_proj':attn+['RowParallelLinear (attention output)'],
        'attention_reduce':[root],
        'post_norm':[root,'GemmaRMSNorm (post-attention)'],
        'post_norm_fused':[root,'Fused attention reduction + GemmaRMSNorm (cross-module)'],
        'shared_up':shared+['MergedColumnParallelLinear (gate/up)'],
        'shared_activation':shared+['SiluAndMul'],
        'shared_down':shared+['RowParallelLinear (down)'],
        'router':moe+['ReplicatedLinear (router)'],
        'topk':moe+['TopK'],
        'routing_fused':moe+['TopK / FusedMoE routing (fused backend; boundary inferred)'],
        'experts':moe+['FusedMoE'],
        'moe_combine':moe,
        'moe_projection_ambiguous':moe+['Projection work (shared expert/router attribution ambiguous)'],
        'moe_other':moe,
    }
    return paths[role]

def generate_modules():
    import html, math
    inventory=defaultdict(lambda:[0,0.0]);examples=defaultdict(list);coverage=Counter();forward_counts=Counter()
    amd_templates={}
    for vendor,path in PATHS.items():
        d,steps,groups,runtimes=load(vendor,path);index=module_index(d['traceEvents'])
        for phase,forward,li,seg in layer_segments(steps,groups):
            linear=li%4!=3;roles=inference_roles(seg,linear,vendor)
            forward_counts[vendor,phase,li]+=1
            for j,(e,role) in enumerate(zip(seg,roles)):
                rt=runtimes[e['args']['correlation']]
                recorded=recorded_path(rt,index)
                start=next((i for i,n in enumerate(recorded) if 'DecoderLayer_' in n),None)
                if start is not None:
                    module_path=recorded[start:];evidence='recorded CPU module + launch correlation'
                    if vendor=='AMD':amd_templates.setdefault((li,role),module_path)
                elif vendor=='AMD' and (li,role) in amd_templates:
                    module_path=amd_templates[li,role];evidence='inferred from AMD prefill module and decode kernel role; IDs are prefill references'
                else:
                    module_path=inferred_path(role,linear)
                    evidence='inferred from AMD module structure and kernel role'
                    if 'ambiguous' in role:evidence+='; leaf module unresolved'
                    if 'fused' in role:evidence+='; spans or changes module boundaries'
                coverage[vendor,phase,evidence]+=1
                key=(vendor,phase,li,' > '.join(module_path),evidence,e['name'])
                inventory[key][0]+=1;inventory[key][1]+=e['dur']
                if forward==0:examples[vendor,phase,li].append(dict(sequence_index=j,module_path=' > '.join(module_path),evidence=evidence,kernel=e['name']))
        del d,steps,groups,runtimes,index
    rows=[]
    for (v,p,l,m,e,n),(calls,total) in inventory.items():
        count=forward_counts[v,p,l]
        rows.append(dict(vendor=v,phase=p,layer=l,attention_type='full_attention' if l%4==3 else 'linear_attention',module_path=m,attribution=e,kernel=n,short_label=short(n),forwards=count,calls=calls,calls_per_forward=calls/count,total_gpu_us=round(total,3),gpu_us_per_forward=round(total/count,3)))
    # Conservation checks against the previously verified flat inventory.
    before=list(csv.DictReader((OUT/'kernels_by_layer.csv').open()))
    original={(r['vendor'],r['phase'],int(r['layer']),r['kernel']):(int(r['calls']),float(r['total_gpu_us'])) for r in before}
    current=defaultdict(lambda:[0,0.0])
    for r in rows:
        a=current[r['vendor'],r['phase'],r['layer'],r['kernel']];a[0]+=r['calls'];a[1]+=r['total_gpu_us']
    assert current.keys()==original.keys()
    for k,(calls,total) in current.items():
        assert calls==original[k][0],k
        assert abs(total-original[k][1])<0.02,(k,total,original[k][1])
    write_csv(OUT/'kernels_by_module.csv',rows)
    write_csv(OUT/'module_sequences.csv',[dict(vendor=v,phase=p,layer=l,**e) for (v,p,l),seq in examples.items() for e in seq])
    bylayer=defaultdict(list)
    for r in rows:bylayer[r['vendor'],r['phase'],r['layer']].append(r)
    explanation=('Module sublists: AMD prefill uses recorded nn.Module ranges linked to CPU kernel launches. '
      'AMD decode uses AMD prefill module references and kernel-role inference because graph replay hides the module stack. '
      'NVIDIA groups are inferred from AMD structure and kernel roles; they are not observed NVIDIA module IDs. '
      'Fused operations spanning modules and ambiguous shared-expert/router projections are labeled explicitly. '
      'Kernels launched directly by a parent module appear under “Direct kernels” rather than an invented child module. '
      'Counts and times cover the same forwards as the flat inventory. Each kernel invocation is counted once. '
      'Module grouping reorders the display by hierarchy; the existing execution sequence preserves GPU start order.')
    def tree(rr):
        root={'children':{},'rows':[]}
        for r in rr:
            node=root
            for name in r['module_path'].split(' > '):node=node['children'].setdefault(name,{'children':{},'rows':[]})
            node['rows'].append(r)
        return root
    def render_node(node,name=None):
        body=''
        if node['rows']:
            body+='<p><b>Direct kernels</b></p><ul>'
            for r in node['rows']:
                body+=f'<li><details><summary><code>{html.escape(r["short_label"])}</code> — {r["calls_per_forward"]:g} calls/fwd; {r["gpu_us_per_forward"]:.3f} µs/fwd</summary><code>{html.escape(r["kernel"])}</code><p>{html.escape(r["attribution"])}</p></details></li>'
            body+='</ul>'
        for child,item in node['children'].items():body+=render_node(item,child)
        if name is None:return body
        return '<details class="module" open><summary>'+html.escape(name)+'</summary>'+body+'</details>'
    def render_md(node,depth=0):
        lines=[]
        for name,item in node['children'].items():
            lines.append('  '*depth+'- **'+name+'**')
            for r in item['rows']:
                lines.append('  '*(depth+1)+f'- `{r["short_label"]}` — {r["calls_per_forward"]:g} calls/fwd; {r["gpu_us_per_forward"]:.3f} µs/fwd')
            lines.extend(render_md(item,depth+1))
        return lines
    page=(OUT/'comparison.html').read_text()
    # Replace an existing generated module panel if run repeatedly.
    page=re.sub(r'<!-- MODULE_PANEL_START -->.*?<!-- MODULE_PANEL_END -->','',page,flags=re.S)
    pat=r'(<details class="layer" data-phase="([^"]+)" data-kind="[^"]+" data-layer="(\d+)"><summary>.*?</summary>)'
    def inject(match):
        p=match[2];l=int(match[3]);cols=[]
        for v in PATHS:
            status='Recorded modules' if v=='AMD' and p=='Prefill' else 'Inferred modules'
            cols.append('<section><h3>'+v+' — '+status+'</h3>'+render_node(tree(bylayer[v,p,l]))+'</section>')
        return match[1]+'<!-- MODULE_PANEL_START --><details class="module-panel" open><summary><b>PyTorch module sublists</b></summary><div class="columns">'+''.join(cols)+'</div></details><!-- MODULE_PANEL_END -->'
    page,n=re.subn(pat,inject,page,flags=re.S);assert n==120
    page=re.sub(r'<!-- MODULE_NOTE_START -->.*?<!-- MODULE_NOTE_END -->','',page,flags=re.S)
    page=page.replace('<nav>','<!-- MODULE_NOTE_START --><p>'+html.escape(explanation)+'</p><!-- MODULE_NOTE_END --><nav>',1)
    if '.module{' not in page:page=page.replace('</style>','.module{margin:8px 0 8px 12px;padding:8px;border-left:2px solid #b5c9e3}.module>summary{font-weight:600}.module-panel{margin-top:14px;background:#f2f6fc;padding:12px}</style>')
    page=page.replace('placeholder="Search kernel"','placeholder="Search kernel or module"')
    (OUT/'comparison.html').write_text(page)
    md=['# Qwen3.5: per-layer PyTorch module sublists','',explanation,'','[Interactive report](comparison.html) · [Exact module/kernel CSV](kernels_by_module.csv) · [Sequence with module attribution](module_sequences.csv)','']
    for p in ['Prefill','Decode (TARGET_VERIFY)']:
        for kind in ['linear_attention','full_attention']:
            md+=['',f'## {p}: {kind}','']
            for l in range(60):
                if (l%4==3)!=(kind=='full_attention'):continue
                md+=['',f'### Layer {l}','']
                for v in PATHS:
                    status='recorded' if v=='AMD' and p=='Prefill' else 'inferred'
                    md+=['',f'#### {v} — {status}','']+render_md(tree(bylayer[v,p,l]))
    (OUT/'modules.md').write_text('\n'.join(md)+'\n')
    for filename in ['overview.md','comparison.md']:
        p=OUT/filename;s=p.read_text()
        if '[PyTorch module sublists]' not in s:
            first,rest=s.split('\n',1);p.write_text(first+'\n\n[PyTorch module sublists](modules.md) · [Module/kernel CSV](kernels_by_module.csv)\n'+rest)
    stats={'module_kernel_rows':len(rows),'kernel_calls_preserved':sum(r['calls'] for r in rows),'flat_inventory_entries_preserved':len(original),'layer_panels':n,'attribution_counts':[dict(vendor=v,phase=p,attribution=e,calls=c) for (v,p,e),c in coverage.items()]}
    (OUT/'module_validation.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))

if __name__=='__main__' and '--report' in __import__('sys').argv:
    generate_modules()
