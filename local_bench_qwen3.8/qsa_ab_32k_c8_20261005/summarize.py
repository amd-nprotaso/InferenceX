import json, statistics, re
from pathlib import Path
r=Path(__file__).resolve().parent
trials={}
for name in ['off_1','on_1','on_2','off_2']:
 d=r/name
 x=json.loads((d/'client-exact-result.json').read_text())
 assert len(x['requests']) == 8
 assert all(t['actual_input_tokens']==32768 and t['actual_output_tokens']==32768 for t in x['requests'])
 s=(d/'server.log').read_text()
 enabled=name.startswith('on')
 assert f'flag={int(enabled)} eligible={enabled}' in s
 assert ('QSA_AB_EXECUTING _forward_flydsl_sparse' in s)==enabled
 vals=[]
 for t in x['requests']:
  chunks=t['stream_chunks']
  lo=next(c for c in chunks if c['completion_tokens']>=4096)
  hi=next(c for c in chunks if c['completion_tokens']>=28672)
  vals.append(1000*(hi['time_s']-lo['time_s'])/(hi['completion_tokens']-lo['completion_tokens']))
 trials[name]={k:x[k] for k in ['batch_wall_s','output_throughput_tok_s','ttft_s','tpot_ms','decode_interactivity_tok_s']}
 trials[name]['steady_decode_ms_per_token']=statistics.mean(vals)
 trials[name]['steady_decode_tokens_per_s_per_user']=1000/statistics.mean(vals)
 groups={}
for flag in ['off','on']:
 groups[flag]={k:statistics.mean(v[k] for n,v in trials.items() if n.startswith(flag)) for k in trials['off_1']}
change={k:100*(groups['on'][k]/groups['off'][k]-1) for k in groups['off']}
report={'configuration':{'concurrency':8,'isl':32768,'osl':32768,'tp':1,'nextn_steps':3,'draft_tokens':4,'simulated_accept_length':2.32,'order':list(trials),'warmup':'8 x (32768 input, 256 output), then flush cache','input':'distinct deterministic synthetic token IDs, identical across trials','dispatch_verified':True},'trials':trials,'means':groups,'change_percent_on_vs_off':change}
(r/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
