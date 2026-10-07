#!/usr/bin/env python3
"""Compare the complete unprofiled requests and record comparability checks."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main() -> None:
    results={arm:json.loads((ROOT/arm/'client-exact-result.json').read_text()) for arm in ['baseline','tuned']}
    payloads={arm:json.loads((ROOT/arm/'client-exact-request.json').read_text()) for arm in results}
    if payloads['baseline']!=payloads['tuned']:raise RuntimeError('Request payload mismatch')
    for arm,r in results.items():
        assert r['actual_input_tokens']==32768 and r['actual_output_tokens']==32768
        assert r['successful_requests']==1
        assert r['server_meta']['cached_tokens']==0,(arm,r['server_meta']['cached_tokens'])
        warm=json.loads((ROOT/arm/'warmup.json').read_text())['meta_info']
        assert warm['prompt_tokens']==32768 and warm['completion_tokens']==32
    rows=[]
    for key in ['ttft_s','e2e_s','client_wall_s','tpot_ms','decode_interactivity_tok_s','output_throughput_tok_s','total_throughput_tok_s']:
        a,b=(results[arm][key] for arm in ['baseline','tuned'])
        rows.append(dict(metric=key,baseline=a,tuned=b,change_percent=100*(b/a-1)))
    with (ROOT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out_hash={arm:hashlib.sha256((ROOT/arm/'client-exact-output.txt').read_bytes()).hexdigest() for arm in results}
    info={arm:json.loads((ROOT/arm/'server_info.json').read_text()) for arm in results}
    args={arm:{k:v for k,v in info[arm].get('server_args',info[arm]).items() if k not in {'internal_states', 'last_gen_throughput'}} for arm in results}
    argdiff={k:[args['baseline'].get(k),args['tuned'].get(k)] for k in args['baseline'].keys()|args['tuned'].keys() if args['baseline'].get(k)!=args['tuned'].get(k)}
    counters=['spec_accept_rate','spec_accept_length','spec_num_correct_drafts','spec_num_proposed_drafts','spec_verify_ct','num_retractions','cached_tokens','finish_reason']
    checks={'identical_request':True,'identical_output_text':out_hash['baseline']==out_hash['tuned'],'output_sha256':out_hash,'server_argument_differences':argdiff,
            'profile':False,'trials_per_arm':1,'speculative_metadata':{arm:{k:r['server_meta'].get(k) for k in counters} for arm,r in results.items()},
            'loaded_qsa':{arm:json.loads((ROOT/arm/'loaded_qsa.json').read_text()) for arm in results}}
    (ROOT/'comparison_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps({'metrics':rows,'checks':checks},indent=2))
if __name__=='__main__':main()
