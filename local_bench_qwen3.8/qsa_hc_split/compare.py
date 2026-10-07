#!/usr/bin/env python3
"""Validate baseline-QSA HC combine split E2E trials."""
from __future__ import annotations
import csv
import hashlib
import json
import statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent
ARMS = ['unset', 'enabled', 'disabled']
LABELS = [label for arm in ARMS for label in [arm, arm+'_repeat']]
METRICS = ['e2e_s', 'ttft_s', 'tpot_ms', 'output_throughput_tok_s']

def main() -> None:
    results = {x: json.loads((ROOT/x/'client-exact-result.json').read_text()) for x in LABELS}
    payloads = [json.loads((ROOT/x/'client-exact-request.json').read_text()) for x in LABELS]
    assert all(p == payloads[0] for p in payloads)
    rows = []
    args = {}
    for label, result in results.items():
        meta = result['server_meta']
        assert result['actual_input_tokens'] == result['actual_output_tokens'] == 32768
        assert result['successful_requests'] == 1
        assert meta['cached_tokens'] == 0
        assert meta.get('num_retractions', 0) == 0
        warm = json.loads((ROOT/label/'warmup.json').read_text())['meta_info']
        assert warm['prompt_tokens'] == 32768 and warm['completion_tokens'] == 32
        info = json.loads((ROOT/label/'server_info.json').read_text())
        args[label] = {k: v for k, v in info.get('server_args', info).items() if k not in {'internal_states', 'startup_time', 'last_gen_throughput'}}
        assert args[label]['random_seed'] == 20260928
        row = {'trial': label, **{k: result[k] for k in METRICS}}
        row.update({k: meta.get(k) for k in ['spec_accept_length', 'spec_accept_rate', 'spec_verify_ct', 'num_retractions']})
        row['decode_ms_per_verification_proxy'] = 1000 * (result['e2e_s']-result['ttft_s'])/meta['spec_verify_ct']
        row['output_sha256'] = hashlib.sha256((ROOT/label/'client-exact-output.txt').read_bytes()).hexdigest()
        rows.append(row)
    with (ROOT/'trials.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for key in METRICS:
        summary[key] = {}
        for arm in ARMS:
            values = [results[x][key] for x in [arm, arm+'_repeat']]
            summary[key][arm] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
        summary[key]['enabled_vs_unset_percent'] = 100*(summary[key]['enabled']['mean']/summary[key]['unset']['mean']-1)
        summary[key]['disabled_vs_enabled_percent'] = 100*(summary[key]['disabled']['mean']/summary[key]['enabled']['mean']-1)
    argdiff = {k: {label: a.get(k) for label, a in args.items()} for k in set().union(*(a.keys() for a in args.values())) if any(a.get(k) != args['unset'].get(k) for a in args.values())}
    checks = {'identical_requests': True, 'fixed_seed': 20260928, 'profile': False, 'trials_per_arm': 2,
              'identical_output_text': len({r['output_sha256'] for r in rows}) == 1,
              'server_argument_differences': argdiff,
              'loaded_qsa': {x: json.loads((ROOT/x/'loaded_qsa.json').read_text()) for x in LABELS}}
    checks['flag_values'] = {x: json.loads((ROOT/x/'flag_probe.json').read_text()) for x in LABELS}
    for label, flag in checks['flag_values'].items():
        arm=label.removesuffix('_repeat')
        assert flag['raw']=={'unset':None,'enabled':'1','disabled':'0'}[arm]
        assert flag['effective']==(arm!='disabled')
        for proc in checks['loaded_qsa'][label]:
            assert proc['libraries']==['/sgl-workspace/aiter/aiter/jit/mha_varlen_fwd_bf16_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale.so']
    data = {'trials': rows, 'summary': summary, 'checks': checks}
    (ROOT/'comparison.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({'summary': summary, 'checks': checks}, indent=2))

if __name__ == '__main__':
    main()
