import ast
import hashlib
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
names = ['ck', 'triton', 'flydsl']
expected = {
    'ck': 'ck_flash_attn_varlen_func',
    'triton': 'sparse_gqa_packed_decode_triton',
    'flydsl': '_forward_flydsl_sparse',
}
results, hashes, configs = {}, {}, {}
for name in names:
    directory = root / name
    data = json.loads((directory / 'client-exact-result.json').read_text())
    requests = data['requests']
    assert len(requests) == 8
    assert all(r['actual_input_tokens'] == 32768 and r['actual_output_tokens'] == 32768 for r in requests)
    hashes[name] = [hashlib.sha256((directory / f'request_{i}/input_ids.json').read_bytes()).hexdigest() for i in range(8)]
    log = (directory / 'server.log').read_text()
    dispatches = {line.split('QSA_THREEWAY_EXECUTING ')[1] for line in log.splitlines() if 'QSA_THREEWAY_EXECUTING ' in line}
    assert dispatches == {expected[name]}, (name, dispatches)
    for line in log.splitlines():
        if 'server_args={' in line:
            configs[name] = ast.literal_eval(line.split('server_args=', 1)[1])
            break
    assert configs[name]['random_seed'] == 42
    steady = []
    for request in requests:
        chunks = request['stream_chunks']
        start = next(c for c in chunks if c['completion_tokens'] >= 4096)
        end = next(c for c in chunks if c['completion_tokens'] >= 28672)
        steady.append(1000 * (end['time_s'] - start['time_s']) / (end['completion_tokens'] - start['completion_tokens']))
    results[name] = {key: data[key] for key in ['batch_wall_s', 'output_throughput_tok_s', 'ttft_s', 'tpot_ms']}
    results[name]['steady_decode_ms_per_token'] = statistics.mean(steady)
    results[name]['spec_accept_length'] = statistics.mean(r['server_meta']['spec_accept_length'] for r in requests)
    results[name]['attention_implementation'] = expected[name]
assert all(h == hashes['ck'] for h in hashes.values())
assert len(set(hashes['ck'])) == 8
differences = {k: {name: cfg.get(k) for name, cfg in configs.items()} for k in configs['ck'] if any(cfg.get(k) != configs['ck'][k] for cfg in configs.values())}
assert not differences, differences
comparisons = {}
for baseline, candidate in [('ck', 'triton'), ('ck', 'flydsl'), ('triton', 'flydsl')]:
    comparisons[f'{candidate}_vs_{baseline}'] = {
        'throughput_gain_percent': 100 * (results[candidate]['output_throughput_tok_s'] / results[baseline]['output_throughput_tok_s'] - 1),
        'tpot_change_percent': 100 * (results[candidate]['tpot_ms'] / results[baseline]['tpot_ms'] - 1),
        'steady_decode_tpot_change_percent': 100 * (results[candidate]['steady_decode_ms_per_token'] / results[baseline]['steady_decode_ms_per_token'] - 1),
    }
report = {
    'configuration': {'concurrency': 8, 'isl': 32768, 'osl': 32768, 'tp': 1, 'random_seed': 42, 'simulated_accept_length': 2.32, 'nextn_steps': 3, 'draft_tokens': 4, 'warmup': '8 requests with ISL=32768, OSL=256, then flush cache'},
    'results': results,
    'comparisons': comparisons,
    'validation': {'all_24_requests_exact_lengths': True, 'kernel_dispatch_verified': True, 'identical_inputs_across_arms': True, 'distinct_inputs_within_batch': True, 'server_argument_differences': differences, 'input_sha256': hashes},
    'limitations': ['One measured batch per implementation; small differences are not statistically established.', 'Application-level serving benchmark, not isolated kernel timing or model accuracy evaluation.', 'CK arm uses an experiment-local override to bypass the current ROCm Triton fallback.'],
}
(root / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'validation'}, indent=2))
