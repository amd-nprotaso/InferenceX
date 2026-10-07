import json
import os
import time
import urllib.request
from pathlib import Path

os.environ['HF_HOME'] = '/data/hf_cache/'
root = Path(os.environ['QSA_E2E_OUTPUT'])
model = '/data/hf_cache/hub/models--amd--Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8/snapshots/1ad36fa866c05631c14adc6a74d7e9908c4e5888'
payload = {'sampling_params': {'max_new_tokens':32768, 'ignore_eos':True, 'temperature':0}, 'stream':True}

from concurrent.futures import ThreadPoolExecutor
import threading
import statistics
output_root = root
barrier = threading.Barrier(9)
def run_one(index):
    root = output_root / f'request_{index}'
    root.mkdir()
    local_payload = dict(payload)
    local_payload['input_ids'] = [1000 + ((j * 7919 + index * 104729) % 29000) for j in range(32768)]
    (root/'input_ids.json').write_text(json.dumps(local_payload['input_ids']))
    (root/'request.json').write_text(json.dumps(local_payload))
    req = urllib.request.Request('http://127.0.0.1:18888/generate', data=json.dumps(local_payload).encode(), headers={'Content-Type':'application/json'})
    barrier.wait()
    start = time.perf_counter()
    chunks=[]
    last_count=0
    last_text=''
    final={}
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            if not raw.startswith(b'data:'): continue
            data=raw[5:].strip()
            if data == b'[DONE]': break
            obj=json.loads(data)
            now=time.perf_counter()-start
            meta=obj.get('meta_info',{})
            count=meta.get('completion_tokens',0)
            if count > last_count:
                chunks.append({'time_s':now,'completion_tokens':count,'delta_tokens':count-last_count})
                if len(chunks)==1 or count//2048 > last_count//2048:
                    print(f'elapsed={now:.3f}s output_tokens={count} average_output_rate={count/now:.2f} tok/s',flush=True)
                last_count=count
            last_text=obj.get('text','')
            final=meta
    end=time.perf_counter()-start
    if not chunks: raise RuntimeError(f'No output: {final}')
    ttft=chunks[0]['time_s']
    last_time=chunks[-1]['time_s']
    output_tokens=final.get('completion_tokens',last_count)
    # Client-observed TPOT convention; server streams groups of tokens.
    tpot=(last_time-ttft)/(output_tokens-chunks[0]['completion_tokens'])
    result={'model':model,'tp':1,'concurrency':1,'requested_input_tokens':32768,'requested_output_tokens':32768,'actual_input_tokens':final.get('prompt_tokens'),'actual_output_tokens':output_tokens,'successful_requests':1,'ttft_s':ttft,'e2e_s':last_time,'client_wall_s':end,'tpot_ms':tpot*1000,'decode_interactivity_tok_s':1/tpot,'output_throughput_tok_s':output_tokens/end,'total_throughput_tok_s':(final.get('prompt_tokens',32768)+output_tokens)/end,'stream_chunk_count':len(chunks),'stream_chunks':chunks,'server_meta':final,'output_preview':last_text[:200],'output_tail':last_text[-200:]}
    (root/'client-exact-result.json').write_text(json.dumps(result,indent=2))
    (root/'client-exact-output.txt').write_text(last_text)
    print(f'Request {index} complete: {output_tokens} tokens in {end:.3f}s', flush=True)
    if final.get('prompt_tokens') != 32768 or output_tokens != 32768: raise RuntimeError('Actual lengths do not match requested lengths')
    return result

def warmup_one(index):
    warm = {'input_ids': [1000 + ((j * 7919 + index * 104729) % 29000) for j in range(32768)], 'sampling_params': {'max_new_tokens': 256, 'ignore_eos': True, 'temperature': 0}}
    req = urllib.request.Request('http://127.0.0.1:18888/generate', data=json.dumps(warm).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)['meta_info']
print('Warming all 8 lanes: ISL=32768 OSL=256', flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:
    warm_results = list(pool.map(warmup_one, range(8)))
(output_root/'warmup_batch.json').write_text(json.dumps(warm_results, indent=2))
with urllib.request.urlopen('http://127.0.0.1:18888/flush_cache', timeout=60) as response:
    print('Flush after representative warmup:', response.status, flush=True)
print('Starting 8 simultaneous requests: ISL=32768 OSL=32768', flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(run_one, index) for index in range(8)]
    start = time.perf_counter()
    barrier.wait()
    results = [future.result() for future in futures]
    wall = time.perf_counter() - start
summary = {
    'model': model, 'tp': 1, 'concurrency': 8, 'successful_requests': len(results),
    'requested_input_tokens': 32768, 'requested_output_tokens': 32768,
    'batch_wall_s': wall,
    'output_throughput_tok_s': sum(r['actual_output_tokens'] for r in results)/wall,
    'total_throughput_tok_s': sum(r['actual_input_tokens']+r['actual_output_tokens'] for r in results)/wall,
    'requests': results,
}
for key in ['e2e_s', 'ttft_s', 'tpot_ms', 'decode_interactivity_tok_s']:
    summary[key] = statistics.mean(r[key] for r in results)
(output_root/'client-exact-result.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k:v for k,v in summary.items() if k != 'requests'}, indent=2), flush=True)
