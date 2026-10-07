import json
import os
import time
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer

os.environ['HF_HOME'] = '/data/hf_cache/'
root = Path(os.environ['QSA_E2E_OUTPUT'])
root.mkdir(parents=True, exist_ok=True)
model = 'amd/Qwen3.8-Flash-Next-Quark-MXFP4'
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
text = 'Explain the design and performance of a distributed database. Discuss replication, indexing, transactions, and recovery. '
ids = tok.apply_chat_template([{'role':'user','content':text * 3000}], tokenize=False, add_generation_prompt=True)
ids = tok.encode(ids, add_special_tokens=False)
# Preserve the chat header and assistant suffix, removing excess user content.
ids = ids[:32768-128] + ids[-128:]
assert len(ids) == 32768
payload = {'input_ids': ids, 'sampling_params': {'max_new_tokens':32768, 'ignore_eos':True, 'temperature':0}, 'stream':True}
(root/'client-exact-request.json').write_text(json.dumps(payload))
print('Starting one request: input_tokens=32768 output_tokens=32768 concurrency=1', flush=True)
req = urllib.request.Request('http://127.0.0.1:30000/generate', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
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
tpot=(last_time-ttft)/(output_tokens-1)
result={'model':model,'tp':1,'concurrency':1,'requested_input_tokens':32768,'requested_output_tokens':32768,'actual_input_tokens':final.get('prompt_tokens'),'actual_output_tokens':output_tokens,'successful_requests':1,'ttft_s':ttft,'e2e_s':last_time,'client_wall_s':end,'tpot_ms':tpot*1000,'decode_interactivity_tok_s':1/tpot,'output_throughput_tok_s':output_tokens/end,'total_throughput_tok_s':(final.get('prompt_tokens',32768)+output_tokens)/end,'stream_chunk_count':len(chunks),'stream_chunks':chunks,'server_meta':final,'output_preview':last_text[:200],'output_tail':last_text[-200:]}
(root/'client-exact-result.json').write_text(json.dumps(result,indent=2))
(root/'client-exact-output.txt').write_text(last_text)
print(json.dumps({k:v for k,v in result.items() if k != 'stream_chunks'},indent=2),flush=True)
if final.get('prompt_tokens') != 32768 or output_tokens != 32768: raise RuntimeError('Actual lengths do not match requested lengths')
