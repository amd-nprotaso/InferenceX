#!/usr/bin/env python3
"""Wait for local server, warm the exact prefill shape, flush, run unprofiled client."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,urllib.request,urllib.error
from pathlib import Path
BASE='http://127.0.0.1:30000'
def api(path: str, body: dict | None = None, timeout: int = 1800) -> bytes:
    req=urllib.request.Request(BASE+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:return response.read()
def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('arm');args=p.parse_args()
    root=Path(__file__).resolve().parent;out=root/args.arm;out.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+1800
    while True:
        try:
            log=(out/'server.log').read_text(errors='replace')
            if 'The server is fired up and ready to roll!' not in log:
                if time.monotonic()>deadline: raise RuntimeError('Server readiness timeout')
                time.sleep(5)
                continue
            api('/get_server_info',timeout=15)
            break
        except (urllib.error.URLError,TimeoutError):
            if time.monotonic()>deadline:raise RuntimeError('Server readiness timeout')
            time.sleep(5)
    print('Server ready',flush=True)
    (out/'server_info.json').write_bytes(api('/get_server_info'))
    warm=json.loads((root/'request.json').read_text());warm['stream']=False;warm['sampling_params']['max_new_tokens']=32
    print('Warmup: 32768 input / 32 output',flush=True)
    (out/'warmup.json').write_bytes(api('/generate',warm))
    api('/flush_cache',{})
    print('Cache flushed; starting measured request without profiling',flush=True)
    env={**os.environ,'HF_HOME':'/data/hf_cache/','QSA_E2E_OUTPUT':str(out),'PYTHONUNBUFFERED':'1'}
    with (out/'client.log').open('w') as log:
        subprocess.run([sys.executable,str(root/'client_exact.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((out/'client-exact-result.json').read_text())
    print(json.dumps({k:result[k] for k in ['actual_input_tokens','actual_output_tokens','ttft_s','e2e_s','tpot_ms','decode_interactivity_tok_s','output_throughput_tok_s']}),flush=True)
if __name__=='__main__':main()
