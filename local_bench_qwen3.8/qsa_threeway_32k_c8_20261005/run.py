import os,sys,json,time,subprocess,signal,urllib.request,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
base=root.parent
def request(path,payload=None):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request('http://127.0.0.1:18888/'+path,data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as r:return json.load(r)
meta={}
for name,path in [('sglang','/sgl-workspace/sglang'),('aiter','/sgl-workspace/aiter')]:
 meta[name]={'repo':path,'head':subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip(),'diff':subprocess.check_output(['git','-C',path,'diff'],text=True)}
meta['hc_mix_sha256']=hashlib.sha256(Path('/sgl-workspace/aiter/aiter/ops/flydsl/hc_mix.py').read_bytes()).hexdigest()
(root/'provenance.json').write_text(json.dumps(meta,indent=2))
for arm in ['ck','triton','flydsl']:
 dest=root/arm; dest.mkdir(exist_ok=True)
 env=os.environ.copy(); env.update(HF_HOME='/data/hf_cache',HF_HUB_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',HIP_VISIBLE_DEVICES='0',QSA_E2E_OUTPUT=str(dest),PYTHONUNBUFFERED='1')
 env['QSA_EXPERIMENT_ARM'] = arm
 env['QSA_AB_FLAG'] = '1' if arm == 'flydsl' else '0'
 env['SGLANG_AITER_QSA_PA_DECODE'] = env['QSA_AB_FLAG']
 (dest/'flag.json').write_text(json.dumps({'SGLANG_AITER_QSA_PA_DECODE':env.get('SGLANG_AITER_QSA_PA_DECODE','unset')}))
 print('START',arm,flush=True)
 with (dest/'server.log').open('w') as log:
  proc=subprocess.Popen(['bash',str(root/'server-start.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (dest/'server_pid').write_text(str(proc.pid))
  try:
   deadline=time.monotonic()+1800
   while time.monotonic()<deadline:
    if proc.poll() is not None:raise RuntimeError(f'{arm} server exited {proc.returncode}')
    try:
     with urllib.request.urlopen('http://127.0.0.1:18888/health',timeout=3) as r:
      if r.status==200:break
    except Exception:pass
    time.sleep(10)
   else:raise TimeoutError('Server readiness')
   print('READY',arm,flush=True)
   (dest/'server_info.json').write_text(json.dumps(request('server_info'),indent=2))
   (dest/'warmup.json').write_text(json.dumps(request('generate',{'text':'Explain database replication.','sampling_params':{'max_new_tokens':32,'temperature':0,'ignore_eos':True}}),indent=2))
   with urllib.request.urlopen('http://127.0.0.1:18888/flush_cache',timeout=30) as r: print('FLUSH',r.status,flush=True)
   with (dest/'client.log').open('w') as client:
    subprocess.run([sys.executable,str(root/'client_exact.py')],env=env,stdout=client,stderr=subprocess.STDOUT,check=True)
   print('DONE',arm,flush=True)
  finally:
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=40)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  time.sleep(10)
print('ALL THREE TRIALS COMPLETE', flush=True)
