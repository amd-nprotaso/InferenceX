#!/usr/bin/env python3
"""Run seeded baseline and selected larger-KV variant, two unprofiled requests each."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'qsa_e2e'))
from run_tuned import stop
MODULE='mha_varlen_fwd_bf16_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale.so'

def mappings(pid: int) -> list[dict]:
    result=[]
    for child in psutil.Process(pid).children(recursive=True):
        try:
            names=sorted({x.split()[-1] for x in Path(f'/proc/{child.pid}/maps').read_text().splitlines() if MODULE in x})
            if names:result.append({'pid':child.pid,'libraries':names})
        except (FileNotFoundError,psutil.NoSuchProcess):pass
    return result

def main() -> None:
    for arm in ['baseline','tuned']:
        out=ROOT/arm;out.mkdir(exist_ok=True)
        env={**os.environ,'HF_HOME':'/data/hf_cache/'};env.pop('AITER_JIT_DIR',None)
        if arm=='tuned':
            choice=json.loads((ROOT/'selected.json').read_text())['name']
            overlay=ROOT/'tuned_jit';overlay.mkdir(exist_ok=True)
            for source in Path('/sgl-workspace/aiter/aiter/jit').glob('*.so'):
                target=overlay/source.name
                if not target.exists():target.symlink_to(source if source.name!=MODULE else ROOT/'micro'/choice/'jit'/MODULE)
            env['AITER_JIT_DIR']=str(overlay)
        with (out/'server.log').open('w') as log:
            server=subprocess.Popen(['bash',str(ROOT/'server-start.sh')],env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/'server_pid').write_text(str(server.pid));print(f'START {arm} pid={server.pid}',flush=True)
        try:
            # Permit baseline loading during microbench preparation, but never time them together.
            deadline=time.monotonic()+1800
            while not (ROOT/'selected.json').exists():
                if server.poll() is not None:raise RuntimeError('Baseline server exited during load')
                if time.monotonic()>deadline:raise RuntimeError('Micro selection timeout')
                time.sleep(5)
            for trial in [1,2]:
                label=arm if trial==1 else arm+'_repeat'
                dest=ROOT/label;dest.mkdir(exist_ok=True)
                if trial>1:(dest/'server.log').symlink_to(out/'server.log')
                with (dest/'driver.log').open('w') as log:
                    client=subprocess.Popen([sys.executable,str(ROOT/'run_client.py'),label],stdout=log,stderr=subprocess.STDOUT)
                while client.poll() is None:
                    if server.poll() is not None:
                        client.terminate();raise RuntimeError(f'{arm} server exited')
                    time.sleep(5)
                if client.returncode:raise RuntimeError(f'{label} client failed')
                result=json.loads((dest/'client-exact-result.json').read_text())
                assert result['actual_input_tokens']==32768 and result['actual_output_tokens']==32768
                (dest/'loaded_qsa.json').write_text(json.dumps(mappings(server.pid),indent=2))
                print('DONE '+label+' '+json.dumps({k:result[k] for k in ['e2e_s','tpot_ms','output_throughput_tok_s']}),flush=True)
        finally:
            try:stop(psutil.Process(server.pid))
            except psutil.NoSuchProcess:pass
            print('STOPPED '+arm,flush=True)
if __name__=='__main__':main()
