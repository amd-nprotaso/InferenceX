#!/usr/bin/env python3
"""Compare HC combine split unset, enabled, disabled with the installed baseline QSA kernel."""
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
    for arm in ['unset','enabled','disabled']:
        if (ROOT/(arm+'_repeat')/'loaded_qsa.json').exists():
            print('SKIP completed '+arm,flush=True);continue
        out=ROOT/arm;out.mkdir(exist_ok=True)
        env={**os.environ,'HF_HOME':'/data/hf_cache/','SPT_NOENV':'1'};env.pop('AITER_JIT_DIR',None)
        env.pop('SGLANG_HC_COMBINE_SPLIT',None)
        if arm != 'unset':env['SGLANG_HC_COMBINE_SPLIT']='1' if arm=='enabled' else '0'
        probe = subprocess.check_output([sys.executable,'-c',
            "import json,os,sglang;from sglang.srt.environ import envs;print(json.dumps({'sglang':sglang.__file__,'raw':os.environ.get('SGLANG_HC_COMBINE_SPLIT'),'effective':envs.SGLANG_HC_COMBINE_SPLIT.get()}))"],env=env,text=True)
        (out/'flag_probe.json').write_text(probe)
        with (out/'server.log').open('w') as log:
            server=subprocess.Popen(['bash',str(ROOT/'server-start.sh')],env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/'server_pid').write_text(str(server.pid));print(f'START {arm} pid={server.pid}',flush=True)
        try:
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
                loaded=mappings(server.pid)
                assert loaded, 'No QSA library mapped'
                for proc in loaded:
                    assert proc['libraries']==['/sgl-workspace/aiter/aiter/jit/'+MODULE], proc
                    procenv=psutil.Process(proc['pid']).environ()
                    proc['hc_flag_os_environment']=procenv.get('SGLANG_HC_COMBINE_SPLIT')
                    assert procenv.get('SGLANG_HC_COMBINE_SPLIT')==env.get('SGLANG_HC_COMBINE_SPLIT'), proc
                (dest/'loaded_qsa.json').write_text(json.dumps(loaded,indent=2))
                (dest/'flag_probe.json').write_text(probe)
                print('DONE '+label+' '+json.dumps({k:result[k] for k in ['e2e_s','tpot_ms','output_throughput_tok_s']}),flush=True)
        finally:
            try:stop(psutil.Process(server.pid))
            except psutil.NoSuchProcess:pass
            print('STOPPED '+arm,flush=True)
if __name__=='__main__':main()
