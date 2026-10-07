#!/usr/bin/env python3
"""Switch only the benchmark-owned server after a successful baseline request."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parent
MODULE='mha_varlen_fwd_bf16_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale.so'

def stop(process: psutil.Process) -> None:
    try:
        children=process.children(recursive=True)
        process.terminate()
        _,alive=psutil.wait_procs([process,*children],timeout=30)
        for child in alive:
            try:child.terminate()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(alive,timeout=15)
        for child in alive:
            try:child.kill()
            except psutil.NoSuchProcess:pass
    except psutil.NoSuchProcess:pass

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-pid',type=int,required=True);args=parser.parse_args()
    baseline=psutil.Process(args.baseline_pid)
    if 'sglang.launch_server' not in baseline.cmdline():raise RuntimeError('Unexpected baseline process')
    deadline=time.monotonic()+3600
    while not (ROOT/'baseline/client-exact-result.json').exists():
        if not baseline.is_running() or baseline.status()==psutil.STATUS_ZOMBIE:raise RuntimeError('Baseline server exited')
        if time.monotonic()>deadline:raise RuntimeError('Baseline request timeout')
        time.sleep(5)
    result=json.loads((ROOT/'baseline/client-exact-result.json').read_text())
    if result['actual_input_tokens']!=32768 or result['actual_output_tokens']!=32768:raise RuntimeError('Baseline lengths mismatch')
    print('Baseline complete: '+json.dumps({k:result[k] for k in ['e2e_s','tpot_ms','output_throughput_tok_s']}),flush=True)
    stop(baseline)
    print('Baseline server stopped',flush=True)
    overlay=ROOT/'tuned_jit'
    # Include any baseline modules that were compiled during the measured request.
    for source in Path('/sgl-workspace/aiter/aiter/jit').glob('*.so'):
        dest=overlay/source.name
        if not dest.exists():dest.symlink_to(source)
    env={**os.environ,'HF_HOME':'/data/hf_cache/','AITER_JIT_DIR':str(overlay)}
    with (ROOT/'tuned/server.log').open('w') as log:
        server=subprocess.Popen(['bash','/var/home/qwen3_8_profiling/server-start.sh'],env=env,stdout=log,stderr=subprocess.STDOUT)
    (ROOT/'tuned/server_pid').write_text(str(server.pid))
    print(f'Tuned server started pid={server.pid}',flush=True)
    try:
        with (ROOT/'tuned/driver.log').open('w') as log:
            client=subprocess.Popen([sys.executable,str(ROOT/'run_client.py'),'tuned'],stdout=log,stderr=subprocess.STDOUT)
        while client.poll() is None:
            if server.poll() is not None:
                client.terminate();raise RuntimeError(f'Tuned server exited: {server.returncode}')
            time.sleep(5)
        if client.returncode:raise RuntimeError(f'Tuned client exited: {client.returncode}')
        mappings=[]
        for child in psutil.Process(server.pid).children(recursive=True):
            try:
                names=sorted({line.split()[-1] for line in Path(f'/proc/{child.pid}/maps').read_text().splitlines() if MODULE in line})
                if names:mappings.append({'pid':child.pid,'name':child.name(),'libraries':names})
            except (FileNotFoundError,psutil.NoSuchProcess):pass
        (ROOT/'tuned/loaded_qsa.json').write_text(json.dumps(mappings,indent=2))
        if not any('/qsa_e2e/compiled/n128/jit/' in name for x in mappings for name in x['libraries']):raise RuntimeError('Tuned QSA library not mapped')
        result=json.loads((ROOT/'tuned/client-exact-result.json').read_text())
        print('Tuned complete: '+json.dumps({k:result[k] for k in ['actual_input_tokens','actual_output_tokens','e2e_s','tpot_ms','output_throughput_tok_s']}),flush=True)
    finally:
        stop(psutil.Process(server.pid))
        print('Tuned server stopped',flush=True)
if __name__=='__main__':main()
