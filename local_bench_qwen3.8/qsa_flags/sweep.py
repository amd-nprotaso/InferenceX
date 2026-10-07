#!/usr/bin/env python3
"""Build gfx950 larger-KV CK candidates and validate the C=1 serving shapes."""
from __future__ import annotations
import json,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from tune_qsa_hip import VARIANTS,build
CANDIDATES={'baseline':{},
 'post_sched':{'replace_flags':['-enable-post-misched=0','-enable-post-misched=1']},
 'occupancy2':{'occupancy':2}, 'occupancy3':{'occupancy':3},
 'preload0':{'replace_flags':['--amdgpu-kernarg-preload-count=32','--amdgpu-kernarg-preload-count=0']},
 'preload16':{'replace_flags':['--amdgpu-kernarg-preload-count=32','--amdgpu-kernarg-preload-count=16']}}
VARIANTS.update(CANDIDATES)
def main() -> None:
    results=[]
    for name in CANDIDATES:
        print('START '+name,flush=True)
        folder=ROOT/'micro'/name
        try:
            cache=build(name,ROOT/'micro',ROOT.parent/'qsa_e2e/stack_baseline_jit')
            cmd=[sys.executable,str(ROOT/'bench.py'),'--aiter-path','/sgl-workspace/aiter',
                 '--jit-dir',str(cache),'--device','1','--isl','32768','--osl','32768',
                 '--conc','1','--repeats','20','--profile','--output',str(folder/'results')]
            with (folder/'run.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
            rows=[json.loads(x) for x in (folder/'results/results.jsonl').read_text().splitlines()]
            assert len(rows)==12
            lat={mode:statistics.median(r['median_us'] for r in rows if r['mode']==mode) for mode in ['decode','verify']}
            result={'name':name,'status':'pass','config':CANDIDATES[name],**lat,'max_abs_error':max(r['max_abs_error'] for r in rows),'output_hashes':[r['output_sha256'] for r in rows]}
            result['selection_score_us']=12*lat['verify']+3*lat['decode']
            print('PASS '+json.dumps(result),flush=True)
        except (subprocess.CalledProcessError,AssertionError) as exc:
            result={'name':name,'status':'failed','error':str(exc)}
            print('FAIL '+name+'; see build/run logs',flush=True)
        results.append(result)
        (ROOT/'micro_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    base=next(r for r in results if r['name']=='baseline' and r['status']=='pass')
    for r in results:
        if r['status']=='pass':r['bitwise_equal_baseline']=r['output_hashes']==base['output_hashes']
    (ROOT/'micro_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    eligible=[r for r in results if r['status']=='pass' and r['name']!='baseline']
    if not eligible:raise RuntimeError('No flag candidate passed')
    winner=min(eligible,key=lambda r:r['selection_score_us'])
    (ROOT/'selected.json').write_text(json.dumps(winner,indent=2)+'\n')
    print('SELECTED '+winner['name'],flush=True)
if __name__=='__main__':main()
