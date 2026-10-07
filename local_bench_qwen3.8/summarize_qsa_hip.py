#!/usr/bin/env python3
"""Compare matched QSA cases; reject missing cases and save speedup tables."""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_cases(path: Path) -> dict[tuple[str, int, int], dict]:
    return {(r['mode'],r['conc'],r['context']):r for r in map(json.loads,path.read_text().splitlines())}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('qsa_hip_tuning'))
    parser.add_argument('--label',default='results')
    args=parser.parse_args()
    baseline=read_cases(args.root/'baseline'/args.label/'results.jsonl')
    output=[]
    for directory in sorted(args.root.iterdir()):
        path=directory/args.label/'results.jsonl'
        if not path.exists(): continue
        cases=read_cases(path)
        if cases.keys()!=baseline.keys():
            print(f'INCOMPLETE {directory.name}: {len(cases)}/{len(baseline)} cases');continue
        ratios=[]
        for mode in ['decode','verify']:
            for conc in [1,4,8,16]:
                keys=[key for key in baseline if key[:2]==(mode,conc)]
                speedups=[baseline[key]['median_us']/cases[key]['median_us'] for key in keys]
                ratios.extend(speedups)
                output.append(dict(variant=directory.name,mode=mode,concurrency=conc,
                    baseline_us=round(statistics.median(baseline[k]['median_us'] for k in keys),3),
                    variant_us=round(statistics.median(cases[k]['median_us'] for k in keys),3),
                    geomean_speedup=round(math.exp(statistics.mean(map(math.log,speedups))),4),
                    worst_case_speedup=round(min(speedups),4),
                    max_abs_error=max(cases[k]['max_abs_error'] for k in keys)))
        print(directory.name, 'geomean',round(math.exp(statistics.mean(map(math.log,ratios))),4),
              'worst',round(min(ratios),4))
    dest=args.root/('comparison_'+args.label+'.csv')
    with dest.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)

if __name__=='__main__': main()
