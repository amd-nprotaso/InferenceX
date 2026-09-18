import json
from collections import Counter

PATHS = {
    'AMD': '/var/home/my_InferenceX/traces/amd/1789642053.9277868-TP-0.trace.json',
    'NVIDIA': '/var/home/my_InferenceX/traces/nvidia/1788441037.314373-TP-0.trace.json',
}

if __name__ == '__main__':
    for vendor, path in PATHS.items():
        d = json.load(open(path))
        events = d['traceEvents']
        print(vendor, {k:v for k,v in d.items() if k != 'traceEvents'})
        print(Counter(e.get('cat') for e in events))
        print(Counter(e.get('name') for e in events if e.get('cat') in ['user_annotation', 'gpu_user_annotation']).most_common(120))
        for cat in ['kernel', 'cuda_runtime']:
            print(cat, next((e for e in events if e.get('cat') == cat), None))
        print('layers', Counter(e.get('name') for e in events if any(s in e.get('name','').lower() for s in ['linear_attention', 'full_attention', 'decoderlayer'])).most_common(30))
        print('modules', Counter(e.get('name') for e in events if e.get('name','').startswith('nn.Module:')).most_common(20))
        print('forward functions', Counter(e.get('name') for e in events if 'qwen3_5' in e.get('name','')).most_common(30))
        print('steps', [e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('step[')][:3])
        print('gpu steps', [e for e in events if e.get('cat')=='gpu_user_annotation' and e.get('name','').startswith('step[')][:3])
        print('kernels', Counter(e['name'] for e in events if e.get('cat')=='kernel').most_common(80))
