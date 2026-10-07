#!/usr/bin/env python3
"""Recompile only the matched CK HIP translation unit; preserve baseline objects.

External dispatch trait stays unchanged intentionally: these private modules are
valid only for the benchmark's QSA D=256, length-one, no-mask path. They are not
production replacements for the general AITER library.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

MODULE = 'mha_varlen_fwd_bf16_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale'
STEM = 'fmha_fwd_d256_bf16_group_b128x64x32x256x32x256_r4x1x1_r4x1x1_w32x32x16_w32x32x16_qr_vr_pssk_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_ntrload_nsink_gfx950'
VARIANTS = {
    'baseline': {},
    'post_sched': {'replace_flags': ['-enable-post-misched=0', '-enable-post-misched=1']},
    'no_preload': {'replace_flags': ['--amdgpu-kernarg-preload-count=32', '--amdgpu-kernarg-preload-count=0']},
    'fast_math': {'extra_flags': ['-ffast-math']},
    'occupancy2': {'occupancy': 2},
    'n128': {'n': 128},
    'm64': {'m': 64, 'warp': 16},
    'm32_w1': {'m': 32, 'waves': 1},
    'm16_w1': {'m': 16, 'waves': 1, 'warp': 16},
    'n128_sched': {'n': 128, 'replace_flags': ['-enable-post-misched=0', '-enable-post-misched=1']},
    'm32_n128': {'m': 32, 'n': 128, 'waves': 1},
    'm16_n128': {'m': 16, 'n': 128, 'waves': 1, 'warp': 16},
}

def build(name: str, root: Path, baseline: Path) -> Path:
    cfg = VARIANTS[name]
    dest = root / name / 'jit'
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(baseline / 'module_aiter_core.so', dest / 'module_aiter_core.so')
    if not cfg:
        shutil.copy2(baseline / (MODULE + '.so'), dest / (MODULE + '.so'))
        return dest
    build_dir = baseline / 'build' / MODULE / 'build'
    ninja = (build_dir / 'build.ninja').read_text()
    variables = dict(line.split(' = ', 1) for line in ninja.splitlines() if ' = ' in line and not line.startswith(' '))
    original = (baseline / 'build' / MODULE / 'blob' / (STEM + '.cpp')).read_text()
    modified = original.replace('sequence<128, 64, 32, 256, 32, 256>',
                                f"sequence<{cfg.get('m', 128)}, {cfg.get('n', 64)}, 32, 256, 32, 256>")
    if 'waves' in cfg:
        modified = modified.replace('sequence<4, 1, 1>', f"sequence<{cfg['waves']}, 1, 1>")
    if 'warp' in cfg:
        modified = modified.replace('sequence<32, 32, 16>', f"sequence<{cfg['warp']}, {cfg['warp']}, 16>")
    if 'occupancy' in cfg:
        modified = modified.replace('                                            -1,', f"                                            {cfg['occupancy']},")
    source = root / name / 'kernel.cpp'
    source.write_text(modified)
    (root / name / 'kernel.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True), fromfile=STEM+'.cpp', tofile='kernel.cpp')))
    flags = shlex.split(variables['cuda_cflags'])
    if 'replace_flags' in cfg:
        old,new = cfg['replace_flags']
        assert old in flags, old
        flags[flags.index(old)] = new
    flags += cfg.get('extra_flags', [])
    obj = root / name / 'kernel.o'
    compile_cmd = [variables['nvcc'], *flags, '-c', str(source), '-o', str(obj)]
    link_line = next(line for line in ninja.splitlines() if line.startswith('build '+MODULE+'.so: link '))
    objects = shlex.split(link_line.split(': link ', 1)[1])
    assert STEM+'.cuda.o' in objects
    inputs = [str(obj) if item == STEM+'.cuda.o' else str(build_dir / item) for item in objects]
    link_cmd = [variables['cxx'], *inputs, *shlex.split(variables['ldflags']), '-o', str(dest / (MODULE+'.so'))]
    (root / name / 'build.json').write_text(json.dumps(dict(config=cfg, compile=compile_cmd, link=link_cmd,
        baseline_sha256=hashlib.sha256((baseline/(MODULE+'.so')).read_bytes()).hexdigest()), indent=2))
    with (root / name / 'build.log').open('w') as log:
        subprocess.run(compile_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(link_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return dest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--output', type=Path, default=Path('qsa_hip_tuning'))
    parser.add_argument('--baseline', type=Path, default=Path('.qsa_aiter_jit'))
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--run-label', default='results')
    parser.add_argument('--reuse', action='store_true')
    args=parser.parse_args()
    root=args.output.resolve(); baseline=args.baseline.resolve()
    for name in args.variants.split(','):
        if name not in VARIANTS: parser.error('Unknown variant '+name)
        print('START '+name, flush=True)
        try:
            dest=root/name/'jit' if args.reuse else build(name,root,baseline)
            command=[sys.executable, str(Path(__file__).with_name('bench_qsa_ck.py').resolve()),
                     '--jit-dir', str(dest), '--output', str(root/name/args.run_label),
                     '--repeats',str(args.repeats), '--profile']
            with (root/name/(args.run_label+'.log')).open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            rows=[json.loads(line) for line in (root/name/args.run_label/'results.jsonl').read_text().splitlines()]
            print('PASS '+name+' '+json.dumps({f"{r['mode']}_c{r['conc']}":round(r['median_us'],2) for r in rows if r['context']==32679}),flush=True)
        except subprocess.CalledProcessError as exc:
            print('FAIL '+name+': '+str(exc),flush=True)
            (root/name/'failure.txt').write_text(str(exc))

if __name__=='__main__': main()
