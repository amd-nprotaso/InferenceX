"""Device-side FlyDSL-vs-Triton comparison for the GDN decode recurrence.

Both kernels run in one process under ``rocprofv3 --kernel-trace``; they have
different kernel names in the trace (``fused_sigmoid_gating_delta_rule_update_kernel``
vs FlyDSL's generated ``kernel_N``), so attribution needs no dispatch counting.

Device-side timing is mandatory here, not a refinement: the kernel is ~10 us
while one Python-side invocation costs ~33 us of CPU, so CUDA events around the
call measure the launch path and report every variant as identical. See
``README.md``.

    python3 compare_decode.py
    python3 compare_decode.py --shapes qwen35-397b-tp4-c4 qwen35-397b-tp4-c32
"""

import argparse
import csv
import glob
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRITON_KERNEL = "fused_sigmoid_gating_delta_rule_update_kernel"


def worker(a) -> None:
    import torch
    from bench_decode import build_inputs, call, load_tunable_module
    from decode_shapes import SHAPES
    import gdn_decode_flydsl as fly

    device = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(device)
    triton_mod = load_tunable_module()

    for name in a.shapes.split(","):
        shape = SHAPES[name]
        inp = build_inputs(shape, device)
        cu, idx = inp["cu"], inp["idx"]
        common = dict(
            A_log=inp["A_log"], a=inp["a"], dt_bias=inp["dt_bias"],
            softplus_beta=1.0, softplus_threshold=20.0,
            q=inp["q"], k=inp["k"], v=inp["v"], b=inp["b"],
            initial_state_indices=idx, cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
        )
        s = inp["state"].clone()
        for _ in range(a.warmup + a.launches):
            triton_mod.fused_sigmoid_gating_delta_rule_update(initial_state_source=s, **common)
        torch.cuda.synchronize()
        s = inp["state"].clone()
        for _ in range(a.warmup + a.launches):
            fly.fused_sigmoid_gating_delta_rule_update(initial_state_source=s, **common)
        torch.cuda.synchronize()
        del inp, s
        torch.cuda.empty_cache()
    Path(a.manifest).write_text(json.dumps({"shapes": a.shapes.split(",")}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", nargs="+",
                   default=["qwen35-397b-tp4-c4", "qwen35-397b-tp4-c8", "qwen35-397b-tp4-c32",
                            "qwen35-397b-tp4-c4-nospec"])
    p.add_argument("--launches", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output", type=Path)
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--manifest", default="", help=argparse.SUPPRESS)
    a = p.parse_args()

    if a._worker:
        a.shapes = a.shapes if isinstance(a.shapes, str) else ",".join(a.shapes)
        worker(a)
        return

    from decode_shapes import SHAPES
    import gdn_decode_flydsl as fly

    tmp = tempfile.TemporaryDirectory(prefix="gdn-decode-cmp-")
    manifest = Path(tmp.name) / "m.json"
    outdir = Path(tmp.name) / "trace"
    cmd = ["rocprofv3", "--kernel-trace", "--output-format", "csv", "-d", str(outdir), "--",
           sys.executable, str(HERE / "compare_decode.py"), "--_worker",
           "--shapes", ",".join(a.shapes), "--device", str(a.device),
           "--launches", str(a.launches), "--warmup", str(a.warmup),
           "--manifest", str(manifest)]
    print(f"capturing {len(a.shapes)} shapes x 2 backends under rocprofv3 ...")
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if not manifest.exists():
        sys.exit(f"worker failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")

    files = glob.glob(str(outdir / "**" / "*kernel_trace.csv"), recursive=True)
    rows = [x for x in csv.DictReader(open(files[0]))]
    rows.sort(key=lambda x: int(x["Start_Timestamp"]))

    def dur(x):
        return (int(x["End_Timestamp"]) - int(x["Start_Timestamp"])) / 1000.0

    tri = [x for x in rows if TRITON_KERNEL in x["Kernel_Name"]]
    fl = [x for x in rows if x["Kernel_Name"].startswith("kernel_")]
    per = a.warmup + a.launches
    if len(tri) != per * len(a.shapes) or len(fl) != per * len(a.shapes):
        sys.exit(f"dispatch mismatch: triton={len(tri)} flydsl={len(fl)} "
                 f"expected {per * len(a.shapes)} each")

    print(f"\n{'shape':28} {'Triton us':>10} {'FlyDSL us':>10} {'speedup':>9} "
          f"{'Tri WGs':>8} {'Fly WGs':>8} {'Fly VGPR':>9} {'GB/s':>8}")
    results = []
    for i, name in enumerate(a.shapes):
        shape = SHAPES[name]
        t = tri[i * per + a.warmup:(i + 1) * per]
        f = fl[i * per + a.warmup:(i + 1) * per]
        tu, fu = statistics.median(map(dur, t)), statistics.median(map(dur, f))
        gbs = 2 * shape.state_bytes() / (fu * 1e-6) / 1e9
        fly_wgs = shape.HV * (shape.V // fly.VTILE) * shape.batch
        print(f"{name:28} {tu:10.2f} {fu:10.2f} {tu / fu:9.3f} "
              f"{shape.workgroups(32):8d} {fly_wgs:8d} {f[0]['VGPR_Count']:>9} {gbs:8.1f}")
        results.append(dict(shape=name, triton_us=tu, flydsl_us=fu, speedup=tu / fu,
                            flydsl_vgpr=f[0]["VGPR_Count"], flydsl_lds=f[0]["LDS_Block_Size"],
                            gbs=gbs))
    print(f"\nFlyDSL geometry: KSPLIT={fly.KSPLIT} VTILE={fly.VTILE} KLOCAL={fly.KLOCAL}, "
          f"block={fly.BLOCK}. GB/s = SSM state read+write only; HBM peak ~8000 GB/s.")
    if a.output:
        a.output.write_text(json.dumps(results, indent=2))
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
