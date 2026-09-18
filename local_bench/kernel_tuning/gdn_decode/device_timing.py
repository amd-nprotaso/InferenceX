"""Device-side timing for the GDN decode sweep, via rocprofv3 --kernel-trace.

Why this exists
---------------
``bench_decode.py`` times with CUDA events around Python calls. For the prefill
chunk_h kernel (~800 us) that is fine. For this kernel it is not: device-side it
is ~10 us, while one Python-side invocation (``o = q.new_empty(...)``, argument
marshalling, Triton launch) costs ~33 us of CPU. Bursting does not help --- the
CPU cannot feed the GPU fast enough, so the GPU idles and every config measures
the same ~33 us regardless of BV. That is a measurement of the launch path, not
of the kernel, and it will hide any real geometry effect.

``rocprofv3 --kernel-trace`` reports the dispatch's begin/end timestamps on the
device, excluding launch overhead --- the same reason ``chunk_gated_delta_rule``'s
README uses it for its resource-footprint table.

How it works
------------
Parent mode builds the config list, then re-execs itself as a worker under
rocprofv3. The worker runs each config for ``warmup + launches`` dispatches in a
deterministic order and writes a manifest. The parent then reads
``*_kernel_trace.csv``, keeps only the recurrence kernel's rows, sorts them by
start timestamp, and slices them per the manifest, discarding each config's
warmup dispatches.

Usage
-----
    python3 device_timing.py                                  # production shape
    python3 device_timing.py --shape qwen35-397b-tp4-c32
    python3 device_timing.py --bv 32 16 8 --num-warps 1 2 --launches 200
"""

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL_SUBSTR = "fused_sigmoid_gating_delta_rule_update_kernel"
HERE = Path(__file__).resolve().parent


def worker(a) -> None:
    import torch
    from bench_decode import build_inputs, call, load_tunable_module
    from decode_shapes import SHAPES

    device = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(device)
    shape = SHAPES[a.shape]
    mod = load_tunable_module()
    inp = build_inputs(shape, device)
    state = inp["state"].clone()
    cu = inp["cu"]

    manifest = []
    for spec in a.configs.split(";"):
        bv, warps, stages = (int(x) for x in spec.split(","))
        os.environ["GDN_DECODE_BV"] = str(bv)
        os.environ["GDN_DECODE_WARPS"] = str(warps)
        os.environ["GDN_DECODE_STAGES"] = str(stages)
        # Compile + warm outside the counted region is impossible here (every
        # dispatch is traced), so record the warmup count and drop it later.
        for _ in range(a.warmup + a.launches):
            call(mod, inp, state, cu)
        torch.cuda.synchronize()
        manifest.append(dict(bv=bv, num_warps=warps, num_stages=stages,
                             warmup=a.warmup, launches=a.launches))
    Path(a.manifest).write_text(json.dumps(manifest))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default="qwen35-397b-tp4-c4")
    p.add_argument("--bv", type=int, nargs="+", default=[64, 32, 16, 8, 4])
    p.add_argument("--num-warps", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--num-stages", type=int, nargs="+", default=[3])
    p.add_argument("--launches", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output", type=Path)
    # worker-mode plumbing
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--configs", default="", help=argparse.SUPPRESS)
    p.add_argument("--manifest", default="", help=argparse.SUPPRESS)
    a = p.parse_args()

    if a._worker:
        worker(a)
        return

    from decode_shapes import SHAPES
    shape = SHAPES[a.shape]
    configs = [(bv, w, s) for bv in a.bv if shape.V % bv == 0
               for w in a.num_warps for s in a.num_stages]
    tmp = tempfile.TemporaryDirectory(prefix="gdn-decode-att-")
    manifest_path = Path(tmp.name) / "manifest.json"
    outdir = Path(tmp.name) / "trace"

    cmd = ["rocprofv3", "--kernel-trace", "--output-format", "csv", "-d", str(outdir), "--",
           sys.executable, str(HERE / "device_timing.py"), "--_worker",
           "--shape", a.shape, "--device", str(a.device),
           "--launches", str(a.launches), "--warmup", str(a.warmup),
           "--configs", ";".join(f"{b},{w},{s}" for b, w, s in configs),
           "--manifest", str(manifest_path)]
    print(f"capturing {len(configs)} configs x {a.warmup}+{a.launches} dispatches "
          f"under rocprofv3 --kernel-trace ...")
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if not manifest_path.exists():
        sys.exit(f"worker failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    manifest = json.loads(manifest_path.read_text())

    files = glob.glob(str(outdir / "**" / "*kernel_trace.csv"), recursive=True)
    if not files:
        sys.exit("rocprofv3 produced no kernel trace")
    rows = [r_ for r_ in csv.DictReader(open(files[0]))
            if KERNEL_SUBSTR in r_["Kernel_Name"]]
    rows.sort(key=lambda r_: int(r_["Start_Timestamp"]))
    expected = sum(m["warmup"] + m["launches"] for m in manifest)
    if len(rows) != expected:
        sys.exit(f"dispatch count mismatch: trace has {len(rows)}, expected {expected}. "
                 "Cannot attribute dispatches to configs; aborting rather than guessing.")

    print(f"\n=== {shape.name}  batch={shape.batch} draft={shape.draft_tokens} T={shape.T} "
          f"HV={shape.HV} K={shape.K} V={shape.V}   (device-side, rocprofv3)")
    print(f"{'BV':>4} {'warps':>6} {'stages':>7} {'WGs':>6} {'wave slot%':>11} "
          f"{'us':>9} {'speedup':>8} {'GB/s':>8}")
    i, stock_us, results = 0, None, []
    state_bytes = shape.state_bytes()
    for m in manifest:
        chunk = rows[i + m["warmup"]: i + m["warmup"] + m["launches"]]
        i += m["warmup"] + m["launches"]
        us = statistics.median(
            (int(r_["End_Timestamp"]) - int(r_["Start_Timestamp"])) / 1000.0 for r_ in chunk)
        r0 = chunk[0]
        if (m["bv"], m["num_warps"], m["num_stages"]) == (32, 1, 3):
            stock_us = us
        gbs = 2 * state_bytes / (us * 1e-6) / 1e9   # read + write of the SSM state
        results.append(dict(**m, us=us, gbs=gbs,
                            vgpr=r0["VGPR_Count"], lds=r0["LDS_Block_Size"],
                            grid=r0["Grid_Size_X"], block=r0["Workgroup_Size_X"]))
    for r_ in results:
        sp = f"{stock_us / r_['us']:8.3f}" if stock_us else f"{'--':>8}"
        tag = "  <- STOCK" if (r_["bv"], r_["num_warps"], r_["num_stages"]) == (32, 1, 3) else ""
        print(f"{r_['bv']:>4} {r_['num_warps']:>6} {r_['num_stages']:>7} "
              f"{shape.workgroups(r_['bv']):>6} "
              f"{shape.wave_slot_pct(r_['bv'], r_['num_warps']):>10.1f}% "
              f"{r_['us']:9.2f} {sp} {r_['gbs']:8.1f}{tag}")
    print(f"\nGB/s counts the SSM state read+write only ({2*state_bytes/2**20:.1f} MiB/call); "
          f"MI355X HBM peak is ~8000 GB/s.")
    if a.output:
        a.output.write_text(json.dumps(dict(shape=shape.name, results=results), indent=2))
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
