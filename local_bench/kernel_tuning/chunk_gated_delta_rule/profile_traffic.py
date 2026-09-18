#!/usr/bin/env python3
"""Measure what chunk_gated_delta_rule_fwd_kernel_h_blockdim64 actually moves.

Why this exists
---------------
`sweep.py` reports two bandwidth figures and they disagree wildly:

    BV=16  0.950 ms   2.83 TB/s (full model)    424 GB/s (h + v_new only)
    BV=8   1.691 ms   2.86 TB/s (full model)    238 GB/s (h + v_new only)

The full model says BV=8 and BV=16 run at the *same* bandwidth and BV=8 is
simply moving 1.8x more bytes. That model assumes zero cache reuse: it charges
the kernel for a full re-read of `w` and `k` per V-block, because neither block
pointer carries an i_v term (chunk_delta_h.py:174-176, :271-273).

That assumption is the weak link. MI350X has 256 MB of Infinity Cache, and the
grid is (cdiv(V,BV), N*H) linearised with dim0 fastest -- so the grid.x programs
that share a head, and therefore share `w` and `k`, have adjacent workgroup IDs
and are co-resident. The cache may be absorbing most of the redundancy.

The distinction decides the whole optimisation:

    fetch_measured / fetch_modelled ~ 1.0
        The re-reads really do reach HBM. Confirmed bandwidth wall; the fix is
        to move fewer bytes (fewer V-blocks per w/k load, or a layout change).

    fetch_measured / fetch_modelled << 1.0, with a high TCC hit rate
        The cache eats the redundancy. The wall is L2/fabric, not HBM, and
        "cut redundant reads" is the wrong fix -- the bytes were never going to
        DRAM in the first place.

So: measure, don't assume.

Usage
-----
    ./profile_traffic.py                        # BV 8,16,32 at w=4 s=3
    ./profile_traffic.py --bv 16 --launches 5
    ./profile_traffic.py --csv traffic.csv

Counters are collected in a separate process per config, because counter
collection serialises dispatches and would contaminate any timing measured in
the same run. Nothing here reports time -- use sweep.py for that.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import tempfile

KERNEL_REGEX = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"

# FETCH_SIZE / WRITE_SIZE are in KB and are already device-wide sums over all
# TCC channels. TCC_HIT_sum / TCC_MISS_sum are request counts, not bytes -- the
# hit *rate* is the useful part, not their absolute magnitude.
COUNTERS = ["FETCH_SIZE", "WRITE_SIZE", "TCC_HIT_sum", "TCC_MISS_sum"]


# ---------------------------------------------------------------- worker side


def _worker(argv) -> int:
    """Run the kernel a fixed number of times under the profiler. No timing."""
    p = argparse.ArgumentParser()
    p.add_argument("--_worker", required=True)
    p.add_argument("--shape", default="")
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--pool-slots", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    import torch

    from bench import Config, _call, _kernel
    from shapes import SHAPES, build_inputs

    spec = json.loads(a._worker)
    cfg = Config(**spec)
    shape = SHAPES[a.shape]
    inp = build_inputs(shape, device="cuda", seed=a.seed, pool_slots=a.pool_slots)

    kern = _kernel()
    kern.configs = [cfg.to_triton()]
    kern.cache.clear()

    # Warm up outside the measured region as far as we can -- the profiler will
    # still capture these dispatches, so the parent divides by the true launch
    # count rather than trying to exclude them.
    inp.restore_state()
    _call(inp)
    torch.cuda.synchronize()

    for _ in range(a.launches):
        inp.restore_state()
        _call(inp)
        torch.cuda.synchronize()

    return 0


# ---------------------------------------------------------------- parent side


def _find_counter_csv(outdir: str) -> str:
    hits = glob.glob(os.path.join(outdir, "**", "*counter_collection.csv"), recursive=True)
    return hits[0] if hits else ""


def _parse_counters(path: str, kernel_regex: str) -> tuple[dict, int]:
    """Sum each counter per dispatch of the target kernel.

    rocprofv3 emits one row per (dispatch, counter). Rows for other kernels are
    dropped -- restore_state() runs a copy each iteration and would otherwise be
    folded into the totals.
    """
    per_dispatch: dict = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Kernel_Name", "") or row.get("Kernel_Name ", "")
            if kernel_regex not in name:
                continue
            disp = row.get("Dispatch_Id") or row.get("Dispatch_ID") or "0"
            ctr = (row.get("Counter_Name") or "").strip()
            try:
                val = float(row.get("Counter_Value") or 0.0)
            except ValueError:
                continue
            per_dispatch.setdefault(disp, {}).setdefault(ctr, 0.0)
            per_dispatch[disp][ctr] += val

    if not per_dispatch:
        return {}, 0

    # Median across dispatches, not mean: the first captured dispatch can carry
    # cold-cache effects that are not representative of steady state.
    agg: dict = {}
    for ctr in COUNTERS:
        vals = sorted(d.get(ctr, 0.0) for d in per_dispatch.values())
        if vals:
            agg[ctr] = vals[len(vals) // 2]
    return agg, len(per_dispatch)


def _profile_one(cfg_spec: dict, a) -> dict:
    outdir = tempfile.mkdtemp(prefix="gdn_pmc_")
    cmd = [
        "rocprofv3",
        "--pmc", *COUNTERS,
        "--kernel-include-regex", KERNEL_REGEX,
        "--output-format", "csv",
        "-d", outdir,
        "--",
        sys.executable, "-W", "ignore", os.path.abspath(__file__),
        "--_worker", json.dumps(cfg_spec),
        "--shape", a.shape,
        "--launches", str(a.launches),
        "--seed", str(a.seed),
    ]
    if a.pool_slots is not None:
        cmd += ["--pool-slots", str(a.pool_slots)]

    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(a.device))
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=a.timeout, env=env
    )

    path = _find_counter_csv(outdir)
    if not path:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"error": f"rc={proc.returncode}: " + (tail[-1][:160] if tail else "no counter csv")}

    agg, ndisp = _parse_counters(path, KERNEL_REGEX)
    if not agg:
        return {"error": f"no rows matched {KERNEL_REGEX} in {os.path.basename(path)}"}
    agg["_dispatches"] = ndisp
    agg["_csv"] = path
    return agg


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--_worker":
        return _worker(argv)

    from shapes import DEFAULT_SHAPE, SHAPES

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default=DEFAULT_SHAPE, choices=sorted(SHAPES))
    p.add_argument("--bv", default="8,16,32")
    p.add_argument("--warps", type=int, default=4)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-slots", type=int, default=None)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--csv", default="")
    a = p.parse_args(argv)

    shape = SHAPES[a.shape]
    bvs = [int(x) for x in a.bv.split(",") if x.strip()]

    print(f"shape       : {shape.name}  B={shape.B} T={shape.T} Hg={shape.Hg} "
          f"H={shape.H} K={shape.K} V={shape.V} NT={shape.NT}")
    print(f"counters    : {' '.join(COUNTERS)}")
    print(f"kernel      : {KERNEL_REGEX}")
    print(f"launches    : {a.launches} per config (+1 warmup), median across dispatches\n")

    rows = []
    for bv in bvs:
        spec = {"BV": bv, "num_warps": a.warps, "num_stages": a.stages,
                "waves_per_eu": None, "matrix_instr_nonkdim": None}
        label = f"BV={bv} w={a.warps} s={a.stages}"
        print(f"  profiling {label} ...", flush=True)
        got = _profile_one(spec, a)
        if "error" in got:
            print(f"    FAILED: {got['error']}")
            rows.append({"bv": bv, "error": got["error"]})
            continue

        # FETCH_SIZE / WRITE_SIZE are KB.
        fetch = got.get("FETCH_SIZE", 0.0) * 1024
        write = got.get("WRITE_SIZE", 0.0) * 1024
        hit = got.get("TCC_HIT_sum", 0.0)
        miss = got.get("TCC_MISS_sum", 0.0)
        hit_rate = hit / (hit + miss) if (hit + miss) else float("nan")

        # Modelled: what the kernel would move with zero cache reuse.
        m_read = (shape.w_read_bytes(bv) + shape.k_read_bytes(bv)
                  + shape.u_read_bytes())
        m_write = shape.h_bytes() + shape.v_new_bytes()
        rows.append({
            "bv": bv, "grid_x": shape.grid(bv)[0],
            "fetch": fetch, "write": write,
            "m_read": m_read, "m_write": m_write,
            "hit_rate": hit_rate,
            "ratio": fetch / m_read if m_read else float("nan"),
            "dispatches": got["_dispatches"],
        })

    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("\nEvery config failed -- check rocprofv3 permissions "
              "(counter collection often needs elevated perf access).")
        return 1

    print(f"\n{'='*100}")
    print(f"{'config':<14}{'gx':>4}{'fetch MB':>11}{'model MB':>10}{'ratio':>8}"
          f"{'write MB':>11}{'model MB':>10}{'L2 hit':>9}")
    print("-" * 100)
    for r in ok:
        print(f"BV={r['bv']:<11}{r['grid_x']:>4}{r['fetch']/1e6:>11.0f}"
              f"{r['m_read']/1e6:>10.0f}{r['ratio']:>8.2f}"
              f"{r['write']/1e6:>11.0f}{r['m_write']/1e6:>10.0f}"
              f"{r['hit_rate']*100:>8.1f}%")
    print("-" * 100)

    print("\nfetch ratio = measured HBM reads / modelled reads assuming zero reuse.")
    print("  ~1.0  -> the w/k re-reads really do hit DRAM: bandwidth wall is real,")
    print("           and the fix is to move fewer bytes.")
    print("  <<1.0 -> cache absorbs the redundancy; the wall is L2/fabric and the")
    print("           traffic model overstates the problem.")

    if len(ok) >= 2:
        lo = min(ok, key=lambda r: r["bv"])
        hi = max(ok, key=lambda r: r["bv"])
        if hi["fetch"]:
            scale = lo["fetch"] / hi["fetch"]
            m_scale = lo["m_read"] / hi["m_read"]
            print(f"\nfetch scaling BV={hi['bv']} -> BV={lo['bv']}: "
                  f"measured {scale:.2f}x, modelled {m_scale:.2f}x")
            print("  If measured tracks modelled, redundant reads dominate and are real.")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["shape", "BV", "grid_x", "fetch_bytes", "model_read_bytes",
                        "fetch_ratio", "write_bytes", "model_write_bytes",
                        "l2_hit_rate", "dispatches"])
            for r in ok:
                w.writerow([shape.name, r["bv"], r["grid_x"], int(r["fetch"]),
                            r["m_read"], f"{r['ratio']:.4f}", int(r["write"]),
                            r["m_write"], f"{r['hit_rate']:.4f}", r["dispatches"]])
        print(f"\ncsv         : {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
