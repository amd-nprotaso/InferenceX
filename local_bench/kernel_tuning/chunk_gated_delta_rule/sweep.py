#!/usr/bin/env python3
"""Sweep Triton configs for chunk_gated_delta_rule_fwd_kernel_h_blockdim64.

Why this kernel: on MI350X it runs 69.16 ms per prefill forward (45 calls)
against 46.96 ms on GB200 for the *identical* Triton source and identical input
shapes -- 1.47x slower. The cause is not code quality, it is launch geometry.
Both platforms launch grid=(4,16)=64 workgroups, which fills 43% of a 148-SM
B200 but only 25% of a 256-CU MI350X. The kernel writes 268 MB of `h` in 1.54 ms
= 174 GB/s on a ~8 TB/s part, so it is nowhere near bandwidth-bound; occupancy
is the only thing holding it back.

Grid is (cdiv(V, BV), N*H) and nothing else -- the T axis carries the recurrent
state and cannot be parallelized. With V=128 that makes BV the only knob that
changes workgroup count:

    BV=64 ->  2*16 =  32 WGs (12%)      BV=16 -> 8*16 = 128 WGs (50%)
    BV=32 ->  4*16 =  64 WGs (25%)      BV= 8 -> 16*16 = 256 WGs (100%)

BV=8 may not compile (tl.dot wants >=16 in the contracted tile); it is swept
anyway so you get the answer rather than a guess. 50% fill is the realistic
ceiling for pure config tuning -- getting past it needs the K-split described in
README.md, which is a code change.

Usage
-----
  ./sweep.py                                  # default sweep, production shape
  ./sweep.py --quick                          # BV only, warps/stages at default
  ./sweep.py --bv 8,16,32 --warps 2,4 --stages 1,2,3
  ./sweep.py --shape qwen35-397b-tp2          # TP2 control
  ./sweep.py --waves-per-eu 1,2,4             # AMD occupancy hint (needs code change to ship)
  ./sweep.py --csv results.csv --reps 30

Read the `fill%` and `WGs` columns first. If the winning config did not raise
workgroup count, occupancy was not the binding constraint after all and the
K-split is not worth building.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The kernel uses tl.make_block_ptr, which triton 3.7 deprecates. That is
# sglang's call to make, not ours, and the warning fires per compile -- once per
# config -- which buries the sweep output.
warnings.filterwarnings("ignore", message=".*make_block_ptr is deprecated.*")

import torch  # noqa: E402

from bench import BASELINE, Config, compare, env_equivalent, run_config  # noqa: E402
from shapes import DEFAULT_SHAPE, SHAPES, build_inputs, device_info  # noqa: E402


def _ints(s: str) -> list:
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Config sweep for the GDN chunk-state Triton kernel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--shape", default=DEFAULT_SHAPE, choices=sorted(SHAPES))
    p.add_argument("--bv", default="8,16,32,64", help="BV tile sizes to try")
    p.add_argument("--warps", default="1,2,4,8", help="num_warps (x64 threads on CDNA)")
    p.add_argument("--stages", default="1,2,3", help="num_stages")
    p.add_argument(
        "--waves-per-eu",
        default="",
        help="AMD occupancy hint; empty to skip. No sglang env knob -- shipping "
        "a winner here needs a chunk_delta_h.py change.",
    )
    p.add_argument(
        "--mfma",
        default="",
        help="matrix_instr_nonkdim values (e.g. 16,32); empty to skip.",
    )
    p.add_argument("--quick", action="store_true", help="BV only, other knobs at default")
    p.add_argument(
        "--isolate",
        action="store_true",
        help="Run each config in its own process. Slower, but survives configs "
        "that abort the compiler instead of raising -- waves_per_eu>=8 trips an "
        "LLVM assertion in Sequence.h and core-dumps, which no try/except can "
        "catch. Implies --no-verify (outputs cannot be compared across processes).",
    )
    p.add_argument("--isolate-timeout", type=int, default=300)
    p.add_argument("--_single", default="", help=argparse.SUPPRESS)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--pool-slots",
        type=int,
        default=None,
        help="Override state-pool rows (default: the traced 1243). Only the "
        "used slots are touched, so this affects memory, not the measurement.",
    )
    p.add_argument("--no-verify", action="store_true", help="Skip output comparison")
    p.add_argument("--csv", default="", help="Write results to this CSV path")
    return p.parse_args(argv)


def build_configs(a) -> list:
    bvs = _ints(a.bv)
    if a.quick:
        warps, stages = [BASELINE["num_warps"]], [BASELINE["num_stages"]]
    else:
        warps, stages = _ints(a.warps), _ints(a.stages)
    wpe = _ints(a.waves_per_eu) or [None]
    mfma = _ints(a.mfma) or [None]

    cfgs = [
        Config(BV=bv, num_warps=w, num_stages=s, waves_per_eu=q, matrix_instr_nonkdim=m)
        for bv, w, s, q, m in itertools.product(bvs, warps, stages, wpe, mfma)
    ]

    # Baseline first: it is the reference for both speedup and correctness, and
    # running it first means a later OOM still leaves a usable comparison.
    base = Config(**BASELINE)
    cfgs = [c for c in cfgs if not c.is_baseline()]
    return [base] + cfgs


RESULT_TAG = "@@RESULT@@"


def _run_single_child(a) -> int:
    """Worker path for --isolate: time one config, print one JSON line."""
    import json

    spec = json.loads(a._single)
    torch.cuda.set_device(a.device)
    dev = device_info(a.device)
    shape = SHAPES[a.shape]
    inp = build_inputs(shape, device=f"cuda:{a.device}", seed=a.seed,
                       pool_slots=a.pool_slots)
    r = run_config(
        Config(**spec), inp,
        reps=a.reps, warmup=a.warmup, keep_output=False,
        device_cus=dev["cus"], warp_size=dev["warp_size"],
    )
    print(RESULT_TAG + json.dumps({
        "ok": r.ok, "ms": r.ms, "ms_std": r.ms_std, "grid": list(r.grid),
        "workgroups": r.workgroups, "block_threads": r.block_threads,
        "fill_pct": r.fill_pct, "gbps": r.gbps, "gbps_hv": r.gbps_hv,
        "bytes_total": r.bytes_total, "error": r.error,
    }))
    return 0


def _run_isolated(cfg: Config, a) -> "object":
    """Parent path for --isolate: one subprocess per config.

    A config that aborts the process (rather than raising) comes back as a
    failed Result with the signal in `error`, and the sweep continues.
    """
    import json
    import subprocess

    from bench import Result

    spec = {
        "BV": cfg.BV, "num_warps": cfg.num_warps, "num_stages": cfg.num_stages,
        "waves_per_eu": cfg.waves_per_eu,
        "matrix_instr_nonkdim": cfg.matrix_instr_nonkdim,
    }
    cmd = [
        sys.executable, "-W", "ignore", os.path.abspath(__file__),
        "--_single", json.dumps(spec), "--shape", a.shape,
        "--reps", str(a.reps), "--warmup", str(a.warmup),
        "--device", str(a.device), "--seed", str(a.seed),
    ]
    if a.pool_slots is not None:
        cmd += ["--pool-slots", str(a.pool_slots)]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=a.isolate_timeout
        )
    except subprocess.TimeoutExpired:
        return Result(config=cfg, ok=False, error=f"timeout >{a.isolate_timeout}s")

    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_TAG):
            d = json.loads(line[len(RESULT_TAG):])
            return Result(
                config=cfg, ok=d["ok"], ms=d["ms"], ms_std=d["ms_std"],
                grid=tuple(d["grid"]), workgroups=d["workgroups"],
                block_threads=d["block_threads"], fill_pct=d["fill_pct"],
                gbps=d["gbps"], gbps_hv=d.get("gbps_hv", float("nan")),
                bytes_total=d.get("bytes_total", 0), error=d["error"],
            )
    tail = (proc.stderr or "").strip().splitlines()
    why = tail[-1][:120] if tail else "no result line"
    return Result(config=cfg, ok=False, error=f"rc={proc.returncode}: {why}")


def main(argv=None) -> int:
    a = parse_args(argv)
    if a._single:
        return _run_single_child(a)
    if a.isolate:
        a.no_verify = True
    torch.cuda.set_device(a.device)
    dev = device_info(a.device)
    shape = SHAPES[a.shape]
    cfgs = build_configs(a)

    print(f"device      : {dev['name']}  {dev['cus']} CUs  wavefront {dev['warp_size']}")
    print(f"shape       : {shape.name}  B={shape.B} T={shape.T} Hg={shape.Hg} "
          f"H={shape.H} K={shape.K} V={shape.V} BT={shape.BT} NT={shape.NT}")
    print(f"              {shape.note}")
    print(f"traffic     : h={shape.h_bytes()/1e6:.0f} MB + v_new="
          f"{shape.v_new_bytes()/1e6:.0f} MB per call")
    print(f"baseline    : BV={BASELINE['BV']} warps={BASELINE['num_warps']} "
          f"stages={BASELINE['num_stages']}  (production default)")
    print(f"sweep       : {len(cfgs)} configs, {a.reps} reps + {a.warmup} warmup")
    print()

    inp = None if a.isolate else build_inputs(
        shape, device=f"cuda:{a.device}", seed=a.seed, pool_slots=a.pool_slots
    )

    results = []
    ref = None
    t0 = time.time()
    for i, cfg in enumerate(cfgs, 1):
        keep = not a.no_verify
        print(f"  [{i:>3}/{len(cfgs)}] {cfg.label():<44}", end="", flush=True)
        if a.isolate:
            r = _run_isolated(cfg, a)
        else:
            r = run_config(
                cfg, inp,
                reps=a.reps, warmup=a.warmup, keep_output=keep,
                device_cus=dev["cus"], warp_size=dev["warp_size"],
            )
        if r.ok:
            if ref is None:
                ref = r
            elif keep:
                compare(ref, r)
            print(f"{r.ms:8.3f} ms   {r.workgroups:>4} WGs  {r.fill_pct:5.1f}% fill")
            # Only the reference needs its tensors kept alive.
            if r is not ref:
                r._h = r._v = None
        else:
            print(f"   FAILED  {r.error}")
        results.append(r)
    elapsed = time.time() - t0

    ok = [r for r in results if r.ok]
    if not ok:
        print("\nEvery config failed. Check the sglang import and GPU state.")
        return 1
    base = next((r for r in ok if r.config.is_baseline()), ok[0])

    print(f"\n{'='*118}")
    print(f"{'config':<44}{'ms':>9}{'±':>7}{'speedup':>9}{'WGs':>6}"
          f"{'blk':>6}{'fill%':>7}{'TB/s':>7}{'h+v':>7}{'MB':>7}{'maxerr':>9}")
    print("-" * 118)
    for r in sorted(ok, key=lambda x: x.ms):
        err = "ref" if r is base else (
            "-" if r.max_err_h != r.max_err_h else f"{r.max_err_h:.2e}"
        )
        tag = "  <-- baseline" if r.config.is_baseline() else ""
        print(f"{r.config.label():<44}{r.ms:9.3f}{r.ms_std:7.3f}"
              f"{base.ms / r.ms:8.2f}x{r.workgroups:6d}{r.block_threads:6d}"
              f"{r.fill_pct:7.1f}{r.gbps/1000:7.2f}{r.gbps_hv:7.0f}"
              f"{r.bytes_total/1e6:7.0f}{err:>9}{tag}")
    failed = [r for r in results if not r.ok]
    for r in failed:
        print(f"{r.config.label():<44}{'FAILED':>9}  {r.error}")
    print("-" * 118)
    print("TB/s = full traffic (h + v_new + u + grid.x x (w + k)); "
          "h+v = GB/s counting only h + v_new, the old column")

    best = min(ok, key=lambda x: x.ms)
    print(f"\nswept {len(results)} configs in {elapsed:.0f}s "
          f"({len(failed)} failed)")
    print(f"best        : {best.config.label()}  {best.ms:.3f} ms  "
          f"({base.ms / best.ms:.2f}x vs baseline {base.ms:.3f} ms)")
    print(f"workgroups  : {base.workgroups} -> {best.workgroups} "
          f"({base.fill_pct:.0f}% -> {best.fill_pct:.0f}% of {dev['cus']} CUs)")

    if best.config.is_baseline():
        print("\nThe production default already won. Occupancy tuning is a dead end "
              "here -- do not build the K-split; re-check the all-reduce instead.")
    elif best.workgroups <= base.workgroups:
        print("\nThe winner did NOT raise workgroup count, so the gain is not from "
              "occupancy. The K-split premise is wrong -- re-profile before building it.")
    else:
        print(f"\nApply to the server with:\n  {env_equivalent(best.config)}")
        if best.fill_pct < 99:
            # Aggregate throughput tracks workgroup count at ~20 GB/s per
            # workgroup (measured across TP4/TP2/4k at BV>=16), so the headroom
            # from unused CUs is real and roughly linear. What it does NOT need
            # is the K-split: b_v sums over every K block (chunk_delta_h.py:178-196)
            # and then updates every K block (:275-293), so splitting K across
            # workgroups would need a grid-wide reduction on all NT iterations.
            # See README.md "Why the K-split cannot work".
            print(f"\nStill only {best.fill_pct:.0f}% filled, and the kernel scales "
                  f"~linearly with workgroups, so the remaining "
                  f"{100 - best.fill_pct:.0f}% is worth roughly "
                  f"{best.ms * (1 - best.fill_pct / 100):.2f} ms.")
            print("  NOT via the K-split (architecturally blocked) and NOT via "
                  "BV<16 (halves per-workgroup rate).")
            print("  The only untried axis is a T-split two-pass scan -- see README.md.")

    if not a.no_verify:
        bad = [r for r in ok if r.max_err_h == r.max_err_h and r.max_err_h > 0.05]
        if bad:
            print("\nWARNING: configs with large output drift vs baseline "
                  "(suspect, do not ship):")
            for r in bad:
                print(f"  {r.config.label():<44} max|dh|={r.max_err_h:.3e}")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wtr = csv.writer(fh)
            wtr.writerow(["shape", "BV", "num_warps", "num_stages", "waves_per_eu",
                          "matrix_instr_nonkdim", "ok", "ms", "ms_std", "speedup",
                          "workgroups", "block_threads", "fill_pct", "gbps",
                          "gbps_hv", "bytes_total",
                          "max_err_h", "max_err_v", "error"])
            for r in results:
                c = r.config
                wtr.writerow([shape.name, c.BV, c.num_warps, c.num_stages,
                              c.waves_per_eu, c.matrix_instr_nonkdim, r.ok,
                              f"{r.ms:.4f}", f"{r.ms_std:.4f}",
                              f"{base.ms / r.ms:.4f}" if r.ok else "",
                              r.workgroups, r.block_threads, f"{r.fill_pct:.1f}",
                              f"{r.gbps:.1f}", f"{r.gbps_hv:.1f}", r.bytes_total,
                              r.max_err_h, r.max_err_v, r.error])
        print(f"\ncsv         : {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
