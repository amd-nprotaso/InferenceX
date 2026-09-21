#!/usr/bin/env python3
"""Concurrency sweep for the GDN chunk-state kernel: sglang Triton vs AITER FlyDSL.

ATT showed the kernel is latency-starved at conc=1: grid = (V/BV) x H x N is
192-384 workgroups of 4 waves on 256 CUs, and SIMDs sit 31-77% EMPTY. N (the
number of sequences in the prefill batch) is the only term in that grid which
grows without shrinking each workgroup's work, so this sweeps it.

"Concurrency" has two defensible readings for a chunked-prefill batch, and they
bracket reality, so both are measured:

  packed  -- the --chunked-prefill-size 16384 budget is FIXED and split across N
             sequences (16384/N tokens each). Total work constant; parallelism
             grows N-fold. This is what a prefill pass looks like when the
             scheduler packs several requests into one budget.
  per-seq -- each of the N sequences contributes a full 16384-token chunk
             (N*16384 tokens). Total work grows N-fold. This is the upper bound
             on available parallelism, and what you get if the scheduler ever
             admits several full chunks at once.

At ISL=32768 with a 16384 budget, one request's chunk fills the budget exactly,
so production at conc>1 most likely still issues N=1 passes -- `packed` at N=1.
Treat the N>1 columns as headroom, not as a prediction. Verify against a real
server trace before acting on them.

  python3 sweep_gdn_chunk_h_conc.py
  FLYDSL_K5_OPT_BV=32 SCHED=1 python3 sweep_gdn_chunk_h_conc.py
"""

from __future__ import annotations

import argparse
import os
import sys

AITER_DIR = os.environ.get("AITER_DIR", "/var/home/my_aiter/aiter")
if AITER_DIR and os.path.isdir(AITER_DIR):
    sys.path.insert(0, AITER_DIR)

import torch  # noqa: E402

from bench_gdn_chunk_h import (  # noqa: E402
    BT, H, V, _import_backends, make_inputs, run_backend, timeit, to_head_major,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--budget", type=int, default=16384,
                    help="chunked-prefill token budget (packed mode)")
    ap.add_argument("--chunk", type=int, default=16384,
                    help="tokens per sequence (per-seq mode)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--modes", nargs="+", default=["packed", "per-seq"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    # Apply the two tuning levers found via ATT, if requested.
    if os.environ.get("SCHED", "0") == "1":
        import aiter.ops.flydsl.linear_attention_prefill_kernels as M
        M._IS_GFX942 = True

    fns = _import_backends(["sglang", "flydsl"])
    props = torch.cuda.get_device_properties(0)
    cus = props.multi_processor_count
    bv_env = os.environ.get("FLYDSL_K5_OPT_BV", "auto")
    print(f"device {torch.cuda.get_device_name(0)}  ({cus} CUs)")
    print(f"FLYDSL_K5_OPT_BV={bv_env}  sched_barrier={os.environ.get('SCHED','0')=='1'}")
    bv = int(bv_env) if bv_env.isdigit() else 16   # heuristic picks 16 on gfx950
    print(f"flydsl grid = (V/BV) x H x N = {V//bv} x {H} x N = {V//bv*H} x N WGs\n")

    for mode in args.modes:
        print(f"===== mode={mode} "
              f"({'fixed %d-token budget split N ways' % args.budget if mode == 'packed' else 'N x %d tokens' % args.chunk}) =====")
        print(f"  {'N':>3} {'tok/seq':>8} {'tok tot':>9} {'WGs':>6} {'WG/CU':>6} "
              f"{'sglang':>9} {'flydsl':>9} {'ratio':>7}")
        for n in args.conc:
            T = args.budget // n if mode == "packed" else args.chunk
            if T < BT:
                print(f"  {n:>3}  tokens/seq {T} < chunk {BT}; skipped")
                continue
            T = (T // BT) * BT                      # whole chunks only
            inp = make_inputs(T, "cuda", nseq=n)
            w_hm, u_hm = to_head_major(inp["w"], inp["u"])
            try:
                sg = timeit(lambda: run_backend("sglang", fns["sglang"], inp,
                                                inp["pool"].clone()),
                            args.iters, args.warmup)
                fd = timeit(lambda: run_backend("flydsl", fns["flydsl"], inp,
                                                inp["pool"].clone(), (w_hm, u_hm)),
                            args.iters, args.warmup)
            except Exception as exc:
                print(f"  {n:>3} ERROR {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:80]}")
                continue
            wgs = (V // bv) * H * n
            print(f"  {n:>3} {T:>8} {T*n:>9} {wgs:>6} {wgs/cus:>6.1f} "
                  f"{sg:>8.3f}m {fd:>8.3f}m {sg/fd:>6.2f}x")
            del inp, w_hm, u_hm
            torch.cuda.empty_cache()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
