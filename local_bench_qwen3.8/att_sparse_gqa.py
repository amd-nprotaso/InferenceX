#!/usr/bin/env python3
"""Standalone single-dispatch driver for _sparse_gqa_chunk_prefill, for ATT.

  rocprofv3 --att=true --att-library-path /opt/rocm/lib \
    --kernel-include-regex 'sparse_gqa' --att-target-cu 1 \
    -d att_gqa -o att -- python3 att_sparse_gqa.py --topk 256

Use a reduced --topk for the ATT run: the kernel is gather-bound and scales
linearly with topk, so a smaller topk has the same instruction mix in a much
smaller trace.
"""

from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, ".")
import bench_sparse_gqa as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-q", type=int, default=2048)
    ap.add_argument("--kv-len", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--block-n", type=int, default=16)
    ap.add_argument("--warps", type=int, default=1)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--warmup-iters", type=int, default=2)
    ap.add_argument("--time", action="store_true")
    args = ap.parse_args()

    B.TOPK = args.topk
    inp = B.build(args.total_q, args.kv_len, False)
    print(f"q[{args.total_q},{B.NUM_Q_HEADS},{B.HEAD_DIM}] "
          f"kv[{args.kv_len},{B.NUM_KV_HEADS},{B.HEAD_DIM}] topk={args.topk} "
          f"BLOCK_N={args.block_n} warps={args.warps} stages={args.stages}")

    def call():
        return B.launch(inp, "chunk", args.block_n, args.warps, args.stages)

    for _ in range(args.warmup_iters):
        call()
    torch.cuda.synchronize()
    print(f"warmup done ({args.warmup_iters} dispatches)")
    if args.time:
        print(f"timed: {B.timeit(call, 10, 3):.3f} ms")
    else:
        call()
        torch.cuda.synchronize()
        print("profiled 1 dispatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
