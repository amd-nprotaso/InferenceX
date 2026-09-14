#!/usr/bin/env python3
"""Isolated timing of the MoE routing front-end: router, sort, and the fusion.

Times, at the Qwen3.5 decode shape, the four things change #2 trades against
each other, each on its own stream with nothing else running:

  router          aiter.topk_softmax           (vllm::moe::topkGatingSoftmax)
  sort:opus       aiter.moe_sorting_opus_fwd   (2 launches, the default)
  sort:flydsl     aiter's vendored unfused FlyDSL oneshot (1 launch)
  fused           this port: router+sort in 1 launch

Usage: python3 bench_route_sort.py [--tokens 16] [--iters 200]
"""

import argparse
import os
import sys

import torch

import aiter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BM = 32


def _bench(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us


def main(M, E, topk, model_dim, iters, max_tokens):
    from moe_fused_route_sort_flydsl import flydsl_fused_route_sort

    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn((M, E), dtype=torch.bfloat16, device=dev)

    tw = torch.empty(M, topk, dtype=torch.float32, device=dev)
    ti = torch.empty(M, topk, dtype=torch.int32, device=dev)
    tei = torch.empty(M, topk, dtype=torch.int32, device=dev)
    aiter.topk_softmax(tw, ti, tei, logits, True)

    max_sorted = M * topk + E * BM - topk
    max_blocks = (max_sorted + BM - 1) // BM

    def alloc():
        return (
            torch.empty(max_sorted, dtype=torch.int32, device=dev),
            torch.empty(max_sorted, dtype=torch.float32, device=dev),
            torch.empty(max_blocks, dtype=torch.int32, device=dev),
            torch.empty(2, dtype=torch.int32, device=dev),
            torch.empty((0, 0), dtype=torch.bfloat16, device=dev),
        )

    out_a = alloc()
    out_b = alloc()
    out_c = alloc()

    ws_size = aiter.moe_sorting_opus_get_workspace_size(M, E, topk, 0)
    ws = torch.empty(ws_size, dtype=torch.uint8, device=dev) if ws_size > 0 else None

    def router():
        aiter.topk_softmax(tw, ti, tei, logits, True)

    def sort_opus():
        aiter.moe_sorting_opus_fwd(ti, tw, *out_a, E, BM, None, None, ws, 0, None, None, None)

    def sort_flydsl():
        from aiter.ops.flydsl.moe_sorting import flydsl_moe_sorting_fwd

        flydsl_moe_sorting_fwd(ti, tw, *out_b, E, BM, None, None)

    def fused():
        flydsl_fused_route_sort(logits, *out_c, E, topk, BM, max_tokens=max_tokens)

    rows = [
        ("router (topk_softmax)", router, 1),
        ("sort:opus", sort_opus, 2),
        ("sort:flydsl-unfused", sort_flydsl, 1),
        ("fused router+sort", fused, 1),
    ]
    print(f"\n=== M={M} E={E} topk={topk} max_tokens={max_tokens} iters={iters}")
    res = {}
    for name, fn, launches in rows:
        try:
            t = _bench(fn, iters)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:24s}  FAILED: {type(exc).__name__}: {exc}")
            continue
        res[name] = t
        print(f"  {name:24s}  {t:7.2f} us  ({launches} launch{'es' if launches > 1 else ''})")
    if "router (topk_softmax)" in res and "sort:opus" in res and "fused router+sort" in res:
        base = res["router (topk_softmax)"] + res["sort:opus"]
        print(f"  {'-> router+opus':24s}  {base:7.2f} us  (3 launches)")
        print(f"  {'-> delta':24s}  {res['fused router+sort'] - base:+7.2f} us")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--model-dim", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=None)
    a = ap.parse_args()
    main(
        a.tokens,
        a.experts,
        a.topk,
        a.model_dim,
        a.iters,
        a.max_tokens if a.max_tokens is not None else a.tokens,
    )
