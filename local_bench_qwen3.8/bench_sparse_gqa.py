#!/usr/bin/env python3
"""Tune SGLang's QSA sparse-attention Triton kernels at Qwen3.8 shapes.

Target: `_sparse_gqa_chunk_prefill` -- 78.44 ms of 910.7 ms prefill GPU time
(8.6%) in only 13 launches (6.03 ms each), the most expensive per-launch kernel
in the trace. Its sibling `_sparse_gqa_prefill` adds 62.29 ms (6.8%). Together
15.5% of prefill, nearly 3x the GDN chunk-state kernel.

THE FINDING THIS EXISTS TO TEST
-------------------------------
`sparse_attn.py:_get_best_config` picks its (BLOCK_N, num_warps, num_stages)
tuple from a table keyed on NVIDIA GPU names:

    table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS

On "AMD Instinct MI355X" that is False, so it silently uses the **NVIDIA L20**
table, and at prefill-scale total_q falls through to the catch-all
`(float("inf"), (16, 1, 2))` -> BLOCK_N=16, **num_warps=1**, num_stages=2.
num_warps=1 is a 64-thread workgroup on AMD (the table was tuned against
32-thread warps), so this is an untuned config from the wrong vendor.

SHAPES
------
From config.json text_config: num_attention_heads=24, num_key_value_heads=2,
head_dim=256, indexer_budget=2048 (topk token indices per query, after the
compress_ratio=4 block expansion).

The two kernels split the prefill by chunk, which is why each runs 13x
(12 full-attention layers + 1 MTP) rather than 26x:
  chunk 1 (prefix_len=0)      -> _sparse_gqa_prefill,       kv_len = 16384
  chunk 2 (prefix_len=16384)  -> _sparse_gqa_chunk_prefill, kv_len = 32768

CAVEAT: shapes are derived from config.json + the kernel's own index
arithmetic, not captured from a live forward pass. The `indices` rank in
particular is inferred (the launcher accepts 2D [M,N] shared across kv heads,
or 3D [M,G,N]); `--indices-3d` switches. Relative config ranking is robust to
this, absolute times less so. Validate against a real run before quoting
end-to-end numbers.

  python3 bench_sparse_gqa.py                  # stock vs swept configs
  python3 bench_sparse_gqa.py --kernel prefill # the chunk-1 sibling
"""

from __future__ import annotations

import argparse
import itertools
import sys

import torch
import triton

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.srt.layers.attention.qsa.sparse_attn import (  # noqa: E402
    _get_best_config,
    _sparse_gqa_chunk_prefill,
    _sparse_gqa_prefill,
)

# Qwen3.8-Flash-Next text_config
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
TOPK = 2048          # indexer_budget
DTYPE = torch.bfloat16


def build(total_q: int, kv_len: int, indices_3d: bool, seed: int = 0):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
    k = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
    v = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)

    # Causal-plausible indices: query i may see [0, prefix + i]. Sample without
    # replacement per row would be far too slow at this size, so use a strided
    # scatter that stays in range and is not degenerate.
    prefix = kv_len - total_q
    base = torch.arange(TOPK, device=dev, dtype=torch.int32)
    rows = torch.arange(total_q, device=dev, dtype=torch.int32).unsqueeze(1)
    visible = (rows + prefix + 1).clamp(min=1)
    idx = (base.unsqueeze(0) * 7919) % visible          # deterministic spread
    idx = idx.to(torch.int32).contiguous()
    if indices_3d:
        idx = idx.unsqueeze(1).expand(total_q, NUM_KV_HEADS, TOPK).contiguous()

    cu_q = torch.tensor([0, total_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32, device=dev)
    return dict(q=q, k=k, v=v, indices=idx, cu_q=cu_q, cu_k=cu_k,
                kv_lens=kv_lens, total_q=total_q, kv_len=kv_len,
                scale=HEAD_DIM ** -0.5)


def launch(inp, which, block_n, warps, stages):
    """Replicates sparse_gqa_fwd_interface_triton_ck / _triton with the tuning
    tuple overridden, so a config can be forced instead of taken from the
    NVIDIA-keyed table."""
    q, k, v, ind = inp["q"], inp["k"], inp["v"], inp["indices"]
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    block_m = max(16, triton.next_power_of_2(group_size))
    out = torch.empty_like(q)
    si_m = ind.stride(0)
    si_g = ind.stride(1) if ind.ndim == 3 else 0
    si_n = ind.stride(2) if ind.ndim == 3 else ind.stride(1)
    grid = (inp["total_q"], (inp["cu_q"].shape[0] - 1) * NUM_KV_HEADS)

    if which == "chunk":
        _sparse_gqa_chunk_prefill[grid](
            q, k, v, out, ind, inp["cu_q"], inp["cu_k"], inp["kv_lens"],
            inp["scale"], TOPK,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            si_m, si_g, si_n,
            NUM_KV_HEADS=NUM_KV_HEADS, GROUP_SIZE=group_size,
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=HEAD_DIM,
            num_warps=warps, num_stages=stages,
        )
    else:
        _sparse_gqa_prefill[(inp["kv_len"], (inp["cu_q"].shape[0] - 1) * NUM_KV_HEADS)](
            q, k, v, out, ind, inp["cu_q"], inp["scale"], TOPK,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            si_m, si_g, si_n,
            NUM_KV_HEADS=NUM_KV_HEADS, GROUP_SIZE=group_size,
            BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=HEAD_DIM,
            num_warps=warps, num_stages=stages,
        )
    return out


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=("chunk", "prefill"), default="chunk")
    ap.add_argument("--total-q", type=int, default=16384)
    ap.add_argument("--kv-len", type=int, default=None,
                    help="default: 32768 for chunk, 16384 for prefill")
    ap.add_argument("--indices-3d", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--block-n", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    kv_len = args.kv_len or (32768 if args.kernel == "chunk" else 16384)
    inp = build(args.total_q, kv_len, args.indices_3d)

    name = torch.cuda.get_device_name(0)
    stock = _get_best_config(args.total_q)
    print(f"device            : {name}")
    print(f"config table used : "
          f"{'_H20_CONFIGS' if 'H20' in name else '_L20_CONFIGS  <-- NVIDIA table on AMD'}")
    print(f"kernel            : _sparse_gqa_{args.kernel}_prefill")
    print(f"shapes            : q[{args.total_q},{NUM_Q_HEADS},{HEAD_DIM}] "
          f"kv[{kv_len},{NUM_KV_HEADS},{HEAD_DIM}] topk={TOPK} "
          f"indices{'3D' if args.indices_3d else '2D'}")
    print(f"grid              : ({args.total_q}, {NUM_KV_HEADS}) = "
          f"{args.total_q * NUM_KV_HEADS} workgroups")
    print(f"STOCK config      : BLOCK_N={stock[0]} num_warps={stock[1]} "
          f"num_stages={stock[2]}\n")

    ref = None
    base_ms = None
    results = []
    combos = [stock] + [c for c in itertools.product(args.block_n, args.warps,
                                                     args.stages)
                        if tuple(c) != tuple(stock)]
    for bn, w, st in combos:
        try:
            out = launch(inp, args.kernel, bn, w, st)
            torch.cuda.synchronize()
            ms = timeit(lambda: launch(inp, args.kernel, bn, w, st),
                        args.iters, args.warmup)
        except Exception as exc:
            print(f"  BLOCK_N={bn:<4} warps={w:<2} stages={st}  SKIP "
                  f"({type(exc).__name__}: {str(exc).splitlines()[0][:60]})")
            continue
        if ref is None:
            ref, base_ms = out.clone(), ms
            tag = "  <-- STOCK"
            dev_s = "reference"
        else:
            d = (out.float() - ref.float()).abs().max().item()
            dev_s = f"maxdiff={d:.2e}"
            tag = ""
        spd = base_ms / ms
        results.append((ms, bn, w, st, spd, dev_s))
        print(f"  BLOCK_N={bn:<4} warps={w:<2} stages={st}  {ms:8.3f} ms  "
              f"{spd:5.2f}x  {dev_s}{tag}")

    print("\n  --- best 5 ---")
    for ms, bn, w, st, spd, dev_s in sorted(results)[:5]:
        print(f"  BLOCK_N={bn:<4} warps={w:<2} stages={st}  {ms:8.3f} ms  "
              f"{spd:5.2f}x  {dev_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
