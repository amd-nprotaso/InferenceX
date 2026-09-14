#!/usr/bin/env python3
"""Per-kernel timing of the AITER MXFP4 2-stage MoE at Qwen3.5-397B shapes.

Compares the parallelism layouts that the server script can actually pick:

  TP4 (what runs today, --ep-size 1): every rank owns all 512 experts but only
      inter_dim/4 = 256 of each -> stage2 K=256, and every rank produces the
      full [tokens*topk, 4096] partial-output tensor.
  EP4 (--ep-size 4):                 every rank owns 128 whole experts
      (inter_dim 1024) and only tokens*topk/4 rows -> stage2 K=1024 and a 4x
      smaller partial-output tensor.

Both arms do the same FLOPs per rank; only the shape of the stage-2 output
traffic changes. Times come from a torch profiler trace so the reduction kernel
is attributed separately from the GEMMs.

Usage: python3 moe_shape_bench.py [--iters 5]
"""

import argparse
import collections
import os

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

QTYPE = aiter.QuantType.per_1x32
WQ = dtypes.fp4x2


def build(token, model_dim, inter_dim, E, topk, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    x = torch.randn((token, model_dim), dtype=dtypes.bf16, device=dev)

    torch_quant = aiter.get_torch_quant(QTYPE)
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtypes.bf16, device=dev)
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16, device=dev)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=WQ)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=WQ)
    del w1, w2
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    w1_s = shuffle_weight(w1_qt, layout=(16, 16))
    w2_s = shuffle_weight(w2_qt, layout=(16, 16))
    del w1_qt, w2_qt
    w1_scale = fp4_utils.e8m0_shuffle(w1_scale)
    w2_scale = fp4_utils.e8m0_shuffle(w2_scale)

    # uniform routing: the best case for load balance, so any gap we measure is
    # the kernel's, not an imbalance artifact.
    score = torch.randn((token, E), dtype=dtypes.bf16, device=dev)
    topk_weights, topk_ids = torch.topk(score.float().softmax(-1), topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    torch.cuda.empty_cache()
    return x, w1_s, w2_s, w1_scale, w2_scale, topk_weights, topk_ids


def run(tag, token, model_dim, inter_dim, E, topk, iters):
    x, w1, w2, w1s, w2s, tw, ti = build(token, model_dim, inter_dim, E, topk)
    kw = dict(w1_scale=w1s, w2_scale=w2s, quant_type=QTYPE,
              activation=aiter.ActivationType.Silu, doweight_stage1=False)

    for _ in range(3):
        fused_moe(x, w1, w2, tw, ti, **kw)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fused_moe(x, w1, w2, tw, ti, **kw)
        torch.cuda.synchronize()

    per = collections.defaultdict(float)
    for ev in prof.events():
        if ev.device_type.name != "CUDA" and str(ev.device_type) != "DeviceType.CUDA":
            continue
        d = getattr(ev, "self_device_time_total", 0) or 0
        if d:
            per[ev.key] += d / iters

    # M*N*K*2 for both stages: stage1 N=2*inter K=model_dim, stage2 N=model_dim K=inter
    rows = token * topk
    f1 = 2 * rows * (2 * inter_dim) * model_dim
    f2 = 2 * rows * model_dim * inter_dim
    print(f"\n===== {tag}: token={token} topk={topk} rows={rows} E={E} "
          f"model_dim={model_dim} inter_dim={inter_dim} =====")
    print(f"stage1 {f1/1e12:.3f} TFLOP   stage2 {f2/1e12:.3f} TFLOP   "
          f"stage2 partial-out [{rows},{model_dim}] bf16 = {rows*model_dim*2/1e9:.3f} GB")
    tot = 0.0
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
        if v < 1.0:
            continue
        tot += v
        note = ""
        if k.startswith("mfma_moe1") or "moe1" in k or "stage1" in k:
            note = f"  [{f1/(v*1e-6)/1e12:.0f} TFLOP/s]"
        elif k.startswith("mfma_moe2") or "moe2" in k:
            note = f"  [{f2/(v*1e-6)/1e12:.0f} TFLOP/s]"
        elif "reduction" in k or "reduce" in k:
            note = f"  [{(rows*model_dim*2 + token*model_dim*2)/(v*1e-6)/1e12:.2f} TB/s]"
        print(f"  {v:9.1f} us  {k[:92]}{note}")
    print(f"  {tot:9.1f} us  TOTAL (kernels >1us)")
    del x, w1, w2, w1s, w2s, tw, ti
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    arms = [
        ("decode conc=1 (bs=4 tok)", 4, 4096, 256, 512, 10),
        ("decode conc=2 (bs=8 tok)", 8, 4096, 256, 512, 10),
        ("decode conc=4 (bs=16 tok)", 16, 4096, 256, 512, 10),
        ("decode conc=8 (bs=32 tok)", 32, 4096, 256, 512, 10),
        ("decode conc=16 (bs=64 tok)", 64, 4096, 256, 512, 10),
        ("decode conc=32 (bs=128 tok)", 128, 4096, 256, 512, 10),
        ("decode conc=64 (bs=256 tok)", 256, 4096, 256, 512, 10),
    ]
    for arm in arms:
        if a.only and a.only not in arm[0]:
            continue
        run(*arm, iters=a.iters)
