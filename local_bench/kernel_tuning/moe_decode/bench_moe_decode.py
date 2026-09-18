#!/usr/bin/env python3
"""Per-kernel timing of the *whole* Qwen3.5-397B MXFP4 decode MoE front-end.

``../../moe_decode_bench.py`` starts from ``topk_weights``/``topk_ids`` and so
never shows the router.  The router is 9.8 us/layer of the 70.3 us this work is
attacking, and change #2 deletes it, so it has to be inside the measured
window: this bench calls ``aiter.fused_moe.fused_topk`` on gating logits and
then ``fused_moe``, which is exactly the SGLang call sequence.

Arms (``--arms``):
  stock  -- aiter router + opus sort, tuned CSV as shipped
  fused  -- SGLANG_FLYDSL_MOE_DECODE=1: FlyDSL fused router+sort (change #2)

Change #1 (the atomic stage-2 epilogue) is a tuned-CSV row, so it is selected
through the environment rather than an arm here::

    AITER_CONFIG_FMOE=$(python3 fmoe_chain.py qwen3_5_397b_fp4_tuned_fmoe.atomic.csv) \
        python3 bench_moe_decode.py

Usage:  python3 bench_moe_decode.py [--tokens 16] [--iters 30] [--arms stock,fused]
"""

import argparse
import collections
import os
import sys

import torch

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QTYPE = aiter.QuantType.per_1x32
WQ = dtypes.fp4x2


def build(token, model_dim, inter_dim, E, topk, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    x = torch.randn((token, model_dim), dtype=dtypes.bf16, device=dev)
    logits = torch.randn((token, E), dtype=dtypes.bf16, device=dev)

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
    torch.cuda.empty_cache()
    return x, logits, w1_s, w2_s, w1_scale, w2_scale


def run(tag, token, model_dim, inter_dim, E, topk, iters):
    import aiter.fused_moe as fm

    x, logits, w1, w2, w1s, w2s = build(token, model_dim, inter_dim, E, topk)
    kw = dict(
        w1_scale=w1s,
        w2_scale=w2s,
        quant_type=QTYPE,
        activation=aiter.ActivationType.Silu,
        doweight_stage1=False,
    )

    def step():
        tw, ti = fm.fused_topk(x, logits, topk, True)
        return fm.fused_moe(x, w1, w2, tw, ti, **kw)

    for _ in range(5):
        step()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            step()
        torch.cuda.synchronize()

    per = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    for ev in prof.events():
        if ev.device_type.name != "CUDA" and str(ev.device_type) != "DeviceType.CUDA":
            continue
        d = getattr(ev, "self_device_time_total", 0) or 0
        if d:
            per[ev.key] += d / iters
            cnt[ev.key] += 1

    print(f"\n===== {tag}: token={token} topk={topk} rows={token * topk} E={E} inter_dim={inter_dim} =====")
    tot = 0.0
    launches = 0
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
        if v < 0.5:
            continue
        n = cnt[k] / iters
        tot += v
        launches += n
        print(f"  {v:9.2f} us  x{n:4.1f}  {k[:96]}")
    print(f"  {tot:9.2f} us  x{launches:4.1f}  TOTAL (kernels >0.5us)")
    del x, logits, w1, w2, w1s, w2s
    torch.cuda.empty_cache()
    return tot, launches


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--model-dim", type=int, default=4096)
    ap.add_argument("--inter-dim", type=int, default=256)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()

    if os.environ.get("SGLANG_FLYDSL_MOE_DECODE") == "1":
        import moe_decode_patch

        moe_decode_patch.install()
        tag = "fused (FlyDSL router+sort)"
    else:
        tag = "stock (aiter router + opus sort)"

    for r in range(a.repeat):
        run(
            f"{tag} rep{r}",
            a.tokens,
            a.model_dim,
            a.inter_dim,
            a.experts,
            a.topk,
            iters=a.iters,
        )

    if os.environ.get("SGLANG_FLYDSL_MOE_DECODE") == "1":
        print("patch stats:", moe_decode_patch.stats())
