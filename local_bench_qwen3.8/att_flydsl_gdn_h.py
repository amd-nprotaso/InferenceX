#!/usr/bin/env python3
"""Standalone driver for the AITER FlyDSL GDN chunk-state kernel, for ATT capture.

Runs ONLY chunk_gated_delta_rule_fwd_h_flydsl_opt -- no sglang, no server, no
other GPU work -- at Qwen3.8-Flash-Next shapes (Hg=16, H=48, K=V=128, BT=64,
bf16, bf16 state), so an Advanced Thread Trace contains just this kernel.

ATT volume is the reason for the structure here: warmup dispatches happen
first (FlyDSL JIT + BV selection + any lazy allocs), then exactly
--profile-iters dispatches of the real thing. Pair with
--kernel-iteration-range so only the post-warmup dispatch is traced:

  rocprofv3 --att=true \
            --att-library-path /opt/rocm/lib \
            --kernel-include-regex 'kernel_0' \
            --kernel-iteration-range 4 \
            -d att_out \
            -- python3 att_flydsl_gdn_h.py

  # smaller trace if the 256MB ATT buffer overflows:
  ... -- python3 att_flydsl_gdn_h.py --T 2048

Env: AITER_DIR (default /var/home/my_aiter/aiter),
     FLYDSL_K5_OPT_BV (16|32|64) to pin the block-V tile.
"""

from __future__ import annotations

import argparse
import os
import sys

AITER_DIR = os.environ.get("AITER_DIR", "/var/home/my_aiter/aiter")
if AITER_DIR and os.path.isdir(AITER_DIR):
    sys.path.insert(0, AITER_DIR)

import torch  # noqa: E402

# Qwen3.8-Flash-Next text_config
HG, H, K, V, BT = 16, 48, 128, 128, 64
DTYPE = torch.bfloat16
STATE_DTYPE = torch.bfloat16


def build(T: int, device="cuda", seed=0):
    torch.manual_seed(seed)
    dev = torch.device(device)
    k = torch.randn(1, T, HG, K, dtype=torch.float32, device=dev)
    k = torch.nn.functional.normalize(k, p=2, dim=-1).to(DTYPE)
    # head-major [B,H,T,*], which is what the AITER entry requires
    w = (torch.randn(1, H, T, K, dtype=DTYPE, device=dev) * 0.1).contiguous()
    u = (torch.randn(1, H, T, V, dtype=DTYPE, device=dev) * 0.1).contiguous()
    cu = torch.tensor([0, T], dtype=torch.long, device=dev)

    from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum

    g_raw = torch.nn.functional.logsigmoid(
        torch.rand(1, T, H, dtype=torch.float32, device=dev)
    )
    g = chunk_local_cumsum(g_raw, chunk_size=BT, cu_seqlens=cu)

    pool = torch.randn(8, H, V, K, dtype=STATE_DTYPE, device=dev) * 0.1
    idx = torch.tensor([3], dtype=torch.int32, device=dev)
    return dict(k=k, w=w, u=u, g=g, cu=cu, pool=pool, idx=idx, T=T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=16384,
                    help="tokens (default 16384 = one prefill chunk)")
    ap.add_argument("--warmup-iters", type=int, default=3)
    ap.add_argument("--profile-iters", type=int, default=1)
    ap.add_argument("--trunc", type=int, choices=(0, 1), default=0,
                    help="bf16_convert_trunc; 0 = round-to-nearest (matches "
                         "sglang bit-for-bit), 1 = AITER default")
    ap.add_argument("--time", action="store_true",
                    help="also report wall time (skip under ATT)")
    args = ap.parse_args()

    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        chunk_gated_delta_rule_fwd_h_flydsl_opt as fwd_h,
    )

    inp = build(args.T)
    print(f"shapes Hg={HG} H={H} K={K} V={V} BT={BT} T={args.T} "
          f"({(args.T + BT - 1)//BT} chunks) dtype={DTYPE}")
    print(f"BV override: {os.environ.get('FLYDSL_K5_OPT_BV', '<auto>')}  "
          f"bf16_convert_trunc={bool(args.trunc)}")

    def call():
        return fwd_h(
            k=inp["k"], w=inp["w"], u=inp["u"], g=inp["g"],
            initial_state=inp["pool"], initial_state_indices=inp["idx"],
            output_final_state=True, cu_seqlens=inp["cu"], chunk_size=BT,
            save_new_value=True, state_dtype=STATE_DTYPE,
            use_exp2=False, g_head_major=False, seq_lens_cpu=[inp["T"]],
            bf16_convert_trunc=bool(args.trunc),
        )

    for _ in range(args.warmup_iters):
        call()
    torch.cuda.synchronize()
    print(f"warmup done ({args.warmup_iters} dispatches)")

    if args.time:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(args.profile_iters):
            call()
        e.record()
        torch.cuda.synchronize()
        print(f"profiled {args.profile_iters} dispatches: "
              f"{s.elapsed_time(e)/args.profile_iters:.3f} ms each")
    else:
        for _ in range(args.profile_iters):
            call()
        torch.cuda.synchronize()
        print(f"profiled {args.profile_iters} dispatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
