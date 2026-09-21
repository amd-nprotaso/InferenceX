#!/usr/bin/env python3
"""Is the AITER FlyDSL chunk-state kernel less accurate than SGLang's Triton one?

bench_gdn_chunk_h.py compares flydsl *against sglang*, which only shows that the
two differ -- it cannot say which is closer to the truth. This script scores both
against an independent fp32 PyTorch reference of the same recurrence.

The reference is transcribed from the Triton kernel body
(sglang/kernels/ops/attention/fla/chunk_delta_h.py:53
 chunk_gated_delta_rule_fwd_kernel_h_blockdim64), per chunk of BT=64:

    h[t]  = state                                  # snapshot BEFORE the update
    bv    = v[t] - w[t] @ state^T
    vnew  = bv
    bv   *= exp(g_last - g[t])
    state*= exp(g_last)
    state+= (k[t]^T @ bv)^T

Two reference modes disentangle "algorithm" from "storage precision":
  fp32   -- everything in fp32; ground truth.
  emul   -- fp32 accumulator, but bf16 rounding at exactly the points the Triton
            kernel rounds (h snapshot, v_new, the state operand of the w@state
            matmul, and bv before the rank update). If a kernel sits near `emul`
            it is doing the same thing the reference kernel does; the gap to
            `fp32` is the format's cost, not the kernel's fault.

Metrics: relative Frobenius error ||x-ref||/||ref|| (scale-free and not
dominated by near-zero elements, unlike a naive mean of elementwise ratios),
plus max abs error expressed in bf16 ULP at the local magnitude.

  python3 check_gdn_chunk_h_accuracy.py
  python3 check_gdn_chunk_h_accuracy.py --T 512 2048 8192
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
    BT, DTYPE, H, HG, K, STATE_DTYPE, V,
    _import_backends, make_inputs, run_backend, to_head_major,
)


def reference(inp, emulate_bf16: bool):
    """fp32 transcription of the Triton h recurrence. Returns (h, v_new, state)."""
    T = inp["T"]
    dev = inp["k"].device
    nt = (T + BT - 1) // BT

    def r(x):  # bf16 round-trip, only in emulation mode
        return x.bfloat16().float() if emulate_bf16 else x

    # head h reads k-head h // (H // Hg)  -- see the kernel's i_h // (H // Hg)
    kx = inp["k"][0].float().repeat_interleave(H // HG, dim=1)   # [T,H,K]
    w_ = inp["w"][0].float()                                      # [T,H,K]
    v_ = inp["u"][0].float()                                      # [T,H,V]
    g_ = inp["g"][0].float()                                      # [T,H]

    state = inp["pool"][inp["indices"].long()][0].float().clone()  # [H,V,K]

    h_out = torch.empty(nt, H, V, K, dtype=torch.float32, device=dev)
    v_out = torch.empty(T, H, V, dtype=torch.float32, device=dev)

    for t in range(nt):
        sl = slice(t * BT, min((t + 1) * BT, T))
        h_out[t] = r(state)
        # b_v = v - w @ state^T ; the kernel feeds the matmul a bf16 state
        bv = v_[sl] - torch.einsum("thk,hvk->thv", w_[sl], r(state))
        v_out[sl] = r(bv)
        g_last = g_[sl][-1]                                       # [H]
        bv = r(bv) * torch.exp(g_last[None, :, None] - g_[sl][:, :, None])
        state = state * torch.exp(g_last)[:, None, None]
        state = state + torch.einsum("thv,thk->hvk", r(bv), kx[sl])

    return h_out.unsqueeze(0), v_out.unsqueeze(0), r(state).unsqueeze(0)


def score(x, ref, label):
    if x is None:
        return f"{label:<7} n/a"
    a = x.float().reshape(-1)
    b = ref.float().reshape(-1)
    if a.numel() != b.numel():
        return f"{label:<7} SHAPE {tuple(x.shape)} vs {tuple(ref.shape)}"
    rel = (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()
    maxabs = (a - b).abs().max().item()
    # bf16 ULP at the magnitude where the worst error occurs
    i = (a - b).abs().argmax()
    mag = max(abs(b[i].item()), 1e-30)
    ulp = 2.0 ** (torch.floor(torch.log2(torch.tensor(mag))).item() - 7)
    return (f"{label:<7} rel_fro={rel:.3e}  max_abs={maxabs:.3e} "
            f"({maxabs/ulp:5.2f} bf16 ULP)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[512, 2048, 8192])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1
    print(f"device : {torch.cuda.get_device_name(0)}")
    print(f"shapes : Hg={HG} H={H} K={K} V={V} BT={BT} dtype={DTYPE} "
          f"state={STATE_DTYPE}\n")

    fns = _import_backends(["sglang", "flydsl"])

    for T in args.T:
        inp = make_inputs(T, "cuda")
        w_hm, u_hm = to_head_major(inp["w"], inp["u"])
        print(f"===== T={T} ({(T+BT-1)//BT} chunks) =====")

        ref32 = reference(inp, emulate_bf16=False)
        refemu = reference(inp, emulate_bf16=True)

        got = {
            "sglang": run_backend("sglang", fns["sglang"], inp, inp["pool"].clone()),
            "flydsl": run_backend("flydsl", fns["flydsl"], inp, inp["pool"].clone(),
                                  (w_hm, u_hm)),
        }
        # Both write the final state back in place into the pool and return the
        # whole pool; the reference returns just this sequence's slot.
        idx = inp["indices"].long()
        got = {n: (t[0], t[1], t[2][idx]) for n, t in got.items()}

        for ridx, rname, ref in ((0, "fp32 truth", ref32), (1, "bf16-emulated", refemu)):
            print(f"  --- vs {rname} reference ---")
            for i, tname in enumerate(("h", "v_new", "state")):
                line = f"    {tname:<6}"
                for bname in ("sglang", "flydsl"):
                    line += "  | " + score(got[bname][i], ref[i], bname)
                print(line)
        # direct: emulated reference vs fp32, i.e. what bf16 storage alone costs
        print("  --- cost of bf16 storage alone (emulated ref vs fp32 ref) ---")
        for i, tname in enumerate(("h", "v_new", "state")):
            print(f"    {tname:<6}  | {score(refemu[i], ref32[i], 'bf16fmt')}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
