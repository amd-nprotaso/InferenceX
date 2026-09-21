#!/usr/bin/env python3
"""Does flydsl_gdn_decode_varlen stay correct on fused-QKV views?

The Qwen3.8-Flash-Next GDN layer hands SGLang three *views* into one fused
``[1, T, (Hg + Hg + HV) * D]`` projection buffer, so q/k/v carry a token stride
of the fused width (10240) rather than their own. The AITER wrapper used to
require ``is_contiguous()`` and rejected them; the kernel itself takes per-tensor
token strides, so the guard was stricter than the kernel.

This checks the relaxed guard three ways at the real serving shapes:
  1. strided views vs contiguous copies through the same AITER kernel
     -- isolates stride handling from everything else;
  2. AITER vs the Triton recurrence SGLang would otherwise run
     -- absolute correctness of the output;
  3. the ssm-state snapshots written into intermediate_states
     -- the verify path commits the accepted prefix from these, so a correct
     output tensor with wrong snapshots would still corrupt generation.
"""

import torch

from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdn_decode_varlen
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)

# Observed live on the server at concurrency 16 (see logs/server_diag2.log).
BS, DRAFT = 16, 4
T = BS * DRAFT
HG, HV, D = 16, 48, 128
SLOTS = 335
DEV = "cuda"


def build():
    torch.manual_seed(0)
    fused_width = (HG + HG + HV) * D  # 10240
    mixed = torch.randn(1, T, fused_width, device=DEV, dtype=torch.bfloat16) * 0.1
    # Exactly how the GDN layer splits its conv output.
    q = mixed[..., : HG * D].view(1, T, HG, D)
    k = mixed[..., HG * D : 2 * HG * D].view(1, T, HG, D)
    v = mixed[..., 2 * HG * D :].view(1, T, HV, D)
    a = (torch.randn(T, HV, device=DEV, dtype=torch.bfloat16) * 0.1)
    b = (torch.randn(T, HV, device=DEV, dtype=torch.bfloat16) * 0.1)
    A_log = torch.randn(HV, device=DEV, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device=DEV, dtype=torch.float32) * 0.1
    state = torch.randn(SLOTS, HV, D, D, device=DEV, dtype=torch.bfloat16) * 0.05
    idx = torch.arange(BS, device=DEV, dtype=torch.int32)
    cu = torch.arange(0, T + 1, DRAFT, device=DEV, dtype=torch.int32)
    return q, k, v, a, b, A_log, dt_bias, state, idx, cu


def run_aiter(q, k, v, a, b, A_log, dt_bias, state, idx, cu):
    inter = torch.zeros(BS + 1, DRAFT, HV, D, D, device=DEV, dtype=torch.bfloat16)
    out = flydsl_gdn_decode_varlen(
        q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        state=state.clone(), state_indices=idx, cu_seqlens=cu,
        softplus_beta=1.0, softplus_threshold=20.0,
        disable_state_update=True,
        intermediate_states=inter,
        intermediate_state_indices=idx,
    )
    return out, inter


def run_triton(q, k, v, a, b, A_log, dt_bias, state, idx, cu):
    inter = torch.zeros(BS + 1, DRAFT, HV, D, D, device=DEV, dtype=torch.bfloat16)
    out = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, dt_bias=dt_bias, q=q, k=k, v=v, a=a, b=b,
        initial_state_source=state.clone(), initial_state_indices=idx,
        cu_seqlens=cu, use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0, softplus_threshold=20.0, is_kda=False,
        disable_state_update=True,
        intermediate_states_buffer=inter,
        intermediate_state_indices=idx,
        cache_steps=DRAFT,
        retrieve_parent_token=None,
    )
    return out, inter


def report(name, x, y):
    x, y = x.float(), y.float()
    denom = y.abs().mean().clamp_min(1e-6)
    max_abs = (x - y).abs().max().item()
    rel = ((x - y).abs().mean() / denom).item()
    print(f"  {name:<34} max|diff|={max_abs:.5f}  mean-rel={rel:.5f}")
    return max_abs, rel


def main():
    t = build()
    q, k, v = t[0], t[1], t[2]
    print(f"q stride={q.stride()} contiguous={q.is_contiguous()}")
    print(f"v stride={v.stride()} contiguous={v.is_contiguous()}")
    assert not q.is_contiguous(), "test is meaningless if the views are contiguous"

    out_view, inter_view = run_aiter(*t)
    # Same math, contiguous inputs: the path the old guard allowed.
    t_contig = (q.contiguous(), k.contiguous(), v.contiguous()) + t[3:]
    out_contig, inter_contig = run_aiter(*t_contig)
    out_tri, inter_tri = run_triton(*t)

    print("\n1. AITER strided views vs AITER contiguous copies (stride handling)")
    a1 = report("out", out_view, out_contig)
    a2 = report("intermediate_states", inter_view, inter_contig)

    print("\n2. AITER strided views vs Triton reference (absolute correctness)")
    b1 = report("out", out_view, out_tri)
    b2 = report("intermediate_states", inter_view, inter_tri)

    # 1 must be bit-identical: same kernel, same values, only addressing differs.
    ok_stride = a1[0] == 0.0 and a2[0] == 0.0
    # 2 is bf16 accumulation order, so compare on a tolerance.
    ok_abs = b1[1] < 0.02 and b2[1] < 0.02

    print()
    print(f"stride handling bit-identical : {'PASS' if ok_stride else 'FAIL'}")
    print(f"matches Triton within 2% rel  : {'PASS' if ok_abs else 'FAIL'}")
    raise SystemExit(0 if (ok_stride and ok_abs) else 1)


if __name__ == "__main__":
    main()
