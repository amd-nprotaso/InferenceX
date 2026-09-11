#!/usr/bin/env python3
"""Correctness gate for the varlen FlyDSL chunk_h kernel.

`compare_chunk_h.py` only builds one packed sequence whose length is a multiple
of 64 -- exactly the shape the old kernel was restricted to. This checks the
cases that restriction used to exclude: ragged token counts and several packed
sequences per call.

Run with the FlyDSL bootstrap DISABLED, so `sglang`'s own Triton kernel is the
reference rather than the thing under test:

    python3 check_varlen.py
    python3 check_varlen.py --verbose

Compares h, v_new and the in-place state update. v_new is compared over the WHOLE
tensor, not per sequence: a tail store that escapes its own sequence lands in the
next one's rows, and that is the most likely silent failure mode.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass

import torch

CHUNK = 64


@dataclass
class Case:
    name: str
    seqlens: list[int]
    Hg: int = 4
    H: int = 16
    K: int = 128
    V: int = 128
    use_state: bool = True
    save_new_value: bool = True
    use_gk: bool = False
    sentinel: tuple[int, ...] = ()   # sequence positions given a -1 state slot


def reference_fn():
    """The unpatched sglang entry point; refuses to run against the hook."""
    mod = importlib.import_module("sglang.kernels.ops.attention.fla.chunk_delta_h")
    fn = mod.chunk_gated_delta_rule_fwd_h
    if fn.__qualname__ != "chunk_gated_delta_rule_fwd_h":
        raise SystemExit(
            f"sglang entry point is patched ({fn.__qualname__}); unset "
            "SGLANG_FLYDSL_GDN_CHUNK_H and drop bootstrap/ from PYTHONPATH so the "
            "reference is Triton, not FlyDSL."
        )
    return fn


def build_inputs(case: Case, device: torch.device, seed: int, gate_scale: float = 0.1):
    """Same production preamble as shapes.build_inputs, generalized to N sequences.

    w and u are not free parameters: they must come out of
    chunk_gated_delta_rule_fwd_intra or the delta rule they encode is unstable
    and h returns NaN, which would make any comparison meaningless.
    """
    from sglang.kernels.ops.attention.fla.chunk_fwd import (
        chunk_gated_delta_rule_fwd_intra,
    )
    from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
    from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

    gen = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.bfloat16
    T = sum(case.seqlens)
    n = len(case.seqlens)

    def randn(*size):
        return torch.randn(*size, generator=gen, device=device, dtype=torch.float32)

    cu = torch.tensor([0, *torch.tensor(case.seqlens).cumsum(0).tolist()], device=device, dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu, CHUNK)

    k = l2norm_fwd(randn(1, T, case.Hg, case.K).to(dtype)).view(1, T, case.Hg, case.K)
    v = (randn(1, T, case.H, case.V) * 0.5).to(dtype)
    beta = torch.sigmoid(randn(1, T, case.H)).to(dtype)

    g = -torch.nn.functional.softplus(randn(1, T, case.H)) * gate_scale
    g = chunk_local_cumsum(g.contiguous(), chunk_size=CHUNK, cu_seqlens=cu, chunk_indices=chunk_indices)

    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g, beta=beta, cu_seqlens=cu, chunk_size=CHUNK, chunk_indices=chunk_indices
    )

    gk = None
    if case.use_gk:
        # Synthetic per-channel gate. Physically inconsistent with the g-derived
        # w/u, but both kernels see identical inputs, so equivalence still holds.
        gk = (-torch.nn.functional.softplus(randn(1, T, case.H, case.K)) * 0.02).contiguous()
        g = None

    state = None
    if case.use_state:
        state = torch.randn(n + 4, case.H, case.V, case.K, generator=gen, device=device, dtype=torch.float32).to(dtype)
        state = state.contiguous() * 0.1
    idx = torch.arange(n, device=device, dtype=torch.int32)
    for pos in case.sentinel:
        idx[pos] = -1
    return dict(k=k, w=w, u=u, g=g, gk=gk, state=state, idx=idx, cu=cu, chunk_indices=chunk_indices)


def diff(a, b):
    if a is None and b is None:
        return 0.0, True
    if (a is None) != (b is None):
        return float("inf"), False
    a32, b32 = a.float(), b.float()
    if not torch.isfinite(a32).all() or not torch.isfinite(b32).all():
        return float("nan"), False
    return (a32 - b32).abs().max().item(), True


def run_case(case: Case, device: torch.device, rtol: float, atol: float, verbose: bool) -> bool:
    ref = reference_fn()
    fly = importlib.import_module("chunk_delta_h_flydsl").chunk_gated_delta_rule_fwd_h
    inp = build_inputs(case, device, seed=1234)

    state_ref = inp["state"].clone() if inp["state"] is not None else None
    state_fly = inp["state"].clone() if inp["state"] is not None else None

    common = dict(
        k=inp["k"],
        w=inp["w"],
        u=inp["u"],
        g=inp["g"],
        gk=inp["gk"],
        initial_state_indices=inp["idx"],
        save_new_value=case.save_new_value,
        cu_seqlens=inp["cu"],
        chunk_indices=inp["chunk_indices"],
        use_exp2=False,
    )
    h_ref, vn_ref = ref(initial_state=state_ref, **common)
    h_fly, vn_fly = fly(initial_state=state_fly, **common)

    checks = [("h", h_ref, h_fly), ("v_new", vn_ref, vn_fly)]
    if state_ref is not None:
        checks.append(("state", state_ref, state_fly))

    ok = True
    report = []
    for name, a, b in checks:
        if a is not None and b is not None and a.shape != b.shape:
            report.append(f"{name}: SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
            ok = False
            continue
        maxerr, finite = diff(a, b)
        close = finite and (a is None or torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol))
        report.append(f"{name} maxerr={maxerr:.3e}")
        ok = ok and close
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {case.name:<44} T={sum(case.seqlens):<6} n={len(case.seqlens)}  " + "  ".join(report))
    if verbose and not ok:
        print(f"        seqlens={case.seqlens} sentinel={case.sentinel} gk={case.use_gk}")
    return ok


def cases() -> list[Case]:
    return [
        # regression: what the old kernel already supported
        Case("aligned single seq", [4096]),
        Case("aligned single seq 32k", [32768]),
        # the shape that made the old guard reject 100% of production calls
        Case("ragged single seq", [29618]),
        Case("ragged single seq (T%64==1)", [4097]),
        Case("ragged single seq (T%64==63)", [4159]),
        # shorter than one chunk
        Case("sub-chunk seq", [17]),
        Case("exactly one chunk", [64]),
        # multi-sequence: what --prefill-max-requests 1 was working around
        Case("2 seqs aligned", [2048, 4096]),
        Case("2 seqs ragged", [26347, 3271]),
        Case("3 seqs ragged", [12345, 6789, 1011]),
        Case("4 seqs mixed", [64, 17, 8192, 1234]),
        Case("production-like 3 seqs", [27074, 3690, 2004]),
        # option matrix
        Case("no save_new_value", [5000, 3000], save_new_value=False),
        Case("state slot -1 on seq 0", [4096, 1000], sentinel=(0,)),
        Case("all slots -1", [3000, 2000], sentinel=(0, 1)),
        Case("gk path", [4096, 1500], use_gk=True),
        # No `initial_state=None` case: the Triton reference dereferences the
        # state pointer unconditionally once initial_state_indices is passed
        # (chunk_delta_h.py:127) and fails to compile, so there is nothing to
        # compare against. Production always passes the ssm_states pool.
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtol", type=float, default=0.016)
    ap.add_argument("--atol", type=float, default=0.016)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--filter", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}")
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    print(f"device={torch.cuda.get_device_name(device)} arch={arch} tol rtol={args.rtol} atol={args.atol}\n")

    selected = [c for c in cases() if args.filter in c.name]
    results = [run_case(c, device, args.rtol, args.atol, args.verbose) for c in selected]
    failed = sum(1 for r in results if not r)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
