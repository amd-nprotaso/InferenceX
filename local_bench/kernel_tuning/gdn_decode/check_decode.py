"""Validate the FlyDSL GDN decode kernel against SGLang's Triton reference.

Checks the returned output *and* the in-place write-back to the state pool, on
every shape in decode_shapes plus ragged / sentinel-slot / single-token edge
cases. The state pool is restored from a pristine copy before each side runs,
since both kernels mutate it.

    python3 check_decode.py
    python3 check_decode.py --rtol 0.02 --atol 0.02 --verbose
"""

import argparse
from dataclasses import dataclass

import torch

from bench_decode import build_inputs
from decode_shapes import SHAPES, DecodeShape
import gdn_decode_flydsl as fly


@dataclass
class Case:
    name: str
    shape: DecodeShape
    sentinel: tuple = ()      # sequence positions given a -1 state slot
    verify: bool = False      # EAGLE target_verify: snapshot each draft step,
                              # no final state write-back


def reference_fn():
    """The unpatched sglang entry point; refuses to run against a hook."""
    import importlib
    mod = importlib.import_module(
        "sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent")
    fn = mod.fused_sigmoid_gating_delta_rule_update
    if fn.__module__ != mod.__name__:
        raise SystemExit(f"sglang entry point is patched ({fn.__module__})")
    return fn


def run_case(case: Case, device, rtol, atol, verbose) -> bool:
    ref = reference_fn()
    inp = build_inputs(case.shape, device)
    idx = inp["idx"].clone()
    for pos in case.sentinel:
        idx[pos] = -1

    common = dict(
        A_log=inp["A_log"], a=inp["a"], dt_bias=inp["dt_bias"],
        softplus_beta=1.0, softplus_threshold=20.0,
        q=inp["q"], k=inp["k"], v=inp["v"], b=inp["b"],
        initial_state_indices=idx, cu_seqlens=inp["cu"],
        use_qk_l2norm_in_kernel=True,
    )
    shape = case.shape
    c_ref = c_fly = None
    if case.verify:
        # Mirror gdn_triton.target_verify: no state write-back, one state
        # snapshot per draft step into [slots, steps, HV, V, K].
        steps = shape.draft_tokens
        buf = torch.randn(8, steps, shape.HV, shape.V, shape.K,
                          device=device, dtype=torch.bfloat16)
        cidx = torch.arange(shape.batch, device=device, dtype=torch.int32)
        common.update(disable_state_update=True,
                      intermediate_state_indices=cidx,
                      cache_steps=steps)
        c_ref, c_fly = buf.clone(), buf.clone()

    s_ref = inp["state"].clone()
    o_ref = ref(initial_state_source=s_ref,
                **(dict(common, intermediate_states_buffer=c_ref) if case.verify else common))
    s_fly = inp["state"].clone()
    o_fly = fly.fused_sigmoid_gating_delta_rule_update(
        initial_state_source=s_fly,
        **(dict(common, intermediate_states_buffer=c_fly) if case.verify else common))
    torch.cuda.synchronize()

    o_ref = o_ref.reshape(o_fly.shape).float()
    checks = [("out", o_ref, o_fly.float()), ("state", s_ref.float(), s_fly.float())]
    if case.verify:
        checks.append(("snapshots", c_ref.float(), c_fly.float()))
    reports, ok = [], True
    for label, r, g in checks:
        err = (r - g).abs().max().item()
        good = torch.allclose(r, g, rtol=rtol, atol=atol)
        ok &= good
        reports.append(f"{label} maxerr={err:.3e}")
    print(f"[{'PASS' if ok else 'FAIL'}] {case.name:<40} "
          f"n={case.shape.batch} T={case.shape.T}  " + "  ".join(reports))
    if verbose and not ok:
        print(f"        sentinel={case.sentinel} note={case.shape.note}")
    return ok


def cases() -> list[Case]:
    out = [Case(name, SHAPES[name]) for name in SHAPES]
    out.append(Case("c4 with slot -1 on seq 0", SHAPES["qwen35-397b-tp4-c4"], sentinel=(0,)))
    out.append(Case("c4 with all slots -1", SHAPES["qwen35-397b-tp4-c4"], sentinel=(0, 1, 2, 3)))
    # The EAGLE verify path is what the server actually runs; test it explicitly.
    out.append(Case("c4 target_verify", SHAPES["qwen35-397b-tp4-c4"], verify=True))
    out.append(Case("c8 target_verify", SHAPES["qwen35-397b-tp4-c8"], verify=True))
    out.append(Case("c4 target_verify, slot -1 seq 0", SHAPES["qwen35-397b-tp4-c4"],
                    sentinel=(0,), verify=True))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.02)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    device = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(device)
    print(f"device={torch.cuda.get_device_properties(device).name} "
          f"KSPLIT={fly.KSPLIT} VTILE={fly.VTILE} KLOCAL={fly.KLOCAL} "
          f"tol rtol={a.rtol} atol={a.atol}\n")
    results = [run_case(c, device, a.rtol, a.atol, a.verbose) for c in cases()]
    print(f"\n{sum(results)}/{len(results)} passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
