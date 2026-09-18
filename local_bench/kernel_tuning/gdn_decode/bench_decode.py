"""Tune the GDN decode recurrence (fused_sigmoid_gating_delta_rule_update).

The prefill side of this kernel family has a bench/compare/ATT pipeline under
``chunk_gated_delta_rule/``; the decode side had none. This is it.

What it tunes
-------------
The installed wrapper hardcodes its launch geometry:

    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    num_stages = 3
    num_warps  = 1
    grid       = (NK, NV, N * HV)

At the production shape that is 256 workgroups of a *single* 64-thread wave,
~3% of an MI355X's wave slots. BV is the only knob that changes the grid:
halving it doubles NV and therefore doubles the workgroup count.

How it stays honest
-------------------
SGLang is never edited. We load a source-patched copy of the installed module
(the same trick ``chunk_gated_delta_rule/profile_flydsl.py`` uses for its
"permuted" arm), so the only differences from stock are BV / num_warps /
num_stages, read from the environment at call time.

Every candidate is checked against the stock config's output *and* its
write-back to the state pool before its timing is reported; a config that
mismatches is labelled MISMATCH, one that raises is labelled FAIL. Correctness
always comes from a single call off a pristine state, since the kernel mutates
the pool in place. Timing uses a burst (see ``run_config``) because a single
bracketed call measures launch overhead, not this ~10 us kernel.

Usage
-----
    python3 bench_decode.py                          # sweep, production shape
    python3 bench_decode.py --shapes qwen35-397b-tp4-c4 qwen35-397b-tp4-c32
    python3 bench_decode.py --bv 8 --num-warps 1 --repeats 200   # single config
    python3 bench_decode.py --output /tmp/gdn_decode.json
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import torch

from decode_shapes import SHAPES, DecodeShape

# The stock wrapper's hardcoded geometry, i.e. what production runs today.
STOCK = dict(bv=32, num_warps=1, num_stages=3)

_PATCHES = {
    "    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)": (
        "    BK = triton.next_power_of_2(K)\n"
        "    import os as _os\n"
        "    BV = int(_os.environ.get('GDN_DECODE_BV', 0)) or min(triton.next_power_of_2(V), 32)"
    ),
    "    num_stages = 3": "    num_stages = int(__import__('os').environ.get('GDN_DECODE_STAGES', 3))",
    "    num_warps = 1": "    num_warps = int(__import__('os').environ.get('GDN_DECODE_WARPS', 1))",
}


def load_tunable_module():
    """Import a BV/warps/stages-parameterized copy of the installed wrapper."""
    from sglang.kernels.ops.attention.fla import fused_sigmoid_gating_recurrent as stock

    source = Path(stock.__file__).read_text()
    for before, after in _PATCHES.items():
        if source.count(before) != 1:
            raise SystemExit(
                f"installed sglang source has changed; cannot find exactly one:\n  {before!r}"
            )
        source = source.replace(before, after)
    tmp = tempfile.TemporaryDirectory(prefix="gdn-decode-")
    path = Path(tmp.name) / "gdn_decode_tunable.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("gdn_decode_tunable", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod._tmpdir = tmp  # keep alive
    return mod


def build_inputs(shape: DecodeShape, device, seed: int = 1234):
    """One decode/verify call's worth of tensors, laid out as forward_decode does.

    q/k/v are packed B=1 with cu_seqlens marking sequence starts, which is how
    gdn_backend.forward_decode calls the kernel. q/k are left unnormalized --
    the kernel l2-normalizes internally (USE_QK_L2NORM_IN_KERNEL=True), which is
    also what keeps the k^T v outer product bounded.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    B, T, N = 1, shape.T, shape.batch
    Hg, HV, K, V = shape.Hg, shape.HV, shape.K, shape.V
    bf = torch.bfloat16

    def rn(*sz, dtype=bf):
        return torch.randn(*sz, generator=g, device=device, dtype=torch.float32).to(dtype)

    return dict(
        # A_log/dt_bias are per-v-head fp32 parameters (layer.A_log, layer.dt_bias).
        A_log=rn(HV, dtype=torch.float32),
        dt_bias=rn(HV, dtype=torch.float32),
        q=rn(B, T, Hg, K),
        k=rn(B, T, Hg, K),
        v=rn(B, T, HV, V),
        # a drives softplus gating; b is the beta gate. Both per (token, v-head).
        a=rn(B, T, HV, dtype=torch.float32),
        b=rn(B, T, HV, dtype=torch.float32),
        # Contiguous state pool: stride(0) == HV*V*K, the fast path in the wrapper.
        state=rn(shape.pool_slots, HV, V, K),
        # Distinct slots, deliberately not 0..N-1, so an indexing bug shows up.
        idx=torch.arange(1, N + 1, device=device, dtype=torch.int32) * 3,
        cu=torch.arange(0, N + 1, device=device, dtype=torch.int32) * shape.draft_tokens,
    )


def call(mod, inp, state, cu):
    return mod.fused_sigmoid_gating_delta_rule_update(
        A_log=inp["A_log"],
        a=inp["a"],
        dt_bias=inp["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        b=inp["b"],
        initial_state_source=state,
        initial_state_indices=inp["idx"],
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
    )


def run_config(mod, inp, shape, bv, warps, stages, repeats, warmup, burst):
    """Time one (BV, warps, stages); returns (median_ms_per_call, out, state_after).

    Correctness is taken from a single call off a pristine state. Timing is a
    burst of ``burst`` back-to-back calls inside one event pair, divided by
    ``burst``: this kernel is ~10 us device-side, so bracketing a single call
    measures mostly launch and Python overhead (~45 us) rather than the kernel.
    The state is restored before each burst but deliberately not between calls
    inside it -- the gated delta rule is contractive (h <- h*exp(g), g<0), so it
    settles rather than diverging, and we assert the result stayed finite.
    """
    os.environ["GDN_DECODE_BV"] = str(bv)
    os.environ["GDN_DECODE_WARPS"] = str(warps)
    os.environ["GDN_DECODE_STAGES"] = str(stages)
    pristine, cu = inp["state"], inp["cu"]

    state = pristine.clone()
    out = call(mod, inp, state, cu)
    torch.cuda.synchronize()
    out, state_after = out.clone(), state.clone()

    s = pristine.clone()
    for _ in range(warmup):
        call(mod, inp, s, cu)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        s.copy_(pristine)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(burst):
            last = call(mod, inp, s, cu)
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / burst)
    if not torch.isfinite(last.float()).all():
        raise RuntimeError("burst produced non-finite output; lower --burst")
    return statistics.median(times), out, state_after


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", nargs="+", default=["qwen35-397b-tp4-c4"], choices=tuple(SHAPES))
    p.add_argument("--bv", type=int, nargs="+", default=[64, 32, 16, 8, 4])
    p.add_argument("--num-warps", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--num-stages", type=int, nargs="+", default=[3])
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--burst", type=int, default=50,
                   help="calls per timed window; amortizes ~45us launch overhead "
                        "over a ~10us kernel")
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.02)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output", type=Path)
    a = p.parse_args()

    device = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_properties(device).name
    mod = load_tunable_module()
    results = []

    for shape_name in a.shapes:
        shape = SHAPES[shape_name]
        inp = build_inputs(shape, device)
        print(f"\n=== {shape.name}  batch={shape.batch} draft={shape.draft_tokens} "
              f"T={shape.T} HV={shape.HV} K={shape.K} V={shape.V}  [{name}]")
        print(f"    {shape.note}")
        print(f"    state touched/layer/call: {shape.state_bytes()/2**20:.1f} MiB\n")

        ref_ms, ref_out, ref_state = run_config(
            mod, inp, shape, STOCK["bv"], STOCK["num_warps"], STOCK["num_stages"],
            a.repeats, a.warmup, a.burst)
        print(f"{'BV':>4} {'warps':>6} {'stages':>7} {'WGs':>6} {'wave slot%':>11} "
              f"{'ms':>9} {'speedup':>8}  status")
        print(f"{STOCK['bv']:>4} {STOCK['num_warps']:>6} {STOCK['num_stages']:>7} "
              f"{shape.workgroups(STOCK['bv']):>6} "
              f"{shape.wave_slot_pct(STOCK['bv'], STOCK['num_warps']):>10.1f}% "
              f"{ref_ms:9.4f} {1.0:8.3f}  STOCK (reference)")

        for bv in a.bv:
            if shape.V % bv:
                continue
            for warps in a.num_warps:
                for stages in a.num_stages:
                    if (bv, warps, stages) == (STOCK["bv"], STOCK["num_warps"], STOCK["num_stages"]):
                        continue
                    try:
                        ms, out, state_after = run_config(
                            mod, inp, shape, bv, warps, stages, a.repeats, a.warmup, a.burst)
                    except Exception as exc:
                        print(f"{bv:>4} {warps:>6} {stages:>7} {'':>6} {'':>11} "
                              f"{'':>9} {'':>8}  FAIL {type(exc).__name__}: {str(exc)[:48]}")
                        continue
                    ok_o = torch.allclose(out.float(), ref_out.float(), rtol=a.rtol, atol=a.atol)
                    ok_s = torch.allclose(state_after.float(), ref_state.float(),
                                          rtol=a.rtol, atol=a.atol)
                    status = "ok" if (ok_o and ok_s) else (
                        "MISMATCH out" if not ok_o else "MISMATCH state")
                    print(f"{bv:>4} {warps:>6} {stages:>7} {shape.workgroups(bv):>6} "
                          f"{shape.wave_slot_pct(bv, warps):>10.1f}% {ms:9.4f} "
                          f"{ref_ms/ms:8.3f}  {status}")
                    results.append(dict(shape=shape.name, BV=bv, num_warps=warps,
                                        num_stages=stages, ms=ms, stock_ms=ref_ms,
                                        speedup=ref_ms / ms,
                                        workgroups=shape.workgroups(bv),
                                        wave_slot_pct=shape.wave_slot_pct(bv, warps),
                                        out_ok=ok_o, state_ok=ok_s))
        del inp
        torch.cuda.empty_cache()

    print("\nSpeedup = stock / candidate (>1 favors the candidate). Kernel-only; "
          "no end-to-end claim.")
    if a.output:
        a.output.write_text(json.dumps(
            dict(device=name, stock=STOCK, rtol=a.rtol, atol=a.atol, results=results),
            indent=2))
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
