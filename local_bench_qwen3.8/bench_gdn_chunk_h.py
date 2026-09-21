#!/usr/bin/env python3
"""Compare SGLang's Triton GDN chunk-state kernel against the AITER ones, at
Qwen3.8-Flash-Next shapes.

The kernel under test is the ``h`` (inter-chunk hidden state) stage of the
chunked gated delta rule -- ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``
in SGLang. In the ISL=32768 prefill trace it is the single largest GDN kernel:
54.5 ms of 910.7 ms total prefill GPU time (6.0%), 72 launches = 36 GDN layers
x 2 chunked-prefill chunks. The whole GDN prefill recurrence is ~12.3%.

Backends
--------
  sglang      sglang.kernels.ops.attention.fla.chunk_delta_h
              .chunk_gated_delta_rule_fwd_h                  (baseline)
  flydsl      aiter.ops.flydsl.linear_attention_prefill_kernels
              .chunk_gated_delta_rule_fwd_h_flydsl_opt       (K5 opt, GQA-aware)
  hip         aiter.ops.chunk_gated_delta_rule_fwd_h
              .chunk_gated_delta_rule_fwd_h_hip_fn

Shapes come from the model config, not from guesswork:
  linear_num_key_heads=16 (Hg), linear_num_value_heads=48 (H),
  linear_key_head_dim=linear_value_head_dim=128, chunk size 64.
T defaults to 16384 -- the server runs --chunked-prefill-size 16384, so a
32768-token prompt is two of these, which is exactly what the trace shows.

Layout note (matters for any adoption decision)
-----------------------------------------------
SGLang produces ``w``/``u`` token-major ``[B, T, H, *]``; both AITER entries
want them head-major ``[B, H, T, *]``. That transpose is NOT part of the
kernel, so it is timed separately and reported as ``+xpose``. Adopting an
AITER kernel means either paying it or teaching
``chunk_gated_delta_rule_fwd_intra`` to emit head-major directly.

Usage
-----
  python3 bench_gdn_chunk_h.py                      # correctness + perf, T=16384
  python3 bench_gdn_chunk_h.py --T 2048 4096 8192 16384 32768
  python3 bench_gdn_chunk_h.py --backends sglang flydsl
  FLYDSL_K5_OPT_BV=32 python3 bench_gdn_chunk_h.py  # BV A/B sweep for tuning
  python3 bench_gdn_chunk_h.py --sweep-bv           # try BV in {16,32,64}

Point at a different AITER checkout with AITER_DIR (default /var/home/my_aiter/aiter).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# The new GDN kernels live in a separate AITER checkout; it must win over the
# one already on sys.path (/sgl-workspace/aiter) or we silently test the old one.
AITER_DIR = os.environ.get("AITER_DIR", "/var/home/my_aiter/aiter")
if AITER_DIR and os.path.isdir(AITER_DIR):
    sys.path.insert(0, AITER_DIR)

import torch  # noqa: E402

# ---------------------------------------------------------------- Qwen3.8 shapes
# amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8, text_config
HG = 16          # linear_num_key_heads
H = 48           # linear_num_value_heads   (GQA ratio H/Hg = 3)
K = 128          # linear_key_head_dim
V = 128          # linear_value_head_dim
BT = 64          # CHUNK_SIZE in sglang's fla
DTYPE = torch.bfloat16
# server runs --mamba-ssm-dtype bfloat16
STATE_DTYPE = torch.bfloat16


def _import_backends(names):
    mods = {}
    if "sglang" in names:
        from sglang.kernels.ops.attention.fla.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h,
        )

        mods["sglang"] = chunk_gated_delta_rule_fwd_h
    if "flydsl" in names:
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            chunk_gated_delta_rule_fwd_h_flydsl_opt,
        )

        mods["flydsl"] = chunk_gated_delta_rule_fwd_h_flydsl_opt
    if "hip" in names:
        from aiter.ops.chunk_gated_delta_rule_fwd_h import (
            chunk_gated_delta_rule_fwd_h_hip_fn,
        )

        mods["hip"] = chunk_gated_delta_rule_fwd_h_hip_fn
    return mods


def make_inputs(T: int, device: str, seed: int = 0, nseq: int = 1):
    """Inputs in SGLang's production layout for one prefill forward pass.

    Mirrors the real call in sglang/kernels/ops/attention/fla/chunk.py:
    ``g`` is already chunk-local-cumsum'd, and the state is an indexed pool
    (the mamba pool) rather than a dense [N, ...] tensor.

    ``T`` is tokens PER SEQUENCE and ``nseq`` the number of sequences packed
    into the batch, so the flat token count is T*nseq and cu_seqlens has
    nseq+1 entries. nseq is the lever that matters for occupancy: the kernel's
    grid is (V/BV) x H x nseq, and the chunk loop is sequential, so nseq is the
    only term that can grow without making each workgroup do less useful work.
    """
    torch.manual_seed(seed)
    g_dev = torch.device(device)
    T_flat = T * nseq

    # k token-major [B,T,Hg,K]; w/u token-major [B,T,H,*] -- what fwd_intra emits.
    k = torch.randn(1, T_flat, HG, K, dtype=DTYPE, device=g_dev)
    k = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(DTYPE)
    w = torch.randn(1, T_flat, H, K, dtype=DTYPE, device=g_dev) * 0.1
    u = torch.randn(1, T_flat, H, V, dtype=DTYPE, device=g_dev) * 0.1

    cu_seqlens = torch.tensor([i * T for i in range(nseq + 1)],
                              dtype=torch.long, device=g_dev)

    # Gate: production passes g through chunk_local_cumsum before fwd_h, so do
    # the same -- using sglang's own op, so both backends see identical values.
    from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum

    g_raw = torch.nn.functional.logsigmoid(
        torch.rand(1, T_flat, H, dtype=torch.float32, device=g_dev)
    )
    g = chunk_local_cumsum(g_raw, chunk_size=BT, cu_seqlens=cu_seqlens)

    # Indexed state pool, as the mamba pool is in production: nseq scattered,
    # non-identity slots so the per-sequence gather is actually exercised.
    pool_size = nseq + 7
    pool = torch.randn(pool_size, H, V, K, dtype=STATE_DTYPE, device=g_dev) * 0.1
    indices = torch.randperm(pool_size, device=g_dev)[:nseq].to(torch.int32)

    return dict(k=k, w=w, u=u, g=g, cu_seqlens=cu_seqlens, pool=pool,
                indices=indices, T=T, nseq=nseq, T_flat=T_flat)


def to_head_major(w, u):
    """[B,T,H,*] -> [B,H,T,*] contiguous, as both AITER entries require."""
    return w.transpose(1, 2).contiguous(), u.transpose(1, 2).contiguous()


# fp32->bf16 conversion mode for the FlyDSL kernel. AITER defaults to
# bf16_convert_trunc=True (truncate), which is biased and ~2.6x less accurate
# than round-to-nearest for zero measured speed benefit at these shapes, so the
# harness defaults to RNE. Flip with --bf16-convert-trunc 1.
BF16_CONVERT_TRUNC = False


def run_backend(name, fn, inp, pool, head_major_wu=None):
    """One invocation. Returns (h, v_new, final_state_pool)."""
    k, g, cu = inp["k"], inp["g"], inp["cu_seqlens"]
    if name == "sglang":
        # token-major w/u; returns (h, v_new); writes final state into pool.
        h, v_new = fn(
            k=k, w=inp["w"], u=inp["u"], g=g,
            initial_state=pool,
            initial_state_indices=inp["indices"],
            cu_seqlens=cu,
            save_new_value=True,
        )
        return h, v_new, pool

    w_hm, u_hm = head_major_wu
    h, v_new, ht = fn(
        k=k, w=w_hm, u=u_hm, g=g,
        initial_state=pool,
        initial_state_indices=inp["indices"],
        output_final_state=True,
        cu_seqlens=cu,
        chunk_size=BT,
        save_new_value=True,
        state_dtype=STATE_DTYPE,
        # SGLang's scalar-g path is natural-exp (use_exp2=False is its default
        # and it asserts use_exp2 is only for the gk path). AITER defaults to
        # True, so this must be set explicitly or the gates are interpreted in
        # log2 space and the outputs diverge by a large factor.
        use_exp2=False,
        g_head_major=False,          # g stays token-major [B,T,H], as sglang emits
        # per-sequence lengths; avoids a GPU->host readback of chunk metadata
        seq_lens_cpu=[inp["T"]] * inp.get("nseq", 1),
        **({"bf16_convert_trunc": BF16_CONVERT_TRUNC} if name == "flydsl" else {}),
    )
    # h is [B,NT,H,V,K] in both. v_new, however, follows the w/u convention:
    # sglang emits token-major [B,T,H,V], AITER head-major [B,H,T,V]. Put it
    # back token-major so the comparison is like-for-like -- and note that a
    # real integration would owe this transpose too, or a consumer that reads
    # head-major (chunk_fwd_o is the only consumer).
    if v_new is not None and v_new.dim() == 4 and v_new.shape[1] == H:
        v_new = v_new.transpose(1, 2)
    return h, v_new, ht


def timeit(fn, iters, warmup):
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
    return start.elapsed_time(end) / iters  # ms


def compare(ref, got, label, atol, rtol):
    if ref is None or got is None:
        return f"{label}: n/a"
    r, gf = ref.float(), got.float()
    if r.shape != gf.shape:
        return f"{label}: SHAPE MISMATCH ref{tuple(r.shape)} vs {tuple(gf.shape)}"
    diff = (r - gf).abs()
    denom = r.abs().clamp_min(1e-6)
    maxabs = diff.max().item()
    relmean = (diff / denom).mean().item()
    ok = torch.allclose(r, gf, atol=atol, rtol=rtol)
    return (f"{label}: {'PASS' if ok else 'FAIL'} "
            f"max_abs={maxabs:.3e} mean_rel={relmean:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[16384],
                    help="tokens per call (default 16384 = one prefill chunk)")
    # "hip" is opt-in: in the current /var/home/my_aiter checkout its prebuilt
    # module_chunk_gdr_fwd_h.so has a pybind11 ABI split against
    # module_aiter_core.so and every call raises. Fix with AITER_REBUILD=1 (or
    # rm -rf $AITER_DIR/aiter/jit) and then pass --backends sglang flydsl hip.
    ap.add_argument("--backends", nargs="+",
                    default=["sglang", "flydsl"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--bv", type=int, choices=(16, 32, 64),
                    help="force FLYDSL_K5_OPT_BV for this process")
    ap.add_argument("--sweep-bv", action="store_true",
                    help="sweep BV in {16,32,64}, one subprocess per value")
    ap.add_argument("--bf16-convert-trunc", type=int, choices=(0, 1), default=0,
                    help="flydsl fp32->bf16 mode: 0=round-to-nearest (default "
                         "here, matches sglang accuracy), 1=truncate (AITER's "
                         "own default, ~2.6x more error, no speed gain)")
    ap.add_argument("--skip-correctness", action="store_true")
    args = ap.parse_args()

    global BF16_CONVERT_TRUNC
    BF16_CONVERT_TRUNC = bool(args.bf16_convert_trunc)

    # aiter reads FLYDSL_K5_OPT_BV once, at module-import time
    # (linear_attention_prefill_kernels.py: _OPT_BV_ENV = os.environ.get(...)).
    # So it must be set before _import_backends, and a sweep needs a fresh
    # interpreter per value -- mutating os.environ mid-run silently re-measures
    # whatever BV was live at import.
    if args.bv is not None:
        os.environ["FLYDSL_K5_OPT_BV"] = str(args.bv)

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    print(f"device      : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"gfx / CUs   : {getattr(props,'gcnArchName','?')} / {props.multi_processor_count}")
    print(f"shapes      : Hg={HG} H={H} K={K} V={V} BT={BT} dtype={DTYPE} "
          f"state={STATE_DTYPE}")
    print(f"aiter       : {AITER_DIR}")

    fns = _import_backends(args.backends)
    import aiter
    print(f"aiter module: {aiter.__file__}\n")

    for T in args.T:
        inp = make_inputs(T, "cuda")
        w_hm, u_hm = to_head_major(inp["w"], inp["u"])
        nt = (T + BT - 1) // BT
        broken: set[str] = set()
        print(f"===== T={T}  ({nt} chunks of {BT}) =====")

        # ---- correctness: every backend against sglang, fresh pool each time
        ref = None
        if not args.skip_correctness and "sglang" in fns:
            ref = run_backend("sglang", fns["sglang"], inp, inp["pool"].clone())
        for name in args.backends:
            if name == "sglang" or name not in fns:
                continue
            if ref is None:
                break
            print(f"  [{name}] vs sglang")
            try:
                got = run_backend(name, fns[name], inp, inp["pool"].clone(),
                                  (w_hm, u_hm))
            except Exception as exc:  # a broken backend must not kill the run
                print(f"      ERROR: {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:150]}")
                broken.add(name)
                continue
            for i, lbl in enumerate(("h      ", "v_new  ", "state  ")):
                print(f"      {compare(ref[i], got[i], lbl, args.atol, args.rtol)}")

        # ---- perf
        print("  --- perf (mean over "
              f"{args.iters} iters, {args.warmup} warmup) ---")
        base = None
        xpose_ms = timeit(lambda: to_head_major(inp["w"], inp["u"]),
                          args.iters, args.warmup)
        for name in args.backends:
            if name not in fns or name in broken:
                continue
            pool = inp["pool"].clone()
            hm = (w_hm, u_hm)
            try:
                ms = timeit(lambda n=name, p=pool, h=hm:
                            run_backend(n, fns[n], inp, p, h),
                            args.iters, args.warmup)
            except Exception as exc:
                print(f"      {name:<8} ERROR: {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:120]}")
                continue
            if name == "sglang":
                base = ms
            tag = f"{ms:8.3f} ms"
            if base:
                tag += f"   {base/ms:5.2f}x vs sglang"
            if name != "sglang":
                tot = ms + xpose_ms
                tag += f"   (+xpose {xpose_ms:.3f} -> {tot:.3f} ms"
                if base:
                    tag += f", {base/tot:.2f}x"
                tag += ")"
            print(f"      {name:<8} {tag}")

        if args.sweep_bv:
            import re
            import subprocess

            print("  --- BV sweep (flydsl, kernel only; one process per BV) ---")
            for bv in (16, 32, 64):
                cmd = [sys.executable, os.path.abspath(__file__),
                       "--T", str(T), "--backends", "flydsl",
                       "--skip-correctness", "--bv", str(bv),
                       "--iters", str(args.iters), "--warmup", str(args.warmup)]
                res = subprocess.run(cmd, capture_output=True, text=True)
                m = re.search(r"^\s*flydsl\s+([0-9.]+) ms", res.stdout, re.M)
                if not m:
                    err = (res.stderr or res.stdout).strip().splitlines()
                    print(f"      BV={bv:<3} FAILED: {err[-1][:110] if err else '?'}")
                    continue
                ms = float(m.group(1))
                line = f"      BV={bv:<3} {ms:8.3f} ms"
                if base:
                    line += f"   {base/ms:5.2f}x vs sglang"
                print(line)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
