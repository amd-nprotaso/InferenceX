"""Time and verify one Triton config of chunk_gated_delta_rule_fwd_kernel_h_blockdim64.

This drives the *production* kernel -- it imports sglang's
`chunk_gated_delta_rule_fwd_h` rather than a copy, so whatever you measure here
is what the server runs.

How the config is applied
-------------------------
The kernel is wrapped in `@triton.autotune` with a single hardcoded config built
from three env vars read at import time (chunk_delta_h.py:23-26):

    SGLANG_GDN_CHUNK_H_BV / _NUM_WARPS / _NUM_STAGES

Re-reading those would mean a fresh process per config. Instead we swap
`Autotuner.configs` in place. That is exact, not a hack: `Autotuner.run` takes
the `len(self.configs) > 1` branch only when there is something to choose
between, and otherwise goes straight to `config = self.configs[0]`
(triton/runtime/autotuner.py:220-262). With one config there is no benchmarking
pass and no cache lookup, so nothing stale can leak between configs.

waves_per_eu / matrix_instr_nonkdim are AMD backend compile options
(triton/backends/amd/compiler.py:47,64), not kernel parameters. They ride along
in Config.kwargs and get parsed out by the backend at launch. Some combinations
fail to compile; callers get a `failed` result rather than an exception.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import triton

from shapes import Inputs, Shape

# Production defaults, from chunk_delta_h.py:23-26. Every speedup is relative
# to this triple.
BASELINE = dict(BV=32, num_warps=4, num_stages=2)


@dataclass
class Config:
    BV: int
    num_warps: int
    num_stages: int
    waves_per_eu: Optional[int] = None
    matrix_instr_nonkdim: Optional[int] = None

    def label(self) -> str:
        s = f"BV={self.BV} w={self.num_warps} s={self.num_stages}"
        if self.waves_per_eu is not None:
            s += f" wpe={self.waves_per_eu}"
        if self.matrix_instr_nonkdim is not None:
            s += f" mfma={self.matrix_instr_nonkdim}"
        return s

    def is_baseline(self) -> bool:
        return (
            self.BV == BASELINE["BV"]
            and self.num_warps == BASELINE["num_warps"]
            and self.num_stages == BASELINE["num_stages"]
            and self.waves_per_eu is None
            and self.matrix_instr_nonkdim is None
        )

    def to_triton(self) -> triton.Config:
        kwargs = {"BV": self.BV}
        if self.waves_per_eu is not None:
            kwargs["waves_per_eu"] = self.waves_per_eu
        if self.matrix_instr_nonkdim is not None:
            kwargs["matrix_instr_nonkdim"] = self.matrix_instr_nonkdim
        return triton.Config(
            kwargs, num_warps=self.num_warps, num_stages=self.num_stages
        )


@dataclass
class Result:
    config: Config
    ok: bool
    ms: float = float("nan")
    ms_std: float = float("nan")
    grid: tuple = ()
    workgroups: int = 0
    block_threads: int = 0
    fill_pct: float = float("nan")
    gbps: float = float("nan")  # full traffic model, incl. w/k re-reads
    gbps_hv: float = float("nan")  # h + v_new only -- what this harness used to report
    bytes_total: int = 0
    max_err_h: float = float("nan")
    max_err_v: float = float("nan")
    error: str = ""
    _h: Optional[torch.Tensor] = field(default=None, repr=False)
    _v: Optional[torch.Tensor] = field(default=None, repr=False)


def _kernel():
    """The Autotuner object wrapping the jitted kernel."""
    from sglang.kernels.ops.attention.fla import chunk_delta_h

    return chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64


def _fwd():
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )

    return chunk_gated_delta_rule_fwd_h


def _call(inp: Inputs):
    return _fwd()(
        k=inp.k,
        w=inp.w,
        u=inp.u,
        g=inp.g,
        initial_state=inp.initial_state,
        initial_state_indices=inp.initial_state_indices,
        cu_seqlens=inp.cu_seqlens,
        chunk_indices=inp.chunk_indices,
    )


def run_config(
    cfg: Config,
    inp: Inputs,
    *,
    reps: int = 20,
    warmup: int = 5,
    keep_output: bool = False,
    device_cus: int = 256,
    warp_size: int = 64,
) -> Result:
    """Compile, verify-runnable, and time one config.

    The state pool is restored before every single launch, including warmups.
    Timing brackets only the call itself, so the restore never lands inside a
    measurement window.
    """
    shape: Shape = inp.shape
    grid = shape.grid(cfg.BV)
    wgs = grid[0] * grid[1]
    res = Result(
        config=cfg,
        ok=False,
        grid=grid,
        workgroups=wgs,
        block_threads=cfg.num_warps * warp_size,
        fill_pct=100.0 * min(wgs, device_cus) / device_cus,
    )

    kern = _kernel()
    saved_configs = kern.configs
    saved_cache = dict(kern.cache)
    try:
        kern.configs = [cfg.to_triton()]
        kern.cache.clear()

        for _ in range(warmup):
            inp.restore_state()
            h, v_new = _call(inp)
        torch.cuda.synchronize()

        times = []
        for _ in range(reps):
            inp.restore_state()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            h, v_new = _call(inp)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        t = torch.tensor(times, dtype=torch.float64)
        # Median, not mean: a single stray launch (allocator growth, a stolen
        # CU) skews the mean and we care about steady state.
        res.ms = float(t.median())
        res.ms_std = float(t.std()) if len(times) > 1 else 0.0

        # Two bandwidth figures, deliberately. `gbps_hv` counts only what the
        # kernel produces (h + v_new); it is BV-independent and was the old
        # column. `gbps` counts everything the kernel moves, including the
        # grid.x-fold re-read of w and k. They diverge sharply with BV, and the
        # divergence is the finding: at BV=16 and BV=8 the full figure pins to
        # ~2.9-3.0 TB/s while the h+v_new figure keeps climbing, which is what a
        # bandwidth wall looks like when you are only counting half the traffic.
        res.bytes_total = shape.total_bytes(cfg.BV)
        res.gbps = res.bytes_total / (res.ms * 1e-3) / 1e9
        res.gbps_hv = (
            (shape.h_bytes() + shape.v_new_bytes()) / (res.ms * 1e-3) / 1e9
        )

        if keep_output:
            res._h = h.detach().clone()
            res._v = v_new.detach().clone() if v_new is not None else None
        res.ok = True
    except Exception as exc:  # compile failure, OOM, bad tile -- all reportable
        res.error = f"{type(exc).__name__}: {exc}".split("\n")[0][:160]
    finally:
        kern.configs = saved_configs
        kern.cache.clear()
        kern.cache.update(saved_cache)
        inp.restore_state()
        torch.cuda.synchronize()

    return res


def compare(ref: Result, cand: Result) -> None:
    """Fill in max abs error of `cand` against `ref`, in place.

    BV only partitions the V axis and each V row accumulates independently, so
    changing BV alone should be near-bitwise identical. num_warps and
    matrix_instr_nonkdim change the tl.dot lowering, so small bf16-level drift
    there is expected and fine. A large error means the config is wrong, not
    merely different.
    """
    if not (ref.ok and cand.ok) or ref._h is None or cand._h is None:
        return
    cand.max_err_h = float((ref._h.float() - cand._h.float()).abs().max())
    if ref._v is not None and cand._v is not None:
        cand.max_err_v = float((ref._v.float() - cand._v.float()).abs().max())


def env_equivalent(cfg: Config) -> str:
    """The server-side env vars that reproduce this config.

    waves_per_eu / matrix_instr_nonkdim have no env knob in sglang -- if one of
    those wins you need a code change in chunk_delta_h.py, so say so.
    """
    s = (
        f"SGLANG_GDN_CHUNK_H_BV={cfg.BV} "
        f"SGLANG_GDN_CHUNK_H_NUM_WARPS={cfg.num_warps} "
        f"SGLANG_GDN_CHUNK_H_NUM_STAGES={cfg.num_stages}"
    )
    extra = [
        n
        for n, v in (
            ("waves_per_eu", cfg.waves_per_eu),
            ("matrix_instr_nonkdim", cfg.matrix_instr_nonkdim),
        )
        if v is not None
    ]
    if extra:
        s += f"   # plus {', '.join(extra)} -- no env knob, needs a code change"
    return s
