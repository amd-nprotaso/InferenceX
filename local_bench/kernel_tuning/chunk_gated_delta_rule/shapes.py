"""Model shapes for chunk_gated_delta_rule_fwd_kernel_h_blockdim64.

The shapes are not invented: they were read out of a real MI350X torch-profiler
trace (isl32768_osl100_c4_amd, TP-0), from the `Input Dims` arg of the
`ChunkGatedDeltaRuleFunction` cpu_op:

    q [1, 32768, 4, 128]   k [1, 32768, 4, 128]   v [1, 32768, 16, 128]
    g [1, 32768, 16]       beta [1, 32768, 16]    scale 0.088388347648318447
    initial_state [1243, 16, 128, 128]

Note q/k carry the grouped head count (Hg=4) while v carries the full head count
(H=16). `chunk_gated_delta_rule_fwd_h` is called one level down from that, with
w/u produced by chunk_gated_delta_rule_fwd_intra -- those are [B, T, H, *], NOT
[B, T, Hg, *]. Getting this wrong silently changes the launch grid, which is the
whole thing we are trying to measure, so the distinction matters:

    k  [B, T, Hg, K]   stride_k = Hg * K
    w  [B, T, H,  K]   stride_w = H  * K
    u  [B, T, H,  V]
    g  [B, T, H]       fp32

The launch grid is (cdiv(V, BV), N * H) -- see chunk_delta_h.py:355. With the
production default BV=32 that is (4, 16) = 64 workgroups, which is what the
trace shows and which is only 25% of the MI350X's 256 CUs. That under-fill is
the thing this directory exists to fix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class Shape:
    """One benchmarkable configuration of the GDN state-update kernel."""

    name: str
    B: int  # batch (sequences packed via cu_seqlens)
    T: int  # total tokens
    Hg: int  # k/q head count (grouped)
    H: int  # v head count
    K: int  # key head dim
    V: int  # value head dim
    BT: int  # chunk size; CHUNK_SIZE in chunk_delta_h.py, always 64
    pool_slots: int  # rows in the mamba/GDN state pool
    note: str = ""

    @property
    def NT(self) -> int:
        """Number of chunks == iterations of the kernel's serial recurrence."""
        return math.ceil(self.T / self.BT)

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.K)

    def grid(self, BV: int) -> tuple:
        """Replicates the launch grid from chunk_delta_h.py:355."""
        return (math.ceil(self.V / BV), self.B * self.H)

    def workgroups(self, BV: int) -> int:
        g = self.grid(BV)
        return g[0] * g[1]

    def h_bytes(self, dtype_size: int = 2) -> int:
        """h is [B, NT, H, V, K] and is written once per chunk iteration.

        This dominates the kernel's memory traffic: 268 MB for the 32k shape.
        """
        return self.B * self.NT * self.H * self.V * self.K * dtype_size

    def v_new_bytes(self, dtype_size: int = 2) -> int:
        return self.B * self.T * self.H * self.V * dtype_size

    # ------------------------------------------------------------------
    # Full traffic model.
    #
    # h_bytes + v_new_bytes is what the kernel *produces*, and for a long time
    # it was the only thing this harness counted. It is not what the kernel
    # *moves*. The w and k block pointers carry no i_v term
    # (chunk_delta_h.py:174-176 and :271-273), so every one of the grid.x
    # V-blocks re-reads the whole of w and the whole of k. That redundancy is
    # the dominant term at small BV and it is the reason BV=8 loses.
    # ------------------------------------------------------------------

    def w_read_bytes(self, BV: int, dtype_size: int = 2) -> int:
        """w is [B, T, H, K], re-read once per V-block."""
        return self.grid(BV)[0] * self.B * self.T * self.H * self.K * dtype_size

    def k_read_bytes(self, BV: int, dtype_size: int = 2) -> int:
        """k is [B, T, Hg, K] but is read H times, not Hg times.

        Each of the H v-heads loads the k-slice for head i_h // (H // Hg)
        (chunk_delta_h.py:108). Head grouping saves storage, not read volume:
        the same k-head is pulled through the cache by H/Hg distinct programs.
        Then re-read again per V-block, hence the grid.x factor.
        """
        return self.grid(BV)[0] * self.B * self.T * self.H * self.K * dtype_size

    def u_read_bytes(self, dtype_size: int = 2) -> int:
        """u is [B, T, H, V] and IS partitioned by i_v -- read once, not grid.x times."""
        return self.B * self.T * self.H * self.V * dtype_size

    def total_bytes(self, BV: int, dtype_size: int = 2) -> int:
        """Everything the kernel moves, assuming zero cache reuse.

        This is an upper bound: it is what HBM sees if the L2/MALL absorbs
        nothing. Compare against rocprofv3 FETCH_SIZE + WRITE_SIZE to find out
        how much of the redundancy the cache actually eats -- that ratio is what
        decides whether the fix is 'move fewer bytes' or something else
        entirely. See profile_traffic.py.
        """
        return (
            self.h_bytes(dtype_size)
            + self.v_new_bytes(dtype_size)
            + self.w_read_bytes(BV, dtype_size)
            + self.k_read_bytes(BV, dtype_size)
            + self.u_read_bytes(dtype_size)
        )


# Qwen3.5-397B-A17B-MXFP4, one 32768-token prefill chunk, TP4 -> 16 v-heads/rank.
# This is the exact shape behind the 69.16 ms / 45 calls per prefill forward
# measured on MI350X (vs 46.96 ms on GB200 for the identical Triton kernel).
QWEN35_397B_TP4 = Shape(
    name="qwen35-397b-tp4",
    B=1,
    T=32768,
    Hg=4,
    H=16,
    K=128,
    V=128,
    BT=64,
    pool_slots=1243,
    note="ISL 32768 prefill chunk, TP4. Production shape. grid=(4,16)=64 WGs @ BV=32.",
)

# Same model at TP2: twice the v-heads per rank, so the grid doubles for free.
# Useful as a control -- if BV tuning is doing what we think, TP2 at BV=32
# should land near TP4 at BV=16 (both 128 workgroups).
QWEN35_397B_TP2 = Shape(
    name="qwen35-397b-tp2",
    B=1,
    T=32768,
    Hg=8,
    H=32,
    K=128,
    V=128,
    BT=64,
    pool_slots=1243,
    note="Same model at TP2. grid=(4,32)=128 WGs @ BV=32.",
)

# Short chunk: exercises NT_BUCKET=1 instead of 2 and a much shorter serial
# recurrence, so it separates 'not enough workgroups' from 'serial loop too long'.
QWEN35_397B_TP4_4K = Shape(
    name="qwen35-397b-tp4-4k",
    B=1,
    T=4096,
    Hg=4,
    H=16,
    K=128,
    V=128,
    BT=64,
    pool_slots=1243,
    note="4096-token chunk, TP4. NT=64 -> NT_BUCKET=1.",
)

SHAPES: Dict[str, Shape] = {
    s.name: s for s in (QWEN35_397B_TP4, QWEN35_397B_TP2, QWEN35_397B_TP4_4K)
}
DEFAULT_SHAPE = QWEN35_397B_TP4.name


@dataclass
class Inputs:
    """Device tensors for one `chunk_gated_delta_rule_fwd_h` call."""

    shape: Shape
    k: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    g: torch.Tensor
    initial_state: torch.Tensor
    initial_state_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    chunk_indices: torch.Tensor
    _state_backup: torch.Tensor

    def restore_state(self) -> None:
        """Undo the kernel's in-place write to the state pool.

        The kernel runs with INPLACE_UPDATE=True and writes the final state ht
        back over initial_state (chunk_delta_h.py:130). Re-running it without
        restoring feeds its own output back in, so timings drift and any
        correctness comparison across configs is meaningless. This is the same
        hazard that made sglang disable autotune on this kernel -- see the
        comment at chunk_delta_h.py:29.

        Only the slots named by initial_state_indices are touched, so restoring
        those is enough and costs ~0.5 MB rather than the pool's 651 MB.
        """
        idx = self.initial_state_indices
        self.initial_state[idx] = self._state_backup


def build_inputs(
    shape: Shape,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    pool_slots: int | None = None,
    gate_scale: float = 0.1,
) -> Inputs:
    """Allocate one call's worth of inputs, via the production pipeline.

    w and u are NOT free parameters. They come out of
    chunk_gated_delta_rule_fwd_intra (fused kkt + solve_tril + recompute_w_u),
    and the delta rule they encode is only stable for w that actually solves
    that triangular system. Handing the kernel random w diverges: over NT=512
    chunk iterations the state blows up and >90% of h comes back NaN, which
    silently invalidates every cross-config comparison and perturbs timing
    through denormal handling.

    So we build q/k/v/beta/g -- the real model tensors, at the shapes read from
    the trace -- and run the same two preamble steps chunk.py:47-59 runs:

        g = chunk_local_cumsum(g, ...)
        w, u, _ = chunk_gated_delta_rule_fwd_intra(k, v, g, beta, ...)

    k is L2-normalized because the model does that (the trace shows
    l2norm_fwd_kernel 90x == 45 layers x {q, k}), and it is also what keeps the
    k^T v outer product bounded.

    initial_state is zeros: these benchmarks model a fresh prefill chunk, and
    the traced run had a 0.0% cache hit rate. The pool is still passed, so
    USE_INITIAL_STATE is on and the load path is still timed.
    """
    from sglang.kernels.ops.attention.fla.chunk_fwd import (
        chunk_gated_delta_rule_fwd_intra,
    )
    from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
    from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)
    slots = shape.pool_slots if pool_slots is None else pool_slots

    def randn(*size):
        return torch.randn(*size, generator=gen, device=dev, dtype=torch.float32)

    # One packed sequence of length T, matching the traced prefill chunk.
    cu_seqlens = torch.tensor([0, shape.T], device=dev, dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, shape.BT)

    k = l2norm_fwd(randn(shape.B, shape.T, shape.Hg, shape.K).to(dtype)).view(
        shape.B, shape.T, shape.Hg, shape.K
    )
    v = (randn(shape.B, shape.T, shape.H, shape.V) * 0.5).to(dtype)
    beta = torch.sigmoid(randn(shape.B, shape.T, shape.H)).to(dtype)

    # Per-token log decay: strictly negative, small. gate_scale ~0.1 puts the
    # within-chunk cumulative decay near -4.5 by the end of a 64-token chunk,
    # i.e. exp(g) spans roughly [0.01, 1.0] -- the range a trained gate produces.
    g = -torch.nn.functional.softplus(randn(shape.B, shape.T, shape.H)) * gate_scale
    g = chunk_local_cumsum(
        g.contiguous(),
        chunk_size=shape.BT,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    w, u, _A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=shape.BT,
        chunk_indices=chunk_indices,
    )

    # Pool layout is [slots, H, V, K]; stride(0) == H*V*K for a contiguous pool,
    # which is what the kernel reads as stride_init_state.
    initial_state = torch.zeros(
        slots, shape.H, shape.V, shape.K, device=dev, dtype=dtype
    )
    initial_state_indices = torch.arange(shape.B, device=dev, dtype=torch.int32)

    return Inputs(
        shape=shape,
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        _state_backup=initial_state[initial_state_indices].clone(),
    )


def device_info(device: torch.device | str = "cuda") -> dict:
    p = torch.cuda.get_device_properties(torch.device(device))
    return {
        "name": p.name,
        "cus": p.multi_processor_count,
        "warp_size": getattr(p, "warp_size", 64),
        "lds_per_cu": getattr(p, "maxSharedMemoryPerMultiProcessor", None)
        or getattr(p, "shared_memory_per_multiprocessor", None),
    }
