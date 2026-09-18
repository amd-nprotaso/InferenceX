# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused MoE routing front-end for Qwen3.5 MXFP4 decode (FlyDSL, gfx950).

Ports ``compile_moe_sorting_oneshot_fused`` from upstream FlyDSL
(``kernels/moe/moe_sorting_kernel.py:696``) into aiter's vendored FlyDSL
namespace, which only carries the *unfused* oneshot sorter.  One launch turns
``gating_logits[M, E]`` into the full aiter ``moe_sorting`` output contract:

    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf

replacing ``topk_softmax`` (the router) *and* both ``opus_moe_sorting_entry``
launches.  Every per-token-K intermediate stays in LDS.

Changes versus the upstream source, all needed for E=512 / topk=10 / T=16:

* ``_pick_layout`` gained ``VPT=32``.  Upstream caps VPT at 16, which for
  E=512 forces ``THREADS_PER_TOKEN=32`` and therefore ``TOKENS_PER_BLOCK=8``
  -- half the 16 tokens of a CONC=4 EAGLE verify step would never be routed,
  because the fused kernel only runs gating in block 0.  ``VPT=32`` gives
  ``TPT=16`` and ``TOKENS_PER_BLOCK=16`` at the same per-thread element count
  (32 experts/thread) and *fewer* butterfly shuffles per top-k iteration.
* ``compile_moe_sorting_oneshot_fused`` asserts ``TOKENS_PER_BLOCK >=
  max_tokens`` so that failure mode is a compile-time error, not silent
  mis-routing.
* imports retargeted at ``aiter.ops.flydsl.kernels`` (buffer_ops,
  kernels_common) instead of the ``kernels.common`` package, which is not
  shipped in the FlyDSL wheel.

Host entry: :func:`flydsl_fused_route_sort`, signature-compatible with the
tuple returned by ``aiter.fused_moe.moe_sorting`` minus the router.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import vector
from flydsl.expr import arith, as_ir_value, gpu, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, T
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import dtype_to_elem_type, get_warp_size

# Threads in the fused block. Everything except the moe_buf zeroing runs in
# block 0, so this sets the parallelism of the whole routing front-end. Swept
# at E=512 / topk=10 / T=16 on gfx950: 256 -> 15.18 us, 512 -> 12.60 us,
# 1024 -> 14.54 us (see README). 512 also picks VPT=16 for the gating layout,
# which is the best point on the scan-vs-shuffle tradeoff at this shape.
BLOCK_SIZE = 512
UNIT_SIZE = 32  # GEMM tile-M, aka block_size in CK
WARP_SIZE = get_warp_size()

# DPP constants for the intra-wave prefix sum.
DPP_ROW_SHR_1 = 0x111
DPP_ROW_SHR_2 = 0x112
DPP_ROW_SHR_4 = 0x114
DPP_ROW_SHR_8 = 0x118
DPP_ROW_MASK = 0xF
DPP_BANK_MASK = 0xF


def _unwrap_val(v):
    """Unwrap DSL value to raw MLIR ir.Value."""
    return v.ir_value() if hasattr(v, "ir_value") else v


def _lds_load_raw(raw_ptr, idx):
    """Load i32 from an LDS pointer at element offset ``idx``."""
    return fx.ptr_load(raw_ptr + fx.Int64(idx))


def _lds_store_raw(raw_ptr, val, idx):
    """Store i32 to an LDS pointer at element offset ``idx``."""
    fx.ptr_store(val, raw_ptr + fx.Int64(idx))


def _dpp_intra_wave_prefix_sum(val, lane, warp_size):
    """Inclusive prefix sum within a single wave using DPP.

    Four DPP ``row_shr`` steps (1, 2, 4, 8) for the intra-row scan, then two
    ``ds_bpermute`` steps (16, 32) for cross-row accumulation within the wave.
    Call inside ``@flyc.kernel`` only -- emits MLIR during tracing.
    """
    val_raw = _unwrap_val(val)
    zero_raw = _unwrap_val(fx.Int32(0))

    for shift, dpp_op, threshold in [
        (1, DPP_ROW_SHR_1, 1),
        (2, DPP_ROW_SHR_2, 2),
        (4, DPP_ROW_SHR_4, 4),
        (8, DPP_ROW_SHR_8, 8),
    ]:
        remote = fly_rocdl.update_dpp(T.i32, zero_raw, val_raw, dpp_op, DPP_ROW_MASK, DPP_BANK_MASK, True)
        val = (lane >= fx.Int32(threshold)).select(val + fx.Int32(remote), val)
        val_raw = _unwrap_val(val)

    src_lane_16 = (lane & fx.Int32(0x30)) - fx.Int32(1)
    remote16 = fly_rocdl.ds_bpermute(T.i32, src_lane_16 * fx.Int32(4), val)
    val = (lane >= fx.Int32(16)).select(val + fx.Int32(remote16), val)

    if warp_size > 32:
        src_lane_32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
        remote32 = fly_rocdl.ds_bpermute(T.i32, src_lane_32 * fx.Int32(4), val)
        val = (lane >= fx.Int32(32)).select(val + fx.Int32(remote32), val)

    return val


# ---------------------------------------------------------------------------
# Gating layout + softmax/top-K body (port of
# kernels/moe/topk_gating_softmax_kernel.py)
# ---------------------------------------------------------------------------
def _pick_layout(num_experts: int, warps_per_block: int, min_tokens_per_block: int):
    """Pick (VPT, THREADS_PER_TOKEN) for the multi-token-per-block fast path.

    Constraints:
      - ``THREADS_PER_TOKEN = num_experts // VPT`` is a power of 2 <= WARP_SIZE
      - ``TOKENS_PER_BLOCK = warps_per_block * WARP_SIZE // THREADS_PER_TOKEN``
        must reach ``min_tokens_per_block``

    Upstream takes the *largest* VPT unconditionally, which minimises loads.
    Here the top-k loop dominates -- it rescans all VPT register slots on every
    one of ``topk`` iterations -- so we take the *smallest* VPT that still
    covers the required tokens per block, trading 2x more butterfly shuffles
    for 4x fewer per-slot compares.  At E=512 / 16 waves that is VPT=8
    (one full wave per token) instead of VPT=32.
    """
    best = None
    for vpt in [1, 2, 4, 8, 16, 32]:
        if num_experts % vpt != 0:
            continue
        tpt = num_experts // vpt
        if tpt > WARP_SIZE or (tpt & (tpt - 1)) != 0:
            continue
        tokens_per_block = warps_per_block * (WARP_SIZE // tpt)
        if best is None:
            best = (vpt, tpt)  # fall back to *some* valid layout
        if tokens_per_block >= min_tokens_per_block:
            return vpt, tpt
    # None reaches min_tokens_per_block; return the widest valid layout so the
    # caller's explicit check reports the shortfall.
    for vpt in [32, 16, 8, 4, 2, 1]:
        if num_experts % vpt != 0:
            continue
        tpt = num_experts // vpt
        if tpt <= WARP_SIZE and (tpt & (tpt - 1)) == 0:
            return vpt, tpt
    return None, None


def _compute_topk_gating_layout(
    num_experts: int,
    topk: int,
    dtype_str: str,
    block_size: int = BLOCK_SIZE,
    min_tokens_per_block: int = 1,
):
    """Resolve the full gating layout dict shared by the emitter and the host."""
    elem_bits = 32 if dtype_str == "f32" else 16
    warps_per_block = block_size // WARP_SIZE

    VPT, THREADS_PER_TOKEN = _pick_layout(num_experts, warps_per_block, min_tokens_per_block)
    if VPT is None:
        raise ValueError(
            f"num_experts={num_experts} is not supported by the multi-token-per-block "
            f"layout: requires num_experts // VPT to be a power of 2 <= "
            f"WARP_SIZE={WARP_SIZE} for some VPT in [32, 16, 8, 4, 2, 1]."
        )
    if topk > num_experts:
        raise ValueError(f"topk={topk} > num_experts={num_experts}")

    TOKENS_PER_WARP = WARP_SIZE // THREADS_PER_TOKEN
    TOKENS_PER_BLOCK = warps_per_block * TOKENS_PER_WARP

    if elem_bits <= 16 and VPT % 8 == 0:
        ATOM_BITS = 128  # 8 bf16/f16 per atom call
    elif elem_bits <= 16 and VPT % 4 == 0:
        ATOM_BITS = 64
    elif elem_bits <= 16 and VPT % 2 == 0:
        ATOM_BITS = 32
    elif elem_bits == 32 and VPT % 2 == 0:
        ATOM_BITS = 64
    else:
        ATOM_BITS = elem_bits
    ELEMS_PER_ATOM = ATOM_BITS // elem_bits
    ATOMS_PER_THREAD = VPT // ELEMS_PER_ATOM

    return dict(
        elem_bits=elem_bits,
        VPT=VPT,
        THREADS_PER_TOKEN=THREADS_PER_TOKEN,
        TOKENS_PER_WARP=TOKENS_PER_WARP,
        TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
        ATOM_BITS=ATOM_BITS,
        ELEMS_PER_ATOM=ELEMS_PER_ATOM,
        ATOMS_PER_THREAD=ATOMS_PER_THREAD,
    )


@flyc.jit
def _emit_topk_gating_softmax_body(
    GatingOutput,
    TopkWeights,
    TopkIndices,
    TokenExpertIndices,
    i32_num_tokens,
    *,
    num_experts: int,
    topk: int,
    dtype_str: str,
    renormalize: bool,
    VPT: int,
    THREADS_PER_TOKEN: int,
    TOKENS_PER_WARP: int,
    TOKENS_PER_BLOCK: int,
    ATOM_BITS: int,
    ELEMS_PER_ATOM: int,
    ATOMS_PER_THREAD: int,
    elem_bits: int,
    on_winner_idx=None,
    on_winner_weight=None,
    emit_tei: bool = True,
):
    """Emit gating logits -> softmax -> top-K into the current kernel body.

    ``on_winner_idx`` / ``on_winner_weight`` replace the corresponding HBM
    store with a caller-supplied LDS sink; both are invoked once per
    (token, k) inside the leader-active region as

        on_winner_idx(local_token_i32, global_token_i32, k_int, expert_idx_i32)
        on_winner_weight(local_token_i32, global_token_i32, k_int, weight_f32)
    """
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    # aiter's dtype_to_elem_type returns the MLIR type directly; upstream
    # FlyDSL's returns a dtype wrapper carrying it as `.ir_type`.
    elem_type = dtype_to_elem_type(dtype_str)
    elem_type = getattr(elem_type, "ir_type", elem_type)
    compute_type = T.f32
    register_addr_space = int(fx.AddressSpace.Register)

    fm_fast = arith.FastMathFlags.fast

    c_zero_f = fx.Float32(0.0)
    c_neg_inf = fx.Float32(float("-inf"))
    c_log2e = fx.Float32(1.4426950408889634)
    c_one_f = fx.Float32(1.0)

    c_warp = fx.Int32(WARP_SIZE)
    c_tpt = fx.Int32(THREADS_PER_TOKEN)
    c_tpw = fx.Int32(TOKENS_PER_WARP)
    c_tpb = fx.Int32(TOKENS_PER_BLOCK)
    c_vpt = fx.Int32(VPT)

    warp_id = tid // c_warp
    lane = tid % c_warp
    token_in_warp = lane // c_tpt
    expert_lane = lane % c_tpt
    local_token = warp_id * c_tpw + token_in_warp
    global_token = bid * c_tpb + local_token

    in_range = global_token < i32_num_tokens
    global_token_safe = in_range.select(global_token, fx.Int32(0))

    def group_reduce(x, mode):
        """Butterfly reduce within a THREADS_PER_TOKEN sub-warp group."""
        width_i32 = c_tpt
        w = x
        for _sh in range_constexpr(int(math.log2(THREADS_PER_TOKEN))):
            off = fx.Int32(THREADS_PER_TOKEN // (2 << _sh))
            peer = w.shuffle_xor(off, width_i32)
            if mode == "max":
                w = w.maximumf(peer)
            else:
                w = w.addf(peer, fastmath=fm_fast)
        return w

    def group_reduce_argmax(val, idx):
        """Butterfly argmax within a group; ties go to the lower expert index."""
        width_i32 = c_tpt
        wv, wi = val, idx
        for _sh in range_constexpr(int(math.log2(THREADS_PER_TOKEN))):
            off = fx.Int32(THREADS_PER_TOKEN // (2 << _sh))
            peer_v = wv.shuffle_xor(off, width_i32)
            peer_i = wi.shuffle_xor(off, width_i32)
            is_greater = peer_v > wv
            is_equal = ArithValue(peer_v) == ArithValue(wv)
            peer_lower_idx = peer_i < wi
            take_peer = is_greater | (is_equal & peer_lower_idx)
            wv = take_peer.select(peer_v, wv)
            wi = take_peer.select(peer_i, wi)
        return wv, wi

    GatingOutput_buf = fx.rocdl.make_buffer_tensor(GatingOutput)
    row_gating = fx.slice(GatingOutput_buf, (global_token_safe, None))
    gating_div = fx.logical_divide(row_gating, fx.make_layout(ELEMS_PER_ATOM, 1))

    # Only materialise the output views for the stores we actually emit;
    # callback callers pass None for the corresponding HBM tensor.
    weights_div = None
    if on_winner_weight is None:
        TopkWeights_buf = fx.rocdl.make_buffer_tensor(TopkWeights)
        row_weights = fx.slice(TopkWeights_buf, (global_token_safe, None))
        weights_div = fx.logical_divide(row_weights, fx.make_layout(1, 1))

    indices_div = None
    if on_winner_idx is None:
        TopkIndices_buf = fx.rocdl.make_buffer_tensor(TopkIndices)
        row_indices = fx.slice(TopkIndices_buf, (global_token_safe, None))
        indices_div = fx.logical_divide(row_indices, fx.make_layout(1, 1))

    tei_div = None
    if emit_tei:
        TokenExpertIndices_buf = fx.rocdl.make_buffer_tensor(TokenExpertIndices)
        row_tei = fx.slice(TokenExpertIndices_buf, (global_token_safe, None))
        tei_div = fx.logical_divide(row_tei, fx.make_layout(1, 1))

    copy_atom_in = fx.make_copy_atom(fx.rocdl.BufferCopy(ATOM_BITS), elem_bits)
    atom_reg_ty_in = fx.MemRefType.get(elem_type, fx.LayoutType.get(ELEMS_PER_ATOM, 1), register_addr_space)
    atom_reg_lay_in = fx.make_layout(ELEMS_PER_ATOM, 1)

    copy_atom_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
    scalar_reg_ty_f32 = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), register_addr_space)
    scalar_reg_lay = fx.make_layout(1, 1)

    def _load_atom_in(divided, atom_index):
        view = fx.slice(divided, (None, atom_index))
        r = fx.memref_alloca(atom_reg_ty_in, atom_reg_lay_in)
        fx.copy(copy_atom_in, view, r)
        return fx.memref_load_vec(r)

    def _store_scalar_f32(divided, index, val):
        r = fx.memref_alloca(scalar_reg_ty_f32, scalar_reg_lay)
        v = fx.Vector.from_elements([val], fx.Float32)
        fx.memref_store_vec(v, r)
        view = fx.slice(divided, (None, index))
        fx.copy(copy_atom_f32, r, view)

    def _store_scalar_i32(divided, index, val):
        # `divided` has f32 element type; reinterpret the i32 bits as f32 and
        # store via the f32 atom (avoids signed-vs-signless legalize failures).
        val_f32 = ArithValue(val).bitcast(T.f32)
        r = fx.memref_alloca(scalar_reg_ty_f32, scalar_reg_lay)
        v = fx.Vector.from_elements([val_f32], fx.Float32)
        fx.memref_store_vec(v, r)
        view = fx.slice(divided, (None, index))
        fx.copy(copy_atom_f32, r, view)

    # Pass 1: load this thread's VPT experts + per-thread max
    col_idx_list = []
    for v in range_constexpr(VPT):
        col_idx_list.append(expert_lane * c_vpt + fx.Int32(v))

    c_atoms_pt = fx.Int32(ATOMS_PER_THREAD)
    x_list = []
    thread_max = c_neg_inf
    for a in range_constexpr(ATOMS_PER_THREAD):
        atom_idx = expert_lane * c_atoms_pt + fx.Int32(a)
        atom_vec = _load_atom_in(gating_div, atom_idx)
        for v in range_constexpr(ELEMS_PER_ATOM):
            val_e = vector.extract(as_ir_value(atom_vec), dynamic_position=[], static_position=[v])
            xv = val_e if dtype_str == "f32" else val_e.extf(compute_type)
            x_list.append(xv)
            thread_max = thread_max.maximumf(xv)

    group_max = group_reduce(thread_max, "max")

    # Pass 2: exp(x - max) and per-token sum
    thread_sum = c_zero_f
    exp_list = []
    for v in range_constexpr(VPT):
        sub = x_list[v] - group_max
        scaled = sub * c_log2e
        ev = scaled.exp2(fastmath=fm_fast)
        exp_list.append(ev)
        thread_sum = thread_sum + ev

    group_sum = group_reduce(thread_sum, "sum")

    # Pass 3: normalise -> softmax probabilities (kept in registers)
    inv_sum = c_one_f / group_sum
    prob_list = []
    for v in range_constexpr(VPT):
        prob_list.append(exp_list[v] * inv_sum)

    # Pass 4: iterative top-K (sub-warp argmax -> mask)
    selected_weights = []
    selected_indices = []
    selected_sum = c_zero_f

    for k_idx in range_constexpr(topk):
        thread_best_val = c_neg_inf
        thread_best_idx = fx.Int32(-1)
        for v in range_constexpr(VPT):
            pv = prob_list[v]
            ci = col_idx_list[v]
            is_better = pv > thread_best_val
            thread_best_val = is_better.select(pv, thread_best_val)
            thread_best_idx = is_better.select(ci, thread_best_idx)

        global_best_val, global_best_idx = group_reduce_argmax(thread_best_val, thread_best_idx)

        selected_weights.append(global_best_val)
        selected_indices.append(global_best_idx)
        selected_sum = selected_sum + global_best_val

        for v in range_constexpr(VPT):
            ci = col_idx_list[v]
            is_winner = ArithValue(ci) == ArithValue(global_best_idx)
            prob_list[v] = is_winner.select(c_neg_inf, prob_list[v])

    # Pass 5: leader writes weights/indices/tei (with optional renorm).
    c_eps = fx.Float32(1e-20)
    denom = selected_sum.maximumf(c_eps)
    inv_denom = c_one_f / denom

    if (expert_lane == fx.Int32(0)) & (global_token < i32_num_tokens):
        num_tokens_v = ArithValue(i32_num_tokens)
        for k_idx in range_constexpr(topk):
            w_val = selected_weights[k_idx]
            if renormalize:
                w_val = w_val * inv_denom
            if on_winner_weight is not None:
                on_winner_weight(local_token, global_token, k_idx, w_val)
            else:
                _store_scalar_f32(weights_div, Int32(k_idx), w_val)

            if on_winner_idx is not None:
                on_winner_idx(local_token, global_token, k_idx, selected_indices[k_idx])
            else:
                _store_scalar_i32(indices_div, Int32(k_idx), selected_indices[k_idx])

            if emit_tei:
                # tei[t, k] = k * num_tokens + t  (matches the vLLM convention).
                tei_val = Int32(k_idx) * num_tokens_v + global_token
                _store_scalar_i32(tei_div, Int32(k_idx), tei_val)


# ---------------------------------------------------------------------------
# Fused (gating + sort) oneshot kernel
# ---------------------------------------------------------------------------
def compute_fused_sub_tokens(num_experts, arch=None):
    """LDS-capacity bound on the oneshot mesh row count, for a given E."""
    if arch is None:
        arch = get_rocm_arch()
    smem_cols = num_experts + 1
    if str(arch).startswith("gfx95"):
        lds_capacity_bytes = 163840
    else:
        lds_capacity_bytes = 65536
    lds_capacity_ints = lds_capacity_bytes // 4
    target_occupancy = 2
    r = lds_capacity_ints // target_occupancy // smem_cols
    sub_unroll = 8
    cumsum_bufs = 2
    if r < (cumsum_bufs + sub_unroll):
        return 0
    return ((r - cumsum_bufs) // sub_unroll) * sub_unroll


@functools.lru_cache(maxsize=256)
def compile_moe_sorting_oneshot_fused(
    *,
    num_experts: int,
    topk: int,
    dtype_str: str = "bf16",
    renormalize: bool = True,
    max_tokens: int = 16,
    unit_size: int = UNIT_SIZE,
    has_mask: bool = False,
    block_size: int = BLOCK_SIZE,
):
    """Compile the fused (gating + sort) oneshot-path MoE kernel.

    Reads ``gating_logits[M, E]`` directly, performs softmax + top-K in block
    0, stages every per-token-K intermediate in LDS, then runs the sort phases
    (count -> prefix-sum -> scatter + padding) without round-tripping routing
    data through HBM.  Blocks > 0 zero ``moe_buf``.

    ``block_size`` is the single knob that matters for speed here: the whole
    routing front-end is one workgroup, so every per-expert loop is
    ``E / block_size`` long and there is no second block to hide latency
    behind.  Upstream hard-codes 256.
    """
    BS = block_size
    arch = get_rocm_arch()
    E = num_experts
    smem_cols = E + 1

    if str(arch).startswith("gfx95"):
        lds_capacity_bytes = 163840
    else:
        lds_capacity_bytes = 65536

    lds_capacity_ints = lds_capacity_bytes // 4
    target_occupancy = 2
    r = lds_capacity_ints // target_occupancy // smem_cols
    sub_unroll = 8
    cumsum_bufs = 2
    if r < (cumsum_bufs + sub_unroll):
        raise ValueError(f"LDS too small for E={E}: need at least {(cumsum_bufs + sub_unroll) * smem_cols * 4} bytes")
    r_for_sub = ((r - cumsum_bufs) // sub_unroll) * sub_unroll
    r_token_min = ((max_tokens + sub_unroll - 1) // sub_unroll) * sub_unroll
    r_for_sub = min(r_for_sub, r_token_min)
    sub_tokens = r_for_sub

    gating_layout = _compute_topk_gating_layout(E, topk, dtype_str, block_size=BS, min_tokens_per_block=max_tokens)
    tokens_per_block = gating_layout["TOKENS_PER_BLOCK"]
    if tokens_per_block < max_tokens:
        # Gating only runs in block 0, so tokens >= TOKENS_PER_BLOCK would be
        # silently dropped from the mesh. Upstream has no such guard because
        # its VPT cap keeps E <= 256 shapes at TOKENS_PER_BLOCK >= 16.
        raise ValueError(
            f"gating layout covers only {tokens_per_block} tokens per block but "
            f"max_tokens={max_tokens} (E={E}, topk={topk}, dtype={dtype_str})"
        )
    if sub_tokens < max_tokens:
        raise ValueError(f"LDS mesh holds only {sub_tokens} tokens but max_tokens={max_tokens} (E={E})")

    _blocks_per_expert = (max_tokens + unit_size - 1) // unit_size
    # Sentinel fill assigns `unit_size` consecutive threads to one expert so a
    # wave writes contiguous runs; needs at least one whole expert per wave.
    _pad_experts_per_step = max(BS // unit_size, 1)

    @fx.struct
    class SharedStorage:
        cumsum: fx.Array[fx.Int32, smem_cols, 16]
        cumdup: fx.Array[fx.Int32, smem_cols, 16]
        mesh: fx.Array[fx.Int32, sub_tokens * smem_cols, 16]
        weights_lds: fx.Array[fx.Int32, max_tokens * topk, 16]

    @flyc.kernel(known_block_size=[BS, 1, 1])
    def moe_sorting_oneshot_fused_kernel(
        gating_logits: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        moe_buf: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        tokens = i32_tokens
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_oob_idx = fx.Int32(0x7FFFFFFF)
        c4_i32 = fx.Int32(4)

        moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf, max_size=True)
        sorted_ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        sorted_w_rsrc = buffer_ops.create_buffer_resource(sorted_weights_out, max_size=True)
        sorted_e_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True)
        nvalid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
        mask_rsrc = buffer_ops.create_buffer_resource(expert_mask_tensor, max_size=True)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        cumsum_mr = lds.cumsum.ptr
        cumdup_mr = lds.cumdup.ptr
        mesh_mr = lds.mesh.ptr
        weights_lds_mr = lds.weights_lds.ptr

        c_topk = fx.Int32(topk)
        c_E = fx.Int32(E)
        c_unit = fx.Int32(unit_size)
        c_sub_tokens = fx.Int32(sub_tokens)
        c_smem_cols = fx.Int32(smem_cols)
        c_sentinel = fx.Int32(topk << 24)

        # =================== MOE_BUF ZEROING (blocks > 0 only) ===============
        is_zero_block = bid != c_zero_i32
        if is_zero_block:
            zero_gid_v4 = (bid - c_one_i32) * fx.Int32(BS) + tid
            num_zero_blocks = gpu.grid_dim.x - c_one_i32
            zero_stride_v4 = num_zero_blocks * fx.Int32(BS)
            i32_moe_buf_v4 = i32_moe_buf_elems >> fx.Int32(2)
            zero_niters = (i32_moe_buf_v4 + zero_stride_v4 - c_one_i32) // zero_stride_v4
            _zs = fx.Index(0)
            _ze = ArithValue(zero_niters).index_cast(T.index)
            _z1 = fx.Index(1)
            c_zero_v4 = fx.Vector.filled(4, 0, fx.Int32)
            c4_i32 = fx.Int32(4)
            for _z in range(_zs, _ze, _z1):
                z_idx_v4 = zero_gid_v4 + fx.Int32(_z) * zero_stride_v4
                z_valid = z_idx_v4 < i32_moe_buf_v4
                # Fold the out-of-range lanes onto element 0 rather than the
                # 0x7FFFFFFF OOB sentinel used elsewhere. The descriptors are
                # built with max_size=True (num_records = 4 GiB), so the
                # hardware does NOT discard a sentinel offset -- it writes ~8
                # GiB past the base and faults. Writing an extra zero into a
                # buffer that is being zeroed is idempotent.
                # Fold out-of-range lanes onto element 0 rather than the
                # 0x7FFFFFFF OOB sentinel used elsewhere: descriptors are built
                # with max_size=True (num_records = 4 GiB), so the hardware does
                # not discard a sentinel offset for a 128-bit store. Writing an
                # extra zero into a buffer being zeroed is idempotent.
                z_elem = z_valid.select(z_idx_v4 * c4_i32, c_zero_i32)
                buffer_ops.buffer_store(c_zero_v4, moe_buf_rsrc, z_elem)

        # =================== SORTING (block 0 only) ==========================
        is_sort_block = bid == c_zero_i32
        if is_sort_block:
            # ========== PHASE 1 (mesh CLEAR ONLY in the fused kernel) ========
            # Gating's on_winner_idx callback fills the mesh directly, so the
            # old "Phase 1 fill" loop is gone. The clear must still happen:
            # Phase 2 reads every cell and treats 0 as "no token".
            for i_clear in range_constexpr(0, sub_tokens * smem_cols, BS):
                idx = fx.Int32(i_clear) + tid
                is_valid = idx < fx.Int32(sub_tokens * smem_cols)
                safe_idx = is_valid.select(idx, c_zero_i32)
                safe_idx_ix = ArithValue(safe_idx).index_cast(T.index)
                _lds_store_raw(mesh_mr, c_zero_i32, safe_idx_ix)
            gpu.barrier()

            # =========== PHASE 0 (fused): Gating + softmax + top-K ===========
            # mesh_LDS is [sub_tokens, smem_cols] row-major; a winning cell
            # holds k_idx + 1 (0 still means "empty").
            # weights_LDS is [max_tokens, topk] row-major, holding the f32
            # weight's i32 bits (Phase 3 stores those bits to sorted_weights,
            # which it also reaches through the i32 path).
            def on_winner_idx(local_token, global_token, k_int, expert_idx):
                mesh_idx = local_token * c_smem_cols + expert_idx
                _lds_store_raw(mesh_mr, fx.Int32(k_int + 1), mesh_idx)

            def on_winner_weight(local_token, global_token, k_int, weight):
                w_bits = ArithValue(weight).bitcast(T.i32)
                w_idx = local_token * c_topk + fx.Int32(k_int)
                _lds_store_raw(weights_lds_mr, fx.Int32(w_bits), w_idx)

            _emit_topk_gating_softmax_body(
                gating_logits,
                None,  # TopkWeights HBM not used in the fused path
                None,  # TopkIndices HBM not used in the fused path
                None,  # TokenExpertIndices HBM not used in the fused path
                i32_tokens,
                num_experts=num_experts,
                topk=topk,
                dtype_str=dtype_str,
                renormalize=renormalize,
                on_winner_idx=on_winner_idx,
                on_winner_weight=on_winner_weight,
                emit_tei=False,
                **gating_layout,
            )
            gpu.barrier()

            # ===================== PHASE 2: Count + Prefix Sum ===============
            c_lane_group_sz = fx.Int32(8)
            lane_group_id = tid // c_lane_group_sz
            lane_group_os = tid % c_lane_group_sz
            width8_i32 = fx.Int32(8)

            is_t0 = tid == c_zero_i32
            _lds_store_raw(cumsum_mr, c_zero_i32, c_zero_i32)
            gpu.barrier()

            # One thread per expert walking its own mesh column. Upstream
            # spreads each expert over an 8-lane group and cross-reduces with
            # three shuffle_xor steps per mesh cell; at sub_tokens=16 that is
            # 48 cross-lane ops per thread to save 14 LDS reads, and the mesh
            # column read is bank-conflict-free anyway (consecutive lanes read
            # consecutive experts, stride 1).
            for i_e in range_constexpr(0, E, BS):
                eid_local = fx.Int32(i_e) + tid
                eid_valid = eid_local < c_E
                safe_eid = eid_valid.select(eid_local, c_zero_i32)

                cnt = c_zero_i32
                for i_sub in range_constexpr(sub_tokens):
                    mesh_rd_addr = fx.Int32(i_sub * smem_cols) + safe_eid
                    mesh_rd_ix = ArithValue(mesh_rd_addr).index_cast(T.index)
                    mesh_val = _lds_load_raw(mesh_mr, mesh_rd_ix)
                    cnt = cnt + (mesh_val != c_zero_i32).select(c_one_i32, c_zero_i32)

                cs_idx = eid_valid.select(eid_local + c_one_i32, c_zero_i32)
                cs_ix = ArithValue(cs_idx).index_cast(T.index)
                cs_val = eid_valid.select(cnt, c_zero_i32)
                _lds_store_raw(cumsum_mr, cs_val, cs_ix)
            gpu.barrier()

            for i_cvt in range_constexpr(0, E, BS):
                cvt_eid = fx.Int32(i_cvt) + tid
                cvt_valid = cvt_eid < c_E
                safe_cvt_idx = cvt_valid.select(cvt_eid + c_one_i32, c_zero_i32)
                cvt_ix = ArithValue(safe_cvt_idx).index_cast(T.index)
                raw_cnt_cvt = _lds_load_raw(cumsum_mr, cvt_ix)
                blocks_cvt = (raw_cnt_cvt + c_unit - c_one_i32) // c_unit
                padded_cvt = (raw_cnt_cvt == c_zero_i32).select(c_zero_i32, blocks_cvt * c_unit)
                _lds_store_raw(cumsum_mr, cvt_valid.select(padded_cvt, c_zero_i32), cvt_ix)
            gpu.barrier()

            if has_mask:
                for i_ep in range_constexpr(0, E, BS):
                    ep_eid = fx.Int32(i_ep) + tid
                    ep_valid = ep_eid < c_E
                    ep_safe_eid = ep_valid.select(ep_eid, c_zero_i32)
                    ep_m = buffer_ops.buffer_load(mask_rsrc, ep_safe_eid, vec_width=1, dtype=T.i32)
                    should_zero = ep_valid & (ep_m == c_zero_i32)
                    ep_cs_ix = ArithValue(ep_valid.select(ep_eid + c_one_i32, c_zero_i32)).index_cast(T.index)
                    _lds_store_raw(
                        cumsum_mr, should_zero.select(c_zero_i32, _lds_load_raw(cumsum_mr, ep_cs_ix)), ep_cs_ix
                    )
                gpu.barrier()

            # Prefix sum over cumsum[1..E] into cumdup[1..E], wave 0 only, in
            # E/WARP_SIZE serial chunks chained through a ds_bpermute carry.
            # A block-wide variant (every wave scans WARP_SIZE experts, wave
            # totals combined through an LDS scratch row) was tried and is NOT
            # worth it: 12.62 -> 12.53 us, and it faulted at block_size 256/512
            # while passing at 1024. Not on the critical path; left as upstream.
            is_wave0 = wave == c_zero_i32
            prev_chunk_total = c_zero_i32

            for chunk_start in range_constexpr(0, E, WARP_SIZE):
                eid_ps = fx.Int32(chunk_start) + lane
                eid_ps_valid = is_wave0 & (eid_ps < c_E)
                safe_eid_ps = eid_ps_valid.select(eid_ps + c_one_i32, c_zero_i32)
                ps_ix = ArithValue(safe_eid_ps).index_cast(T.index)
                val = eid_ps_valid.select(_lds_load_raw(cumsum_mr, ps_ix), c_zero_i32)

                val = _dpp_intra_wave_prefix_sum(val, lane, WARP_SIZE)
                val = val + prev_chunk_total

                _lds_store_raw(
                    cumdup_mr, eid_ps_valid.select(val, c_zero_i32), eid_ps_valid.select(eid_ps + c_one_i32, c_zero_i32)
                )

                last_addr = fx.Int32((WARP_SIZE - 1) * 4)
                prev_chunk_total = fly_rocdl.ds_bpermute(T.i32, last_addr, val)
                prev_chunk_total = fx.Int32(prev_chunk_total)

            _lds_store_raw(cumdup_mr, is_t0.select(c_zero_i32, _lds_load_raw(cumdup_mr, c_zero_i32)), c_zero_i32)
            gpu.barrier()

            cs_E_ix_ps = ArithValue(c_E).index_cast(T.index)
            total_padded = _lds_load_raw(cumdup_mr, cs_E_ix_ps)
            buffer_ops.buffer_store(total_padded, nvalid_rsrc, c_zero_i32)
            buffer_ops.buffer_store(tokens, nvalid_rsrc, c_one_i32)
            gpu.barrier()

            for i_cp in range_constexpr(0, E + 1, BS):
                cp_idx = fx.Int32(i_cp) + tid
                cp_valid = cp_idx <= c_E
                safe_cp_idx = cp_valid.select(cp_idx, c_zero_i32)
                cp_ix = ArithValue(safe_cp_idx).index_cast(T.index)
                cp_val = _lds_load_raw(cumdup_mr, cp_ix)
                _lds_store_raw(cumsum_mr, cp_val, cp_ix)
            gpu.barrier()

            if has_mask:
                for i_ml in range_constexpr(0, E, BS):
                    ml_eid = fx.Int32(i_ml) + tid
                    ml_valid = ml_eid < c_E
                    safe_ml_eid = ml_valid.select(ml_eid, c_zero_i32)
                    ml_mask = buffer_ops.buffer_load(mask_rsrc, safe_ml_eid, vec_width=1, dtype=T.i32)
                    ml_val = ml_valid.select(ml_mask, c_zero_i32)
                    ml_ix = ArithValue(ml_valid.select(ml_eid + c_one_i32, c_zero_i32)).index_cast(T.index)
                    _lds_store_raw(cumdup_mr, ml_val, ml_ix)
                _lds_store_raw(cumdup_mr, is_t0.select(c_zero_i32, _lds_load_raw(cumdup_mr, c_zero_i32)), c_zero_i32)
                gpu.barrier()

                prev_chunk_total_m = c_zero_i32
                for chunk_start_m in range_constexpr(0, E, WARP_SIZE):
                    eid_m = fx.Int32(chunk_start_m) + lane
                    eid_m_valid = is_wave0 & (eid_m < c_E)
                    safe_eid_m = eid_m_valid.select(eid_m + c_one_i32, c_zero_i32)
                    m_ix = ArithValue(safe_eid_m).index_cast(T.index)
                    mval = eid_m_valid.select(_lds_load_raw(cumdup_mr, m_ix), c_zero_i32)

                    mval = _dpp_intra_wave_prefix_sum(mval, lane, WARP_SIZE)
                    mval = mval + prev_chunk_total_m
                    _lds_store_raw(
                        cumdup_mr,
                        eid_m_valid.select(mval, c_zero_i32),
                        eid_m_valid.select(eid_m + c_one_i32, c_zero_i32),
                    )

                    last_addr_m = fx.Int32((WARP_SIZE - 1) * 4)
                    prev_chunk_total_m = fly_rocdl.ds_bpermute(T.i32, last_addr_m, mval)
                    prev_chunk_total_m = fx.Int32(prev_chunk_total_m)

                _lds_store_raw(cumdup_mr, is_t0.select(c_zero_i32, _lds_load_raw(cumdup_mr, c_zero_i32)), c_zero_i32)
                gpu.barrier()
            else:
                for i_ml in range_constexpr(0, E, BS):
                    ml_eid = fx.Int32(i_ml) + tid
                    ml_valid = ml_eid < c_E
                    safe_ml_eid = ml_valid.select(ml_eid, c_zero_i32)
                    ml_ix = ArithValue(safe_ml_eid).index_cast(T.index)
                    _lds_store_raw(cumdup_mr, ml_valid.select(safe_ml_eid, c_zero_i32), ml_ix)
                gpu.barrier()

            for i_eid in range_constexpr(0, E, BS):
                eid_wr = fx.Int32(i_eid) + tid
                eid_wr_valid = eid_wr < c_E
                safe_eid_wr = eid_wr_valid.select(eid_wr, c_zero_i32)

                cs_start_ix = ArithValue(safe_eid_wr).index_cast(T.index)
                cs_end_ix = ArithValue(safe_eid_wr + c_one_i32).index_cast(T.index)
                e_start = _lds_load_raw(cumsum_mr, cs_start_ix)
                e_end = eid_wr_valid.select(_lds_load_raw(cumsum_mr, cs_end_ix), e_start)
                local_eid = _lds_load_raw(cumdup_mr, cs_start_ix)

                _lds_store_raw(cumdup_mr, e_start, cs_start_ix)

                blk_start = e_start // c_unit
                blk_end = e_end // c_unit
                # An expert receives at most `max_tokens` rows, so it spans at
                # most ceil(max_tokens / unit_size) GEMM tiles. Upstream loops
                # `max_tokens` times regardless, which at unit_size=32 issues
                # 16 predicated stores per expert to write one.
                for j_blk in range_constexpr(_blocks_per_expert):
                    blk_idx = blk_start + fx.Int32(j_blk)
                    blk_valid = eid_wr_valid & (blk_idx < blk_end)
                    safe_blk = blk_valid.select(blk_idx, c_oob_idx)
                    buffer_ops.buffer_store(local_eid, sorted_e_rsrc, safe_blk)
            gpu.barrier()

            cs_E_ix = ArithValue(c_E).index_cast(T.index)
            cumE = _lds_load_raw(cumsum_mr, cs_E_ix)
            _lds_store_raw(cumdup_mr, cumE, cs_E_ix)
            gpu.barrier()

            # ====================== PHASE 3: Scatter =========================
            for i_e2 in range_constexpr(0, E, BS // 8):
                eid_sc = fx.Int32(i_e2) + lane_group_id
                eid_sc_valid = eid_sc < c_E
                safe_eid_sc = eid_sc_valid.select(eid_sc, c_E)

                sc_expert_enabled = eid_sc_valid
                if has_mask:
                    sc_mask_val = buffer_ops.buffer_load(
                        mask_rsrc, eid_sc_valid.select(eid_sc, c_zero_i32), vec_width=1, dtype=T.i32
                    )
                    sc_expert_enabled = eid_sc_valid & (sc_mask_val != c_zero_i32)

                cs_sc_ix = ArithValue(safe_eid_sc).index_cast(T.index)
                position = _lds_load_raw(cumsum_mr, cs_sc_ix)

                for i_sub2 in range_constexpr(0, sub_tokens, 8):
                    my_sub = fx.Int32(i_sub2) + lane_group_os
                    my_sub_valid = sc_expert_enabled & (my_sub < c_sub_tokens)
                    safe_my_sub = my_sub_valid.select(my_sub, c_zero_i32)
                    my_mesh_addr = safe_my_sub * c_smem_cols + safe_eid_sc
                    my_mesh_ix = ArithValue(my_mesh_addr).index_cast(T.index)
                    my_x = _lds_load_raw(mesh_mr, my_mesh_ix)
                    my_has_token = my_sub_valid & (my_x != c_zero_i32)
                    local_cnt = my_has_token.select(c_one_i32, c_zero_i32)

                    cnt_raw = _unwrap_val(local_cnt)
                    zero_raw = _unwrap_val(c_zero_i32)

                    remote = fly_rocdl.update_dpp(
                        T.i32, zero_raw, cnt_raw, DPP_ROW_SHR_1, DPP_ROW_MASK, DPP_BANK_MASK, True
                    )
                    should_add = lane_group_os >= c_one_i32
                    local_cnt = should_add.select(local_cnt + fx.Int32(remote), local_cnt)

                    cnt_raw = _unwrap_val(local_cnt)
                    remote = fly_rocdl.update_dpp(
                        T.i32, zero_raw, cnt_raw, DPP_ROW_SHR_2, DPP_ROW_MASK, DPP_BANK_MASK, True
                    )
                    should_add = lane_group_os >= fx.Int32(2)
                    local_cnt = should_add.select(local_cnt + fx.Int32(remote), local_cnt)

                    cnt_raw = _unwrap_val(local_cnt)
                    remote = fly_rocdl.update_dpp(
                        T.i32, zero_raw, cnt_raw, DPP_ROW_SHR_4, DPP_ROW_MASK, DPP_BANK_MASK, True
                    )
                    should_add = lane_group_os >= fx.Int32(4)
                    local_cnt = should_add.select(local_cnt + fx.Int32(remote), local_cnt)

                    last_lane_of_group = tid | fx.Int32(7)
                    last_addr = last_lane_of_group * c4_i32
                    batch_total = fly_rocdl.ds_bpermute(T.i32, last_addr, local_cnt)
                    batch_total = fx.Int32(batch_total)

                    slot = position + local_cnt - c_one_i32
                    safe_x = my_has_token.select(my_x, c_one_i32)
                    topk_slot_sc = safe_x - c_one_i32
                    packed_id = (topk_slot_sc << fx.Int32(24)) | my_sub
                    safe_slot = my_has_token.select(slot, c_oob_idx)
                    buffer_ops.buffer_store(packed_id, sorted_ids_rsrc, safe_slot)

                    # Fused: the weight comes from LDS (gating staged it there)
                    # instead of HBM. `my_sub` is the per-token local index
                    # 0..max_tokens-1; topk_slot_sc identifies the K rank.
                    w_lds_idx = my_has_token.select(my_sub * c_topk + topk_slot_sc, c_zero_i32)
                    w_val_i32 = _lds_load_raw(weights_lds_mr, w_lds_idx)
                    buffer_ops.buffer_store(w_val_i32, sorted_w_rsrc, safe_slot)

                    position = position + batch_total

                _lds_store_raw(cumsum_mr, position, cs_sc_ix)
            gpu.barrier()

            # ====================== PHASE 4: sentinel padding ================
            # The tail of each expert's padded tile gets the (topk<<24)|tokens
            # sentinel and weight 0. Upstream maps one thread per expert and
            # loops `unit_size` slots: consecutive lanes then write addresses
            # `unit_size` dwords apart, so every store instruction fans out to
            # one cache line per lane. Mapping `unit_size` consecutive lanes to
            # one expert instead makes each store a handful of contiguous runs.
            sentinel_val = c_sentinel | tokens
            c_zero_as_i32 = c_zero_i32
            pad_slot_in_blk = tid % c_unit
            pad_grp = tid // c_unit
            for i_pad in range_constexpr(0, E, _pad_experts_per_step):
                eid_pad = fx.Int32(i_pad) + pad_grp
                pad_valid = eid_pad < c_E
                safe_eid_pad = pad_valid.select(eid_pad, c_zero_i32)

                cs_pad_ix = ArithValue(safe_eid_pad).index_cast(T.index)
                cdp_ix = ArithValue(safe_eid_pad + c_one_i32).index_cast(T.index)
                pad_start = _lds_load_raw(cumsum_mr, cs_pad_ix)
                pad_end = pad_valid.select(_lds_load_raw(cumdup_mr, cdp_ix), pad_start)

                # padded_count - count < unit_size always, so one slot per
                # thread covers the whole gap.
                pad_slot = pad_start + pad_slot_in_blk
                pad_slot_valid = pad_valid & (pad_slot < pad_end)
                safe_pad_slot = pad_slot_valid.select(pad_slot, c_oob_idx)
                buffer_ops.buffer_store(sentinel_val, sorted_ids_rsrc, safe_pad_slot)
                buffer_ops.buffer_store(c_zero_as_i32, sorted_w_rsrc, safe_pad_slot)

    @flyc.jit
    def launch_moe_sorting_oneshot_fused(
        gating_logits: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids_out: fx.Tensor,
        moe_buf: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        n_grid_blocks: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = moe_sorting_oneshot_fused_kernel(
            gating_logits,
            sorted_token_ids,
            sorted_weights_out,
            sorted_expert_ids,
            num_valid_ids_out,
            moe_buf,
            expert_mask_tensor,
            i32_tokens,
            i32_moe_buf_elems,
        )
        launcher.launch(
            grid=(n_grid_blocks, 1, 1),
            block=(BS, 1, 1),
            stream=stream,
        )

    return launch_moe_sorting_oneshot_fused


# ---------------------------------------------------------------------------
# Host entry
# ---------------------------------------------------------------------------
_oneshot_fused_cf_cache = {}
_dummy_mask_cache = {}
_dummy_buf_cache = {}

_DTYPE_STR = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
}


@functools.lru_cache(maxsize=64)
def fused_route_sort_supported(
    num_experts: int, topk: int, max_tokens: int, dtype_str: str, block_size: int = BLOCK_SIZE
) -> bool:
    """Whether the fused kernel can be compiled for this shape."""
    if dtype_str not in ("bf16", "f16", "f32"):
        return False
    try:
        layout = _compute_topk_gating_layout(
            num_experts, topk, dtype_str, block_size=block_size, min_tokens_per_block=max_tokens
        )
    except ValueError:
        return False
    if layout["TOKENS_PER_BLOCK"] < max_tokens:
        return False
    return compute_fused_sub_tokens(num_experts) >= max_tokens


def _pad_max_tokens(m: int) -> int:
    """LDS mesh rows are allocated in multiples of 8, with a floor of 8."""
    return ((max(m, 8) + 7) // 8) * 8


def flydsl_fused_route_sort(
    gating_logits,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    num_experts,
    topk,
    unit_size=UNIT_SIZE,
    expert_mask=None,
    renormalize=True,
    max_tokens=None,
    block_size=BLOCK_SIZE,
):
    """Fused router + sort. All output tensors are caller-allocated.

    ``max_tokens`` fixes the LDS geometry and the compiled kernel variant. Pass
    the CUDA-graph capacity (not the live token count) so capture and replay
    select the same kernel; it defaults to ``gating_logits.shape[0]``.
    """
    M = gating_logits.shape[0]
    if max_tokens is None:
        max_tokens = M
    max_tokens = _pad_max_tokens(max_tokens)

    device = gating_logits.device
    if moe_buf.numel():
        moe_buf_i32 = moe_buf.view(torch.int32)
        moe_buf_elems = moe_buf_i32.numel()
    else:
        # `reduce`-epilogue stage 2 owns its own [M, topk, model_dim] partial,
        # so moe_sorting gets a (0, 0) placeholder and there is nothing to
        # zero. A 0-element tensor has no usable data_ptr for the buffer
        # descriptor, so pass a cached 1-element dummy with elems = 0.
        moe_buf_i32 = _dummy_buf_cache.get(device)
        if moe_buf_i32 is None:
            moe_buf_i32 = torch.zeros(1, dtype=torch.int32, device=device)
            _dummy_buf_cache[device] = moe_buf_i32
        moe_buf_elems = 0

    has_mask = expert_mask is not None
    if has_mask:
        mask_tensor = expert_mask
    else:
        mask_tensor = _dummy_mask_cache.get(device)
        if mask_tensor is None:
            mask_tensor = torch.ones(1, dtype=torch.int32, device=device)
            _dummy_mask_cache[device] = mask_tensor

    dtype_str = _DTYPE_STR[gating_logits.dtype]

    target_occupancy = 2
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    n_zero_blocks = min((moe_buf_elems + block_size - 1) // block_size, num_cu * target_occupancy)
    n_grid_blocks = 1 + n_zero_blocks

    cache_key = (
        num_experts,
        topk,
        max_tokens,
        unit_size,
        n_grid_blocks,
        dtype_str,
        renormalize,
        has_mask,
        block_size,
    )
    args = (
        gating_logits,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf_i32,
        mask_tensor,
        M,
        moe_buf_elems,
        n_grid_blocks,
    )
    stream = torch.cuda.current_stream()
    cf = _oneshot_fused_cf_cache.get(cache_key)
    if cf is not None:
        cf(*args, fx.Stream(stream))
    else:
        launch_fn = compile_moe_sorting_oneshot_fused(
            num_experts=num_experts,
            topk=topk,
            dtype_str=dtype_str,
            renormalize=renormalize,
            max_tokens=max_tokens,
            unit_size=unit_size,
            has_mask=has_mask,
            block_size=block_size,
        )
        launch_fn(*args, stream=stream)
        _oneshot_fused_cf_cache[cache_key] = flyc.compile(launch_fn, *args, fx.Stream(stream))

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf
