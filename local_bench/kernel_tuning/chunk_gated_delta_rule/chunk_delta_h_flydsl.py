"""MI355X BF16 K128/BV8 chunk-state recurrence with explicit MFMA fragments.

Two kernels share the same math:

``build``        the original shape-specialized kernel. ``length`` is baked in
                 as the loop trip count, so it only accepts one sequence whose
                 token count is a multiple of 64, and every distinct length is a
                 separate compile.
``build_varlen`` the general kernel. Sequence count, per-sequence length and
                 chunk count are all runtime values, so one compile serves every
                 batch shape. The ragged tail chunk is masked by hardware: each
                 per-sequence buffer descriptor stops at that sequence's last
                 token, so loads past it read 0 and stores past it are dropped.

``chunk_gated_delta_rule_fwd_h`` picks between them; see ``_static_allowed``.
"""

import os
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

PREFETCH_DEPTH = 2
CHUNK_SIZE = 64
VALUE_TILE = 8


@flyc.jit
def hw_exp2(x):
    return fx.Float32(fx.rocdl.exp2(T.f32, fx.Float32(x).ir_value()))


@flyc.jit
def load_vec(buf, offset, width: fx.Constexpr[int], dtype: fx.Constexpr):
    view = fx.make_view(fx.get_iter(buf) + offset, fx.make_layout(width, 1))
    reg = fx.make_rmem_tensor(width, dtype)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(width * dtype.width), dtype)
    fx.copy_atom_call(atom, view, reg)
    return reg.load()


@flyc.jit
def store_vec(buf, offset, value, dtype: fx.Constexpr):
    reg = fx.make_rmem_tensor(value.numel, dtype)
    reg.store(value)
    view = fx.make_view(fx.get_iter(buf) + offset, fx.make_layout(value.numel, 1))
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(value.numel * dtype.width), dtype)
    fx.copy_atom_call(atom, reg, view)


@flyc.jit
def lds_load(ptr, offset, width: fx.Constexpr[int]):
    reg = fx.make_rmem_tensor(width, fx.BFloat16)
    view = fx.make_view(ptr + offset, fx.make_layout(width, 1))
    atom = fx.make_copy_atom(fx.UniversalCopy(width * 16), fx.BFloat16)
    fx.copy_atom_call(atom, view, reg)
    return reg.load()


@flyc.jit
def lds_store(ptr, offset, value):
    reg = fx.make_rmem_tensor(value.numel, fx.BFloat16)
    reg.store(value)
    view = fx.make_view(ptr + offset, fx.make_layout(value.numel, 1))
    atom = fx.make_copy_atom(fx.UniversalCopy(value.numel * 16), fx.BFloat16)
    fx.copy_atom_call(atom, reg, view)


@flyc.jit
def mma(a, b, c):
    ra = fx.make_rmem_tensor(8, fx.BFloat16)
    rb = fx.make_rmem_tensor(8, fx.BFloat16)
    rc = fx.make_rmem_tensor(4, fx.Float32)
    ra.store(a)
    rb.store(b)
    rc.store(c)
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
    fx.mma_atom_call(atom, rc, ra, rb, rc)
    return rc.load()


@lru_cache(maxsize=128)
def build(
    length: int,
    H: int,
    Hg: int,
    V: int,
    state_stride: int,
    depth: int = 2,
    has_state: bool = True,
    has_g: bool = True,
    has_gk: bool = False,
    save: bool = True,
    use_exp2: bool = False,
):
    @fx.struct
    class Storage:
        h: fx.Array[fx.BFloat16, 16 * 128, 16]
        v: fx.Array[fx.BFloat16, 16 * 64, 16]
        k: fx.Array[fx.BFloat16, 2 * 64 * 128, 16]

    @flyc.kernel
    def kernel(
        k: fx.Tensor,
        w: fx.Tensor,
        u: fx.Tensor,
        g: fx.Tensor,
        gk: fx.Tensor,
        state: fx.Tensor,
        indices: fx.Tensor,
        h: fx.Tensor,
        vn: fx.Tensor,
    ):
        tid = fx.Int64(fx.thread_idx.x)
        lane = tid % 64
        wave = tid // 64
        head = fx.Int64(fx.block_idx.x)
        vb = fx.Int64(fx.block_idx.y) * VALUE_TILE
        # MFMA keeps 16 physical columns; only VALUE_TILE columns own values.
        # The remaining columns have zero state/input and never write globally.
        # Keep all lanes active for MFMA, LDS staging and workgroup barriers.
        col = lane % 16
        group = lane // 16
        kr = wave * 16 + group * 4
        slot = fx.Int64(indices[0])
        sb = head * V * 128 + (vb + col) * 128
        kb = fx.rocdl.make_buffer_tensor(k)
        wb = fx.rocdl.make_buffer_tensor(w)
        ub = fx.rocdl.make_buffer_tensor(u)
        hb = fx.rocdl.make_buffer_tensor(h)
        ob = fx.rocdl.make_buffer_tensor(vn)
        # Rebase the 64-bit pointer before creating the 32-bit-offset buffer resource.
        state_slot = fx.make_view(fx.get_iter(state) + slot * state_stride, fx.make_layout(H * V * 128, 1))
        ss = fx.rocdl.make_buffer_tensor(state_slot)
        gkb = fx.rocdl.make_buffer_tensor(gk)
        gg = fx.Tensor(fx.make_view(fx.get_iter(g), fx.make_layout(length * H, 1)))
        storage = fx.SharedAllocator().allocate(Storage).peek()
        hp = storage.h.ptr
        vp = storage.v.ptr
        kp_base = storage.k.ptr
        h1 = fx.Vector.filled(4, 0.0, fx.Float32)
        h2 = fx.Vector.filled(4, 0.0, fx.Float32)

        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        tiled = fx.make_tiled_mma(atom, fx.make_layout((4, 1, 1), (1, 4, 0)))
        cp = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans16_64b(), fx.BFloat16)
        tc = fx.make_tiled_copy_A(cp, tiled).get_slice(fx.Int32(tid))
    
        if const_expr(has_state):
            if (slot >= 0) & (col < VALUE_TILE):
                h1 = load_vec(ss, sb + kr, 4, fx.BFloat16).to(fx.Float32)
                h2 = load_vec(ss, sb + kr + 64, 4, fx.BFloat16).to(fx.Float32)

        def prefetch(chunk):
            kw = []
            for j in range_constexpr(4):
                tr = tid // 16 + j * 16
                kk = tid % 16 * 8
                kw.append(load_vec(kb, ((chunk * 64 + tr) * Hg + head // (H // Hg)) * 128 + kk, 8, fx.BFloat16))
            for j in range_constexpr(4):
                kw.append(
                    load_vec(wb, ((chunk * 64 + wave * 16 + col) * H + head) * 128 + j * 32 + group * 8, 8, fx.BFloat16)
                )
            uv = []
            gv = []
            for e in range_constexpr(4):
                off = ((chunk * 64 + kr + e) * H + head) * V + vb + col
                uv_value = fx.Float32(0.0)
                if col < VALUE_TILE:
                    uv_value = load_vec(ub, off, 1, fx.BFloat16)[0].to(fx.Float32)
                uv.append(uv_value)
                gv_value = fx.Float32(0.0)
                if const_expr(has_g):
                    gv_value = fx.Float32(gg[(chunk * 64 + kr + e) * H + head])
                gv.append(gv_value)
            kw.append(fx.Vector.from_elements(uv, fx.Float32))
            kw.append(fx.Vector.from_elements(gv, fx.Float32))
            last_g = fx.Float32(0.0)
            if const_expr(has_g):
                last_g = fx.Float32(gg[(chunk * 64 + 63) * H + head])
            kw.append(last_g)
            return kw

        # Carry immutable global tiles independently of the recurrent FP32 state.
        prefetched = []
        for stage in range_constexpr(depth):
            prefetched += prefetch(fx.Int64(stage % (length // 64)))
        for it, carry in range(fx.Int64(0), fx.Int64(length // 64), fx.Int64(1), init=[h1, h2] + prefetched):
            it = fx.Int64(it)
            a1, a2 = carry[0], carry[1]
            # Alternate k storage: a wave may still be reading the previous tile.
            kp = kp_base + (it % 2) * 8192
            next_data = prefetch((it + depth) % (length // 64))
            outbase = ((it * H + head) * V + vb + col) * 128 + kr
            if col < VALUE_TILE:
                store_vec(hb, outbase, a1.to(fx.BFloat16), fx.BFloat16)
                store_vec(hb, outbase + 64, a2.to(fx.BFloat16), fx.BFloat16)
            lds_store(hp, col * 128 + (kr ^ ((col % 8) * 8)), a1.to(fx.BFloat16))
            lds_store(hp, col * 128 + ((kr + 64) ^ ((col % 8) * 8)), a2.to(fx.BFloat16))
            for j in range_constexpr(4):
                tr = tid // 16 + j * 16
                kk = tid % 16 * 8
                kval = carry[2 + j]
                lds_store(kp, tr * 128 + (kk ^ ((tr % 4) * 16)), kval)
            fx.gpu.barrier()
            accum = []
            bv = fx.Vector.filled(4, 0.0, fx.Float32)
            for half in range_constexpr(2):
                for sub in range_constexpr(2):
                    kk = half * 64 + sub * 32 + group * 8
                    wa = carry[6 + half * 2 + sub]
                    hh = lds_load(hp, col * 128 + (kk ^ ((col % 8) * 8)), 8)
                    bv = mma(wa, hh, bv)
                accum.append(bv)
            predicted = accum[1]
            vals = []
            decay = hw_exp2(carry[12] * 1.4426950408889634)
            for e in range_constexpr(4):
                tr = kr + e
                off = ((it * 64 + tr) * H + head) * V + vb + col
                uv = carry[10][e]
                vv = uv - predicted[e]
                if const_expr(save):
                    if col < VALUE_TILE:
                        store_vec(ob, off, fx.Vector.from_elements([vv.to(fx.BFloat16)], fx.BFloat16), fx.BFloat16)
                diff = carry[12] - carry[11][e]
                gate = hw_exp2((diff <= 0).select(diff, float("-inf")) * 1.4426950408889634)
                vals.append((vv * gate).to(fx.BFloat16))
            lds_store(vp, col * 64 + (kr ^ ((col % 8) * 8)), fx.Vector.from_elements(vals, fx.BFloat16))
            fx.gpu.barrier()
            updates = []
            for half in range_constexpr(2):
                thr = tiled.thr_slice(fx.Int32(tid))
                ka = fx.make_view(
                    kp + half * 64,
                    fx.make_composed_layout(fx.static(fx.SwizzleType.get(2, 4, 3)), fx.make_layout((64, 64), (1, 128))),
                )
                frag = thr.make_fragment_A(ka)
                fx.copy(cp, tc.partition_S(ka), tc.retile(frag))
                update = fx.Vector.filled(4, 0.0, fx.Float32)
                for sub in range_constexpr(2):
                    tr = sub * 32 + group * 8
                    kv = frag[None, None, sub].load()
                    vv = lds_load(vp, col * 64 + (tr ^ ((col % 8) * 8)), 8)
                    update = mma(kv, vv, update)
                updates.append(update)
            factor1 = fx.Vector.filled(4, decay, fx.Float32)
            factor2 = factor1
            if const_expr(has_gk):
                channels1 = load_vec(gkb, ((it * 64 + 63) * H + head) * 128 + kr, 4, fx.Float32)
                channels2 = load_vec(gkb, ((it * 64 + 63) * H + head) * 128 + kr + 64, 4, fx.Float32)
                scale1 = []
                scale2 = []
                for e in range_constexpr(4):
                    exponent1 = channels1[e]
                    exponent2 = channels2[e]
                    if const_expr(not use_exp2):
                        exponent1 = exponent1 * 1.4426950408889634
                        exponent2 = exponent2 * 1.4426950408889634
                    scale1.append(hw_exp2(exponent1))
                    scale2.append(hw_exp2(exponent2))
                if const_expr(has_g):
                    a1 = a1 * decay
                    a2 = a2 * decay
                factor1 = fx.Vector.from_elements(scale1, fx.Float32)
                factor2 = fx.Vector.from_elements(scale2, fx.Float32)
            result = yield [fx.fma(a1, factor1, updates[0]), fx.fma(a2, factor2, updates[1])] + carry[13:] + next_data
        if const_expr(has_state):
            if (slot >= 0) & (col < VALUE_TILE):
                store_vec(ss, sb + kr, result[0].to(fx.BFloat16), fx.BFloat16)
                store_vec(ss, sb + kr + 64, result[1].to(fx.BFloat16), fx.BFloat16)

    @flyc.jit
    def launch(
        k: fx.Tensor,
        w: fx.Tensor,
        u: fx.Tensor,
        g: fx.Tensor,
        gk: fx.Tensor,
        state: fx.Tensor,
        indices: fx.Tensor,
        h: fx.Tensor,
        vn: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(k, w, u, g, gk, state, indices, h, vn).launch(grid=(H, V // VALUE_TILE), block=(256,), stream=stream)

    return launch


@lru_cache(maxsize=64)
def build_varlen(
    H: int,
    Hg: int,
    V: int,
    state_stride: int,
    depth: int = 2,
    has_state: bool = True,
    has_g: bool = True,
    has_gk: bool = False,
    save: bool = True,
    use_exp2: bool = False,
):
    """Shape-generic twin of ``build``.

    ``length`` is gone from the cache key: sequence count, per-sequence length
    and chunk count arrive as runtime values, so a single compiled module serves
    every batch shape. FlyDSL erases tensor extents from its own module cache key
    (``dynamic_layout=True``), so nothing here re-specializes per shape.
    """

    @fx.struct
    class Storage:
        h: fx.Array[fx.BFloat16, 16 * 128, 16]
        v: fx.Array[fx.BFloat16, 16 * 64, 16]
        k: fx.Array[fx.BFloat16, 2 * 64 * 128, 16]

    @flyc.kernel
    def kernel(
        k: fx.Tensor,
        w: fx.Tensor,
        u: fx.Tensor,
        g: fx.Tensor,
        gk: fx.Tensor,
        state: fx.Tensor,
        indices: fx.Tensor,
        h: fx.Tensor,
        vn: fx.Tensor,
        cu: fx.Tensor,
        coff: fx.Tensor,
    ):
        tid = fx.Int64(fx.thread_idx.x)
        lane = tid % 64
        wave = tid // 64
        head = fx.Int64(fx.block_idx.x)
        vb = fx.Int64(fx.block_idx.y) * VALUE_TILE
        seq = fx.Int64(fx.block_idx.z)
        # MFMA keeps 16 physical columns; only VALUE_TILE columns own values.
        # The remaining columns have zero state/input and never write globally.
        # Keep all lanes active for MFMA, LDS staging and workgroup barriers.
        col = lane % 16
        group = lane // 16
        kr = wave * 16 + group * 4

        bos = fx.Int64(cu[seq])
        eos = fx.Int64(cu[seq + 1])
        tn = eos - bos
        nt = (tn + 63) // 64
        last_chunk = nt - 1
        boh = fx.Int64(coff[seq])
        slot = fx.Int64(indices[seq])
        sb = head * V * 128 + (vb + col) * 128

        # Per-sequence buffer bounds. Everything is indexed from the tensor base
        # with an absolute token id, so capping the descriptor at `eos` masks the
        # ragged tail chunk in hardware: reads past the sequence return 0 and
        # writes past it are dropped. The dropped writes matter as much as the
        # reads -- an unmasked v_new tail would land in the NEXT sequence's rows.
        kb = fx.rocdl.make_buffer_tensor(k, max_size=False, num_records_bytes=fx.Int64(eos * (Hg * 128 * 2)))
        wb = fx.rocdl.make_buffer_tensor(w, max_size=False, num_records_bytes=fx.Int64(eos * (H * 128 * 2)))
        ub = fx.rocdl.make_buffer_tensor(u, max_size=False, num_records_bytes=fx.Int64(eos * (H * V * 2)))
        ob = fx.rocdl.make_buffer_tensor(vn, max_size=False, num_records_bytes=fx.Int64(eos * (H * V * 2)))
        gb = fx.rocdl.make_buffer_tensor(g, max_size=False, num_records_bytes=fx.Int64(eos * (H * 4)))
        gkb = fx.rocdl.make_buffer_tensor(gk, max_size=False, num_records_bytes=fx.Int64(eos * (H * 128 * 4)))
        # h is chunk-indexed, never token-indexed: every (boh + it) is in range.
        hb = fx.rocdl.make_buffer_tensor(h)
        # Rebase the 64-bit pointer before creating the 32-bit-offset buffer resource.
        state_slot = fx.make_view(fx.get_iter(state) + slot * state_stride, fx.make_layout(H * V * 128, 1))
        ss = fx.rocdl.make_buffer_tensor(state_slot)
        storage = fx.SharedAllocator().allocate(Storage).peek()
        hp = storage.h.ptr
        vp = storage.v.ptr
        kp_base = storage.k.ptr
        h1 = fx.Vector.filled(4, 0.0, fx.Float32)
        h2 = fx.Vector.filled(4, 0.0, fx.Float32)

        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        tiled = fx.make_tiled_mma(atom, fx.make_layout((4, 1, 1), (1, 4, 0)))
        cp = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans16_64b(), fx.BFloat16)
        tc = fx.make_tiled_copy_A(cp, tiled).get_slice(fx.Int32(tid))

        if const_expr(has_state):
            if (slot >= 0) & (col < VALUE_TILE):
                h1 = load_vec(ss, sb + kr, 4, fx.BFloat16).to(fx.Float32)
                h2 = load_vec(ss, sb + kr + 64, 4, fx.BFloat16).to(fx.Float32)

        def clamp_chunk(c):
            # Trailing prefetches have nothing left to read; re-reading the last
            # chunk keeps every address in range and the result is discarded.
            c = fx.Int64(c)
            return (c <= last_chunk).select(c, last_chunk)

        def chunk_last_token(c):
            # Reference: last_idx = min((i_t + 1) * BT, T) - 1. The decay gate
            # must sample g at the last VALID token, not at chunk*64 + 63.
            end = (fx.Int64(c) + 1) * 64
            return (end <= tn).select(end, tn) - 1

        def prefetch(chunk):
            base = bos + chunk * 64
            kw = []
            for j in range_constexpr(4):
                tr = tid // 16 + j * 16
                kk = tid % 16 * 8
                kw.append(load_vec(kb, ((base + tr) * Hg + head // (H // Hg)) * 128 + kk, 8, fx.BFloat16))
            for j in range_constexpr(4):
                kw.append(
                    load_vec(wb, ((base + wave * 16 + col) * H + head) * 128 + j * 32 + group * 8, 8, fx.BFloat16)
                )
            uv = []
            gv = []
            for e in range_constexpr(4):
                off = ((base + kr + e) * H + head) * V + vb + col
                uv_value = fx.Float32(0.0)
                if col < VALUE_TILE:
                    uv_value = load_vec(ub, off, 1, fx.BFloat16)[0].to(fx.Float32)
                uv.append(uv_value)
                gv_value = fx.Float32(0.0)
                if const_expr(has_g):
                    gv_value = load_vec(gb, (base + kr + e) * H + head, 1, fx.Float32)[0]
                gv.append(gv_value)
            kw.append(fx.Vector.from_elements(uv, fx.Float32))
            kw.append(fx.Vector.from_elements(gv, fx.Float32))
            last_g = fx.Float32(0.0)
            if const_expr(has_g):
                last_g = load_vec(gb, (bos + chunk_last_token(chunk)) * H + head, 1, fx.Float32)[0]
            kw.append(last_g)
            return kw

        # Carry immutable global tiles independently of the recurrent FP32 state.
        prefetched = []
        for stage in range_constexpr(depth):
            prefetched += prefetch(clamp_chunk(stage))
        for it, carry in range(fx.Int64(0), nt, fx.Int64(1), init=[h1, h2] + prefetched):
            it = fx.Int64(it)
            a1, a2 = carry[0], carry[1]
            # Alternate k storage: a wave may still be reading the previous tile.
            kp = kp_base + (it % 2) * 8192
            next_data = prefetch(clamp_chunk(it + depth))
            outbase = (((boh + it) * H + head) * V + vb + col) * 128 + kr
            if col < VALUE_TILE:
                store_vec(hb, outbase, a1.to(fx.BFloat16), fx.BFloat16)
                store_vec(hb, outbase + 64, a2.to(fx.BFloat16), fx.BFloat16)
            lds_store(hp, col * 128 + (kr ^ ((col % 8) * 8)), a1.to(fx.BFloat16))
            lds_store(hp, col * 128 + ((kr + 64) ^ ((col % 8) * 8)), a2.to(fx.BFloat16))
            for j in range_constexpr(4):
                tr = tid // 16 + j * 16
                kk = tid % 16 * 8
                kval = carry[2 + j]
                lds_store(kp, tr * 128 + (kk ^ ((tr % 4) * 16)), kval)
            fx.gpu.barrier()
            accum = []
            bv = fx.Vector.filled(4, 0.0, fx.Float32)
            for half in range_constexpr(2):
                for sub in range_constexpr(2):
                    kk = half * 64 + sub * 32 + group * 8
                    wa = carry[6 + half * 2 + sub]
                    hh = lds_load(hp, col * 128 + (kk ^ ((col % 8) * 8)), 8)
                    bv = mma(wa, hh, bv)
                accum.append(bv)
            predicted = accum[1]
            vals = []
            decay = hw_exp2(carry[12] * 1.4426950408889634)
            for e in range_constexpr(4):
                tr = kr + e
                off = ((bos + it * 64 + tr) * H + head) * V + vb + col
                uv = carry[10][e]
                vv = uv - predicted[e]
                if const_expr(save):
                    if col < VALUE_TILE:
                        store_vec(ob, off, fx.Vector.from_elements([vv.to(fx.BFloat16)], fx.BFloat16), fx.BFloat16)
                diff = carry[12] - carry[11][e]
                gate = hw_exp2((diff <= 0).select(diff, float("-inf")) * 1.4426950408889634)
                vals.append((vv * gate).to(fx.BFloat16))
            lds_store(vp, col * 64 + (kr ^ ((col % 8) * 8)), fx.Vector.from_elements(vals, fx.BFloat16))
            fx.gpu.barrier()
            updates = []
            for half in range_constexpr(2):
                thr = tiled.thr_slice(fx.Int32(tid))
                ka = fx.make_view(
                    kp + half * 64,
                    fx.make_composed_layout(fx.static(fx.SwizzleType.get(2, 4, 3)), fx.make_layout((64, 64), (1, 128))),
                )
                frag = thr.make_fragment_A(ka)
                fx.copy(cp, tc.partition_S(ka), tc.retile(frag))
                update = fx.Vector.filled(4, 0.0, fx.Float32)
                for sub in range_constexpr(2):
                    tr = sub * 32 + group * 8
                    kv = frag[None, None, sub].load()
                    vv = lds_load(vp, col * 64 + (tr ^ ((col % 8) * 8)), 8)
                    update = mma(kv, vv, update)
                updates.append(update)
            factor1 = fx.Vector.filled(4, decay, fx.Float32)
            factor2 = factor1
            if const_expr(has_gk):
                gklast = (bos + chunk_last_token(it)) * H + head
                channels1 = load_vec(gkb, gklast * 128 + kr, 4, fx.Float32)
                channels2 = load_vec(gkb, gklast * 128 + kr + 64, 4, fx.Float32)
                scale1 = []
                scale2 = []
                for e in range_constexpr(4):
                    exponent1 = channels1[e]
                    exponent2 = channels2[e]
                    if const_expr(not use_exp2):
                        exponent1 = exponent1 * 1.4426950408889634
                        exponent2 = exponent2 * 1.4426950408889634
                    scale1.append(hw_exp2(exponent1))
                    scale2.append(hw_exp2(exponent2))
                if const_expr(has_g):
                    a1 = a1 * decay
                    a2 = a2 * decay
                factor1 = fx.Vector.from_elements(scale1, fx.Float32)
                factor2 = fx.Vector.from_elements(scale2, fx.Float32)
            result = yield [fx.fma(a1, factor1, updates[0]), fx.fma(a2, factor2, updates[1])] + carry[13:] + next_data
        if const_expr(has_state):
            if (slot >= 0) & (col < VALUE_TILE):
                store_vec(ss, sb + kr, result[0].to(fx.BFloat16), fx.BFloat16)
                store_vec(ss, sb + kr + 64, result[1].to(fx.BFloat16), fx.BFloat16)

    @flyc.jit
    def launch(
        k: fx.Tensor,
        w: fx.Tensor,
        u: fx.Tensor,
        g: fx.Tensor,
        gk: fx.Tensor,
        state: fx.Tensor,
        indices: fx.Tensor,
        h: fx.Tensor,
        vn: fx.Tensor,
        cu: fx.Tensor,
        coff: fx.Tensor,
        n: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(k, w, u, g, gk, state, indices, h, vn, cu, coff).launch(
            grid=(H, V // VALUE_TILE, n), block=(256,), stream=stream
        )

    return launch


# Static specializations already built. The static kernel bakes `length` in, so
# admitting a new one costs a FlyDSL compile. Naturally 64-aligned batches are
# ~1/64 of real traffic, so compiling on demand would stall the serving loop for
# a path it rarely takes. Serving therefore uses varlen unless the exact
# specialization is already warm; benchmarks with a fixed ISL can opt in.
_STATIC_READY: set = set()
_STATIC_ON_DEMAND = os.environ.get("SGLANG_FLYDSL_GDN_STATIC", "0") == "1"


def _arg_signature(t: torch.Tensor):
    """Mirror FlyDSL's own module cache key: extents are runtime, so only dtype,
    rank and which axis carries unit stride may vary the compiled code."""
    strides = t.stride()
    return (t.dtype, t.dim(), strides.index(1) if 1 in strides else -1)


def _run(launch, args, shape_exact: bool, device: torch.device):
    with torch.cuda.device(device):
        stream = fx.Stream(torch.cuda.current_stream(device))
        if shape_exact:
            key = (device.index, tuple((t.dtype, tuple(t.shape), t.stride()) for t in args if torch.is_tensor(t)))
        else:
            key = (device.index, tuple(_arg_signature(t) for t in args if torch.is_tensor(t)))
        if not hasattr(launch, "_compiled"):
            launch._compiled = {}
        if key not in launch._compiled:
            # flyc.compile() also executes this first call (jit_function.py:1662).
            launch._compiled[key] = flyc.compile(launch, *args, stream)
        else:
            launch._compiled[key](*args, stream)


def warmup_static(length: int, H: int, Hg: int, V: int, state_stride: int, **kwargs) -> None:
    """Admit one static specialization so serving may use it without a JIT stall."""
    _STATIC_READY.add(
        (
            length,
            H,
            Hg,
            V,
            state_stride,
            PREFETCH_DEPTH,
            kwargs.get("has_state", True),
            kwargs.get("has_g", True),
            kwargs.get("has_gk", False),
            kwargs.get("save", True),
            kwargs.get("use_exp2", False),
        )
    )


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    use_exp2: bool = False,
):
    """Specialized MI355X BF16 K128 prefill; h retains the production (V,K) layout.

    Accepts any number of packed sequences and any token count: the ragged tail
    chunk is masked by per-sequence buffer bounds. Requires K=128, V divisible by
    16 and H divisible by Hg. Optional state and -1 slots are safe. State inner
    dimensions must be contiguous; the slot stride may be int64.
    """
    if torch.cuda.get_device_properties(k.device).gcnArchName.split(":")[0] != "gfx950":
        raise ValueError("FlyDSL chunk_h targets gfx950")
    B, length, Hg, K = k.shape
    H, V = u.shape[-2:]
    if B != 1 or K != 128 or length == 0 or V % 16 or H % Hg:
        raise ValueError("FlyDSL chunk_h requires B=1, K=128, V%16=0, H%Hg=0")
    if use_exp2 and g is not None:
        raise ValueError("use_exp2 covers only gk")
    for t in (k, w, u):
        if t.dtype != torch.bfloat16 or not t.is_cuda or not t.is_contiguous() or t.device != k.device:
            raise ValueError("k/w/u must be contiguous BF16 tensors on the same GPU")
    if w.shape != (B, length, H, K) or u.shape != (B, length, H, V):
        raise ValueError("inconsistent w/u shapes")
    for tensor, shape in ((g, (B, length, H)), (gk, (B, length, H, K))):
        if tensor is not None and (
            tensor.shape != shape
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or tensor.device != k.device
        ):
            raise ValueError("gate tensors must be contiguous FP32 with matching shape and device")
    if initial_state is not None:
        if (
            initial_state.shape[1:] != (H, V, K)
            or initial_state.stride()[1:] != (V * K, K, 1)
            or initial_state.dtype != k.dtype
            or initial_state.device != k.device
        ):
            raise ValueError("state must be BF16 [slots,H,V,K] with contiguous inner dimensions")

    n_seq = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
    if n_seq < 1:
        raise ValueError("cu_seqlens must describe at least one sequence")
    if initial_state_indices is None:
        initial_state_indices = torch.zeros(n_seq, dtype=torch.int32, device=k.device)
    if (
        initial_state_indices.numel() != n_seq
        or initial_state_indices.dtype not in (torch.int32, torch.int64)
        or initial_state_indices.device != k.device
    ):
        raise ValueError("state indices must hold one int32/int64 GPU slot per sequence")

    state_stride = 0 if initial_state is None else initial_state.stride(0)
    flags = (
        PREFETCH_DEPTH,
        initial_state is not None,
        g is not None,
        gk is not None,
        save_new_value,
        use_exp2,
    )
    vn = torch.empty_like(u) if save_new_value else None

    static_key = (length, H, Hg, V, state_stride) + flags
    use_static = (
        n_seq == 1
        and length % CHUNK_SIZE == 0
        and (_STATIC_ON_DEMAND or static_key in _STATIC_READY)
    )

    if use_static:
        h = torch.empty((B, length // CHUNK_SIZE, H, V, K), dtype=k.dtype, device=k.device)
        launch = build(length, H, Hg, V, state_stride, *flags)
        args = (
            k,
            w,
            u,
            g if g is not None else k,
            gk if gk is not None else k,
            initial_state if initial_state is not None else k,
            initial_state_indices,
            h,
            vn if vn is not None else k,
        )
        _run(launch, args, shape_exact=True, device=k.device)
        _STATIC_READY.add(static_key)
        return h, vn

    from sglang.kernels.ops.attention.fla.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )

    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=k.device)
        n_chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
    else:
        # Mirror the reference: h is sized by the total chunk count across
        # sequences, which is exactly len(chunk_indices).
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        n_chunks = len(chunk_indices)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)

    h = torch.empty((B, n_chunks, H, V, K), dtype=k.dtype, device=k.device)
    launch = build_varlen(H, Hg, V, state_stride, *flags)
    args = (
        k,
        w,
        u,
        g if g is not None else k,
        gk if gk is not None else k,
        initial_state if initial_state is not None else k,
        initial_state_indices,
        h,
        vn if vn is not None else k,
        cu_seqlens,
        chunk_offsets,
        n_seq,
    )
    _run(launch, args, shape_exact=False, device=k.device)
    return h, vn
