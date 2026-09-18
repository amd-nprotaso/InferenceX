"""FlyDSL GDN decode recurrence for MI355X (gfx950).

Drop-in replacement for SGLang's Triton
``fused_sigmoid_gating_delta_rule_update``, restricted to the GDN decode /
verify configuration that actually runs in serving. Unsupported inputs raise
rather than silently falling back, so a measured arm can never turn out to have
been the baseline -- the same discipline the prefill FlyDSL kernels use.

Math (per sequence n, v-head hv, v column v), identical to the Triton kernel:

    x     = a + dt_bias
    g     = -exp(A_log) * softplus(x)          # softplus guarded at threshold
    beta  = sigmoid(b)
    q, k  = l2norm(q), l2norm(k)               # over K, eps 1e-6
    h    *= exp(g)
    dv    = v - sum_k h[k,v] * k[k]            # delta rule
    h    += k[:,None] * (dv * beta)[None,:]
    o[v]  = sum_k h[k,v] * q[k]                # then * scale, folded into q

Why a different decomposition than Triton
-----------------------------------------
The Triton kernel launches ``grid = (1, cdiv(V, BV), N * HV)`` with
``BV = min(next_pow2(V), 32)`` and ``num_warps = 1``: single-wave workgroups,
256 waves at concurrency 4, about 3% of this GPU's wave slots (see
``README.md``). It is nowhere near bandwidth bound -- ~400 GB/s of state
traffic against ~8 TB/s of HBM.

Here the K axis is split across lanes instead of living inside one lane:

    lane (tid) owns  v = vb + tid // KSPLIT,  k in [ (tid % KSPLIT) * KLOCAL, +KLOCAL )

so ``KSPLIT`` lanes cooperate on one v column and the two reductions over K
(the delta-rule projection and the output projection) become ``log2(KSPLIT)``
``shuffle_xor`` steps instead of a serial per-lane sum. Each lane carries only
``KLOCAL`` floats of state, and the state rows load fully coalesced: the
``KSPLIT`` lanes of a v group cover one contiguous K row of ``K * 2`` bytes.

With the defaults below (KSPLIT=8, KLOCAL=16, VTILE=32) the launch is
``(HV, V // 32, N)`` workgroups of 256 threads = 1024 waves at concurrency 4,
4x the Triton kernel's occupancy. No LDS and no barriers are needed: the
l2-norm reductions ride the same lane group as the delta-rule reduction.

Two implementation constraints worth knowing
--------------------------------------------
1. The token loop is **unrolled at compile time** (``T`` baked into ``build``),
   not a runtime ``range``. Any global store inside a FlyDSL runtime carry loop
   segfaults this version of the compiler, and decode ``T`` is a small constant
   anyway (``draft_token_num``, 4 under EAGLE, 1 without), so unrolling costs
   nothing and schedules better. The wrapper requires uniform sequence lengths
   and raises otherwise.
2. Only one lane per v group writes the output, but a plain ``if`` around a
   store is what triggers (1). Instead the non-writing lanes are steered past
   the output buffer's ``num_records`` and the hardware drops their stores --
   the same masking ``chunk_delta_h_flydsl.build_varlen`` uses for its ragged
   tail. This *requires* ``max_size=False`` with an explicit
   ``num_records_bytes``; with the default unbounded descriptor the steered
   stores are not dropped, they scribble 512 MB past the tensor.

Limits: K == 128, V % VTILE == 0, HV % Hg == 0, bf16 q/k/v/state, fp32 a/b/
A_log/dt_bias, contiguous inner state dims, uniform sequence length.
"""

import functools
import operator
import os
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

# Lanes cooperating on one v column. Must be a power of two <= 64, and
# BLOCK == VTILE * KSPLIT.
KSPLIT = int(os.environ.get("SGLANG_FLYDSL_GDN_DECODE_KSPLIT", 8))
BLOCK = int(os.environ.get("SGLANG_FLYDSL_GDN_DECODE_BLOCK", 256))
VTILE = BLOCK // KSPLIT
KDIM = 128
KLOCAL = KDIM // KSPLIT
# Cross-checks cu_seqlens really is uniform. Off by default: it needs a
# device->host sync, which is illegal inside CUDA-graph capture.
_VALIDATE_UNIFORM = os.environ.get("SGLANG_FLYDSL_GDN_DECODE_VALIDATE") == "1"
HALF = 8  # b128 is the widest buffer load: 8 bf16, so KLOCAL comes in chunks of 8
NHALF = KLOCAL // HALF
assert KLOCAL % HALF == 0 and NHALF >= 1, f"KSPLIT={KSPLIT} gives KLOCAL={KLOCAL}"
assert VTILE * KSPLIT == BLOCK, f"BLOCK={BLOCK} != VTILE*KSPLIT"


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
def lane_group_sum(x):
    """Sum ``x`` across the KSPLIT consecutive lanes that share a v column.

    Unrolled explicitly rather than in a ``range_constexpr``: the AST rewriter
    rebinds names captured in those bodies, which breaks an accumulator.
    """
    s = x
    if const_expr(KSPLIT > 1):
        s = s + fx.gpu.shuffle_xor(s, 1, 64)
    if const_expr(KSPLIT > 2):
        s = s + fx.gpu.shuffle_xor(s, 2, 64)
    if const_expr(KSPLIT > 4):
        s = s + fx.gpu.shuffle_xor(s, 4, 64)
    if const_expr(KSPLIT > 8):
        s = s + fx.gpu.shuffle_xor(s, 8, 64)
    if const_expr(KSPLIT > 16):
        s = s + fx.gpu.shuffle_xor(s, 16, 64)
    if const_expr(KSPLIT > 32):
        s = s + fx.gpu.shuffle_xor(s, 32, 64)
    return s


@lru_cache(maxsize=32)
def build(HV: int, Hg: int, V: int, TSTATIC: int, out_bytes: int, state_stride: int,
          stride_q: int, stride_k: int, stride_v: int, stride_a: int, stride_b: int,
          scale: float, softplus_beta: float, softplus_threshold: float,
          has_state: bool, update_state: bool, cache_states: bool, cache_steps: int,
          a_bf16: bool, b_bf16: bool):
    group = HV // Hg  # v-heads per k-head
    # The gate inputs are whatever the model produced (bf16 in serving, fp32 in
    # the standalone harness); the Triton kernel converts on load, so match it.
    A_DT = fx.BFloat16 if a_bf16 else fx.Float32
    B_DT = fx.BFloat16 if b_bf16 else fx.Float32
    # Offset (in elements) guaranteed past the output descriptor, so the
    # hardware drops the stores from lanes that are not the group's writer.
    OOB = out_bytes  # bytes >= elements for bf16, so this is always out of range

    @flyc.kernel
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        A_log: fx.Tensor,
        dt_bias: fx.Tensor,
        state: fx.Tensor,
        indices: fx.Tensor,
        out: fx.Tensor,
        cu: fx.Tensor,
        icache: fx.Tensor,
        icache_idx: fx.Tensor,
    ):
        tid = fx.Int64(fx.thread_idx.x)
        hv = fx.Int64(fx.block_idx.x)
        vb = fx.Int64(fx.block_idx.y) * VTILE
        seq = fx.Int64(fx.block_idx.z)

        kpart = tid % KSPLIT
        vcol = vb + tid // KSPLIT
        kbase = kpart * KLOCAL
        ih = hv // group

        bos = fx.Int64(cu[seq])
        slot = fx.Int64(indices[seq])
        # EAGLE target-verify: each draft step's post-update state is snapshotted
        # into intermediate_states_buffer, laid out exactly like the state pool
        # ([hv][v][k]) with a per-slot pitch of cache_steps * HV * V * K.
        cslot = fx.Int64(0)
        if const_expr(cache_states):
            cslot = fx.Int64(icache_idx[seq])

        qb = fx.rocdl.make_buffer_tensor(q)
        kb = fx.rocdl.make_buffer_tensor(k)
        vv_b = fx.rocdl.make_buffer_tensor(v)
        ab = fx.rocdl.make_buffer_tensor(a)
        bb = fx.rocdl.make_buffer_tensor(b)
        # Bounded on purpose: the predicated output store below relies on
        # out-of-range offsets being dropped. See the module docstring.
        ob = fx.rocdl.make_buffer_tensor(out, max_size=False,
                                         num_records_bytes=fx.Int64(out_bytes))
        # Rebase to the slot before building the 32-bit-offset buffer resource;
        # a whole pool can exceed the 32-bit offset range.
        state_slot = fx.make_view(fx.get_iter(state) + slot * state_stride,
                                  fx.make_layout(HV * V * KDIM, 1))
        ss = fx.rocdl.make_buffer_tensor(state_slot)
        if const_expr(cache_states):
            cache_slot = fx.make_view(
                fx.get_iter(icache) + cslot * (cache_steps * HV * V * KDIM),
                fx.make_layout(cache_steps * HV * V * KDIM, 1))
            cb = fx.rocdl.make_buffer_tensor(cache_slot)

        alog = fx.Float32(A_log[hv])
        dtb = fx.Float32(dt_bias[hv])
        neg_exp_alog = -fx.exp(alog)

        sbase = hv * V * KDIM + vcol * KDIM + kbase
        # h[j] holds this lane's j-th b128 chunk of its K slice, in fp32.
        h = [fx.Vector.filled(HALF, 0.0, fx.Float32) for _ in range(NHALF)]
        if const_expr(has_state):
            if slot >= 0:
                h = [load_vec(ss, sbase + j * HALF, HALF, fx.BFloat16).to(fx.Float32)
                     for j in range(NHALF)]

        for t in range_constexpr(TSTATIC):
            tok = bos + t
            koff = tok * stride_k + ih * KDIM + kbase
            qoff = tok * stride_q + ih * KDIM + kbase
            kvec = [load_vec(kb, koff + j * HALF, HALF, fx.BFloat16).to(fx.Float32)
                    for j in range(NHALF)]
            qvec = [load_vec(qb, qoff + j * HALF, HALF, fx.BFloat16).to(fx.Float32)
                    for j in range(NHALF)]

            # l2 norms over the full K: summed inside the lane, then across the group
            # functools.reduce, not a `for` statement: FlyDSL's AST rewriter turns
            # statement-level loops in a kernel body into runtime scf.for, which
            # would make `j` a DSL value and break the Python-list indexing.
            sk_local = functools.reduce(operator.add, [x * x for x in kvec])
            sq_local = functools.reduce(operator.add, [x * x for x in qvec])
            sk = lane_group_sum(sk_local.reduce(fx.ReductionOp.ADD, fx.Float32(0.0)))
            sq = lane_group_sum(sq_local.reduce(fx.ReductionOp.ADD, fx.Float32(0.0)))
            rk = fx.Float32(1.0) / fx.sqrt(sk + fx.Float32(1e-6))
            rq = scale / fx.sqrt(sq + fx.Float32(1e-6))
            kvec = [x * rk for x in kvec]
            qvec = [x * rq for x in qvec]

            av = fx.Float32(load_vec(ab, tok * stride_a + hv, 1, A_DT)[0].to(fx.Float32))
            bv_in = fx.Float32(load_vec(bb, tok * stride_b + hv, 1, B_DT)[0].to(fx.Float32))
            vval = fx.Float32(load_vec(vv_b, tok * stride_v + hv * V + vcol, 1,
                                       fx.BFloat16)[0].to(fx.Float32))

            # g = -exp(A_log) * softplus(x), matching the Triton guard exactly.
            x = av + dtb
            bx = softplus_beta * x
            sp = (bx <= softplus_threshold).select(
                (fx.Float32(1.0) / softplus_beta) * fx.log(fx.Float32(1.0) + fx.exp(bx)), x)
            decay = fx.exp(neg_exp_alog * sp)
            beta = fx.Float32(1.0) / (fx.Float32(1.0) + fx.exp(-bv_in))

            hd = [x * decay for x in h]

            dv_local = functools.reduce(operator.add,
                                        [hd[j] * kvec[j] for j in range(NHALF)])
            dv = lane_group_sum(dv_local.reduce(fx.ReductionOp.ADD, fx.Float32(0.0)))
            vnew = (vval - dv) * beta

            h = [hd[j] + kvec[j] * vnew for j in range(NHALF)]

            ov_local = functools.reduce(operator.add,
                                        [h[j] * qvec[j] for j in range(NHALF)])
            ov = lane_group_sum(ov_local.reduce(fx.ReductionOp.ADD, fx.Float32(0.0)))
            # Predicated by address, not by control flow: see the docstring.
            oaddr = (kpart == 0).select((tok * HV + hv) * V + vcol, fx.Int64(OOB))
            store_vec(ob, oaddr,
                      fx.Vector.from_elements([ov.to(fx.BFloat16)], fx.BFloat16),
                      fx.BFloat16)

            if const_expr(cache_states):
                if cslot >= 0:
                    for j in range_constexpr(NHALF):
                        store_vec(cb, t * (HV * V * KDIM) + sbase + j * HALF,
                                  h[j].to(fx.BFloat16), fx.BFloat16)

        if const_expr(has_state and update_state):
            if slot >= 0:
                for j in range_constexpr(NHALF):
                    store_vec(ss, sbase + j * HALF, h[j].to(fx.BFloat16), fx.BFloat16)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        A_log: fx.Tensor,
        dt_bias: fx.Tensor,
        state: fx.Tensor,
        indices: fx.Tensor,
        out: fx.Tensor,
        cu: fx.Tensor,
        icache: fx.Tensor,
        icache_idx: fx.Tensor,
        n: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(q, k, v, a, b, A_log, dt_bias, state, indices, out, cu,
               icache, icache_idx).launch(
            grid=(HV, V // VTILE, n), block=(BLOCK,), stream=stream
        )

    return launch


_UNSUPPORTED = dict(
    is_kda=False,
    lower_bound=None,
    retrieve_parent_token=None,
    cache_ring=False,
    replayssm_rawv=None,
    replayssm_rawk=None,
    replayssm_g=None,
    replayssm_beta=None,
)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale=None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens=None,
    disable_state_update: bool = False,
    intermediate_states_buffer=None,
    intermediate_state_indices=None,
    cache_steps=None,
    **kwargs,
):
    """FlyDSL decode recurrence; same signature subset as the Triton entry point."""
    for name, expected in _UNSUPPORTED.items():
        if kwargs.get(name, expected) != expected:
            raise ValueError(f"FlyDSL GDN decode does not support {name}={kwargs[name]!r}")
    if torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx950":
        raise ValueError("FlyDSL GDN decode targets gfx950")
    if not use_qk_l2norm_in_kernel:
        raise ValueError("FlyDSL GDN decode requires use_qk_l2norm_in_kernel=True")
    if cu_seqlens is None:
        raise ValueError("FlyDSL GDN decode requires cu_seqlens (varlen decode)")
    if initial_state_source is None:
        raise ValueError("FlyDSL GDN decode requires a state pool")

    B, T, Hg, K = k.shape
    HV, V = v.shape[-2:]
    if B != 1 or K != KDIM or V % VTILE or HV % Hg:
        raise ValueError(f"FlyDSL GDN decode requires B=1, K={KDIM}, V%{VTILE}=0, HV%Hg=0")
    for t_ in (q, k, v):
        if t_.dtype != torch.bfloat16 or not t_.is_contiguous():
            raise ValueError("q/k/v must be contiguous bf16")
    for nm, t_ in (("a", a), ("b", b), ("A_log", A_log), ("dt_bias", dt_bias)):
        if t_.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"{nm} must be fp32 or bf16, got {t_.dtype}")
        if not t_.is_contiguous():
            raise ValueError(f"{nm} must be contiguous")
    if initial_state_source.dtype != torch.bfloat16:
        raise ValueError("state pool must be bf16")
    if initial_state_source.stride()[1:] != (V * K, K, 1):
        raise ValueError("state pool inner dims must be contiguous [HV, V, K]")

    # EAGLE target-verify snapshots each draft step's state. The buffer is
    # [slots, cache_steps, HV, V, K]; the per-slot pitch comes from the
    # ALLOCATED step dim, not the runtime step count (they differ under
    # --speculative-adaptive), matching the Triton wrapper.
    cache_states = intermediate_states_buffer is not None
    if cache_states:
        if intermediate_state_indices is None:
            raise ValueError("intermediate_states_buffer needs intermediate_state_indices")
        if intermediate_states_buffer.dtype != initial_state_source.dtype:
            raise ValueError("intermediate state buffer dtype must match the state pool")
        if not intermediate_states_buffer.is_contiguous():
            raise ValueError("intermediate state buffer must be contiguous")
        cstep = intermediate_states_buffer.shape[1]
        if intermediate_states_buffer.shape[2:] != (HV, V, K):
            raise ValueError(
                f"intermediate state buffer must be [slots, steps, {HV}, {V}, {K}], "
                f"got {tuple(intermediate_states_buffer.shape)}")
    else:
        cstep = 0

    # The token loop is unrolled at compile time, so every sequence in the batch
    # must have the same length. Derived with host-side arithmetic only: this
    # runs inside CUDA-graph capture, where any device->host sync (.item(),
    # .tolist()) raises hipErrorStreamCaptureUnsupported and kills the server.
    # T // n is exactly how gdn_backend derives draft_token_num.
    n = cu_seqlens.numel() - 1
    if n <= 0 or T % n:
        raise ValueError(f"cu_seqlens implies a ragged batch: T={T} n={n}")
    tstatic = T // n
    if _VALIDATE_UNIFORM:  # opt-in; syncs, so never on under graph capture
        lens = torch.diff(cu_seqlens)
        if not bool((lens == tstatic).all().item()):
            raise ValueError(
                f"FlyDSL GDN decode requires a uniform sequence length, got {lens.tolist()}")

    out = q.new_empty(1, T, HV, V)
    launch = build(
        HV, Hg, V, tstatic, out.numel() * out.element_size(),
        initial_state_source.stride(0),
        q.stride()[1], k.stride()[1], v.stride()[1],
        a.stride()[1] if a.ndim == 3 else a.stride()[-2],
        b.stride()[1] if b.ndim == 3 else b.stride()[-2],
        float(K ** -0.5) if scale is None else float(scale),
        float(softplus_beta), float(softplus_threshold),
        True, not disable_state_update, cache_states, int(cstep),
        a.dtype == torch.bfloat16, b.dtype == torch.bfloat16,
    )
    launch(q, k, v, a, b, A_log, dt_bias, initial_state_source,
           initial_state_indices, out, cu_seqlens,
           intermediate_states_buffer if cache_states else initial_state_source,
           intermediate_state_indices if cache_states else initial_state_indices,
           fx.Int32(n))
    return out
