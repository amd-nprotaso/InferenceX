# SPDX-License-Identifier: Apache-2.0
"""FlyDSL depthwise causal prefill convolution for SGLang's packed GDN input.

Uses the cached-builder/compiled-dispatch pattern of conv3d_implicit.py, but
maps channels to lanes rather than using an implicit GEMM for depthwise work.
"""

from functools import lru_cache
from typing import Optional, Sequence

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr


@lru_cache(maxsize=128)
def _build(dim: int, width: int, xs: tuple, ws: tuple, ss: tuple,
           os: tuple, has_bias: bool, has_cache: bool, has_indices: bool,
           has_initial: bool, silu: bool, pad_slot: Optional[int],
           batch: int, max_len: int, block: int, tokens: int, dtype: torch.dtype,
           spans: tuple, prefetch: int):
    element = {torch.bfloat16: fx.BFloat16, torch.float16: fx.Float16,
               torch.float32: fx.Float32}[dtype]
    @flyc.jit
    def load_pair(tensor: fx.Tensor, offset: fx.Int64, stride: fx.Int64):
        return fx.Vector.from_elements([tensor[offset], tensor[offset + stride]], element)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(x: fx.Tensor, w: fx.Tensor, bias: fx.Tensor, state: fx.Tensor,
               starts: fx.Tensor, indices: fx.Tensor, initial: fx.Tensor,
               out: fx.Tensor):
        # Explicit physical offsets below already include the original strides.
        # Rebase to unit-stride views so Tensor indexing does not apply them twice.
        x = fx.Tensor(fx.make_view(fx.get_iter(x), fx.make_layout(spans[0], 1)))
        w = fx.Tensor(fx.make_view(fx.get_iter(w), fx.make_layout(spans[1], 1)))
        state = fx.Tensor(fx.make_view(fx.get_iter(state), fx.make_layout(spans[2], 1)))
        out = fx.Tensor(fx.make_view(fx.get_iter(out), fx.make_layout(spans[3], 1)))
        seq = fx.block_idx.y
        chunk = fx.block_idx.x
        c = (fx.block_idx.z * block + fx.thread_idx.x) * 2
        start = fx.Int64(starts[seq])
        length = fx.Int64(starts[seq + 1]) - start
        offset = chunk * tokens
        slot = fx.Int64(seq)
        if const_expr(has_indices):
            slot = fx.Int64(indices[seq])
        valid_slot = fx.Int32(1)
        if const_expr(pad_slot is not None):
            valid_slot = fx.Int32(slot != pad_slot)
        if (c < dim) & (offset < length) & (valid_slot != 0):
            use_history = fx.Int32(0)
            if const_expr(has_initial):
                use_history = fx.Int32(initial[seq])
            raw_weights = [load_pair(w, c * ws[0] + j * ws[1], fx.Int64(ws[0])) for j in range_constexpr(width)]
            # Only chunk zero reads/writes the cache. Other chunks obtain their
            # entire halo from immutable x, avoiding inter-block state races.
            history = []
            for j in range_constexpr(width - 1):
                hvalue = fx.Vector.filled(2, 0.0, fx.Float32)
                if chunk == 0:
                    if const_expr(has_cache):
                        if use_history != 0:
                            hvalue = load_pair(state, slot * ss[0] + c * ss[1] + j * ss[2], fx.Int64(ss[1])).to(fx.Float32)
                else:
                    hvalue = load_pair(x, c * xs[0] + (start + offset - (width - 1) + j) * xs[1], fx.Int64(xs[0])).to(fx.Float32)
                history.append(hvalue)
            weights = [raw_weights[j].to(fx.Float32) for j in range_constexpr(width)]
            base = fx.Vector.filled(2, 0.0, fx.Float32)
            if const_expr(has_bias):
                base = load_pair(bias, fx.Int64(c), fx.Int64(1)).to(fx.Float32)
            # Snapshot the whole old window before any state stores (short
            # sequences must shift old history, not read overwritten values).
            if const_expr(has_cache):
                if chunk == 0:
                    for j in range_constexpr(width - 1):
                        tail = length - (width - 1) + j
                        value = fx.Vector.filled(2, 0.0, fx.Float32)
                        if tail >= 0:
                            value = load_pair(x, c * xs[0] + (start + tail) * xs[1], fx.Int64(xs[0])).to(fx.Float32)
                        else:
                            for h in range_constexpr(width - 1):
                                if tail + width - 1 == h:
                                    value = history[h]
                        for lane in range_constexpr(2):
                            state[slot * ss[0] + (c + lane) * ss[1] + j * ss[2]] = element(value[lane])
            for t in range_constexpr(tokens):
                if const_expr(t % prefetch == 0):
                    prefetched = []
                    for p in range_constexpr(min(prefetch, tokens - t)):
                        position = offset + t + p
                        safe_position = (position < length).select(position, length - 1)
                        raw = load_pair(x, c * xs[0] + (start + safe_position) * xs[1], fx.Int64(xs[0]))
                        prefetched.append(raw)
                value = prefetched[t % prefetch].to(fx.Float32)
                acc = base
                for j in range_constexpr(width - 1):
                    acc = acc + history[j] * weights[j]
                acc = acc + value * weights[width - 1]
                if const_expr(silu):
                    acc = acc / (fx.Vector.filled(2, 1.0, fx.Float32) + fx.exp(-acc))
                if offset + t < length:
                    for lane in range_constexpr(2):
                        out[(c + lane) * os[0] + (start + offset + t) * os[1]] = element(acc[lane])
                for j in range_constexpr(width - 2):
                    history[j] = history[j + 1]
                history[width - 2] = value

    @flyc.jit
    def launch(x: fx.Tensor, w: fx.Tensor, bias: fx.Tensor, state: fx.Tensor,
               starts: fx.Tensor, indices: fx.Tensor, initial: fx.Tensor,
               out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(x, w, bias, state, starts, indices, initial, out).launch(
            grid=((max_len + tokens - 1) // tokens, batch, (dim + block * 2 - 1) // (block * 2)),
            block=(block, 1, 1), stream=stream,
        )
    return launch


def causal_conv1d_fn(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor], query_start_loc: torch.Tensor,
    seq_lens_cpu: Sequence[int], cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu", pad_slot_id: Optional[int] = -1,
    validate_data: bool = False, *, block: int = 128, tokens: int = 16,
    prefetch: int = 4,
) -> torch.Tensor:
    """SGLang prefill API; returns new output and updates conv_states in place.

    Active cache indices must be unique and in bounds. CPU lengths must match
    GPU cumulative offsets. Padded-slot outputs are unspecified, like SGLang.
    No host tensor reads occur unless validate_data=True (not graph safe).
    """
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("expected x [channels,total_tokens], weight [channels,width]")
    dim, total = x.shape
    if dim % 2:
        raise ValueError("experimental pair mapping requires even channels")
    width = weight.shape[1]
    batch = len(seq_lens_cpu)
    if weight.shape[0] != dim or width not in (2, 3, 4, 5):
        raise ValueError("weight must have matching channels and width 2..5")
    if activation not in (None, False, True, "silu", "swish"):
        raise ValueError("activation must be None, silu, or swish")
    if block not in (64, 128, 256) or not width - 1 <= tokens <= 64:
        raise ValueError("block must be 64/128/256 and width-1 <= tokens <= 64")
    if prefetch not in (1, 2, 4, 8, 16):
        raise ValueError("prefetch must be 1/2/4/8/16")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("expected a GPU BF16/FP16/FP32 tensor")
    if 1 not in x.stride() or 1 not in weight.stride():
        raise ValueError("input and weight need at least one unit-stride axis")
    tensors = [weight, query_start_loc, bias, conv_states, cache_indices, has_initial_state]
    if any(t is not None and t.device != x.device for t in tensors):
        raise ValueError("all tensors must be on the input device")
    if weight.dtype != x.dtype or (conv_states is not None and conv_states.dtype != x.dtype):
        raise ValueError("input, weight, and state must have matching dtype")
    if query_start_loc.shape != (batch + 1,) or not query_start_loc.is_contiguous():
        raise ValueError("query_start_loc must be contiguous [batch+1]")
    if query_start_loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("query_start_loc must contain integers")
    for name, tensor in (("cache_indices", cache_indices), ("has_initial_state", has_initial_state)):
        if tensor is not None and (tensor.shape != (batch,) or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous [batch]")
    if cache_indices is not None and cache_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("cache_indices must contain integers")
    if bias is not None and (bias.shape != (dim,) or not bias.is_contiguous()):
        raise ValueError("bias must be contiguous [channels]")
    if conv_states is not None and (conv_states.ndim != 3 or conv_states.shape[1] != dim or conv_states.shape[2] < width - 1):
        raise ValueError("state must be [slots,channels,at_least_width-1]")
    if conv_states is not None and 1 not in conv_states.stride():
        raise ValueError("state needs at least one unit-stride axis")
    if conv_states is not None and cache_indices is None and conv_states.shape[0] < batch:
        raise ValueError("state needs at least batch slots without cache_indices")
    if has_initial_state is not None and conv_states is None:
        raise ValueError("initial history requires conv_states")
    if any(n < 0 for n in seq_lens_cpu) or sum(seq_lens_cpu) > total:
        raise ValueError("invalid sequence lengths")
    if validate_data:
        starts = query_start_loc.cpu().tolist()
        expected = [0]
        for n in seq_lens_cpu:
            expected.append(expected[-1] + n)
        if starts != expected:
            raise ValueError("GPU offsets do not match CPU sequence lengths")
        if conv_states is not None:
            slots = list(range(batch)) if cache_indices is None else cache_indices.cpu().tolist()
            active = [s for s, n in zip(slots, seq_lens_cpu) if s != pad_slot_id and n > 0]
            if len(set(active)) != len(active) or any(s < 0 or s >= conv_states.shape[0] for s in active):
                raise ValueError("active cache indices must be unique and in bounds")
    out = torch.empty_like(x)
    max_len = max(seq_lens_cpu, default=0)
    if not max_len or not dim:
        return out
    ss = conv_states.stride() if conv_states is not None else (0, 0, 0)
    def span(t: Optional[torch.Tensor]) -> int:
        return 1 if t is None else 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
    launch = _build(dim, width, x.stride(), weight.stride(), ss, out.stride(),
                    bias is not None, conv_states is not None, cache_indices is not None,
                    has_initial_state is not None, activation in (True, "silu", "swish"),
                    pad_slot_id, batch, max_len, block, tokens, x.dtype,
                    tuple(span(t) for t in (x, weight, conv_states, out)), prefetch)
    # Absent optional pointers are never dereferenced by the specialized kernel.
    args = (x, weight, bias if bias is not None else x,
            conv_states if conv_states is not None else x, query_start_loc,
            cache_indices if cache_indices is not None else query_start_loc,
            has_initial_state if has_initial_state is not None else query_start_loc, out)
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device)
        compiled = getattr(launch, "_compiled", None)
        # Signature includes dtype/strides in FlyDSL; retain one fast path per
        # concrete tensor signature and device, not merely per logical shape.
        key = (x.device.index, tuple((t.dtype, tuple(t.shape), t.stride()) for t in args))
        if compiled is None:
            launch._compiled = {}
        if key not in launch._compiled:
            launch._compiled[key] = flyc.compile(launch, *args, fx.Stream(stream))
        else:
            launch._compiled[key](*args, fx.Stream(stream))
    return out
