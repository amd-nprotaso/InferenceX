# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime patch that routes the Qwen3.5 decode MoE front-end through the
fused FlyDSL router+sort kernel.

Two seams, both in ``aiter.fused_moe``:

``fused_topk``
    aiter's router (``vllm::moe::topkGatingSoftmax``).  When the decode gate
    matches we do *not* run it: the gating logits are stashed and zero-filled
    ``topk_weights`` / ``topk_ids`` of the right shape and device are returned
    so every downstream shape check still works.

``moe_sorting``
    consumes the stash and produces the whole 5-tensor sort contract from the
    logits in one launch.  If the stash is missing or does not match the call
    (different M/E/topk, EP mask with ``num_local_tokens``, ...) the original
    router is run into the caller's ``topk_ids`` / ``topk_weights`` first and
    the original sorter takes over -- so a gate miss costs a kernel launch,
    never correctness.

Everything is off unless ``SGLANG_FLYDSL_MOE_DECODE=1``.

Why the stash and not a direct call: SGLang runs the router in
``select_experts`` and only later hands ``topk_ids`` to ``fused_moe``.  Fusing
router into sort means the router's *output* has to disappear, and the only
place both the logits and the sort are visible is across those two calls.

CUDA graphs: the first call per (shape, dtype) JIT-compiles, so the fused
kernel must be exercised during SGLang's pre-capture warmup -- which it is,
because capture is preceded by warmup replays of the same batch size.  No
allocation happens on the cached path beyond the output tensors, which is
exactly what the unpatched ``moe_sorting`` does too.

Limitations, all fail-safe (they fall back to the stock path):
  * softmax scoring only (that is the only route into ``aiter.fused_topk``),
  * ``num_local_tokens`` must be None,
  * M must fit the fused kernel's tokens-per-block (16 for E=512 bf16).
"""

import os
import sys

import torch

_ENABLED = os.environ.get("SGLANG_FLYDSL_MOE_DECODE") == "1"
# Upper bound on tokens routed through the fused kernel. The fused gating body
# runs in a single block, so this can never exceed TOKENS_PER_BLOCK; the
# compile-time check in the kernel module enforces that.
_MAX_TOKENS = int(os.environ.get("SGLANG_FLYDSL_MOE_DECODE_MAX_TOKENS", "16"))
# Log a running (fused / fallback / miss) tally every N fused calls. The import
# banner only proves the hook landed; this proves the fused kernel is actually
# being taken during a benchmark window, and how often the gate misses.
_LOG_EVERY = int(os.environ.get("SGLANG_FLYDSL_MOE_LOG_EVERY", "0"))

_installed = False
_pending = None  # (gating_logits, renormalize, M, E, topk)
_stats = {"fused": 0, "fallback": 0, "miss": 0}
_placeholder_cache = {}  # (device, M, topk) -> (zero topk_weights, zero topk_ids)


def _log(msg):
    print(f"[FlyDSL-moe] {msg}", file=sys.stderr, flush=True)


def _gate_ok(gating_output, topk, renormalize):
    from moe_fused_route_sort_flydsl import (  # noqa: PLC0415
        _DTYPE_STR,
        fused_route_sort_supported,
    )

    if gating_output.dim() != 2 or not gating_output.is_contiguous():
        return False
    M, E = gating_output.shape
    if M > _MAX_TOKENS:
        return False
    dtype_str = _DTYPE_STR.get(gating_output.dtype)
    if dtype_str is None:
        return False
    return fused_route_sort_supported(E, topk, _MAX_TOKENS, dtype_str)


def install():
    """Patch aiter (and SGLang's re-export) in place. Idempotent."""
    global _installed
    if _installed or not _ENABLED:
        return
    _installed = True

    import aiter
    import aiter.fused_moe as fm
    from aiter import dtypes

    from moe_fused_route_sort_flydsl import flydsl_fused_route_sort

    _orig_fused_topk = fm.fused_topk
    _orig_moe_sorting = fm.moe_sorting

    def fused_topk(hidden_states, gating_output, topk, renormalize, topk_ids=None, topk_weights=None):
        global _pending
        if not _gate_ok(gating_output, topk, renormalize):
            _pending = None
            return _orig_fused_topk(hidden_states, gating_output, topk, renormalize, topk_ids, topk_weights)

        M = gating_output.shape[0]
        device = gating_output.device
        # Return cached, permanently-zero placeholders instead of filling the
        # caller's buffers. Nothing downstream reads them once moe_sorting is
        # fused (only their shape/device/dtype matter), and zeroing per call
        # costs two FillFunctor launches -- 2.8 us/layer, over half the win.
        # Cached tensors also keep their addresses stable across CUDA-graph
        # capture and replay.
        key = (device, M, topk)
        placeholder = _placeholder_cache.get(key)
        if placeholder is None:
            placeholder = (
                torch.zeros(M, topk, dtype=dtypes.fp32, device=device),
                torch.zeros(M, topk, dtype=dtypes.i32, device=device),
            )
            _placeholder_cache[key] = placeholder
        _pending = (gating_output, bool(renormalize), M, gating_output.shape[1], topk)
        return placeholder

    def moe_sorting(
        topk_ids,
        topk_weight,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size=fm.BLOCK_SIZE_M,
        expert_mask=None,
        num_local_tokens=None,
        dispatch_policy=0,
        return_local_topk_ids=False,
        accumulate=True,
        flat=False,
        output_aux=False,
    ):
        global _pending
        pending, _pending = _pending, None

        M, topk = topk_ids.shape
        usable = (
            pending is not None
            and pending[2] == M
            and pending[3] == num_experts
            and pending[4] == topk
            and num_local_tokens is None
            and dispatch_policy == 0
            and not return_local_topk_ids
            and not flat
            and not output_aux
        )

        if pending is not None and not usable:
            # The router was skipped but this sort cannot use the stash --
            # recompute topk_weight / topk_ids in place and fall through.
            _stats["fallback"] += 1
            tei = torch.empty(M, topk, dtype=dtypes.i32, device=topk_ids.device)
            aiter.topk_softmax(topk_weight, topk_ids, tei, pending[0], pending[1])

        if not usable:
            if pending is None:
                _stats["miss"] += 1
            return _orig_moe_sorting(
                topk_ids,
                topk_weight,
                num_experts,
                model_dim,
                moebuf_dtype,
                block_size,
                expert_mask,
                num_local_tokens,
                dispatch_policy,
                return_local_topk_ids,
                accumulate,
                flat,
                output_aux,
            )

        gating_logits, renormalize = pending[0], pending[1]
        device = topk_ids.device
        block_size = int(block_size)
        max_num_tokens_padded = int(M * topk + num_experts * block_size - topk)
        max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
        sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
        sorted_weights = torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device)
        sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
        num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
        if (expert_mask is not None) or accumulate:
            moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
        else:
            moe_buf = torch.empty((0, 0), dtype=moebuf_dtype, device=device)

        _stats["fused"] += 1
        if _LOG_EVERY and _stats["fused"] % _LOG_EVERY == 0:
            _log(f"M={M} E={num_experts} topk={topk} stats={_stats}")
        return flydsl_fused_route_sort(
            gating_logits,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_experts,
            topk,
            unit_size=block_size,
            expert_mask=expert_mask,
            renormalize=renormalize,
            max_tokens=_MAX_TOKENS,
        )

    fm.fused_topk = fused_topk
    fm.moe_sorting = moe_sorting

    # SGLang binds `fused_topk` at import time; rebind the module attribute too
    # when that module is already loaded (the import hook runs it first).
    sgl_topk = sys.modules.get("sglang.srt.layers.moe.topk")
    if sgl_topk is not None and hasattr(sgl_topk, "aiter_fused_topk"):
        sgl_topk.aiter_fused_topk = fused_topk

    _log(f"fused router+sort enabled (max_tokens={_MAX_TOKENS})")


def stats():
    return dict(_stats)
