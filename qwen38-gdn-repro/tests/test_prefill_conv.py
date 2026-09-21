"""Accuracy of the AITER FlyDSL prefill causal-conv1d against Triton and fp32.

The conv is simple enough to have a trustworthy independent reference
(``reference_causal_conv1d`` in conftest), so these tests check ground truth
rather than just agreement with the incumbent: the AITER kernel must be at least
as accurate as the Triton kernel it replaces, measured against fp32.

Both kernels also mutate ``conv_states`` in place -- the state carries a
sequence's history into the next forward pass, so a kernel that produces perfect
outputs while corrupting state would break generation on the following step.
Every test checks the state too.
"""

import pytest
import torch
from conftest import DEVICE, assert_close, reference_causal_conv1d, rel_error

from aiter.ops.flydsl.causal_conv1d_flydsl import (
    causal_conv1d_prefill_flydsl_fn as aiter_conv,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn as triton_conv

PAD_SLOT_ID = -1
# bf16 rounding on a width<=5 dot product; both kernels land near 2e-3 against
# fp32, so this is roughly 2.5x headroom rather than a fitted threshold.
OUT_REL_TOL, OUT_MAX_TOL = 5e-3, 2e-2


def make_case(dim, width, seq_lens, *, bias=True, dtype=torch.bfloat16,
              n_slots=None, cache_indices=None, has_initial_state=True):
    b = len(seq_lens)
    total = sum(seq_lens)
    n_slots = n_slots if n_slots is not None else b + 2
    x = torch.randn(dim, total, device=DEVICE, dtype=dtype) * 0.5
    weight = torch.randn(dim, width, device=DEVICE, dtype=dtype) * 0.5
    bias_t = torch.randn(dim, device=DEVICE, dtype=dtype) * 0.1 if bias else None
    states = torch.randn(n_slots, dim, width - 1, device=DEVICE, dtype=dtype) * 0.3
    qsl = torch.zeros(b + 1, device=DEVICE, dtype=torch.int32)
    qsl[1:] = torch.tensor(seq_lens, device=DEVICE, dtype=torch.int32).cumsum(0)
    if cache_indices is None:
        cache_indices = torch.arange(b, device=DEVICE, dtype=torch.int32)
    if isinstance(has_initial_state, bool):
        has_initial_state = torch.full((b,), has_initial_state, device=DEVICE, dtype=torch.bool)
    return x, weight, bias_t, states, qsl, cache_indices, has_initial_state


def run_all(x, w, bias, states, qsl, seq_lens, ci, his, activation):
    """Run fp32 reference, Triton and AITER on identical inputs."""
    ref_out, ref_state = reference_causal_conv1d(
        x, w, bias, states, seq_lens, ci, his, activation, PAD_SLOT_ID
    )
    st_t = states.clone()
    out_t = triton_conv(x, w, bias, st_t, qsl, seq_lens, ci, his, activation, PAD_SLOT_ID, False)
    st_a = states.clone()
    out_a = aiter_conv(x, w, bias, st_a, qsl, seq_lens, ci, his, activation, PAD_SLOT_ID, False)
    return (ref_out, ref_state), (out_t, st_t), (out_a, st_a)


def assert_no_worse_than_triton(out_a, out_t, ref_out, label):
    """AITER must not be less accurate than the kernel it replaces.

    Compared against fp32, not against each other: two kernels can agree closely
    while both drifting from the true value.
    """
    err_a, err_t = rel_error(out_a, ref_out), rel_error(out_t, ref_out)
    assert err_a <= max(err_t * 1.5, 1e-4), (
        f"{label}: AITER rel-err {err_a:.3e} worse than Triton {err_t:.3e}"
    )


@pytest.mark.parametrize("width", [2, 3, 4, 5])
def test_widths(width):
    """Qwen3.8 uses width 4; the wrapper claims 2..5, so all four are covered."""
    seq_lens = [7, 13, 4]
    case = make_case(256, width, seq_lens)
    (ref_out, ref_st), (out_t, st_t), (out_a, st_a) = run_all(*case[:4], case[4], seq_lens, case[5], case[6], "silu")
    assert_close(out_a, ref_out, f"out(width={width})", OUT_REL_TOL, OUT_MAX_TOL)
    assert_close(st_a, ref_st, f"state(width={width})", OUT_REL_TOL, OUT_MAX_TOL)
    assert_no_worse_than_triton(out_a, out_t, ref_out, f"width={width}")


@pytest.mark.parametrize("activation", ["silu", None])
@pytest.mark.parametrize("bias", [True, False])
def test_activation_and_bias(activation, bias):
    seq_lens = [16, 1, 33]
    case = make_case(512, 4, seq_lens, bias=bias)
    (ref_out, ref_st), (out_t, st_t), (out_a, st_a) = run_all(*case[:4], case[4], seq_lens, case[5], case[6], activation)
    tag = f"act={activation},bias={bias}"
    assert_close(out_a, ref_out, f"out({tag})", OUT_REL_TOL, OUT_MAX_TOL)
    assert_close(st_a, ref_st, f"state({tag})", OUT_REL_TOL, OUT_MAX_TOL)
    assert_no_worse_than_triton(out_a, out_t, ref_out, tag)


@pytest.mark.parametrize("seq_lens", [
    [1],              # single token
    [1, 1, 1, 1],     # all minimal
    [1024],           # one long sequence
    [3, 512, 1, 77],  # ragged, the realistic varlen case
    [64] * 16,        # wide batch
])
def test_varlen_shapes(seq_lens):
    case = make_case(256, 4, seq_lens)
    (ref_out, ref_st), (out_t, st_t), (out_a, st_a) = run_all(*case[:4], case[4], seq_lens, case[5], case[6], "silu")
    assert_close(out_a, ref_out, f"out(lens={seq_lens[:4]})", OUT_REL_TOL, OUT_MAX_TOL)
    assert_close(st_a, ref_st, f"state(lens={seq_lens[:4]})", OUT_REL_TOL, OUT_MAX_TOL)
    assert_no_worse_than_triton(out_a, out_t, ref_out, f"lens={seq_lens[:4]}")


@pytest.mark.parametrize("pattern", ["all", "none", "mixed"])
def test_initial_state(pattern):
    """has_initial_state selects between the cached history and zeros per sequence.

    The 'mixed' case is the one that catches a kernel reading the state for a
    sequence that should have started cold -- which a uniformly-true or
    uniformly-false test cannot see.
    """
    seq_lens = [9, 5, 12, 3]
    b = len(seq_lens)
    if pattern == "mixed":
        his = torch.tensor([True, False, True, False], device=DEVICE)
    else:
        his = torch.full((b,), pattern == "all", device=DEVICE, dtype=torch.bool)
    case = make_case(256, 4, seq_lens, has_initial_state=his)
    (ref_out, ref_st), (out_t, st_t), (out_a, st_a) = run_all(*case[:4], case[4], seq_lens, case[5], case[6], "silu")
    assert_close(out_a, ref_out, f"out(his={pattern})", OUT_REL_TOL, OUT_MAX_TOL)
    assert_close(st_a, ref_st, f"state(his={pattern})", OUT_REL_TOL, OUT_MAX_TOL)


def test_scattered_cache_indices():
    """Slots are assigned by the scheduler, so they are neither ordered nor dense."""
    seq_lens = [11, 6, 20]
    ci = torch.tensor([7, 2, 5], device=DEVICE, dtype=torch.int32)
    case = make_case(256, 4, seq_lens, n_slots=10, cache_indices=ci)
    (ref_out, ref_st), _, (out_a, st_a) = run_all(*case[:4], case[4], seq_lens, case[5], case[6], "silu")
    assert_close(out_a, ref_out, "out(scattered)", OUT_REL_TOL, OUT_MAX_TOL)
    assert_close(st_a, ref_st, "state(scattered)", OUT_REL_TOL, OUT_MAX_TOL)
    # Slots nobody owns must be byte-identical to what went in.
    untouched = [i for i in range(10) if i not in (7, 2, 5)]
    assert torch.equal(st_a[untouched], case[3][untouched]), "kernel wrote to an unowned slot"


def test_pad_slot_is_not_written():
    """A padded request must leave its slot alone.

    Its *output* is explicitly unspecified, so only the state is asserted.
    """
    seq_lens = [8, 4, 6]
    ci = torch.tensor([1, PAD_SLOT_ID, 3], device=DEVICE, dtype=torch.int32)
    case = make_case(256, 4, seq_lens, n_slots=6, cache_indices=ci)
    states = case[3]
    _, _, (out_a, st_a) = run_all(*case[:4], case[4], seq_lens, case[5], case[6], "silu")
    untouched = [0, 2, 4, 5]
    assert torch.equal(st_a[untouched], states[untouched]), "pad slot leaked into another slot"


def test_transposed_input_layout():
    """Production passes ``mixed_qkv.transpose(0, 1)`` -- a non-contiguous view.

    The same class of layout assumption that silently disabled the decode kernel
    on Qwen3.8, so it is asserted here rather than assumed.
    """
    seq_lens = [32, 16]
    total, dim, width = sum(seq_lens), 256, 4
    token_major = torch.randn(total, dim, device=DEVICE, dtype=torch.bfloat16) * 0.5
    x = token_major.transpose(0, 1)                      # [dim, total], not contiguous
    assert not x.is_contiguous()
    weight = torch.randn(dim, width, device=DEVICE, dtype=torch.bfloat16) * 0.5
    states = torch.randn(4, dim, width - 1, device=DEVICE, dtype=torch.bfloat16) * 0.3
    qsl = torch.tensor([0, 32, 48], device=DEVICE, dtype=torch.int32)
    ci = torch.arange(2, device=DEVICE, dtype=torch.int32)
    his = torch.ones(2, device=DEVICE, dtype=torch.bool)

    ref_out, ref_st = reference_causal_conv1d(x, weight, None, states, seq_lens, ci, his, "silu")
    st_a = states.clone()
    out_a = aiter_conv(x, weight, None, st_a, qsl, seq_lens, ci, his, "silu", PAD_SLOT_ID, False)
    assert_close(out_a, ref_out, "out(transposed)", OUT_REL_TOL, OUT_MAX_TOL)
    assert_close(st_a, ref_st, "state(transposed)", OUT_REL_TOL, OUT_MAX_TOL)


@pytest.mark.slow
def test_qwen38_production_shape():
    """The real layer: fused QKV width 10240 = (16+16+48)*128, conv width 4."""
    seq_lens = [16384]     # one chunked-prefill chunk
    case = make_case(10240, 4, seq_lens)
    (ref_out, ref_st), (out_t, st_t), (out_a, st_a) = run_all(*case[:4], case[4], seq_lens, case[5], case[6], "silu")
    assert_close(out_a, ref_out, "out(qwen3.8)", OUT_REL_TOL, OUT_MAX_TOL)
    assert_close(st_a, ref_st, "state(qwen3.8)", OUT_REL_TOL, OUT_MAX_TOL)
    assert_no_worse_than_triton(out_a, out_t, ref_out, "qwen3.8")


def test_does_not_mutate_inputs():
    """Only conv_states may change; x/weight/bias must survive the call."""
    seq_lens = [12, 8]
    x, w, bias, states, qsl, ci, his = make_case(256, 4, seq_lens)
    x0, w0, b0 = x.clone(), w.clone(), bias.clone()
    aiter_conv(x, w, bias, states.clone(), qsl, seq_lens, ci, his, "silu", PAD_SLOT_ID, False)
    assert torch.equal(x, x0), "kernel modified x"
    assert torch.equal(w, w0), "kernel modified weight"
    assert torch.equal(bias, b0), "kernel modified bias"
