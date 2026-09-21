"""Accuracy of the AITER FlyDSL GDN decode recurrence against the Triton kernel.

Triton is the reference here, not ground truth -- see the note in conftest. The
bar is "drop-in replacement for the kernel that ships today".

Three properties get asserted beyond a tolerance, because for them an exact
answer exists:

* fused-QKV **views** and contiguous **copies** must give bit-identical results
  (only addressing differs -- this is the bug patch 0005 exists for);
* ``disable_state_update=True`` must leave the state pool byte-unchanged (the
  MTP verify path commits the accepted prefix itself);
* slots no request owns must be byte-unchanged.
"""

import pytest
import torch
from conftest import DEVICE, assert_bitwise, assert_close

from aiter.ops.flydsl.linear_attention_kernels import (
    flydsl_gdn_decode_varlen as aiter_decode,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as triton_decode,
)

# Qwen3.8-Flash-Next linear-attention geometry.
HG, HV, D = 16, 48, 128
BETA, THRESHOLD = 1.0, 20.0

# Observed agreement is ~1e-8 mean / ~1e-3 worst-element; a genuinely broken
# kernel is off by O(1), so these leave room for bf16 noise without going blind.
REL_TOL, MAX_TOL = 1e-3, 2e-2


def build(batch, tokens_per_seq, *, ab_dtype=torch.float32, hg=HG, hv=HV, d=D,
          n_extra_slots=2, state_indices=None):
    """Inputs in the layout the GDN layer really produces: q/k/v as three views
    into one fused ``[1, T, (hg + hg + hv) * d]`` projection buffer."""
    t = batch * tokens_per_seq
    fused_width = (hg + hg + hv) * d
    mixed = torch.randn(1, t, fused_width, device=DEVICE, dtype=torch.bfloat16) * 0.2
    q = mixed[..., : hg * d].view(1, t, hg, d)
    k = mixed[..., hg * d : 2 * hg * d].view(1, t, hg, d)
    v = mixed[..., 2 * hg * d :].view(1, t, hv, d)
    a = torch.randn(t, hv, device=DEVICE, dtype=ab_dtype) * 0.2
    b = torch.randn(t, hv, device=DEVICE, dtype=ab_dtype) * 0.2
    A_log = torch.randn(hv, device=DEVICE, dtype=torch.float32) * 0.2
    dt_bias = torch.randn(hv, device=DEVICE, dtype=torch.float32) * 0.2
    n_slots = batch + n_extra_slots
    state = torch.randn(n_slots, hv, d, d, device=DEVICE, dtype=torch.bfloat16) * 0.05
    if state_indices is None:
        state_indices = torch.arange(batch, device=DEVICE, dtype=torch.int32)
    cu = torch.arange(0, t + 1, tokens_per_seq, device=DEVICE, dtype=torch.int32)
    return dict(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
                state=state, state_indices=state_indices, cu_seqlens=cu)


def run_aiter(t, *, snapshots=False, disable_state_update=False, steps=None, q=None):
    state = t["state"].clone()
    kw = dict(q=q if q is not None else t["q"], k=t["k"], v=t["v"], a=t["a"], b=t["b"],
              A_log=t["A_log"], dt_bias=t["dt_bias"], state=state,
              state_indices=t["state_indices"], cu_seqlens=t["cu_seqlens"],
              softplus_beta=BETA, softplus_threshold=THRESHOLD,
              disable_state_update=disable_state_update)
    inter = None
    if snapshots:
        n = t["cu_seqlens"].numel() - 1
        inter = torch.zeros(n + 1, steps, t["v"].shape[-2], D, D,
                            device=DEVICE, dtype=torch.bfloat16)
        kw.update(intermediate_states=inter,
                  intermediate_state_indices=t["state_indices"])
    out = aiter_decode(**kw)
    return out, state, inter


def run_triton(t, *, snapshots=False, disable_state_update=False, steps=None):
    state = t["state"].clone()
    inter = None
    kw = dict(A_log=t["A_log"], a=t["a"], dt_bias=t["dt_bias"],
              softplus_beta=BETA, softplus_threshold=THRESHOLD,
              q=t["q"], k=t["k"], v=t["v"], b=t["b"],
              initial_state_source=state, initial_state_indices=t["state_indices"],
              cu_seqlens=t["cu_seqlens"], use_qk_l2norm_in_kernel=True, is_kda=False,
              disable_state_update=disable_state_update)
    if snapshots:
        n = t["cu_seqlens"].numel() - 1
        inter = torch.zeros(n + 1, steps, t["v"].shape[-2], D, D,
                            device=DEVICE, dtype=torch.bfloat16)
        kw.update(intermediate_states_buffer=inter,
                  intermediate_state_indices=t["state_indices"],
                  cache_steps=steps, retrieve_parent_token=None)
    out = triton_decode(**kw)
    return out, state, inter


# --------------------------------------------------------------------------
# decode: one token per sequence, state written back
# --------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [1, 2, 5, 8, 16, 33])
def test_decode_matches_triton(batch):
    t = build(batch, 1)
    out_a, st_a, _ = run_aiter(t)
    out_t, st_t, _ = run_triton(t)
    assert_close(out_a, out_t, f"out(bs={batch})", REL_TOL, MAX_TOL)
    assert_close(st_a, st_t, f"state(bs={batch})", REL_TOL, MAX_TOL)


def test_decode_actually_advances_state():
    """Guards against a kernel that agrees with Triton by writing nothing."""
    t = build(4, 1)
    _, st_a, _ = run_aiter(t)
    assert not torch.equal(st_a[:4], t["state"][:4]), "state was never written"


def test_unowned_slots_untouched():
    t = build(4, 1, n_extra_slots=3)
    _, st_a, _ = run_aiter(t)
    assert torch.equal(st_a[4:], t["state"][4:]), "kernel wrote past its slots"


def test_scattered_state_indices():
    """Slot assignment comes from the scheduler: unordered and sparse."""
    idx = torch.tensor([5, 0, 3], device=DEVICE, dtype=torch.int32)
    t = build(3, 1, n_extra_slots=5, state_indices=idx)
    out_a, st_a, _ = run_aiter(t)
    out_t, st_t, _ = run_triton(t)
    assert_close(out_a, out_t, "out(scattered)", REL_TOL, MAX_TOL)
    assert_close(st_a, st_t, "state(scattered)", REL_TOL, MAX_TOL)
    untouched = [1, 2, 4, 6, 7]
    assert torch.equal(st_a[untouched], t["state"][untouched])


@pytest.mark.parametrize("ab_dtype", [torch.float32, torch.bfloat16])
def test_gating_input_dtypes(ab_dtype):
    """The wrapper accepts fp32 or bf16 for a/b/A_log/dt_bias."""
    t = build(4, 1, ab_dtype=ab_dtype)
    out_a, st_a, _ = run_aiter(t)
    out_t, st_t, _ = run_triton(t)
    assert_close(out_a, out_t, f"out(ab={ab_dtype})", REL_TOL, MAX_TOL)
    assert_close(st_a, st_t, f"state(ab={ab_dtype})", REL_TOL, MAX_TOL)


# --------------------------------------------------------------------------
# target-verify: several draft tokens per sequence, per-step snapshots
# --------------------------------------------------------------------------

@pytest.mark.parametrize("batch,steps", [(1, 2), (4, 4), (16, 4), (8, 3)])
def test_verify_matches_triton(batch, steps):
    """The MTP path: --speculative-num-draft-tokens tokens verified together."""
    t = build(batch, steps)
    out_a, st_a, inter_a = run_aiter(t, snapshots=True, disable_state_update=True, steps=steps)
    out_t, st_t, inter_t = run_triton(t, snapshots=True, disable_state_update=True, steps=steps)
    assert_close(out_a, out_t, f"out(bs={batch},steps={steps})", REL_TOL, MAX_TOL)
    # The accepted prefix is committed from these snapshots, so a correct output
    # with wrong snapshots still corrupts the next step.
    assert_close(inter_a, inter_t, f"snapshots(bs={batch},steps={steps})", REL_TOL, MAX_TOL)


def test_verify_suppresses_state_writeback():
    """disable_state_update must leave the pool byte-identical, not merely close."""
    t = build(8, 4)
    _, st_a, _ = run_aiter(t, snapshots=True, disable_state_update=True, steps=4)
    assert_bitwise(st_a, t["state"], "state pool under disable_state_update")


def test_verify_writes_every_step():
    """Each draft step must leave a distinct snapshot; an all-zero or constant
    buffer would still pass a comparison against a equally-broken kernel."""
    t = build(4, 4)
    _, _, inter = run_aiter(t, snapshots=True, disable_state_update=True, steps=4)
    used = inter[:4]
    assert used.abs().sum() > 0, "snapshot buffer was never written"
    for s in range(1, 4):
        assert not torch.equal(used[:, s], used[:, s - 1]), f"step {s} duplicates {s-1}"


# --------------------------------------------------------------------------
# the layout property patch 0005 turns on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("batch,steps", [(4, 4), (16, 4), (8, 1)])
def test_fused_views_bit_identical_to_contiguous(batch, steps):
    """q/k/v as views into the fused QKV buffer vs contiguous copies.

    Same kernel, same values, only the token stride differs -- so anything short
    of bit-identical means the stride is being mishandled.
    """
    t = build(batch, steps)
    assert not t["q"].is_contiguous(), "fixture no longer exercises the strided path"

    snap = steps > 1
    out_view, st_view, inter_view = run_aiter(
        t, snapshots=snap, disable_state_update=snap, steps=steps if snap else None)

    t_contig = dict(t)
    t_contig["q"] = t["q"].contiguous()
    t_contig["k"] = t["k"].contiguous()
    t_contig["v"] = t["v"].contiguous()
    out_c, st_c, inter_c = run_aiter(
        t_contig, snapshots=snap, disable_state_update=snap, steps=steps if snap else None)

    assert_bitwise(out_view, out_c, f"out(bs={batch},steps={steps})")
    assert_bitwise(st_view, st_c, f"state(bs={batch},steps={steps})")
    if snap:
        assert_bitwise(inter_view, inter_c, f"snapshots(bs={batch},steps={steps})")


def test_does_not_mutate_inputs():
    """Only `state` (and the snapshot buffer) may change."""
    t = build(4, 4)
    originals = {n: t[n].clone() for n in ("q", "k", "v", "a", "b", "A_log", "dt_bias")}
    run_aiter(t, snapshots=True, disable_state_update=True, steps=4)
    for n, orig in originals.items():
        assert torch.equal(t[n], orig), f"kernel modified {n}"


# --------------------------------------------------------------------------
# guards: unsupported input must RAISE, never compute something wrong
# --------------------------------------------------------------------------
# The dispatcher catches ValueError and falls back to Triton. A guard that is
# too loose returns garbage; one that is too tight silently disables the kernel
# (exactly what happened on Qwen3.8). Both directions are worth pinning.

def _kw(t, **over):
    kw = dict(q=t["q"], k=t["k"], v=t["v"], a=t["a"], b=t["b"], A_log=t["A_log"],
              dt_bias=t["dt_bias"], state=t["state"].clone(),
              state_indices=t["state_indices"], cu_seqlens=t["cu_seqlens"],
              softplus_beta=BETA, softplus_threshold=THRESHOLD)
    kw.update(over)
    return kw


def test_rejects_wrong_head_dim():
    t = build(2, 1, d=64, hg=4, hv=8)
    with pytest.raises(ValueError, match="head_k_dim"):
        aiter_decode(**_kw(t))


def test_rejects_non_bf16_qkv():
    t = build(2, 1)
    with pytest.raises(ValueError, match="bf16"):
        aiter_decode(**_kw(t, q=t["q"].float()))


def test_rejects_hv_not_multiple_of_hg():
    t = build(2, 1, hg=5, hv=12)
    with pytest.raises(ValueError, match="multiple"):
        aiter_decode(**_kw(t))


def test_rejects_non_divisible_batch():
    """T not divisible by the sequence count cannot be a uniform batch."""
    t = build(3, 4)                                     # T = 12
    bad = torch.tensor([0, 2, 4, 6, 9, 12], device=DEVICE, dtype=torch.int32)  # n=5
    with pytest.raises(ValueError, match="divisible"):
        aiter_decode(**_kw(t, cu_seqlens=bad))


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN GAP: a ragged batch whose token total happens to divide evenly "
           "passes the guard and is computed wrong (~60x rel error). Detecting it "
           "needs a host read of cu_seqlens, which raises "
           "hipErrorStreamCaptureUnsupported under CUDA-graph capture, so the "
           "wrapper can only check divisibility. Uniform lengths are a CALLER "
           "precondition -- speculative decode always satisfies it. If this test "
           "starts passing, a real uniformity check was added: delete the xfail.",
)
def test_ragged_but_divisible_is_rejected():
    t = build(3, 4)                                     # T = 12, n = 3
    ragged = torch.tensor([0, 3, 7, 12], device=DEVICE, dtype=torch.int32)  # lens 3,4,5
    with pytest.raises(ValueError):
        aiter_decode(**_kw(t, cu_seqlens=ragged))


def test_rejects_non_bf16_state():
    t = build(2, 1)
    with pytest.raises(ValueError, match="state pool must be bf16"):
        aiter_decode(**_kw(t, state=t["state"].float()))


def test_rejects_broken_head_stride():
    """Head-major contiguity *within* a token is the one layout rule that stands.

    A tensor whose head stride is not the head dim genuinely cannot be addressed
    by the kernel, so it must raise -- this is the boundary patch 0005 moved to,
    and it should not drift further.
    """
    t = build(2, 1)
    wide = torch.randn(1, 2, HG, 2 * D, device=DEVICE, dtype=torch.bfloat16)
    q_bad = wide[..., :D]                      # stride[-2] == 2*D, not D
    assert q_bad.stride()[-2] != D
    with pytest.raises(ValueError, match="head-major contiguous"):
        aiter_decode(**_kw(t, q=q_bad))


def test_accepts_fused_qkv_view():
    """The positive half of the previous test: the production layout must NOT raise.

    Without patch 0005 this is the call that failed, silently costing ~5% e2e.
    """
    t = build(4, 4)
    assert not t["q"].is_contiguous()
    out = aiter_decode(**_kw(t))
    assert out.shape == (1, 16, HV, D)
    assert torch.isfinite(out.float()).all()


# --------------------------------------------------------------------------
# meta: the tolerances must be tight enough to fail on a real regression
# --------------------------------------------------------------------------

def test_tolerances_would_catch_a_regression():
    """A suite that passes on a broken kernel is worse than no suite.

    Pins the sensitivity of `assert_close`: a 0.5% systematic drift -- far
    smaller than any real kernel bug -- must fail. Stops the tolerances from
    being quietly relaxed until they accept anything.
    """
    t = build(4, 1)
    out, _, _ = run_aiter(t)
    with pytest.raises(AssertionError):
        assert_close(out * 1.005, out, "drift", REL_TOL, MAX_TOL)
    # A single bad element must be caught too, not averaged away.
    corrupted = out.clone()
    corrupted.view(-1)[0] += out.abs().max()
    with pytest.raises(AssertionError):
        assert_close(corrupted, out, "single-element", REL_TOL, MAX_TOL)
