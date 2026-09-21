"""Shared fixtures and comparison helpers for the GDN kernel accuracy tests.

What "accuracy" means here differs per kernel, deliberately:

* **prefill conv** is checked against an *independent fp32 PyTorch reference*
  (``reference_causal_conv1d`` below). A causal depthwise conv is short enough to
  reimplement without much risk of the reference itself being wrong, so this is a
  real ground-truth check -- and the AITER kernel is additionally required to be
  no worse than the Triton kernel it replaces.

* **GDN decode** is checked against the **Triton kernel**, not ground truth. The
  gated delta-rule recurrence (l2-norm, sigmoid gating, softplus, delta
  projection, per-step state snapshots) is intricate enough that a hand-written
  fp32 reference would more likely encode my misreading than catch a kernel bug.
  Triton is the incumbent that ships today, so "matches Triton" is the property
  that actually matters for a drop-in replacement. Where a stronger claim IS
  available -- strided vs contiguous inputs must be *bit-identical*, and a
  suppressed state write-back must leave the pool untouched -- the tests assert
  exactly that instead of a tolerance.
"""

import os

import pytest
import torch

# The kernels are gfx950 FlyDSL and are reached through the AITER wrappers, both
# of which want these set before import.
os.environ.setdefault("SGLANG_USE_AITER", "1")
os.environ.setdefault("AITER_FLYDSL_FORCE", "1")
os.environ.setdefault("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")

DEVICE = "cuda"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: production-sized shapes")


@pytest.fixture(scope="session", autouse=True)
def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no GPU", allow_module_level=True)


@pytest.fixture(autouse=True)
def _seed():
    # Every test draws its own inputs; fix the seed so a failure is reproducible.
    torch.manual_seed(1234)


def rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Mean |a-e| divided by the mean magnitude of `expected`.

    Scale-relative rather than per-element relative: these tensors contain
    near-zero entries where a per-element ratio explodes without indicating any
    real disagreement.
    """
    a, e = actual.float(), expected.float()
    return ((a - e).abs().mean() / e.abs().mean().clamp_min(1e-9)).item()


def max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Largest single-element disagreement, normalised by the tensor's scale."""
    a, e = actual.float(), expected.float()
    return ((a - e).abs().max() / e.abs().max().clamp_min(1e-9)).item()


def assert_close(actual, expected, name, rel_tol, max_tol):
    """Assert agreement on both the average and the worst element.

    Mean error alone hides a single catastrophically wrong element (one bad lane,
    one out-of-bounds write); max error alone is noisy in bf16. Both must pass.
    """
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    assert torch.isfinite(actual.float()).all(), f"{name}: contains NaN/Inf"
    rel, mx = rel_error(actual, expected), max_error(actual, expected)
    assert rel <= rel_tol and mx <= max_tol, (
        f"{name}: mean-rel={rel:.3e} (tol {rel_tol:.0e}), "
        f"max-rel={mx:.3e} (tol {max_tol:.0e})"
    )


def assert_bitwise(actual, expected, name):
    """Exact equality -- for cases where only addressing may differ, not math."""
    assert torch.equal(actual, expected), (
        f"{name}: expected bit-identical results, "
        f"max|diff|={(actual.float() - expected.float()).abs().max().item():.3e}"
    )


def reference_causal_conv1d(
    x, weight, bias, conv_states, seq_lens, cache_indices, has_initial_state,
    activation, pad_slot_id=-1,
):
    """Independent fp32 reference for the varlen causal conv1d.

    Mirrors the documented ``causal_conv1d_fn`` contract: ``x`` is
    ``[dim, total_tokens]`` with sequences concatenated left to right; each
    sequence's history is either zeros or ``conv_states[cache_indices[i]]``; the
    state is left holding the last ``width-1`` inputs of that sequence.

    Returns ``(out, new_states)`` and does not mutate its arguments, so a caller
    can compare an in-place kernel against it.
    """
    dim, _ = x.shape
    width = weight.shape[1]
    xf = x.float()
    wf = weight.float()
    bf = None if bias is None else bias.float()
    new_states = None if conv_states is None else conv_states.float().clone()
    out = torch.zeros_like(xf)

    start = 0
    for i, n in enumerate(seq_lens):
        if n == 0:
            continue
        slot = i if cache_indices is None else int(cache_indices[i])
        seq = xf[:, start : start + n]

        if (
            has_initial_state is not None
            and bool(has_initial_state[i])
            and new_states is not None
            and slot != pad_slot_id
        ):
            hist = new_states[slot][:, -(width - 1):]
        else:
            hist = torch.zeros(dim, width - 1, device=x.device, dtype=torch.float32)

        padded = torch.cat([hist, seq], dim=1)            # [dim, width-1+n]
        # Depthwise causal conv: output t sees inputs t-width+1 .. t.
        acc = torch.zeros(dim, n, device=x.device, dtype=torch.float32)
        for w in range(width):
            acc += wf[:, w : w + 1] * padded[:, w : w + n]
        if bf is not None:
            acc += bf[:, None]
        if activation in ("silu", "swish"):
            acc = acc * torch.sigmoid(acc)
        out[:, start : start + n] = acc

        if new_states is not None and slot != pad_slot_id:
            tail = padded[:, -(width - 1):] if width > 1 else padded[:, :0]
            new_states[slot][:, -(width - 1):] = tail
        start += n

    return out, new_states
