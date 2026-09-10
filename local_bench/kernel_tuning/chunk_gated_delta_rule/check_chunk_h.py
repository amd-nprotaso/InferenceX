"""Correctness checks for the specialized FlyDSL chunk-state kernel."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import Config, _kernel
from chunk_delta_h_flydsl import chunk_gated_delta_rule_fwd_h
from shapes import SHAPES, build_inputs


def reference(inp, state, indices, g, gk, save, exp2):
    s = inp.shape
    h = torch.empty((1, s.NT, s.H, s.V, s.K), dtype=inp.k.dtype, device=inp.k.device)
    vn = torch.empty_like(inp.u) if save else None
    kernel = _kernel()
    kernel.configs = [Config(16, 4, 3, waves_per_eu=4).to_triton()]
    kernel.cache.clear()
    kernel[(triton.cdiv(s.V, 16), s.H)](
        k=inp.k,
        v=inp.u,
        w=inp.w,
        v_new=vn,
        g=g,
        gk=gk,
        h=h,
        initial_state=state if state is not None else inp.initial_state,
        initial_state_indices=indices,
        stride_init_state=0 if state is None else state.stride(0),
        cu_seqlens=None,
        chunk_offsets=None,
        T=s.T,
        H=s.H,
        Hg=s.Hg,
        K=s.K,
        V=s.V,
        BT=64,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=state is not None,
        INPLACE_UPDATE=state is not None,
        SAVE_NEW_VALUE=save,
        IS_VARLEN=False,
        NT_BUCKET=2,
        USE_EXP2=exp2,
    )
    return h, vn


def extra_checks() -> None:
    s = replace(SHAPES["qwen35-397b-tp4"], T=64)
    i = build_inputs(s, device="cuda", seed=19, pool_slots=2)
    i.initial_state[0].normal_(0, 0.01)
    h0 = i.initial_state[0].float().clone()
    predicted = torch.einsum("thk,hvk->thv", i.w[0].float(), h0.to(torch.bfloat16).float())
    v = i.u[0].float() - predicted
    gate = torch.exp(i.g[0, -1, None] - i.g[0])
    corrected = (v * gate[..., None]).to(torch.bfloat16).float()
    grouped_k = i.k[0].repeat_interleave(s.H // s.Hg, dim=1).float()
    ht = h0 * torch.exp(i.g[0, -1])[:, None, None] + torch.einsum("thk,thv->hvk", grouped_k, corrected)
    got = chunk_gated_delta_rule_fwd_h(
        i.k, i.w, i.u, i.g, initial_state=i.initial_state, initial_state_indices=i.initial_state_indices
    )
    torch.testing.assert_close(got[0][0, 0], h0.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(got[1][0].float(), v.to(torch.bfloat16).float(), rtol=0.016, atol=0.016)
    torch.testing.assert_close(i.initial_state[0].float(), ht.to(torch.bfloat16).float(), rtol=0.016, atol=0.016)
    print("independent FP32 reference: pass", flush=True)

    pitch = (1 << 31) + 512
    state = torch.empty_strided((2, s.H, s.V, s.K), (pitch, s.V * s.K, s.K, 1), device="cuda", dtype=torch.bfloat16)
    state.copy_(i.initial_state)
    before = state.clone()
    i.initial_state_indices.fill_(1)
    ref = reference(i, state, i.initial_state_indices, i.g, None, True, False)
    expected = state.clone()
    state.copy_(before)
    got = chunk_gated_delta_rule_fwd_h(
        i.k, i.w, i.u, i.g, initial_state=state, initial_state_indices=i.initial_state_indices
    )
    assert all(torch.equal(a, b) for a, b in zip(ref, got))
    assert torch.equal(expected, state)
    print("slot offset > 2**31 elements: exact pass", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/tmp/gdn_fly_correctness.json")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    rows = []
    for shape in SHAPES.values():
        if a.smoke:
            shape = replace(shape, T=64)
        inp = build_inputs(shape, device="cuda", seed=0, pool_slots=2)
        original = inp.initial_state.clone()
        original[0].normal_(0, 0.01)
        gk = -torch.rand((1, shape.T, shape.H, shape.K), device="cuda") * 0.1
        for gate in ("g", "gk", "gk_exp2", "none"):
            for use_state in (True, False):
                for save in (True, False):
                    g = inp.g if gate == "g" else None
                    kg = gk if gate.startswith("gk") else None
                    exp2 = gate == "gk_exp2"
                    state = inp.initial_state if use_state else None
                    inp.initial_state.copy_(original)
                    ref = reference(inp, state, inp.initial_state_indices, g, kg, save, exp2)
                    expected_state = inp.initial_state.clone()
                    inp.initial_state.copy_(original)
                    got = chunk_gated_delta_rule_fwd_h(
                        inp.k,
                        inp.w,
                        inp.u,
                        g,
                        kg,
                        state,
                        inp.initial_state_indices,
                        save,
                        inp.cu_seqlens,
                        inp.chunk_indices,
                        exp2,
                    )
                    torch.cuda.synchronize()
                    exact = []
                    for x, y in zip([*ref, expected_state], [*got, inp.initial_state]):
                        if x is None:
                            assert y is None
                        else:
                            assert torch.isfinite(y).all(), (shape.name, gate, "nonfinite")
                            exact.append(torch.equal(x, y))
                    row = dict(shape=shape.name, tokens=shape.T, gate=gate, state=use_state, save=save, exact=exact)
                    print(json.dumps(row), flush=True)
                    rows.append(row)
                    assert all(exact), row
        # -1 must neither read nor write the pool, while outputs use zero state.
        inp.initial_state_indices.fill_(-1)
        inp.initial_state.copy_(original)
        ref = reference(inp, inp.initial_state, inp.initial_state_indices, inp.g, None, True, False)
        got = chunk_gated_delta_rule_fwd_h(
            inp.k, inp.w, inp.u, inp.g, initial_state=inp.initial_state, initial_state_indices=inp.initial_state_indices
        )
        assert all(torch.equal(x, y) for x, y in zip(ref, got))
        assert torch.equal(original, inp.initial_state)
        inp.initial_state_indices.zero_()
        # Non-contiguous slot pitch and an indirect nonzero slot.
        envelope = torch.zeros((2, 3, shape.H, shape.V, shape.K), device="cuda", dtype=inp.k.dtype)
        strided = envelope[:, 1]
        strided.copy_(original)
        before = envelope.clone()
        inp.initial_state_indices.fill_(1)
        ref = reference(inp, strided, inp.initial_state_indices, inp.g, None, True, False)
        expected = envelope.clone()
        envelope.copy_(before)
        args = (inp.k, inp.w, inp.u, inp.g, None, strided, inp.initial_state_indices)
        got = chunk_gated_delta_rule_fwd_h(*args)
        assert all(torch.equal(x, y) for x, y in zip(ref, got))
        assert torch.equal(expected, envelope)
        # Warm the concrete signature before capture; restore before each launch.
        envelope.copy_(before)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = chunk_gated_delta_rule_fwd_h(*args)
        for _ in range(3):
            envelope.copy_(before)
            graph.replay()
            torch.cuda.synchronize()
            assert all(torch.equal(x, y) for x, y in zip(ref, captured))
            assert torch.equal(expected, envelope)
        rows.append(dict(shape=shape.name, padded=True, envelope=True, graph_replays=3))
    extra_checks()
    rows.append(dict(independent_fp32=True, large_slot_stride=True))
    Path(a.output).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
