"""Device/ATT worker for isolated Triton and FlyDSL chunk-state experiments.

Run through rocprofv3; stdout reports correctness and CUDA-event cross-checks.
The installed SGLang source is never edited by the grid-permutation experiment.
"""

import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
from pathlib import Path

import torch
from bench import Config
from chunk_delta_h_flydsl import chunk_gated_delta_rule_fwd_h
from shapes import SHAPES, build_inputs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("flydsl", "triton", "permuted"), default="flydsl")
    p.add_argument("--shape", choices=tuple(SHAPES), default="qwen35-397b-tp4")
    p.add_argument("--prefetch-depth", type=int, default=2)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--launches", type=int, default=60)
    a = p.parse_args()
    import chunk_delta_h_flydsl

    chunk_delta_h_flydsl.PREFETCH_DEPTH = a.prefetch_depth
    from sglang.kernels.ops.attention.fla import chunk_delta_h as production

    candidate = production
    temporary = tempfile.TemporaryDirectory(prefix="gdn-grid-")
    if a.backend == "permuted":
        source = Path(production.__file__).read_text()
        replacements = {
            "i_v, i_nh = tl.program_id(0), tl.program_id(1)": "i_v, i_nh = tl.program_id(1), tl.program_id(0)",
            'return (triton.cdiv(V, meta["BV"]), N * H)': 'return (N * H, triton.cdiv(V, meta["BV"]))',
        }
        for before, after in replacements.items():
            assert source.count(before) == 1, "installed source has changed"
            source = source.replace(before, after)
        path = Path(temporary.name) / "gdn_permuted.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location("gdn_permuted", path)
        candidate = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = candidate
        spec.loader.exec_module(candidate)
    production.chunk_gated_delta_rule_fwd_kernel_h_blockdim64.configs = [Config(16, 4, 3, waves_per_eu=4).to_triton()]
    inp = build_inputs(SHAPES[a.shape], device="cuda", seed=0)
    kwargs = dict(
        k=inp.k,
        w=inp.w,
        u=inp.u,
        g=inp.g,
        initial_state=inp.initial_state,
        initial_state_indices=inp.initial_state_indices,
        cu_seqlens=inp.cu_seqlens,
        chunk_indices=inp.chunk_indices,
    )
    inp.restore_state()
    ref = production.chunk_gated_delta_rule_fwd_h(**kwargs)
    state = inp.initial_state.clone()
    fn = chunk_gated_delta_rule_fwd_h if a.backend == "flydsl" else candidate.chunk_gated_delta_rule_fwd_h
    if a.backend != "flydsl":
        kernel = candidate.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
        kernel.configs = [Config(16, 4, a.stages, waves_per_eu=4).to_triton()]
        kernel.cache.clear()
    inp.restore_state()
    got = fn(**kwargs)
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(ref, got))
    assert torch.equal(state, inp.initial_state)
    times = []
    for _ in range(a.launches):
        inp.restore_state()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**kwargs)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    print(
        json.dumps(
            dict(
                backend=a.backend,
                stages=a.stages,
                launches=a.launches,
                exact=True,
                event_median_ms=statistics.median(times),
            )
        ),
        flush=True,
    )
    temporary.cleanup()


if __name__ == "__main__":
    main()
