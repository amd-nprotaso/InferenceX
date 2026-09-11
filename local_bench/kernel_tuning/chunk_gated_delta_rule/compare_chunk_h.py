"""Compare a FlyDSL chunk-state kernel with installed SGLang on Qwen3.5 shapes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import statistics
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import torch

from shapes import SHAPES, Inputs, Shape, build_inputs


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def load_flydsl(path: Path) -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location("comparison_flydsl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load FlyDSL source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.chunk_gated_delta_rule_fwd_h


def kwargs_for(inp: Inputs) -> dict[str, Any]:
    return dict(
        k=inp.k, w=inp.w, u=inp.u, g=inp.g,
        initial_state=inp.initial_state,
        initial_state_indices=inp.initial_state_indices,
        cu_seqlens=inp.cu_seqlens,
        chunk_indices=inp.chunk_indices,
    )


def compare_outputs(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], rtol: float, atol: float
) -> dict[str, Any]:
    checks = {}
    for label, got, want in zip(("h", "v_new", "final_state"), actual, expected, strict=True):
        if not bool(torch.isfinite(want).all()) or not bool(torch.isfinite(got).all()):
            raise AssertionError(f"Nonfinite {label}; timings would be invalid")
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol, msg=lambda msg: f"{label}: {msg}")
        checks[label] = dict(exact=torch.equal(got, want), max_abs_error=(got.float() - want.float()).abs().max().item())
    return checks


def capture(
    fn: Callable[..., Any], kwargs: dict[str, Any], inp: Inputs, warmup: int
) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...]]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            inp.restore_state()
            fn(**kwargs)
    stream.synchronize()
    inp.restore_state()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = fn(**kwargs)
    torch.cuda.synchronize()
    return graph, tuple(outputs)


def measure(graph: torch.cuda.CUDAGraph, inp: Inputs, repeats: int) -> list[float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        inp.restore_state()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def benchmark(
    shape: Shape, functions: dict[str, Callable[..., Any]], args: argparse.Namespace
) -> dict[str, Any]:
    inp = build_inputs(shape, device=f"cuda:{args.device}", seed=args.seed, pool_slots=args.pool_slots)
    kwargs = kwargs_for(inp)
    inp.restore_state()
    reference = functions["sglang"](**kwargs)
    expected = (*reference, inp.initial_state[inp.initial_state_indices].clone())
    inp.restore_state()
    actual = functions["flydsl"](**kwargs)
    eager_checks = compare_outputs((*actual, inp.initial_state[inp.initial_state_indices]), expected, args.rtol, args.atol)
    del actual
    graphs = {}
    graph_checks = {}
    for name, fn in functions.items():
        graph, outputs = capture(fn, kwargs, inp, args.warmup)
        inp.restore_state()
        graph.replay()
        graph_checks[name] = compare_outputs(
            (*outputs, inp.initial_state[inp.initial_state_indices]), expected, args.rtol, args.atol
        )
        graphs[name] = (graph, outputs)
    del expected, reference
    trials = []
    for trial in range(args.trials):
        order = ("sglang", "flydsl") if trial % 2 == 0 else ("flydsl", "sglang")
        timings = {}
        for name in order:
            graph, _ = graphs[name]
            # Rewarm each backend immediately before its timing block.
            for _ in range(args.warmup):
                inp.restore_state()
                graph.replay()
            torch.cuda.synchronize()
            samples = measure(graph, inp, args.repeats)
            timings[name] = dict(
                median_ms=statistics.median(samples),
                stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
                samples_ms=samples,
            )
        trials.append(dict(trial=trial + 1, order=order, timings=timings))
    medians = {name: statistics.median(t["timings"][name]["median_ms"] for t in trials) for name in functions}
    return dict(
        shape=asdict(shape), eager_correctness=eager_checks, graph_correctness=graph_checks,
        trials=trials, median_ms=medians, speedup=medians["sglang"] / medians["flydsl"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", choices=tuple(SHAPES), default=list(SHAPES))
    parser.add_argument("--tokens", nargs="+", type=positive, help="Override lengths for each selected head geometry; multiples of 64")
    parser.add_argument("--flydsl-file", type=Path, default=Path(__file__).with_name("chunk_delta_h_flydsl.py"))
    parser.add_argument("--device", type=int, default=0, help="Logical GPU index")
    parser.add_argument("--warmup", type=positive, default=10)
    parser.add_argument("--repeats", type=positive, default=100)
    parser.add_argument("--trials", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-slots", type=positive, default=None, help="Default: shape's traced state-pool size")
    parser.add_argument("--rtol", type=float, default=0.016)
    parser.add_argument("--atol", type=float, default=0.016)
    parser.add_argument("--output", type=Path, help="Save metadata, correctness checks, and every timing sample as JSON")
    args = parser.parse_args()
    if args.tokens and any(length % 64 for length in args.tokens):
        parser.error("--tokens must be positive multiples of 64")
    if args.rtol < 0 or args.atol < 0:
        parser.error("tolerances must be nonnegative")
    if not args.flydsl_file.is_file():
        parser.error(f"FlyDSL source does not exist: {args.flydsl_file}")
    if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
        parser.error("--device must name an available GPU")
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    if not getattr(properties, "gcnArchName", "").startswith("gfx950"):
        parser.error("this FlyDSL specialization requires gfx950")

    from sglang.kernels.ops.attention.fla import chunk_delta_h as production

    sglang_fn = production.chunk_gated_delta_rule_fwd_h
    if Path(inspect.getfile(sglang_fn)).resolve() != Path(production.__file__).resolve():
        parser.error("SGLang entry point is patched; disable the FlyDSL import hook before comparing")
    functions = {"sglang": sglang_fn, "flydsl": load_flydsl(args.flydsl_file.resolve())}
    shapes = []
    seen = set()
    for name in args.shapes:
        original = SHAPES[name]
        for length in args.tokens or [original.T]:
            key = (original.Hg, original.H, length)
            if key in seen:
                continue
            seen.add(key)
            shape = original if length == original.T else replace(original, name=f"{name}-t{length}", T=length, note=f"Length override from {name}")
            if args.pool_slots is not None:
                shape = replace(shape, pool_slots=args.pool_slots)
            shapes.append(shape)
    versions = {}
    for package in ("torch", "triton", "flydsl", "sglang"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    report = dict(
        method="GPU events around one graph replay; state reset excluded; median of alternating trial medians",
        model="Qwen3.5-397B-A17B-MXFP4 (BF16 GDN tensors)",
        argv=sys.argv, gpu=properties.name, architecture=properties.gcnArchName,
        versions=versions, options={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        flydsl_source=str(args.flydsl_file.resolve()),
        flydsl_sha256=hashlib.sha256(args.flydsl_file.read_bytes()).hexdigest(),
        sglang_source=production.__file__,
        sglang_sha256=hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest(),
        sglang_configs=[str(config) for config in production.chunk_gated_delta_rule_fwd_kernel_h_blockdim64.configs],
        environment={key: value for key, value in os.environ.items() if key.startswith(("SGLANG_GDN_", "SGLANG_FLYDSL_GDN_")) or key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
        results=[],
    )
    print(f"GPU: {properties.name} | SGLang defaults: {report['sglang_configs']}", flush=True)
    print(f"{'Shape':<34} {'T':>7} {'Hg/H':>7} {'SGLang ms':>11} {'FlyDSL ms':>11} {'Speedup':>9}", flush=True)
    with torch.inference_mode():
        for shape in shapes:
            row = benchmark(shape, functions, args)
            report["results"].append(row)
            print(f"{shape.name:<34} {shape.T:7d} {shape.Hg:2d}/{shape.H:<4d} {row['median_ms']['sglang']:11.6f} {row['median_ms']['flydsl']:11.6f} {row['speedup']:8.3f}x", flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Correctness passed. Speedup = SGLang / FlyDSL (>1 favors FlyDSL). Kernel-only; no full-model throughput claim.")


if __name__ == "__main__":
    main()
