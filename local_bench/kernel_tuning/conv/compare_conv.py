"""Compare FlyDSL and installed SGLang prefill convolution on Qwen3.5 TP=4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import statistics
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch

# Qwen3.5-397B-A17B: per-rank Q/K = 4x128, V = 16x128 at TP4.
# Geometry and 32K prefill budget: ../chunk_gated_delta_rule/shapes.py.
CHANNELS = 2 * 4 * 128 + 16 * 128
SHAPES = {
    "qwen35-397b-tp4": [32768],
    "qwen35-397b-tp4-4k": [4096],
    "qwen35-397b-tp4-packed4": [8192] * 4,
}


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def load_flydsl(path: Path) -> Callable[..., torch.Tensor]:
    spec = importlib.util.spec_from_file_location("comparison_flydsl_conv", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load FlyDSL source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.causal_conv1d_fn


def compare_outputs(
    actual: torch.Tensor, expected: torch.Tensor,
    state: torch.Tensor, expected_state: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    for tensor in (actual, expected, state, expected_state):
        if not bool(torch.isfinite(tensor).all()):
            raise AssertionError("Nonfinite output/state; timings would be invalid")
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
    # State copies are exact, including unused pool slots.
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    return dict(output_exact=torch.equal(actual, expected), state_exact=True,
                max_abs_error=(actual.float() - expected.float()).abs().max().item())


def benchmark(
    lengths: list[int], functions: dict[str, Callable[..., torch.Tensor]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = f"cuda:{args.device}"
    gen = torch.Generator(device=device).manual_seed(args.seed)
    batch, total = len(lengths), sum(lengths)
    slots = args.pool_slots
    def randn(*size: int) -> torch.Tensor:
        return torch.randn(size, device=device, dtype=torch.bfloat16, generator=gen)
    x = randn(total, CHANNELS).T
    weight = randn(CHANNELS, 4) * 0.25
    state = randn(slots, CHANNELS, 3)
    if args.state_layout == "channel-last":
        state = state.transpose(1, 2).contiguous().transpose(1, 2)
    original = state.clone()
    indices = torch.randperm(slots, device=device, generator=gen)[:batch]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    initial = [args.history == "cached" or (args.history == "mixed" and i % 2 == 0)
               for i in range(batch)]
    kwargs = dict(x=x, weight=weight, bias=None, conv_states=state,
                  query_start_loc=torch.tensor(starts, device=device, dtype=torch.int32),
                  seq_lens_cpu=lengths, cache_indices=indices,
                  has_initial_state=torch.tensor(initial, device=device, dtype=torch.bool),
                  activation="silu", pad_slot_id=-1)
    def restore() -> None:
        state.copy_(original)

    expected = functions["sglang"](**kwargs).clone()
    expected_state = state.clone()
    restore()
    actual = functions["flydsl"](**kwargs)
    eager = compare_outputs(actual, expected, state, expected_state, args)
    del actual
    graphs = {}
    graph_checks = {}
    for name, fn in functions.items():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(args.warmup):
                restore()
                fn(**kwargs)
        stream.synchronize()
        restore()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = fn(**kwargs)
        torch.cuda.synchronize()
        restore()
        graph.replay()
        graph_checks[name] = compare_outputs(output, expected, state, expected_state, args)
        graphs[name] = (graph, output)
    del expected, expected_state
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    trials = []
    for trial in range(args.trials):
        order = ("sglang", "flydsl") if trial % 2 == 0 else ("flydsl", "sglang")
        timings = {}
        for name in order:
            graph, _ = graphs[name]
            for _ in range(args.warmup):
                restore()
                graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.repeats):
                restore()
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            timings[name] = dict(median_ms=statistics.median(samples),
                                 stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
                                 samples_ms=samples)
        trials.append(dict(trial=trial + 1, order=order, timings=timings))
    medians = {name: statistics.median(t["timings"][name]["median_ms"] for t in trials)
               for name in functions}
    return dict(lengths=lengths, total_tokens=total, channels=CHANNELS, width=4,
                dtype="bfloat16", pool_slots=slots, initial_history=initial,
                strides={key: list(value.stride()) for key, value in kwargs.items()
                         if isinstance(value, torch.Tensor)},
                eager_correctness=eager, graph_correctness=graph_checks,
                trials=trials, median_ms=medians, speedup=medians["sglang"] / medians["flydsl"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--shapes", nargs="+", choices=tuple(SHAPES))
    selection.add_argument("--lengths", nargs="+", type=positive,
                           help="Custom packed sequence lengths, e.g. 1 7 8192")
    parser.add_argument("--flydsl-file", type=Path, default=Path(__file__).with_name("causal_conv1d_flydsl.py"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=positive, default=10)
    parser.add_argument("--repeats", type=positive, default=100)
    parser.add_argument("--trials", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-slots", type=positive, default=1243)
    parser.add_argument("--history", choices=("fresh", "cached", "mixed"), default="fresh")
    parser.add_argument("--state-layout", choices=("channel-first", "channel-last"), default="channel-first")
    parser.add_argument("--channels-per-thread", type=int, choices=(1, 2))
    parser.add_argument("--prefetch", type=int, choices=(1, 2, 4, 8, 16),
                        help="FlyDSL input prefetch window; omitted uses the kernel default")
    parser.add_argument("--block", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--tokens", type=int, choices=range(3, 65), default=16,
                        help="FlyDSL tokens per block (not sequence length)")
    parser.add_argument("--rtol", type=float, default=0.016)
    parser.add_argument("--atol", type=float, default=0.016)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = {"custom-tp4": args.lengths} if args.lengths else {
        name: SHAPES[name] for name in args.shapes or SHAPES}
    if any(len(lengths) > args.pool_slots for lengths in cases.values()):
        parser.error("--pool-slots must be at least the number of packed sequences")
    if any(not math.isfinite(t) or t < 0 for t in (args.rtol, args.atol)):
        parser.error("tolerances must be finite and nonnegative")
    if not args.flydsl_file.is_file():
        parser.error(f"FlyDSL source does not exist: {args.flydsl_file}")
    if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
        parser.error("--device must name an available GPU")
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    if torch.version.hip is None:
        parser.error("the FlyDSL kernel requires a ROCm GPU")
    from sglang.kernels.ops.mamba import causal_conv1d_triton as production
    baseline = production.causal_conv1d_fn
    if Path(inspect.getfile(baseline)).resolve() != Path(production.__file__).resolve():
        parser.error("SGLang entry point is patched; disable the import hook before comparing")
    tuning = dict(block=args.block, tokens=args.tokens)
    if args.channels_per_thread is not None:
        tuning["channels_per_thread"] = args.channels_per_thread
    if args.prefetch is not None:
        tuning["prefetch"] = args.prefetch
    functions = dict(sglang=baseline, flydsl=partial(load_flydsl(args.flydsl_file.resolve()), **tuning))
    versions = {}
    for package in ("torch", "triton", "flydsl", "sglang"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    sources = {}
    for name, path in (("flydsl", args.flydsl_file), ("sglang", Path(production.__file__))):
        sources[name] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report = dict(model="Qwen3.5-397B-A17B-MXFP4", tp=4, argv=sys.argv,
                  method="GPU events around one graph replay; state reset excluded; median of alternating trial medians",
                  gpu=properties.name, architecture=getattr(properties, "gcnArchName", "unknown"),
                  versions=versions, sources=sources,
                  options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  environment={k: v for k, v in os.environ.items() if k.startswith("SGLANG_FLYDSL_")
                               or k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
                  results=[])
    print(f"GPU: {properties.name} | TP=4, channels={CHANNELS}, BF16, width=4, SiLU", flush=True)
    print(f"{'Shape':<34} {'Tokens':>7} {'Batch':>5} {'SGLang ms':>11} {'FlyDSL ms':>11} {'Speedup':>9}", flush=True)
    with torch.inference_mode():
        for name, lengths in cases.items():
            row = dict(name=name, **benchmark(lengths, functions, args))
            report["results"].append(row)
            print(f"{name:<34} {sum(lengths):7d} {len(lengths):5d} {row['median_ms']['sglang']:11.6f} {row['median_ms']['flydsl']:11.6f} {row['speedup']:8.3f}x", flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Correctness passed. Speedup = SGLang / FlyDSL (>1 favors FlyDSL). Kernel-only timing.")


if __name__ == "__main__":
    main()
