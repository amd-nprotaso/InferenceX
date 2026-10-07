#!/usr/bin/env python3
"""Benchmark Qwen3.8's post-compaction CK attention, not the full QSA layer.

Source-derived TP=1 shapes: BF16, Hq=24, Hkv=2, D=256, budget=2048,
compression=4. Each query is an independent varlen sequence of length one,
including speculative verify rows. ISL/OSL select visible-context samples;
the kernel reads only selected KV (2048 plus 0..3 tail tokens at long context).
Synthetic packed KV excludes indexer, top-k, compaction, and model execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    result = []
    # Sample early, middle and final decode contexts, with all compression residues.
    contexts = sorted({args.isl + i for i in range(1, min(args.osl, 4) + 1)}
                      | {args.isl + max(1, args.osl // 2), args.isl + args.osl})
    for conc in args.conc:
        for mode in args.modes.split(","):
            width = 1 if mode == "decode" else args.verify_tokens
            for context in contexts:
                visible = [context + i for _ in range(conc) for i in range(width)]
                lengths = [min(n // 4 * 4, 2048) + n % 4 for n in visible]
                result.append(dict(conc=conc, mode=mode, context=context,
                                   rows=conc * width, lengths=lengths))
    return result


def trace_evidence(directory: Path) -> dict[str, Any]:
    examples = json.loads((directory / "amd_kernel_examples.json").read_text())
    names = [name for name in examples if "ck_tile::kentry" in name and "FmhaFwdKernel<" in name]
    phases = []
    with (directory / "amd_phase_kernels.csv").open() as stream:
        for row in csv.DictReader(stream):
            if any("FmhaFwdKernel<" in str(value) for value in row.values()):
                phases.append(row)
    return dict(kernel_names=names, phase_statistics=phases,
                shape_provenance="Local SGLang source plus trace report; AMD trace has no input tensor shapes",
                recorded_isl=32768, recorded_osl=32768, recorded_concurrency=1)


def run_case(torch: Any, attention: Any, case: dict[str, Any], args: argparse.Namespace,
             expected_names: list[str], destination: Path) -> dict[str, Any]:
    rows, lengths = case["rows"], case["lengths"]
    q = torch.randn(rows, 24, 256, dtype=torch.bfloat16, device="cuda")
    # Production scratch has capacity rows*2051 even when the compact total is smaller.
    k = torch.randn(rows * 2051, 2, 256, dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    cuq = torch.arange(rows + 1, dtype=torch.int32, device=q.device)
    cuk = torch.tensor([0] + lengths, dtype=torch.int32, device=q.device).cumsum(0, dtype=torch.int32)

    def launch() -> Any:
        out = attention(q=q, k=k, v=v, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
                        max_seqlen_q=1, max_seqlen_k=2051,
                        softmax_scale=256 ** -0.5, causal=True)
        return out[0] if isinstance(out, tuple) else out

    actual = launch()
    import hashlib
    output_hash = hashlib.sha256(actual.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    # Independent FP32 attention reference, one compact sequence at a time.
    offset = 0
    max_error = 0.0
    for row, length in enumerate(lengths):
        kr = k[offset:offset + length].float().repeat_interleave(12, dim=1)
        vr = v[offset:offset + length].float().repeat_interleave(12, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), kr) / 16
        expected = torch.einsum("hk,khd->hd", scores.softmax(-1), vr)
        torch.testing.assert_close(actual[row].float(), expected, atol=0.003, rtol=0.03)
        max_error = max(max_error, (actual[row].float() - expected).abs().max().item())
        offset += length
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    # Multiple calls in one graph avoid CPU launch starvation for small kernels.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.inner):
            launch()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / args.inner)
    result = dict(case, q_shape=list(q.shape), kv_capacity_shape=list(k.shape),
                  compact_kv_tokens=sum(lengths), max_seqlen_q=1, max_seqlen_k=2051,
                  median_us=statistics.median(samples), min_us=min(samples),
                  p95_us=sorted(samples)[min(len(samples)-1, int(0.95 * len(samples)))],
                  max_abs_error=max_error, samples_us=samples, output_sha256=output_hash)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            launch()
            torch.cuda.synchronize()
        path = destination / f"profile_{case['mode']}_c{case['conc']}_ctx{case['context']}.json"
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        ck = [e for e in kernels if "ck_tile::kentry" in e["name"] and "FmhaFwdKernel<" in e["name"]]
        result.update(profile=str(path), ck_kernel_names=sorted({e["name"] for e in ck}),
                      ck_kernel_us=sum(e["dur"] for e in ck),
                      exact_trace_symbol_match=any(e["name"] in expected_names for e in ck))
        if not ck:
            raise RuntimeError(f"No CK FmhaFwdKernel found; inspect {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isl", type=positive, default=32678)
    parser.add_argument("--osl", type=positive, default=32678)
    parser.add_argument("--conc", default="1,4,8,16")
    parser.add_argument("--modes", default="decode,verify")
    parser.add_argument("--verify-tokens", type=positive, default=4)
    parser.add_argument("--aiter-path", type=Path, default=Path("/var/home/my_aiter/aiter"))
    parser.add_argument("--trace-dir", type=Path, default=Path("/var/home/qwen_trace_analysis"))
    parser.add_argument("--output", type=Path, default=Path("qsa_ck_results"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--jit-dir", type=Path, default=Path(".qsa_aiter_jit"),
                        help="isolated AITER cache built against the active PyTorch")
    parser.add_argument("--warmup", type=positive, default=10)
    parser.add_argument("--inner", type=positive, default=20)
    parser.add_argument("--repeats", type=positive, default=30)
    parser.add_argument("--profile", action="store_true", help="confirm CK dispatch and save per-case traces")
    parser.add_argument("--dry-run", action="store_true", help="print shapes without importing torch/AITER")
    args = parser.parse_args()
    try:
        args.conc = [positive(value) for value in args.conc.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    if any(mode not in {"decode", "verify"} for mode in args.modes.split(",")):
        parser.error("--modes must contain decode and/or verify")
    if args.verify_tokens > 4:
        parser.error("local QSA supports at most four speculative tokens")
    evidence = trace_evidence(args.trace_dir)
    matrix = cases(args)
    if args.dry_run:
        print(json.dumps(dict(cases=matrix, trace=evidence), indent=2))
        return
    sys.path.insert(0, str(args.aiter_path.resolve()))
    os.environ["AITER_JIT_DIR"] = str(args.jit_dir.resolve())
    import torch
    import aiter

    if not Path(aiter.__file__).resolve().is_relative_to(args.aiter_path.resolve()):
        raise RuntimeError(f"Wrong AITER imported: {aiter.__file__}")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Requires ROCm PyTorch and an accessible AMD GPU")
    torch.cuda.set_device(args.device)
    torch.manual_seed(2026)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    torch=torch.__version__, hip=torch.version.hip, aiter=aiter.__file__,
                    gpu=str(torch.cuda.get_device_properties(args.device)), trace=evidence,
                    timing="GPU events around graph replay / inner; includes AITER call kernels, excludes gather/indexer",
                    limitations="Synthetic compact KV; extrapolated concurrency; context samples, not full generation")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with torch.inference_mode(), (args.output / "results.jsonl").open("w") as stream:
        for case in matrix:
            result = run_case(torch, aiter.flash_attn_varlen_func, case, args,
                              evidence["kernel_names"], args.output)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            print(f"{case['mode']:6s} C={case['conc']:2d} rows={case['rows']:2d} "
                  f"context={case['context']} KV={min(case['lengths'])}..{max(case['lengths'])} "
                  f"median={result['median_us']:.3f} us "
                  f"symbol_match={result.get('exact_trace_symbol_match', 'unprofiled')}", flush=True)


if __name__ == "__main__":
    main()
