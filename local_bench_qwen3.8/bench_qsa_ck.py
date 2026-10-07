#!/usr/bin/env python3
"""Benchmark Qwen3.8's post-compaction attention core, not the full QSA layer.

Source-derived TP=1 shapes: BF16, Hq=24, Hkv=2, D=256, budget=2048,
compression=4. Each query is an independent varlen sequence of length one,
including speculative verify rows. ISL/OSL select visible-context samples;
the kernel reads only selected KV (2048 plus 0..3 tail tokens at long context).
Synthetic packed KV excludes indexer, top-k, compaction, and model execution.

Three candidate backends run on identical inputs (`--backends`):

  ck              AITER `flash_attn_varlen_func`; BF16 D=256 falls through to
                  CK Tile `FmhaFwdKernel` group mode. This is production today.
                  Its `TileFmhaShape<128,64,32,256,32,256>` puts only seqlen_q
                  on the M axis, so a length-one query occupies 1 of 128 MFMA
                  rows.

  pa_decode_tile  FlyDSL `kernels/attention/pa_decode_tile.py`. BF16 paged
                  decode; `TOTAL_ROWS = query_length * query_group_size`
                  flattens the GQA group onto M, so the same query occupies 12
                  of 16 rows. Needs the vectorized-5D paged cache, which this
                  script builds from the same packed K/V the other backends
                  read, so all three attend over bit-identical data.

  fa_gfx950       FlyDSL `kernels/attention/flash_attn_interface` ->
                  `flash_attn_gfx950.py` dual-wave, varlen packed ABI (same
                  tensors as `ck`, no layout change). That kernel documents
                  D=64/128; at D=256 it either routes to the generic BLOCK_M
                  128/256 builder or fails to build. Either outcome is recorded
                  rather than hidden -- the point of including it is to measure
                  whether a non-decode-specialised FlyDSL kernel does anything
                  for this shape.

Per-backend caveat carried into the results: `ck` and `fa_gfx950` allocate
their output inside the timed region (the graph pool absorbs it),
`pa_decode_tile` writes into a preallocated buffer because its ABI has no
allocating form. Preallocated `pmax`/`psum`/`pout` partials are likewise
required under graph capture.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
KV_CAPACITY = 2051  # indexer budget 2048 plus the incomplete compression group


class Unavailable(RuntimeError):
    """A backend cannot serve this shape; the reason is recorded, not raised."""


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def non_negative(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
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


# ── inputs shared by every backend ─────────────────────────────────────────


def make_inputs(torch: Any, case: dict[str, Any]) -> dict[str, Any]:
    """Packed varlen Q/K/V, identical for every backend in this case."""
    rows, lengths = case["rows"], case["lengths"]
    q = torch.randn(rows, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    # Production scratch has capacity rows*2051 even when the compact total is smaller.
    k = torch.randn(rows * KV_CAPACITY, NUM_KV_HEADS, HEAD_DIM, dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    cuq = torch.arange(rows + 1, dtype=torch.int32, device=q.device)
    cuk = torch.tensor([0] + lengths, dtype=torch.int32, device=q.device).cumsum(0, dtype=torch.int32)
    return dict(q=q, k=k, v=v, cu_seqlens_q=cuq, cu_seqlens_k=cuk, lengths=lengths)


def reference_error(torch: Any, inputs: dict[str, Any], actual: Any,
                    atol: float, rtol: float) -> float:
    """Independent FP32 attention reference, one compact sequence at a time."""
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    group = NUM_Q_HEADS // NUM_KV_HEADS
    offset = 0
    max_error = 0.0
    for row, length in enumerate(inputs["lengths"]):
        kr = k[offset:offset + length].float().repeat_interleave(group, dim=1)
        vr = v[offset:offset + length].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), kr) * (HEAD_DIM ** -0.5)
        expected = torch.einsum("hk,khd->hd", scores.softmax(-1), vr)
        torch.testing.assert_close(actual[row].float(), expected, atol=atol, rtol=rtol)
        max_error = max(max_error, (actual[row].float() - expected).abs().max().item())
        offset += length
    return max_error


def paged_from_packed(torch: Any, inputs: dict[str, Any], block_size: int,
                      blocks_per_seq: int) -> dict[str, Any]:
    """Re-lay the packed K/V as FlyDSL's vectorized-5D paged cache.

    key   [num_blocks, num_kv_heads, head_dim // EPV, block_size, EPV]
    value [num_blocks, num_kv_heads, block_size // EPV, v_head_dim, EPV]

    with ``EPV = 16 // element_size`` (8 for BF16) -- the layout produced by
    ``create_kv_cache`` + ``shuffle_value_cache_layout`` in FlyDSL's
    ``tests/kernels/test_pa.py``. Sequence ``r`` owns blocks
    ``[r*blocks_per_seq, (r+1)*blocks_per_seq)``, so ``block_tables`` is an
    identity range; tail slots past ``context_lengths[r]`` stay zero and are
    masked by the kernel's ``causal_bound``.
    """
    k, v, lengths = inputs["k"], inputs["v"], inputs["lengths"]
    rows = len(lengths)
    capacity = blocks_per_seq * block_size
    dense_k = torch.zeros(rows, capacity, NUM_KV_HEADS, HEAD_DIM, dtype=k.dtype, device=k.device)
    dense_v = torch.zeros_like(dense_k)
    offset = 0
    for row, length in enumerate(lengths):
        dense_k[row, :length] = k[offset:offset + length]
        dense_v[row, :length] = v[offset:offset + length]
        offset += length

    epv = 16 // k.element_size()
    if HEAD_DIM % epv or block_size % epv:
        raise Unavailable(f"vectorized-5D needs head_dim and block_size divisible by {epv}")
    num_blocks = rows * blocks_per_seq
    key_cache = (dense_k.view(num_blocks, block_size, NUM_KV_HEADS, HEAD_DIM // epv, epv)
                 .permute(0, 2, 3, 1, 4).contiguous())
    # [nb, H, D, block_size] is create_kv_cache's pre-shuffle value layout.
    value_linear = (dense_v.view(num_blocks, block_size, NUM_KV_HEADS, HEAD_DIM)
                    .permute(0, 2, 3, 1).contiguous())
    value_cache = (value_linear.view(num_blocks, NUM_KV_HEADS, HEAD_DIM, block_size // epv, epv)
                   .permute(0, 1, 3, 2, 4).contiguous())
    block_tables = torch.arange(num_blocks, dtype=torch.int32,
                                device=k.device).view(rows, blocks_per_seq)
    context_lengths = torch.tensor(lengths, dtype=torch.int32, device=k.device)
    del dense_k, dense_v, value_linear
    return dict(key_cache=key_cache, value_cache=value_cache,
                block_tables=block_tables, context_lengths=context_lengths)


# ── backends ───────────────────────────────────────────────────────────────


class Backend:
    name = ""
    # Substrings that identify this backend's kernels in a profiler trace.
    kernel_tags: tuple[str, ...] = ()

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def setup(self) -> dict[str, Any]:
        """Import and validate; raise Unavailable with a reason. Returns metadata."""
        raise NotImplementedError

    def build(self, torch: Any, case: dict[str, Any],
              inputs: dict[str, Any]) -> tuple[Callable[[], Any], dict[str, Any]]:
        """Return (launch, per-case metadata). ``launch()`` yields [rows, 24, 256]."""
        raise NotImplementedError

    def matches(self, kernel_name: str) -> bool:
        return all(tag in kernel_name for tag in self.kernel_tags)


class CKBackend(Backend):
    name = "ck"
    kernel_tags = ("ck_tile::kentry", "FmhaFwdKernel<")

    def setup(self) -> dict[str, Any]:
        import aiter

        if not Path(aiter.__file__).resolve().is_relative_to(self.args.aiter_path.resolve()):
            raise Unavailable(f"wrong AITER imported: {aiter.__file__}")
        self.attention = aiter.flash_attn_varlen_func
        return dict(module=aiter.__file__, entry="aiter.flash_attn_varlen_func")

    def build(self, torch, case, inputs):
        attention = self.attention
        q, k, v = inputs["q"], inputs["k"], inputs["v"]
        cuq, cuk = inputs["cu_seqlens_q"], inputs["cu_seqlens_k"]

        def launch():
            out = attention(q=q, k=k, v=v, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
                            max_seqlen_q=1, max_seqlen_k=KV_CAPACITY,
                            softmax_scale=HEAD_DIM ** -0.5, causal=True)
            return out[0] if isinstance(out, tuple) else out

        return launch, dict(max_seqlen_q=1, max_seqlen_k=KV_CAPACITY,
                            kv_capacity_shape=list(k.shape))


class FlydslBackend(Backend):
    """Shared sys.path wiring for the two FlyDSL candidates."""

    def _import_flydsl(self) -> dict[str, Any]:
        root = self.args.flydsl_path.resolve()
        if not (root / "kernels" / "attention").is_dir():
            raise Unavailable(f"no kernels/attention under --flydsl-path {root}")
        for entry in (str(root), str(root / "python")):
            if entry not in sys.path:
                sys.path.append(entry)
        try:
            import flydsl  # noqa: F401
        except ImportError as exc:
            raise Unavailable(f"flydsl package not importable: {exc}") from exc
        return dict(flydsl=flydsl.__file__, kernels_root=str(root))


class PaDecodeTileBackend(FlydslBackend):
    name = "pa_decode_tile"
    kernel_tags = ("pa_decode_tile",)

    def setup(self) -> dict[str, Any]:
        meta = self._import_flydsl()
        try:
            from kernels.attention.pa_decode_tile import compile_pa_decode_tile, pa_decode_tile
        except ImportError as exc:
            raise Unavailable(f"pa_decode_tile not importable: {exc}") from exc
        block_size = self.args.pa_block_size
        if block_size not in (16, 64):
            raise Unavailable(f"pa_decode_tile supports block_size 16 or 64, got {block_size}")
        # Surface the compile-time asserts (head_dim % 64, v_head_dim % 64,
        # v_head_dim % 64 for the 4-warp PV split, LDS budget) here rather than
        # mid-capture. D=256 is permitted by those asserts but is not covered by
        # FlyDSL's own test matrix, which stops at head_size=192.
        self.compile_pa_decode_tile = compile_pa_decode_tile
        self.block_size = block_size
        self._compile(num_partitions=1)
        self.pa_decode_tile = pa_decode_tile
        meta.update(entry="kernels.attention.pa_decode_tile.pa_decode_tile",
                    block_size=block_size)
        return meta

    def _compile(self, num_partitions: int) -> None:
        """``num_partitions`` is a compile-time constant, so each split count is
        its own build; probe the exact one before it runs under graph capture."""
        try:
            self.compile_pa_decode_tile(
                head_dim=HEAD_DIM, v_head_dim=HEAD_DIM,
                query_group_size=NUM_Q_HEADS // NUM_KV_HEADS,
                block_size=self.block_size, num_partitions=num_partitions,
                softmax_scale=HEAD_DIM ** -0.5, query_dtype="bf16",
                per_token_kv=False, query_length=1, trans_v=True, kv_dtype="bf16")
        except Exception as exc:  # noqa: BLE001 -- assert/ValueError/compiler error
            raise Unavailable(f"build at D={HEAD_DIM} np={num_partitions} failed: "
                              f"{type(exc).__name__}: {exc}") from exc

    def _partitions(self, rows: int) -> int:
        if self.args.pa_partitions:
            return self.args.pa_partitions
        from kernels.attention.pa_decode_fp8 import KV_COMPUTE_BLOCK, get_recommended_splits

        return get_recommended_splits(rows, NUM_KV_HEADS,
                                      split_kv_blocks=KV_COMPUTE_BLOCK // self.block_size)

    def build(self, torch, case, inputs):
        rows = case["rows"]
        blocks_per_seq = -(-KV_CAPACITY // self.block_size)
        paged = paged_from_packed(torch, inputs, self.block_size, blocks_per_seq)
        group = NUM_Q_HEADS // NUM_KV_HEADS
        partitions = self._partitions(rows)
        self._compile(partitions)
        out = torch.empty(rows, NUM_Q_HEADS, HEAD_DIM,
                          dtype=inputs["q"].dtype, device=inputs["q"].device)
        # query_length == 1 here: every verify row is its own sequence with its
        # own compacted KV, so TOTAL_ROWS is the GQA group alone.
        scalar_shape = (rows, NUM_KV_HEADS, partitions, 1 * group)
        pmax = torch.empty(*scalar_shape, dtype=torch.float32, device=out.device)
        psum = torch.empty_like(pmax)
        pout = torch.empty(*scalar_shape, HEAD_DIM, dtype=out.dtype, device=out.device)
        call = self.pa_decode_tile

        def launch():
            call(output=out, query=inputs["q"],
                 key_cache=paged["key_cache"], value_cache=paged["value_cache"],
                 block_tables=paged["block_tables"],
                 context_lengths=paged["context_lengths"],
                 key_scale=None, value_scale=None,
                 softmax_scale=HEAD_DIM ** -0.5,
                 num_partitions=partitions, pmax=pmax, psum=psum, pout=pout)
            return out

        meta = dict(block_size=self.block_size, blocks_per_seq=blocks_per_seq,
                    num_partitions=partitions, query_length=1,
                    query_group_size=group, total_rows=group,
                    m_tiles=-(-group // 16),
                    key_cache_shape=list(paged["key_cache"].shape),
                    value_cache_shape=list(paged["value_cache"].shape),
                    grid_ctas=rows * NUM_KV_HEADS * partitions)
        return launch, meta


class FlashAttnGfx950Backend(FlydslBackend):
    name = "fa_gfx950"
    # The dual-wave and generic builders both emit names carrying these.
    kernel_tags = ("flash_attn",)

    def setup(self) -> dict[str, Any]:
        meta = self._import_flydsl()
        try:
            from kernels.attention.flash_attn_interface import flydsl_flash_attn_func
        except ImportError as exc:
            raise Unavailable(f"flash_attn_interface not importable: {exc}") from exc
        self.attention = flydsl_flash_attn_func
        meta.update(entry="kernels.attention.flash_attn_interface.flydsl_flash_attn_func",
                    note=("flash_attn_gfx950 documents D=64/128 and seq_len>=384; at D=256 "
                          "this call routes to the generic BLOCK_M 128/256 builder or fails"))
        return meta

    def build(self, torch, case, inputs):
        attention = self.attention
        q, k, v = inputs["q"], inputs["k"], inputs["v"]
        cuq, cuk = inputs["cu_seqlens_q"], inputs["cu_seqlens_k"]

        def launch():
            out = attention(q, k, v, causal=True, num_kv_heads=NUM_KV_HEADS,
                            cu_seqlens_q=cuq, cu_seqlens_kv=cuk,
                            max_seqlen_q=1, max_seqlen_kv=KV_CAPACITY,
                            cross_seqlen=True)
            return out[0] if isinstance(out, tuple) else out

        # Build eagerly so an unsupported D=256 surfaces as this case's reason
        # instead of a failure inside graph capture.
        try:
            launch()
        except Exception as exc:  # noqa: BLE001
            raise Unavailable(f"{type(exc).__name__}: {exc}") from exc
        return launch, dict(max_seqlen_q=1, max_seqlen_kv=KV_CAPACITY, cross_seqlen=True)


BACKENDS: dict[str, type[Backend]] = {
    CKBackend.name: CKBackend,
    PaDecodeTileBackend.name: PaDecodeTileBackend,
    FlashAttnGfx950Backend.name: FlashAttnGfx950Backend,
}


# ── measurement ────────────────────────────────────────────────────────────


def time_launch(torch: Any, launch: Callable[[], Any], args: argparse.Namespace) -> dict[str, Any]:
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
    return dict(median_us=statistics.median(samples), min_us=min(samples),
                p95_us=sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))],
                samples_us=samples)


def profile_launch(torch: Any, backend: Backend, launch: Callable[[], Any],
                   case: dict[str, Any], expected_names: list[str],
                   destination: Path) -> dict[str, Any]:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        launch()
        torch.cuda.synchronize()
    path = destination / f"profile_{backend.name}_{case['mode']}_c{case['conc']}_ctx{case['context']}.json"
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if not kernels:
        raise RuntimeError(f"No GPU kernels recorded; inspect {path}")
    per_name: dict[str, float] = {}
    for event in kernels:
        per_name[event["name"]] = per_name.get(event["name"], 0.0) + event["dur"]
    matched = [e for e in kernels if backend.matches(e["name"])]
    result = dict(profile=str(path),
                  kernel_us_by_name=dict(sorted(per_name.items(), key=lambda kv: -kv[1])),
                  total_kernel_us=sum(per_name.values()),
                  matched_kernel_names=sorted({e["name"] for e in matched}),
                  matched_kernel_us=sum(e["dur"] for e in matched),
                  dispatch_confirmed=bool(matched))
    if backend.name == "ck":
        result["exact_trace_symbol_match"] = any(e["name"] in expected_names for e in matched)
        if not matched:
            raise RuntimeError(f"No CK FmhaFwdKernel found; inspect {path}")
    return result


def run_case(torch: Any, backends: list[Backend], case: dict[str, Any],
             args: argparse.Namespace, expected_names: list[str],
             destination: Path) -> list[dict[str, Any]]:
    inputs = make_inputs(torch, case)
    shared = dict(case, q_shape=[case["rows"], NUM_Q_HEADS, HEAD_DIM],
                  compact_kv_tokens=sum(case["lengths"]))
    results = []
    for backend in backends:
        record = dict(shared, backend=backend.name)
        launch = None
        try:
            launch, meta = backend.build(torch, case, inputs)
            record.update(meta)
            actual = launch()
            record["max_abs_error"] = reference_error(torch, inputs, actual, args.atol, args.rtol)
            record.update(time_launch(torch, launch, args))
            if args.profile:
                record.update(profile_launch(torch, backend, launch, case,
                                             expected_names, destination))
            record["status"] = "ok"
        except Unavailable as exc:
            record.update(status="unavailable", reason=str(exc))
        except Exception as exc:  # noqa: BLE001 -- one backend must not sink the sweep
            record.update(status="error", reason=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc())
            if args.strict:
                raise
        finally:
            del launch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        results.append(record)
    return results


def report(record: dict[str, Any]) -> str:
    head = (f"{record['backend']:>14s} {record['mode']:6s} C={record['conc']:2d} "
            f"rows={record['rows']:2d} context={record['context']} "
            f"KV={min(record['lengths'])}..{max(record['lengths'])}")
    if record["status"] != "ok":
        return f"{head} {record['status'].upper()}: {record['reason']}"
    tail = f" median={record['median_us']:.3f} us err={record['max_abs_error']:.2e}"
    if "dispatch_confirmed" in record:
        tail += f" dispatch={record['dispatch_confirmed']}"
    if "exact_trace_symbol_match" in record:
        tail += f" symbol_match={record['exact_trace_symbol_match']}"
    return head + tail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isl", type=positive, default=32678)
    parser.add_argument("--osl", type=positive, default=32678)
    parser.add_argument("--conc", default="1,4,8,16")
    parser.add_argument("--modes", default="decode,verify")
    parser.add_argument("--verify-tokens", type=positive, default=4)
    parser.add_argument("--backends", default=",".join(BACKENDS),
                        help=f"comma-separated subset of {','.join(BACKENDS)}")
    parser.add_argument("--aiter-path", type=Path, default=Path("/var/home/my_aiter/aiter"))
    parser.add_argument("--flydsl-path", type=Path, default=Path("/var/home/main_FlyDSL/FlyDSL"),
                        help="FlyDSL checkout providing kernels/attention and python/flydsl")
    parser.add_argument("--pa-block-size", type=positive, default=64,
                        help="pa_decode_tile paged block size (16 or 64)")
    parser.add_argument("--pa-partitions", type=non_negative, default=0,
                        help="pa_decode_tile KV split count; 0 uses get_recommended_splits")
    parser.add_argument("--trace-dir", type=Path, default=Path("/var/home/qwen_trace_analysis"))
    parser.add_argument("--output", type=Path, default=Path("qsa_ck_results"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--jit-dir", type=Path, default=Path(".qsa_aiter_jit"),
                        help="isolated AITER cache built against the active PyTorch")
    parser.add_argument("--warmup", type=positive, default=10)
    parser.add_argument("--inner", type=positive, default=20)
    parser.add_argument("--repeats", type=positive, default=30)
    parser.add_argument("--atol", type=float, default=0.003)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--profile", action="store_true", help="confirm dispatch and save per-case traces")
    parser.add_argument("--strict", action="store_true", help="abort instead of recording a backend error")
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
    selected = [name for name in args.backends.split(",") if name]
    unknown = [name for name in selected if name not in BACKENDS]
    if unknown:
        parser.error(f"unknown backend(s) {unknown}; choose from {list(BACKENDS)}")
    if not selected:
        parser.error("--backends must name at least one backend")
    evidence = trace_evidence(args.trace_dir)
    matrix = cases(args)
    if args.dry_run:
        print(json.dumps(dict(cases=matrix, backends=selected, trace=evidence), indent=2))
        return
    sys.path.insert(0, str(args.aiter_path.resolve()))
    os.environ["AITER_JIT_DIR"] = str(args.jit_dir.resolve())
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Requires ROCm PyTorch and an accessible AMD GPU")
    torch.cuda.set_device(args.device)
    torch.manual_seed(2026)
    args.output.mkdir(parents=True, exist_ok=True)

    active: list[Backend] = []
    backend_status: dict[str, Any] = {}
    for name in selected:
        backend = BACKENDS[name](args)
        try:
            backend_status[name] = dict(status="ok", **backend.setup())
            active.append(backend)
        except Unavailable as exc:
            backend_status[name] = dict(status="unavailable", reason=str(exc))
            print(f"{name:>14s} unavailable: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            backend_status[name] = dict(status="error", reason=f"{type(exc).__name__}: {exc}",
                                        traceback=traceback.format_exc())
            print(f"{name:>14s} setup error: {exc}", file=sys.stderr)
            if args.strict:
                raise
    if not active:
        raise RuntimeError(f"No backend is usable: {json.dumps(backend_status, indent=2)}")

    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    torch=torch.__version__, hip=torch.version.hip,
                    backends=backend_status,
                    gpu=str(torch.cuda.get_device_properties(args.device)), trace=evidence,
                    timing="GPU events around graph replay / inner; includes backend call kernels, excludes gather/indexer",
                    allocation="ck and fa_gfx950 allocate output inside the timed region; pa_decode_tile writes a preallocated buffer",
                    limitations="Synthetic compact KV; extrapolated concurrency; context samples, not full generation")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with torch.inference_mode(), (args.output / "results.jsonl").open("w") as stream:
        for case in matrix:
            for record in run_case(torch, active, case, args, evidence["kernel_names"], args.output):
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(report(record), flush=True)


if __name__ == "__main__":
    main()
