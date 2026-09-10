"""Reader/classifier for the PyTorch (Kineto) Chrome traces SGLang writes.

Used by analyze_traces.py and compare_traces.py. Nothing in here is specific to
one model -- it does assume the trace was produced by torch.profiler through
SGLang's --profile path, i.e.:

  * events live in a pretty-printed "traceEvents" array, one JSON object per
    block, object braces at indent level 2 (what libkineto's writer emits);
  * GPU work is cat=kernel/gpu_memcpy/gpu_memset with args.correlation;
  * the launching HIP/CUDA API call is cat=cuda_runtime with the same
    correlation, on the CPU thread that issued it;
  * SGLang wraps each forward pass in a user_annotation named
    "step[<FORWARD_MODE> bs=<n> toks=<n>]" on that same thread.

Those three facts are what let us say "this kernel cost X us and it ran during
prefill, under aten::mm". Traces are read as a stream: a single 32K-context
rank trace is several GB of JSON uncompressed, so we never json.load() the file.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import orjson as _json

    def _loads(s):
        return _json.loads(s)
except ImportError:  # pragma: no cover - orjson is present in the bench image
    import json as _json

    def _loads(s):
        return _json.loads(s)


# --------------------------------------------------------------------------
# event categories
# --------------------------------------------------------------------------

# Actual work on the device. These are what "kernel time" is measured over.
GPU_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})

# Everything we need to read to attribute that work to a phase and a host op.
KEEP_CATS = GPU_CATS | frozenset(
    {"cuda_runtime", "cpu_op", "user_annotation", "gpu_user_annotation"}
)

STEP_RE = re.compile(r"^step\[(?P<mode>[A-Za-z_]+)(?P<rest>.*)\]$")
BS_RE = re.compile(r"bs=(\d+)")
TOKS_RE = re.compile(r"toks=(\d+)")
RANK_RE = re.compile(r"TP-(\d+)")
TAG_RE = re.compile(r"isl(\d+)_osl(\d+)_c(\d+)")


# --------------------------------------------------------------------------
# streaming reader
# --------------------------------------------------------------------------


def _open_text(path: Path):
    """Return (text stream, subprocess or None). zcat beats gzip.open here
    because it moves decompression onto a second core."""
    if str(path).endswith(".gz"):
        proc = subprocess.Popen(
            ["zcat", str(path)], stdout=subprocess.PIPE, bufsize=1 << 20
        )
        return io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"), proc
    return open(path, "r", encoding="utf-8", errors="replace"), None


def _block_cat(head: str) -> str | None:
    """Pull the "cat" value out of an event block's first payload line without
    parsing it. Skipping the JSON parse matters: python_function events are
    ~95% of the blocks in an SGLang trace and we never look at them."""
    i = head.find('"cat": "')
    if i < 0:
        return None
    i += 8
    j = head.find('"', i)
    return head[i:j] if j > 0 else None


class TruncatedTrace(RuntimeError):
    """The trace file did not decompress cleanly -- profiling was almost
    certainly killed before the writer finished. Any timings read out of it
    cover only part of the run, so callers must opt in to using them."""


def iter_events(path: Path, keep_cats=KEEP_CATS, allow_truncated: bool = False):
    """Yield decoded event dicts whose cat is in keep_cats.

    Relies on the writer's layout: inside traceEvents, each object starts on a
    line beginning with "  {" and ends on a line beginning with "  }". JSON
    forbids raw newlines inside strings, and every nested line is indented 4 or
    more, so no other line can begin with "  }". A parse failure raises rather
    than silently dropping events.

    Raises TruncatedTrace when the decompressor reports a CRC/length error, so
    a half-written trace is never mistaken for a complete one.
    """
    fh, proc = _open_text(path)
    drained = False
    try:
        it = iter(fh)
        for line in it:
            if '"traceEvents"' in line:
                break
        else:
            raise ValueError(f"{path}: no traceEvents array found")

        buf: list[str] = []
        for line in it:
            if not buf:
                if line.startswith("  {"):
                    buf.append(line)
                elif line.startswith("  ]"):
                    break
                continue

            buf.append(line)
            if not line.startswith("  }"):
                continue

            head = buf[1] if len(buf) > 1 else buf[0]
            cat = _block_cat(head)
            if cat is None or cat not in keep_cats:
                buf = []
                continue

            blk = "".join(buf).rstrip().rstrip(",")
            buf = []
            try:
                yield _loads(blk)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{path}: could not parse event block: {exc}\n{blk[:400]}") from exc

        # read to EOF so the decompressor gets to verify its trailer; without
        # this a truncated .gz just looks like a short trace
        fh.read()
        drained = True
    finally:
        try:
            fh.close()
        except Exception:  # noqa: BLE001
            pass
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            rc = proc.wait()
            if drained and rc != 0 and not allow_truncated:
                raise TruncatedTrace(
                    f"{path}: decompression failed (zcat exit {rc}); the trace is "
                    f"truncated or corrupt. Re-collect it, or pass --allow-truncated "
                    f"to analyse the partial data."
                )


# --------------------------------------------------------------------------
# kernel name handling
# --------------------------------------------------------------------------


_DEMANGLERS = ("llvm-cxxfilt", "llvm-cxxfilt-18", "llvm-cxxfilt-17", "c++filt")


def _demangler() -> str | None:
    from shutil import which

    for tool in _DEMANGLERS:
        if which(tool):
            return tool
    return None


def demangle(names: list[str]) -> dict[str, str]:
    """Batch-demangle Itanium-mangled names.

    ROCm traces mix already-readable names ("void at::native::...") with raw
    mangled ones ("_ZN5aiter26cross_device_reduce_1stage..."); only the latter
    are sent through. Note that hipcc emits type codes the stock demanglers do
    not know (DF16b for __bf16, among others), so a good fraction of aiter/ck
    kernels come back untouched -- _mangled_prefix() in shorten() covers those.
    Returns only the names that actually changed.
    """
    mangled = sorted({n for n in names if n.startswith("_Z")})
    tool = _demangler()
    if not mangled or tool is None:
        return {}
    try:
        out = subprocess.run(
            [tool], input="\n".join(mangled), capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    lines = out.stdout.splitlines()
    if len(lines) != len(mangled):
        return {}
    return {m: d for m, d in zip(mangled, lines) if d and d != m}


_MANGLED_COMPONENT = re.compile(r"(\d+)")


def _mangled_prefix(name: str) -> str:
    """Recover "ns::ns::symbol" from an Itanium nested name the demanglers
    refused. _ZN5aiter26cross_device_reduce_1stageIDF16b... is a run of
    <length><identifier> pairs after _ZN; we read pairs until one stops
    matching, which is exactly where the template arguments begin.
    """
    i = 3 if name.startswith("_ZN") else 2
    parts: list[str] = []
    while i < len(name) and name[i].isdigit():
        j = i
        while j < len(name) and name[j].isdigit():
            j += 1
        n = int(name[i:j])
        if n <= 0 or j + n > len(name):
            break
        parts.append(name[j : j + n])
        i = j + n
    return "::".join(parts)


_SHORTEN_STRIP = re.compile(r"^(void|virtual|static|__global__)\s+")
_ANON_NS = re.compile(r"\(anonymous namespace\)::")


def shorten(name: str, max_len: int = 80) -> str:
    """Collapse a kernel signature down to something you can put in a table.

    Drops the leading return type and the "(anonymous namespace)::" noise, cuts
    at the first template/argument bracket, and keeps the trailing namespace
    components -- "at::native::vectorized_elementwise_kernel",
    "aiter::fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu". Names that carry no
    bracket (most hand-written HIP kernels) pass through unchanged.
    """
    s = name.strip()
    if s.startswith("_Z"):
        pref = _mangled_prefix(s)
        if pref:
            return pref if len(pref) <= max_len else pref[: max_len - 1] + "…"

    s = _ANON_NS.sub("", _SHORTEN_STRIP.sub("", s))

    cut = len(s)
    for ch in ("<", "("):
        i = s.find(ch)
        if 0 <= i < cut:
            cut = i
    s = s[:cut].strip()

    if not s:
        s = name.strip()[:max_len]

    # keep at most the last three "::" components so aiter/at::native kernels
    # stay distinguishable without dragging the whole namespace along
    parts = [p for p in s.split("::") if p]
    if len(parts) > 3:
        parts = parts[-3:]
    s = "::".join(parts)

    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


# --------------------------------------------------------------------------
# categorisation
# --------------------------------------------------------------------------

# Ordered most-specific first: the first pattern that matches wins, so MoE GEMMs
# are counted as MoE rather than as generic GEMM, and quantised MoE stays MoE.
_CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    ("comm", re.compile(
        r"nccl|rccl|cross_device_reduce|quickreduce|all_?reduce|all_?gather|"
        r"reduce_?scatter|all_?to_?all|broadcast|ncclDevKernel", re.I)),
    ("moe", re.compile(
        r"moe|fmoe|expert|topk_?gating|topkGatingSoftmax|routing", re.I)),
    ("attention", re.compile(
        r"attn|attention|flash|fmha|paged|_mla_|mla_|softmax_lse|"
        r"decode_attn|prefill_attn|rotary|_rope|apply_rope", re.I)),
    ("linear_attn", re.compile(
        r"gdn|delta_rule|causal_conv1d|recompute_w_u|chunk_state|chunk_scan|"
        r"l2norm|mamba|ssm|_gate_sigmoid", re.I)),
    # gemm(?!a) so Qwen's _gemma_* kernels are not read as GEMMs
    ("gemm", re.compile(
        r"Cijk_|gemm(?!a)|Tensile|rocblas|hipblas|wv_?splitk|"
        r"^mfma_|cutlass|_mm_|addmm|bmm", re.I)),
    ("quant", re.compile(
        r"quant|dequant|mxfp4|fp4|fp8|float8|scaled_|per_token_scale|"
        r"per_group_scale|smooth", re.I)),
    ("norm", re.compile(r"rmsnorm|rms_norm|layer_norm|layernorm|norm_fwd|_norm_kernel", re.I)),
    ("activation", re.compile(r"act_and_mul|silu|swiglu|gelu|geglu|sigmoid_mul|relu", re.I)),
    ("kv_cache", re.compile(
        r"reshape_and_cache|kv_cache|set_kv_buffer|copy_kv|cache_kernel|page_table|"
        r"slot_clear|_slot_", re.I)),
    ("sampling", re.compile(
        r"sampling|top_?k|top_?p|min_p|multinomial|argmax|penalt|logits_processor|"
        r"verify_tree|build_tree|eagle|draft_|speculative", re.I)),
    ("embedding", re.compile(r"embedding|index_select|gather_kernel|scatter_add", re.I)),
    ("index_copy", re.compile(
        r"index_elementwise|index_put|CatArrayBatchedCopy|copy_kernel|"
        r"FillFunctor|direct_copy|cat_kernel|unrolled_elementwise", re.I)),
    ("elementwise", re.compile(r"elementwise|reduce_kernel|Functor|_fused_|fused_", re.I)),
]


def classify(name: str, cat: str) -> str:
    """Map a kernel to a coarse bucket. cat comes from the trace event so that
    memcpy/memset are never mistaken for compute."""
    if cat == "gpu_memcpy":
        return "memcpy"
    if cat == "gpu_memset":
        return "memset"
    for label, pat in _CATEGORY_RULES:
        if pat.search(name):
            return label
    return "other"


# --------------------------------------------------------------------------
# interval attribution
# --------------------------------------------------------------------------


@dataclass
class _Span:
    start: float
    end: float
    name: str


def _enclosing(spans: list[_Span], points: list[float]) -> list[list[int]]:
    """For each point (sorted ascending) return the indices of the spans that
    contain it, outermost first.

    A linear sweep with an explicit stack -- spans nest (a cpu_op inside a
    user_annotation inside a step), so a plain interval lookup would only find
    one level. O(n log n) for the sort, O(n) for the sweep.
    """
    order = sorted(range(len(spans)), key=lambda i: (spans[i].start, -spans[i].end))
    out: list[list[int]] = []
    stack: list[int] = []
    si = 0
    for pt in points:
        while si < len(order) and spans[order[si]].start <= pt:
            stack.append(order[si])
            si += 1
        while stack and spans[stack[-1]].end < pt:
            stack.pop()
        # entries below the top may have already ended; drop them lazily
        live = [i for i in stack if spans[i].start <= pt <= spans[i].end]
        stack = live
        out.append(list(live))
    return out


def busy_time(starts: np.ndarray, durs: np.ndarray) -> float:
    """Union of [start, start+dur) intervals. Kernels on different streams
    overlap, so summing durations overstates how long the GPU was busy."""
    if len(starts) == 0:
        return 0.0
    order = np.argsort(starts, kind="stable")
    s = starts[order]
    e = s + durs[order]
    total = 0.0
    cur_s, cur_e = s[0], e[0]
    for i in range(1, len(s)):
        if s[i] > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s[i], e[i]
        elif e[i] > cur_e:
            cur_e = e[i]
    return float(total + (cur_e - cur_s))


# --------------------------------------------------------------------------
# per-trace parse
# --------------------------------------------------------------------------


@dataclass
class TraceMeta:
    path: str
    run: str
    rank: int
    isl: int | None = None
    osl: int | None = None
    conc: int | None = None
    n_gpu_ops: int = 0
    n_kernels: int = 0
    kernel_time_us: float = 0.0
    gpu_busy_us: float = 0.0
    wall_us: float = 0.0
    streams: int = 0
    truncated: bool = False
    phases: dict = field(default_factory=dict)


def trace_identity(path: Path) -> tuple[str, int, int | None, int | None, int | None]:
    """Derive (run label, tp rank, isl, osl, concurrency) from the file path.

    Traces land under <PROFILE_DIR>/isl32768_osl4_c8/<unix-ts>/<ts>-TP-<r>.trace.json.gz
    when produced by qwen3.5_fp4_sglang_profile.sh. Runs collected straight from
    qwen3.5_fp4_sglang_bench.sh --profile skip the tag directory, so we fall back
    to the timestamp directory as the label.
    """
    m = RANK_RE.search(path.name)
    rank = int(m.group(1)) if m else 0

    isl = osl = conc = None
    run = path.parent.name
    for parent in path.parents:
        m = TAG_RE.search(parent.name)
        if m:
            isl, osl, conc = (int(x) for x in m.groups())
            run = parent.name
            break
    return run, rank, isl, osl, conc


def parse_trace(
    path: Path, with_cpu_op: bool = True, allow_truncated: bool = False
) -> tuple[list[dict], TraceMeta]:
    """Read one rank trace and return (gpu op rows, meta).

    Each row is one GPU-side event with the phase and host op it belongs to.
    """
    path = Path(path)
    run, rank, isl, osl, conc = trace_identity(path)

    gpu_ops: list[dict] = []
    # correlation -> (launch ts, launch tid, api name)
    launches: dict[int, tuple[float, int, str]] = {}
    annots: dict[int, list[_Span]] = {}
    cpu_ops: dict[int, list[_Span]] = {}
    truncated = False

    def _events():
        # iter_events raises from its finally, i.e. after every readable event
        # has already been yielded -- so with allow_truncated we keep the
        # partial data and just flag it.
        nonlocal truncated
        try:
            yield from iter_events(path)
        except TruncatedTrace:
            if not allow_truncated:
                raise
            truncated = True

    for ev in _events():
        cat = ev.get("cat")
        if ev.get("ph") != "X":
            continue
        args = ev.get("args") or {}

        if cat in GPU_CATS:
            gpu_ops.append(
                {
                    "cat": cat,
                    "name": ev.get("name", ""),
                    "ts_us": float(ev.get("ts", 0.0)),
                    "dur_us": float(ev.get("dur", 0.0)),
                    "device": args.get("device", ev.get("pid")),
                    "stream": args.get("stream", ev.get("tid")),
                    "corr": args.get("correlation", -1),
                    "grid": "x".join(str(v) for v in args.get("grid", [])) or "",
                    "block": "x".join(str(v) for v in args.get("block", [])) or "",
                    "bytes": args.get("bytes", 0) or 0,
                }
            )
        elif cat == "cuda_runtime":
            corr = args.get("correlation")
            if corr is not None:
                launches[corr] = (
                    float(ev.get("ts", 0.0)),
                    ev.get("tid", 0),
                    ev.get("name", ""),
                )
        elif cat == "user_annotation":
            tid = ev.get("tid", 0)
            ts = float(ev.get("ts", 0.0))
            annots.setdefault(tid, []).append(
                _Span(ts, ts + float(ev.get("dur", 0.0)), ev.get("name", ""))
            )
        elif cat == "cpu_op" and with_cpu_op:
            tid = ev.get("tid", 0)
            ts = float(ev.get("ts", 0.0))
            cpu_ops.setdefault(tid, []).append(
                _Span(ts, ts + float(ev.get("dur", 0.0)), ev.get("name", ""))
            )

    # ---- attribute each launch site to a step / annotation / aten op --------
    # group launch correlations by the thread that issued them, then sweep
    by_tid: dict[int, list[tuple[float, int]]] = {}
    for corr, (ts, tid, _api) in launches.items():
        by_tid.setdefault(tid, []).append((ts, corr))

    corr_phase: dict[int, tuple[str, int, int]] = {}
    corr_region: dict[int, str] = {}
    corr_cpu_op: dict[int, str] = {}

    for tid, pts in by_tid.items():
        pts.sort()
        times = [p[0] for p in pts]

        spans = annots.get(tid, [])
        if spans:
            for (ts, corr), idxs in zip(pts, _enclosing(spans, times)):
                step_name = ""
                for i in idxs:  # outermost first
                    if spans[i].name.startswith("step["):
                        step_name = spans[i].name
                        break
                if step_name:
                    m = STEP_RE.match(step_name)
                    if m:
                        rest = m.group("rest")
                        bs = BS_RE.search(rest)
                        tk = TOKS_RE.search(rest)
                        corr_phase[corr] = (
                            m.group("mode"),
                            int(bs.group(1)) if bs else -1,
                            int(tk.group(1)) if tk else -1,
                        )
                elif idxs:
                    # No step[...] ancestor. SGLang's EAGLE/MTP path annotates
                    # "draft" / "draft_extend" *inside* scheduler.run_batch, so
                    # prefer the innermost non-scheduler annotation and only
                    # fall back to the outermost scheduler bookkeeping scope.
                    pick = spans[idxs[0]].name
                    for i in reversed(idxs):  # innermost first
                        if not spans[i].name.startswith("scheduler."):
                            pick = spans[i].name
                            break
                    corr_phase[corr] = (pick, -1, -1)
                if idxs:
                    corr_region[corr] = spans[idxs[-1]].name

        ops = cpu_ops.get(tid, [])
        if ops:
            for (ts, corr), idxs in zip(pts, _enclosing(ops, times)):
                if idxs:
                    corr_cpu_op[corr] = ops[idxs[-1]].name

    # ---- stitch onto the GPU rows -----------------------------------------
    name_cache: dict[str, tuple[str, str]] = {}
    all_names = {r["name"] for r in gpu_ops}
    dem = demangle(list(all_names))

    for row in gpu_ops:
        raw = row["name"]
        hit = name_cache.get(raw)
        if hit is None:
            full = dem.get(raw, raw)
            hit = (shorten(full), classify(full, row["cat"]))
            name_cache[raw] = hit
        row["kernel"], row["category"] = hit

        corr = row.pop("corr")
        ph = corr_phase.get(corr)
        row["phase"] = ph[0] if ph else "unattributed"
        row["batch_size"] = ph[1] if ph else -1
        row["step_tokens"] = ph[2] if ph else -1
        row["region"] = corr_region.get(corr, "")
        row["cpu_op"] = corr_cpu_op.get(corr, "")
        row["run"] = run
        row["rank"] = rank

    meta = TraceMeta(
        path=str(path), run=run, rank=rank, isl=isl, osl=osl, conc=conc,
        n_gpu_ops=len(gpu_ops), truncated=truncated,
    )
    if gpu_ops:
        ts = np.fromiter((r["ts_us"] for r in gpu_ops), dtype=float, count=len(gpu_ops))
        du = np.fromiter((r["dur_us"] for r in gpu_ops), dtype=float, count=len(gpu_ops))
        is_k = np.fromiter(
            (r["cat"] == "kernel" for r in gpu_ops), dtype=bool, count=len(gpu_ops)
        )
        meta.n_kernels = int(is_k.sum())
        meta.kernel_time_us = float(du[is_k].sum())
        meta.gpu_busy_us = busy_time(ts, du)
        meta.wall_us = float((ts + du).max() - ts.min())
        meta.streams = len({r["stream"] for r in gpu_ops})
        phases: dict[str, float] = {}
        for r in gpu_ops:
            if r["cat"] == "kernel":
                phases[r["phase"]] = phases.get(r["phase"], 0.0) + r["dur_us"]
        meta.phases = phases

    return gpu_ops, meta


def find_traces(roots: list[str]) -> list[Path]:
    """Expand roots (files or directories) into a sorted list of trace files."""
    out: list[Path] = []
    for root in roots:
        p = Path(root)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for pat in ("*.trace.json.gz", "*.trace.json", "*.pt.trace.json",
                        "*.pt.trace.json.gz"):
                out.extend(p.rglob(pat))
        else:
            print(f"warning: no such path {root}", file=sys.stderr)
    return sorted(set(out))


def run_sort_key(run: str):
    """Order runs by concurrency when the tag carries one, else by name."""
    m = TAG_RE.search(run)
    if m:
        return (0, int(m.group(3)), int(m.group(1)), run)
    return (1, 0, 0, run)


def fmt_us(us: float) -> str:
    if us >= 1_000_000:
        return f"{us / 1_000_000:.3f} s"
    if us >= 1000:
        return f"{us / 1000:.3f} ms"
    return f"{us:.1f} us"
