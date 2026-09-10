#!/usr/bin/env python3
"""Advanced Thread Trace (ATT) of chunk_gated_delta_rule_fwd_kernel_h_blockdim64.

    rocprofv3 --att=true --att-library-path /opt/rocm/lib

Why a second script instead of a flag on trace_kernel.py
--------------------------------------------------------
ATT serialises dispatches and instruments the instruction stream, so the wall
time of a traced dispatch is meaningless. `trace_kernel.py` exists to measure
time; this exists to explain it. Mixing them in one run would invite reading an
ATT-inflated duration as a real one.

What it answers
---------------
The README's central claim is that the kernel is *per-workgroup latency-bound*:
every workgroup delivers ~20 GB/s regardless of shape, and aggregate throughput
is just that times workgroup count. That was inferred from scaling behaviour
across TP2/TP4/4k. It was never shown directly, because it needs to know what
the workgroup is waiting on, and hardware counters are denied in this container
(`--pmc` hangs: no CAP_PERFMON -- see README).

ATT needs none of that. It samples the instruction stream on one CU and reports,
per instruction address, how many times it issued (`Hitcount`), how long it took
(`Latency`), and how long the wave sat blocked on it (`Stall`). Bucketing that
by instruction class turns "latency-bound" into a specific answer: MFMA issue,
LDS round-trips, VMEM waits, or s_waitcnt on global loads.

Scope of the sample
-------------------
ATT instruments **one CU on one shader engine** (`--att-target-cu`,
`--att-shader-engine-mask`), not the whole device. That is the right sample for
this kernel precisely because the README's finding is that every workgroup
behaves identically -- if per-workgroup behaviour is uniform, one CU is
representative. It is the wrong tool for anything about device fill, which is
what `trace_kernel.py` covers.

Two traps, both of which produce output that looks fine
-------------------------------------------------------
**1. `--att-target-cu` defaults to 1, and CU 1 is wrong here.** At CU 1 the
decoder resolves the sampled PCs into a *different code object* -- torch's, not
Triton's -- and emits a complete, well-formed, entirely false profile: 927
instructions, no MFMA, no LDS, and 99.9% of stall on a single `s_waitcnt
vmcnt(0)`. It is not obviously garbage; it reads like a memory-bound kernel.
CU 0 decodes into code object 19, the one holding
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, and yields 1837 instructions
with 32 MFMA. This script defaults to CU 0 and refuses any decode containing no
MFMA, because this kernel provably has some.

**2. ATT arms all four agents unless told otherwise.** The app runs on one of
them and the decoder emits its stats CSV for whichever agent it enumerated last,
which is an empty trace. The tell is three 424-byte `.att` files sitting next to
one real one. `--att-gpu-index` pins it.

`--att-serialize-all` aborts the profiler on this workload; do not enable it.

The first matching dispatches are the ones traced, so the worker runs a warmup
before them and `--kernels` traces several consecutive dispatches; the summary
reports the last, which is past first-touch.

Usage
-----
    ./trace_att.py                                  # baseline + winner, TP4
    ./trace_att.py --configs 16/4/3/4 --kernels 2
    ./trace_att.py --shape qwen35-397b-tp4-4k       # far smaller trace
    ./trace_att.py --target-cu 0 --se-mask 0x1
    ./trace_att.py --keep-raw                       # keep the .out disassembly

Raw ATT output goes to `traces_att/<shape>/<config>/`. The per-code-object
`.out` disassembly dumps are deleted after parsing unless `--keep-raw`: they are
regenerable and the ones belonging to torch/hipBLASLt run to hundreds of MB.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KERNEL_NAME = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
HERE = os.path.dirname(os.path.abspath(__file__))
ATT_LIBRARY_PATH = "/opt/rocm/lib"

# Instruction classes, in match order -- first hit wins, so the specific
# patterns have to precede the generic `v_` / `s_` catch-alls.
#
# The split that matters for this kernel is MFMA (the tl.dot lowering) against
# the three ways a wave can be made to wait: LDS, VMEM, and the s_waitcnt that
# actually blocks on either. s_waitcnt is broken out separately because on CDNA
# the stall is attributed to the wait, not to the load that caused it -- so
# `vmem` stall being near zero while `wait` stall is large is the normal
# signature of a memory-bound kernel, not evidence that memory is free.
CLASSES = [
    ("mfma",   re.compile(r"^v_mfma")),
    ("lds",    re.compile(r"^ds_")),
    ("vmem_ld", re.compile(r"^(buffer_load|global_load|flat_load|scratch_load)")),
    ("vmem_st", re.compile(r"^(buffer_store|global_store|flat_store|scratch_store)")),
    ("smem",   re.compile(r"^s_(load|buffer_load)")),
    ("wait",   re.compile(r"^s_(waitcnt|barrier|sleep|setprio)")),
    ("branch", re.compile(r"^s_(branch|cbranch|endpgm|call|setpc|getpc)")),
    ("valu",   re.compile(r"^v_")),
    ("salu",   re.compile(r"^s_")),
]


# ---------------------------------------------------------------- worker side


def _worker(argv) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--_worker", required=True)
    p.add_argument("--shape", default="")
    p.add_argument("--launches", type=int, default=3)
    p.add_argument("--pool-slots", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    import warnings

    warnings.filterwarnings("ignore", message=".*make_block_ptr is deprecated.*")

    import torch

    from bench import Config, _call, _kernel
    from shapes import SHAPES, build_inputs

    cfg = Config(**json.loads(a._worker))
    inp = build_inputs(
        SHAPES[a.shape], device="cuda", seed=a.seed, pool_slots=a.pool_slots
    )

    kern = _kernel()
    kern.configs = [cfg.to_triton()]
    kern.cache.clear()

    for _ in range(a.launches):
        inp.restore_state()
        _call(inp)
        torch.cuda.synchronize()
    return 0


# ---------------------------------------------------------------- parent side


def _classify(instr: str) -> str:
    op = instr.strip().split()[0] if instr.strip() else ""
    for name, pat in CLASSES:
        if pat.match(op):
            return name
    return "other"


def _target_codeobj(outdir: str) -> int:
    """Which code object holds our kernel.

    rocprofv3 dumps every loaded code object as a raw ELF next to the trace.
    The Triton kernel lives in exactly one of them; torch and hipBLASLt own the
    rest. Finding it by symbol name is the only reliable way to filter the
    stats, because a single traced window routinely contains samples from more
    than one code object (see `_parse_stats`).
    """
    needle = KERNEL_NAME.encode()
    for p in glob.glob(os.path.join(outdir, "*code_object_id_*.out")):
        try:
            with open(p, "rb") as fh:
                if needle in fh.read():
                    m = re.search(r"code_object_id_(\d+)\.out$", p)
                    if m:
                        return int(m.group(1))
        except OSError:
            continue
    return -1


def _parse_stats(path: str, codeobj: int = -1) -> dict:
    """Bucket one dispatch's per-instruction ATT stats by instruction class.

    `codeobj` filters to a single code object. This is not optional in
    practice: the traced window regularly straddles the dispatch boundary and
    picks up the *preceding* kernel, so a raw aggregate mixes our kernel's
    instructions with torch's. Passing -1 disables the filter.
    """
    buckets: dict = {}
    top: list = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if codeobj >= 0:
                try:
                    if int(row.get("CodeObj") or -1) != codeobj:
                        continue
                except ValueError:
                    continue
            instr = row.get("Instruction", "")
            try:
                hit = int(float(row.get("Hitcount") or 0))
                lat = int(float(row.get("Latency") or 0))
                stall = int(float(row.get("Stall") or 0))
                idle = int(float(row.get("Idle") or 0))
            except ValueError:
                continue
            cls = _classify(instr)
            b = buckets.setdefault(
                cls, {"hit": 0, "lat": 0, "stall": 0, "idle": 0, "n": 0}
            )
            b["hit"] += hit
            b["lat"] += lat
            b["stall"] += stall
            b["idle"] += idle
            b["n"] += 1
            top.append((stall, lat, hit, instr.strip()[:56], cls))
    top.sort(reverse=True)
    return {"buckets": buckets, "top": top[:12]}


def _run_att(cfg_spec: dict, outdir: str, a) -> dict:
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        "rocprofv3",
        "--att=true",
        "--att-library-path", ATT_LIBRARY_PATH,
        "--kernel-include-regex", KERNEL_NAME,
        "--att-consecutive-kernels", str(a.kernels),
        "--att-target-cu", str(a.target_cu),
        "--att-shader-engine-mask", a.se_mask,
        "--att-simd-select", a.simd_select,
        "--att-buffer-size", str(a.buffer_size),
        # Without this, ATT arms all four agents, the app runs on one of them,
        # and the decoder emits its stats CSV for whichever agent it enumerated
        # last -- which is an empty trace. The three 424-byte .att files next to
        # one real one are the tell.
        "--att-gpu-index", str(a.gpu_index),
        "-d", outdir,
        "-o", "att",
        "--",
        sys.executable, "-W", "ignore", os.path.join(HERE, "trace_att.py"),
        "--_worker", json.dumps(cfg_spec),
        "--shape", a.shape,
        "--launches", str(a.launches),
        "--seed", str(a.seed),
    ]
    if a.activity:
        cmd += ["--att-activity", str(a.activity)]
    if a.pool_slots is not None:
        cmd += ["--pool-slots", str(a.pool_slots)]

    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(a.device))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=a.timeout, env=env, cwd=HERE
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout >{a.timeout}s", "cmd": cmd}

    stats = sorted(glob.glob(os.path.join(outdir, "**", "stats_ui_output_*.csv"),
                             recursive=True))
    if not stats:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"error": f"rc={proc.returncode}: " +
                         (tail[-1][:200] if tail else "no ATT stats csv"),
                "cmd": cmd}

    # Traced dispatches come out numbered; the last is the one furthest past
    # first-touch, so it is the one to report.
    def _disp(p):
        m = re.search(r"dispatch_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    stats.sort(key=_disp)
    chosen = stats[-1]
    codeobj = _target_codeobj(outdir)
    if codeobj < 0:
        return {"error": "no dumped code object contains the kernel symbol",
                "cmd": cmd}
    got = _parse_stats(chosen, codeobj)
    got["stats_csv"] = chosen
    got["codeobj"] = codeobj

    # Even filtered, a window can miss our kernel entirely -- the trace buffer
    # sometimes fills on the preceding dispatch. This kernel provably contains
    # MFMA (it is a tl.dot chain), so zero MFMA after filtering means the sample
    # holds nothing of ours and the run has to be repeated.
    if not got["buckets"].get("mfma", {}).get("n"):
        return {"error": f"no MFMA in code object {codeobj} after filtering -- "
                         "the traced window caught the preceding dispatch",
                "cmd": cmd, "retryable": True}
    got["n_dispatches"] = len(stats)
    got["att_files"] = sorted(glob.glob(os.path.join(outdir, "*.att")))
    got["ui_dirs"] = sorted(
        d for d in glob.glob(os.path.join(outdir, "ui_output_*")) if os.path.isdir(d)
    )

    if not a.keep_raw:
        # The per-code-object disassembly dumps include torch's and hipBLASLt's
        # and run to hundreds of MB each. Regenerable, so drop them.
        freed = 0
        for p in glob.glob(os.path.join(outdir, "*code_object*.out")):
            freed += os.path.getsize(p)
            os.remove(p)
        got["freed"] = freed
    return got


def _parse_configs(s: str) -> list:
    out = []
    for tok in s.replace(" ", "").split(","):
        if not tok:
            continue
        parts = tok.split("/")
        if len(parts) < 3:
            raise SystemExit(f"bad config {tok!r}: want BV/warps/stages[/wpe]")
        out.append({
            "BV": int(parts[0]), "num_warps": int(parts[1]),
            "num_stages": int(parts[2]),
            "waves_per_eu": int(parts[3]) if len(parts) > 3 else None,
            "matrix_instr_nonkdim": None,
        })
    return out


ORDER = ["mfma", "vmem_ld", "vmem_st", "lds", "smem", "wait", "valu", "salu",
         "branch", "other"]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--_worker":
        return _worker(argv)

    from bench import Config
    from shapes import DEFAULT_SHAPE, SHAPES

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--shape", default=DEFAULT_SHAPE, choices=sorted(SHAPES))
    p.add_argument("--configs", default="32/4/2,16/4/3",
                   help="BV/warps/stages[/waves_per_eu], comma separated")
    p.add_argument("--launches", type=int, default=3,
                   help="Kernel launches in the worker. ATT traces the first "
                        "--kernels of them.")
    p.add_argument("--kernels", type=int, default=2,
                   help="--att-consecutive-kernels; the last traced dispatch is "
                        "the one summarised")
    # rocprofv3 defaults --att-target-cu to 1. Do NOT inherit that default: on
    # this kernel CU 1 decodes into the wrong code object and yields a plausible
    # but entirely false profile (see the module docstring). CU 0 is correct.
    p.add_argument("--target-cu", type=int, default=0)
    p.add_argument("--se-mask", default="0x1", help="--att-shader-engine-mask")
    p.add_argument("--simd-select", default="0xF", help="--att-simd-select")
    p.add_argument("--buffer-size", type=int, default=256 * 1024 * 1024)
    p.add_argument("--activity", type=int, default=0,
                   help="--att-activity period (gfx9), 1-16; 0 disables")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--gpu-index", type=int, default=0, help="--att-gpu-index")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-slots", type=int, default=None)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--retries", type=int, default=4,
                   help="Re-runs when the traced window misses the dispatch")
    p.add_argument("--keep-raw", action="store_true",
                   help="Keep the per-code-object .out disassembly dumps")
    p.add_argument("--outdir", default=os.path.join(HERE, "traces_att"))
    p.add_argument("--csv", default="")
    a = p.parse_args(argv)

    shape = SHAPES[a.shape]
    specs = _parse_configs(a.configs)

    print(f"shape       : {shape.name}  B={shape.B} T={shape.T} H={shape.H} "
          f"K={shape.K} V={shape.V} NT={shape.NT}")
    print(f"kernel      : {KERNEL_NAME}")
    print(f"mode        : rocprofv3 --att=true --att-library-path {ATT_LIBRARY_PATH}")
    print(f"sample      : CU {a.target_cu}, SE mask {a.se_mask}, SIMD "
          f"{a.simd_select} -- one CU, not the device")
    print(f"dispatches  : {a.launches} launched, first {a.kernels} traced, "
          f"last one summarised")
    print(f"out         : {a.outdir}/{shape.name}/\n")

    rows = []
    for spec in specs:
        cfg = Config(**spec)
        label = cfg.label()
        slug = label.replace(" ", "_").replace("=", "")
        outdir = os.path.join(a.outdir, shape.name, slug)
        print(f"  tracing {label} ...", flush=True)
        # Whether the traced window lands on our dispatch or on the one before
        # it is not deterministic -- roughly two runs in three land correctly.
        # Retry rather than report a hole.
        for attempt in range(1, a.retries + 1):
            got = _run_att(spec, outdir, a)
            if "error" not in got or not got.get("retryable"):
                break
            print(f"    attempt {attempt}/{a.retries}: {got['error']}", flush=True)
        if "error" in got:
            print(f"    FAILED: {got['error']}")
            rows.append({"label": label, "error": got["error"]})
            continue
        tot = sum(b["stall"] for b in got["buckets"].values())
        print(f"    {got['n_dispatches']} dispatch(es) traced, "
              f"{sum(b['n'] for b in got['buckets'].values())} distinct "
              f"instructions, {tot} total stall cycles")
        rows.append({"label": label, **got})

    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("\nEvery config failed.")
        if rows:
            print("last cmdline:\n  " + " ".join(rows[-1].get("cmd", [])))
        return 1

    for r in ok:
        b = r["buckets"]
        tot_stall = sum(x["stall"] for x in b.values()) or 1
        tot_lat = sum(x["lat"] for x in b.values()) or 1
        tot_hit = sum(x["hit"] for x in b.values()) or 1
        print(f"\n{'='*96}")
        print(f"{r['label']}   (code object {r['codeobj']}, "
              f"{os.path.basename(r['stats_csv'])})")
        print(f"{'class':<10}{'instrs':>8}{'hits':>12}{'hit%':>7}"
              f"{'latency':>12}{'lat%':>7}{'stall':>12}{'stall%':>8}")
        print("-" * 96)
        for cls in ORDER:
            if cls not in b:
                continue
            x = b[cls]
            print(f"{cls:<10}{x['n']:>8}{x['hit']:>12}"
                  f"{100*x['hit']/tot_hit:>7.1f}{x['lat']:>12}"
                  f"{100*x['lat']/tot_lat:>7.1f}{x['stall']:>12}"
                  f"{100*x['stall']/tot_stall:>8.1f}")
        print("-" * 96)
        print(f"{'total':<10}{sum(x['n'] for x in b.values()):>8}"
              f"{tot_hit:>12}{100.0:>7.1f}{tot_lat:>12}{100.0:>7.1f}"
              f"{tot_stall:>12}{100.0:>8.1f}")

        print(f"\n  top stall sites:")
        print(f"  {'stall':>10}{'latency':>10}{'hits':>9}  {'class':<8}instruction")
        for stall, lat, hit, instr, cls in r["top"][:8]:
            print(f"  {stall:>10}{lat:>10}{hit:>9}  {cls:<8}{instr}")

    if len(ok) >= 2:
        print(f"\n{'='*96}")
        print(f"{'class':<10}" + "".join(f"{r['label'][:20]:>22}" for r in ok))
        print(f"{'':10}" + "".join(f"{'stall%':>22}" for r in ok))
        print("-" * 96)
        for cls in ORDER:
            if not any(cls in r["buckets"] for r in ok):
                continue
            line = f"{cls:<10}"
            for r in ok:
                t = sum(x["stall"] for x in r["buckets"].values()) or 1
                s = r["buckets"].get(cls, {}).get("stall", 0)
                line += f"{100*s/t:>22.1f}"
            print(line)
        print("-" * 96)

    print("\nreading this table:")
    print("  `wait` is s_waitcnt/s_barrier. On CDNA the stall from a memory op is")
    print("  charged to the wait that blocks on it, not to the load itself, so a")
    print("  large `wait` share with near-zero `vmem_ld` share is the signature of")
    print("  memory latency, not of free memory.")
    print("  `mfma` stall is issue-limited compute. If it dominates, the tile is")
    print("  the constraint and more workgroups will not help.")
    print("  Compare stall SHARES across configs, never stall TOTALS: the totals")
    print("  sum over whatever waves the traced CU happened to hold, and that")
    print("  count changes with the config (BV=32 puts 2 workgroups x 4 waves on")
    print("  a CU, BV=8 puts 1 x 2), so a smaller total can mean fewer waves")
    print("  rather than less waiting.")

    freed = sum(r.get("freed", 0) for r in ok)
    if freed:
        print(f"\ndeleted {freed/1e9:.1f} GB of regenerable .out disassembly "
              f"(--keep-raw to retain)")
    print("\nraw ATT output:")
    for r in ok:
        print(f"  {r['label']:<22}{r['stats_csv']}")
        for d in r["ui_dirs"]:
            print(f"  {'':<22}{d}/  (rocprof-compute viewer input)")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["shape", "config", "class", "instructions", "hits",
                        "latency", "stall", "idle", "stall_pct"])
            for r in ok:
                b = r["buckets"]
                t = sum(x["stall"] for x in b.values()) or 1
                for cls in ORDER:
                    if cls not in b:
                        continue
                    x = b[cls]
                    w.writerow([shape.name, r["label"], cls, x["n"], x["hit"],
                                x["lat"], x["stall"], x["idle"],
                                f"{100*x['stall']/t:.2f}"])
        print(f"\ncsv         : {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
