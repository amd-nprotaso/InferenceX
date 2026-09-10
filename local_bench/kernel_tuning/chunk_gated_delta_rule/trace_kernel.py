#!/usr/bin/env python3
"""Capture rocprofv3 kernel-dispatch traces of
chunk_gated_delta_rule_fwd_kernel_h_blockdim64 at the production model shapes.

Why this and not profile_traffic.py
-----------------------------------
`profile_traffic.py` wants hardware counters (`rocprofv3 --pmc`). Counter
collection is still denied in this container -- re-confirmed 2026-09-08:
`perf_event_paranoid=4`, `CapEff=0xa80425fb` (no CAP_PERFMON, no CAP_SYS_ADMIN),
and `rocprofv3 --pmc FETCH_SIZE WRITE_SIZE` on a bare matmul hangs until killed
and writes no CSV. `--kernel-trace` needs none of that and works fine, so this
script collects what is actually obtainable here:

    per-dispatch device duration, launch geometry (grid / workgroup size),
    and the compiled resource footprint (VGPR / AGPR / SGPR / LDS / scratch)

That last group is the part `sweep.py` cannot see. sweep.py times the call with
CUDA events, which includes launch overhead and tells you nothing about why a
config is fast. The trace gives the true device-side dispatch time and the
register/LDS/scratch numbers that explain the occupancy story -- in particular
whether a config that raises workgroup count is quietly spilling to scratch.

What it does NOT give: memory traffic. FETCH_SIZE/WRITE_SIZE stay out of reach
until the container is relaunched with --cap-add=CAP_PERFMON. See README.md.

Method
------
One rocprofv3 process per config, wrapping a worker that builds inputs through
the production preamble (`shapes.build_inputs`) and launches the kernel N times
with `restore_state()` between launches -- the same in-place-write hazard
`bench.py` handles, and it matters more here because a trace of a kernel eating
its own output is a trace of the wrong kernel.

The first dispatch of each run is dropped: it is the warmup, and it carries
cold-cache and first-touch effects that are not steady state.

Usage
-----
    ./trace_kernel.py                                # 3 configs, production shape
    ./trace_kernel.py --shape qwen35-397b-tp2
    ./trace_kernel.py --configs 32/4/2,16/4/3        # BV/warps/stages[/wpe]
    ./trace_kernel.py --launches 20 --pftrace        # also emit Perfetto traces
    ./trace_kernel.py --csv trace_summary.csv

Raw traces are kept under `traces/<shape>/<config>/` so they can be re-read or
loaded into Perfetto (`--pftrace`, then open the .pftrace at ui.perfetto.dev).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KERNEL_NAME = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
HERE = os.path.dirname(os.path.abspath(__file__))
META_TAG = "@@META@@"

# MI350X (CDNA4) LDS per CU. Used only to report how many workgroups the LDS
# request allows to be co-resident, which is the ceiling `fill%` cannot see.
LDS_PER_CU = 160 * 1024


# ---------------------------------------------------------------- worker side


def _worker(argv) -> int:
    """Launch the kernel a fixed number of times under the profiler."""
    p = argparse.ArgumentParser()
    p.add_argument("--_worker", required=True)
    p.add_argument("--shape", default="")
    p.add_argument("--launches", type=int, default=10)
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

    # Warmup. The profiler captures it too; the parent drops dispatch #1.
    inp.restore_state()
    _call(inp)
    torch.cuda.synchronize()

    for _ in range(a.launches):
        inp.restore_state()
        _call(inp)
        torch.cuda.synchronize()

    # rocprofv3's LDS_Block_Size column reads 0 for these dispatches even though
    # the kernel allocates 76-103 KB of LDS -- Triton requests it as a dynamic
    # group segment, which that column does not account for. Take LDS and the
    # register/spill counts from the compiled kernel instead, where they are
    # authoritative, and hand them to the parent on stdout.
    for _dev, entry in getattr(kern.fn, "device_caches", {}).items():
        for ck in entry[0].values():
            print(META_TAG + json.dumps({
                "shared": getattr(ck.metadata, "shared", None),
                "n_regs": getattr(ck, "n_regs", None),
                "n_spills": getattr(ck, "n_spills", None),
            }))
            break
    return 0


# ---------------------------------------------------------------- parent side


def _find(outdir: str, suffix: str) -> str:
    hits = glob.glob(os.path.join(outdir, "**", f"*{suffix}"), recursive=True)
    return hits[0] if hits else ""


def _parse_trace(path: str, kernel_name: str) -> dict:
    """Pull the target kernel's dispatches out of a kernel_trace.csv.

    Returns per-dispatch durations in ms plus the launch geometry and resource
    footprint, which are constant across dispatches of one compiled config.
    """
    durations, geom = [], {}
    others: dict = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Kernel_Name", "")
            try:
                ns = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if kernel_name not in name:
                # Everything else in the run -- the restore_state copy, the
                # preamble kernels. Kept so the summary can show what share of
                # the traced run is actually the kernel under test.
                short = name.split("(")[0][:60]
                o = others.setdefault(short, [0, 0.0])
                o[0] += 1
                o[1] += ns / 1e6
                continue
            durations.append(ns / 1e6)
            if not geom:
                wg = tuple(
                    int(row[f"Workgroup_Size_{d}"] or 1) for d in ("X", "Y", "Z")
                )
                # rocprofv3 reports Grid_Size in work-items, not workgroups.
                items = tuple(int(row[f"Grid_Size_{d}"] or 1) for d in ("X", "Y", "Z"))
                geom = {
                    "workgroup": wg,
                    "grid_items": items,
                    "workgroups": (
                        (items[0] // wg[0]) * (items[1] // wg[1]) * (items[2] // wg[2])
                    ),
                    "block_threads": wg[0] * wg[1] * wg[2],
                    "vgpr": int(row.get("VGPR_Count") or 0),
                    "agpr": int(row.get("Accum_VGPR_Count") or 0),
                    "sgpr": int(row.get("SGPR_Count") or 0),
                    "lds": int(row.get("LDS_Block_Size") or 0),
                    "scratch": int(row.get("Scratch_Size") or 0),
                }
    return {"durations": durations, "geom": geom, "others": others}


def _trace_one(cfg_spec: dict, outdir: str, a) -> dict:
    os.makedirs(outdir, exist_ok=True)
    fmts = ["csv", "pftrace"] if a.pftrace else ["csv"]
    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "--output-format", *fmts,
        "-d", outdir,
        "-o", "gdn",
        "--",
        sys.executable, "-W", "ignore", os.path.join(HERE, "trace_kernel.py"),
        "--_worker", json.dumps(cfg_spec),
        "--shape", a.shape,
        "--launches", str(a.launches),
        "--seed", str(a.seed),
    ]
    if a.pool_slots is not None:
        cmd += ["--pool-slots", str(a.pool_slots)]

    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(a.device))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=a.timeout, env=env, cwd=HERE
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout >{a.timeout}s"}

    path = _find(outdir, "kernel_trace.csv")
    if not path:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"error": f"rc={proc.returncode}: " + (tail[-1][:160] if tail else "no trace csv")}

    got = _parse_trace(path, KERNEL_NAME)
    if not got["durations"]:
        return {"error": f"no {KERNEL_NAME} dispatches in {os.path.basename(path)}"}
    got["csv"] = path
    got["pftrace"] = _find(outdir, ".pftrace")
    got["meta"] = {}
    for line in (proc.stdout or "").splitlines():
        if line.startswith(META_TAG):
            got["meta"] = json.loads(line[len(META_TAG):])
    return got


def _parse_configs(s: str) -> list:
    """`BV/warps/stages[/waves_per_eu]`, comma separated."""
    out = []
    for tok in s.replace(" ", "").split(","):
        if not tok:
            continue
        parts = tok.split("/")
        if len(parts) < 3:
            raise SystemExit(f"bad config {tok!r}: want BV/warps/stages[/wpe]")
        out.append({
            "BV": int(parts[0]),
            "num_warps": int(parts[1]),
            "num_stages": int(parts[2]),
            "waves_per_eu": int(parts[3]) if len(parts) > 3 else None,
            "matrix_instr_nonkdim": None,
        })
    return out


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
    p.add_argument(
        "--configs",
        default="32/4/2,16/4/3,16/4/3/4",
        help="BV/warps/stages[/waves_per_eu], comma separated. Default is the "
        "production baseline, the env-only winner, and the wpe=4 winner.",
    )
    p.add_argument("--launches", type=int, default=10)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-slots", type=int, default=None)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--pftrace", action="store_true",
                   help="Also emit Perfetto .pftrace alongside the CSV")
    p.add_argument("--outdir", default=os.path.join(HERE, "traces"))
    p.add_argument("--csv", default="", help="Write the summary table here")
    a = p.parse_args(argv)

    shape = SHAPES[a.shape]
    specs = _parse_configs(a.configs)

    print(f"shape       : {shape.name}  B={shape.B} T={shape.T} Hg={shape.Hg} "
          f"H={shape.H} K={shape.K} V={shape.V} NT={shape.NT}")
    print(f"kernel      : {KERNEL_NAME}")
    print(f"mode        : rocprofv3 --kernel-trace  (counters unavailable: "
          f"no CAP_PERFMON, see README)")
    print(f"launches    : {a.launches} per config, first dispatch dropped as warmup")
    print(f"traces      : {a.outdir}/{shape.name}/\n")

    rows = []
    for spec in specs:
        cfg = Config(**spec)
        label = cfg.label()
        slug = label.replace(" ", "_").replace("=", "")
        outdir = os.path.join(a.outdir, shape.name, slug)
        print(f"  tracing {label} ...", flush=True)
        got = _trace_one(spec, outdir, a)
        if "error" in got:
            print(f"    FAILED: {got['error']}")
            rows.append({"label": label, "error": got["error"]})
            continue

        # Dispatches come out of the CSV in dispatch order, so the warmup is
        # durations[0] -- not the fastest one, which is what sorting first would
        # have thrown away.
        d = got["durations"][1:] or got["durations"]
        g = got["geom"]
        rows.append({
            "label": label, "cfg": cfg, "n": len(d),
            "min": min(d), "med": statistics.median(d), "max": max(d),
            "mean": statistics.fmean(d),
            "std": statistics.stdev(d) if len(d) > 1 else 0.0,
            "total": sum(got["durations"]),
            "others": got["others"], "csv": got["csv"],
            "pftrace": got.get("pftrace", ""),
            "meta": got.get("meta", {}),
            **g,
        })
        print(f"    {len(d)} dispatches, median {statistics.median(d):.3f} ms")

    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("\nEvery config failed. Check the sglang import and rocprofv3.")
        return 1

    cus = 256
    base = ok[0]
    print(f"\n{'='*112}")
    print(f"{'config':<22}{'n':>4}{'median ms':>11}{'min':>9}{'max':>9}{'±':>8}"
          f"{'speedup':>9}{'WGs':>6}{'blk':>6}{'fill%':>7}")
    print("-" * 112)
    for r in ok:
        print(f"{r['label']:<22}{r['n']:>4}{r['med']:>11.3f}{r['min']:>9.3f}"
              f"{r['max']:>9.3f}{r['std']:>8.3f}{base['med']/r['med']:>8.2f}x"
              f"{r['workgroups']:>6}{r['block_threads']:>6}"
              f"{100.0*min(r['workgroups'], cus)/cus:>7.1f}")
    print("-" * 112)
    print("Device-side dispatch duration from the trace -- no launch overhead, "
          "unlike sweep.py's CUDA-event timing.")

    print(f"\n{'config':<22}{'VGPR':>7}{'SGPR':>7}{'regs':>7}{'spill':>7}"
          f"{'LDS KB':>9}{'WG/CU':>7}{'scratch B':>11}{'grid (items)':>22}")
    print("-" * 112)
    for r in ok:
        gi = "x".join(str(x) for x in r["grid_items"])
        wg = "x".join(str(x) for x in r["workgroup"])
        m = r["meta"]
        lds = m.get("shared") or 0
        wg_per_cu = LDS_PER_CU // lds if lds else 0
        print(f"{r['label']:<22}{r['vgpr']:>7}{r['sgpr']:>7}"
              f"{str(m.get('n_regs', '-')):>7}{str(m.get('n_spills', '-')):>7}"
              f"{lds/1024:>9.1f}{wg_per_cu:>7}{r['scratch']:>11}"
              f"{gi + '  wg ' + wg:>22}")
    print("-" * 112)
    print("LDS and regs/spill come from the compiled Triton kernel, not the "
          "trace: rocprofv3's LDS_Block_Size column reads 0 for these dispatches "
          "because Triton requests LDS as a dynamic group segment.")
    print(f"WG/CU is the LDS-imposed ceiling on co-resident workgroups "
          f"({LDS_PER_CU//1024} KB LDS per CU) -- an upper bound fill% cannot see.")

    spilled = [r for r in ok
               if r["scratch"] > 0 or (r["meta"].get("n_spills") or 0) > 0]
    if spilled:
        print("Register spilling detected -- these configs pay for occupancy in "
              "memory traffic:")
        for r in spilled:
            print(f"  {r['label']:<22}scratch={r['scratch']} B  "
                  f"spills={r['meta'].get('n_spills')}")
    else:
        print("Zero scratch and zero spills everywhere: the occupancy "
              "differences are geometry, not register pressure.")

    print(f"\nother kernels in the {base['label']} run (restore_state + preamble):")
    for name, (n, ms) in sorted(
        base["others"].items(), key=lambda kv: -kv[1][1]
    )[:5]:
        print(f"  {n:>5}x {ms:8.3f} ms total   {name}")

    print("\nraw traces:")
    for r in ok:
        print(f"  {r['label']:<22}{r['csv']}")
        if r["pftrace"]:
            print(f"  {'':<22}{r['pftrace']}  (open at ui.perfetto.dev)")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["shape", "config", "dispatches", "median_ms", "min_ms",
                        "max_ms", "std_ms", "workgroups", "block_threads",
                        "vgpr", "agpr", "sgpr", "lds_bytes", "n_regs",
                        "n_spills", "scratch_bytes", "trace_csv"])
            for r in ok:
                m = r["meta"]
                w.writerow([shape.name, r["label"], r["n"], f"{r['med']:.4f}",
                            f"{r['min']:.4f}", f"{r['max']:.4f}",
                            f"{r['std']:.4f}", r["workgroups"],
                            r["block_threads"], r["vgpr"], r["agpr"], r["sgpr"],
                            m.get("shared"), m.get("n_regs"), m.get("n_spills"),
                            r["scratch"], r["csv"]])
        print(f"\ncsv         : {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
