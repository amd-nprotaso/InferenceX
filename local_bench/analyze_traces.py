#!/usr/bin/env python3
"""Extract kernel calls and their time from SGLang torch-profiler traces.

Companion to qwen3.5_fp4_sglang_profile.sh: point it at the PROFILE_DIR that
script wrote and it turns every <run>/<ts>/<ts>-TP-<rank>.trace.json.gz into
per-kernel timing tables.

  ./analyze_traces.py /var/home/sglang_profiling -o trace_analysis
  ./analyze_traces.py /var/home/sglang_profiling/isl32768_osl4_c8 --ranks 0
  ./analyze_traces.py <dir> --top 40 --phase EXTEND

Outputs (under -o):
  cache/<run>__TP<r>.parquet  every GPU op, one row each (also the parse cache)
  traces.csv                  one row per rank trace: kernel count, busy time
  kernels.csv                 per run x rank x phase x kernel timing
  kernels_by_run.csv          same, averaged across TP ranks
  categories.csv              per run x phase x category rollup
  phases.csv                  per run x phase rollup
  report.txt                  the human-readable version of the above

"kernel time" is the sum of kernel durations. "gpu busy" is the union of their
intervals -- with several streams in flight the two differ, and busy time is the
one to compare against wall clock.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sglang_trace_lib import (  # noqa: E402
    RANK_RE,
    TruncatedTrace,
    find_traces,
    fmt_us,
    parse_trace,
    run_sort_key,
)

EVENT_COLS = [
    "run", "rank", "cat", "category", "kernel", "phase", "batch_size",
    "step_tokens", "region", "cpu_op", "device", "stream", "ts_us", "dur_us",
    "grid", "block", "bytes", "name",
]


def _cache_path(outdir: Path, path: Path) -> Path:
    from sglang_trace_lib import trace_identity

    run, rank, *_ = trace_identity(path)
    return outdir / "cache" / f"{run}__TP{rank}.parquet"


def _worker(args) -> dict:
    path, outdir, force, with_cpu_op, allow_truncated = args
    path, outdir = Path(path), Path(outdir)
    cache = _cache_path(outdir, path)
    try:
        if not force and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            df = pd.read_parquet(cache, columns=["cat", "dur_us", "ts_us", "stream", "phase"])
            from sglang_trace_lib import TraceMeta, busy_time, trace_identity

            run, rank, isl, osl, conc = trace_identity(path)
            k = df[df["cat"] == "kernel"]
            meta = TraceMeta(
                path=str(path), run=run, rank=rank, isl=isl, osl=osl, conc=conc,
                n_gpu_ops=len(df), n_kernels=len(k),
                kernel_time_us=float(k["dur_us"].sum()),
                gpu_busy_us=busy_time(df["ts_us"].to_numpy(), df["dur_us"].to_numpy()),
                wall_us=float((df["ts_us"] + df["dur_us"]).max() - df["ts_us"].min()) if len(df) else 0.0,
                streams=int(df["stream"].nunique()),
                phases=k.groupby("phase")["dur_us"].sum().to_dict(),
            )
            return {"ok": True, "cached": True, "meta": asdict(meta), "cache": str(cache)}

        rows, meta = parse_trace(
            path, with_cpu_op=with_cpu_op, allow_truncated=allow_truncated
        )
        if not rows:
            # nothing to aggregate; don't leave a cache entry that would make
            # the next run think this trace was analysed successfully
            cache.unlink(missing_ok=True)
            why = "no GPU events in trace"
            if meta.truncated:
                # Kineto flushes CPU events before GPU ones, so a trace cut off
                # early can hold gigabytes of python_function/cpu_op records and
                # still contain not one kernel. Nothing to salvage here.
                why = ("trace is truncated before the GPU event section -- only "
                       "host-side events survived, so there is no kernel timing "
                       "to recover")
            return {"ok": False, "path": str(path), "error": why, "kind": "truncated"}
        cache.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows).reindex(columns=EVENT_COLS)
        # --no-cpu-op leaves whole columns absent; keep them typed so the later
        # concat across traces has consistent dtypes
        for col in ("cat", "category", "kernel", "phase", "region", "cpu_op", "grid", "block", "name"):
            df[col] = df[col].fillna("").astype(str)
        df.to_parquet(cache, index=False, compression="zstd")
        return {"ok": True, "cached": False, "meta": asdict(meta), "cache": str(cache)}
    except TruncatedTrace as exc:
        cache.unlink(missing_ok=True)
        # expected enough (killed profiling runs) that a traceback is noise
        return {"ok": False, "path": str(path), "error": str(exc), "kind": "truncated"}
    except Exception as exc:  # noqa: BLE001
        cache.unlink(missing_ok=True)
        return {"ok": False, "path": str(path),
                "error": f"{exc}\n{traceback.format_exc()}", "kind": "error"}


def _pct(series: pd.Series, q: float) -> float:
    return float(np.percentile(series.to_numpy(), q)) if len(series) else 0.0


def aggregate(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the kernel / category / phase tables from the per-event frame."""
    gpu = df[df["cat"].isin(["kernel", "gpu_memcpy", "gpu_memset"])]

    key = ["run", "rank", "phase", "category", "kernel"]
    g = gpu.groupby(key, dropna=False)["dur_us"]
    kern = g.agg(
        calls="count",
        total_us="sum",
        mean_us="mean",
        min_us="min",
        max_us="max",
        std_us="std",
    ).reset_index()
    q = g.agg(
        p50_us=lambda s: _pct(s, 50),
        p90_us=lambda s: _pct(s, 90),
        p99_us=lambda s: _pct(s, 99),
    ).reset_index()
    kern = kern.merge(q, on=key, how="left")

    # share of the rank's total GPU op time, and of its own phase
    rank_tot = kern.groupby(["run", "rank"])["total_us"].transform("sum")
    phase_tot = kern.groupby(["run", "rank", "phase"])["total_us"].transform("sum")
    kern["pct_of_rank"] = 100.0 * kern["total_us"] / rank_tot.replace(0, np.nan)
    kern["pct_of_phase"] = 100.0 * kern["total_us"] / phase_tot.replace(0, np.nan)
    kern = kern.sort_values(["run", "rank", "phase", "total_us"], ascending=[True, True, True, False])

    # across TP ranks: mean per rank, so numbers stay per-GPU comparable
    nranks = kern.groupby("run")["rank"].transform("nunique")
    by_run = (
        kern.assign(_n=nranks)
        .groupby(["run", "phase", "category", "kernel"], dropna=False)
        .apply(
            lambda d: pd.Series(
                {
                    "ranks": d["rank"].nunique(),
                    "calls_per_rank": d["calls"].sum() / d["rank"].nunique(),
                    "total_us_per_rank": d["total_us"].sum() / d["rank"].nunique(),
                    "mean_us": np.average(d["mean_us"], weights=d["calls"].clip(lower=1)),
                    "max_us": d["max_us"].max(),
                    "total_us_all_ranks": d["total_us"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    tot = by_run.groupby("run")["total_us_per_rank"].transform("sum")
    by_run["pct_of_run"] = 100.0 * by_run["total_us_per_rank"] / tot.replace(0, np.nan)
    by_run = by_run.sort_values(["run", "phase", "total_us_per_rank"], ascending=[True, True, False])

    cats = (
        kern.groupby(["run", "phase", "category"], dropna=False)
        .agg(calls=("calls", "sum"), total_us=("total_us", "sum"),
             kernels=("kernel", "nunique"), ranks=("rank", "nunique"))
        .reset_index()
    )
    cats["total_us_per_rank"] = cats["total_us"] / cats["ranks"]
    tot = cats.groupby("run")["total_us_per_rank"].transform("sum")
    cats["pct_of_run"] = 100.0 * cats["total_us_per_rank"] / tot.replace(0, np.nan)
    cats = cats.sort_values(["run", "phase", "total_us"], ascending=[True, True, False])

    phases = (
        kern.groupby(["run", "phase"], dropna=False)
        .agg(calls=("calls", "sum"), total_us=("total_us", "sum"),
             kernels=("kernel", "nunique"), ranks=("rank", "nunique"))
        .reset_index()
    )
    phases["total_us_per_rank"] = phases["total_us"] / phases["ranks"]
    tot = phases.groupby("run")["total_us_per_rank"].transform("sum")
    phases["pct_of_run"] = 100.0 * phases["total_us_per_rank"] / tot.replace(0, np.nan)
    phases = phases.sort_values(["run", "total_us"], ascending=[True, False])

    return {"kernels": kern, "kernels_by_run": by_run, "categories": cats, "phases": phases}


def write_report(out: Path, traces: pd.DataFrame, tabs: dict[str, pd.DataFrame], top: int) -> str:
    lines: list[str] = []
    w = lines.append

    w("=" * 100)
    w("SGLang kernel-time report")
    w("=" * 100)
    w("")
    w("-- traces --------------------------------------------------------------")
    t = traces.copy()
    t["kernel_time"] = t["kernel_time_us"].map(fmt_us)
    t["gpu_busy"] = t["gpu_busy_us"].map(fmt_us)
    t["wall"] = t["wall_us"].map(fmt_us)
    t["busy%"] = (100.0 * t["gpu_busy_us"] / t["wall_us"].replace(0, np.nan)).round(1)
    cols = ["run", "rank", "conc", "n_kernels", "kernel_time", "gpu_busy", "wall",
            "busy%", "streams"]
    if t.get("truncated", pd.Series(dtype=bool)).any():
        cols.append("truncated")
    w(t[cols].to_string(index=False))
    w("")

    for run in sorted(tabs["phases"]["run"].unique(), key=run_sort_key):
        w("")
        w("=" * 100)
        w(f"RUN {run}")
        w("=" * 100)

        ph = tabs["phases"][tabs["phases"]["run"] == run]
        w("")
        w("-- phases (per-rank GPU op time) ---------------------------------------")
        p = ph.copy()
        p["time"] = p["total_us_per_rank"].map(fmt_us)
        p["pct"] = p["pct_of_run"].round(1)
        w(p[["phase", "calls", "kernels", "time", "pct"]].to_string(index=False))

        ct = tabs["categories"][tabs["categories"]["run"] == run]
        w("")
        w("-- categories ----------------------------------------------------------")
        c = ct.copy()
        c["time"] = c["total_us_per_rank"].map(fmt_us)
        c["pct"] = c["pct_of_run"].round(2)
        w(c[["phase", "category", "calls", "kernels", "time", "pct"]].to_string(index=False))

        kr = tabs["kernels_by_run"]
        kr = kr[kr["run"] == run]
        for phase in ph["phase"].tolist():
            sub = kr[kr["phase"] == phase].head(top)
            if sub.empty:
                continue
            w("")
            w(f"-- top {min(top, len(sub))} kernels in phase {phase} ".ljust(72, "-"))
            s = sub.copy()
            s["total"] = s["total_us_per_rank"].map(fmt_us)
            s["mean"] = s["mean_us"].round(1)
            s["max"] = s["max_us"].round(1)
            s["calls"] = s["calls_per_rank"].round(1)
            s["pct"] = s["pct_of_run"].round(2)
            w(s[["category", "kernel", "calls", "total", "mean", "max", "pct"]]
              .to_string(index=False, max_colwidth=70))

    text = "\n".join(lines)
    (out / "report.txt").write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract kernel calls and their time from SGLang torch-profiler traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Outputs")[0],
    )
    ap.add_argument("roots", nargs="*", default=["/var/home/sglang_profiling"],
                    help="trace files or directories to scan (default: /var/home/sglang_profiling)")
    ap.add_argument("-o", "--outdir", default="./trace_analysis", help="output directory")
    ap.add_argument("-j", "--jobs", type=int, default=min(16, os.cpu_count() or 4),
                    help="parallel trace parsers (each holds one trace in memory)")
    ap.add_argument("--ranks", default="", help="comma-separated TP ranks to parse (default: all)")
    ap.add_argument("--top", type=int, default=25, help="kernels per phase in report.txt")
    ap.add_argument("--phase", default="", help="restrict tables to these phases (comma-separated)")
    ap.add_argument("--force", action="store_true", help="re-parse even if a cache entry is current")
    ap.add_argument("--no-cpu-op", action="store_true",
                    help="skip aten-op attribution (faster, drops the cpu_op column)")
    ap.add_argument("--allow-truncated", action="store_true",
                    help="analyse traces whose gzip stream is incomplete (a killed "
                         "profiling run); their numbers cover only part of the run")
    ap.add_argument("--list", action="store_true", help="list the traces that would be parsed and exit")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    traces = find_traces(args.roots)
    if args.ranks:
        want = {int(r) for r in args.ranks.split(",") if r.strip()}
        traces = [t for t in traces
                  if (m := RANK_RE.search(t.name)) and int(m.group(1)) in want]
    if not traces:
        print("no traces found", file=sys.stderr)
        return 1

    if args.list:
        for t in traces:
            print(f"{t.stat().st_size / 2**20:9.1f} MiB  {t}")
        return 0

    print(f"=== {len(traces)} trace(s) -> {outdir}  (jobs={args.jobs})")
    metas, caches, failed = [], [], []
    work = [(str(t), str(outdir), args.force, not args.no_cpu_op, args.allow_truncated)
            for t in traces]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(_worker, w): w[0] for w in work}
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            src = Path(futs[fut])
            if not res["ok"]:
                failed.append(res)
                if res.get("kind") == "truncated":
                    print(f"[{n}/{len(work)}] SKIPPED (truncated) {src}", file=sys.stderr)
                else:
                    print(f"[{n}/{len(work)}] FAILED {src}\n{res['error']}", file=sys.stderr)
                continue
            m = res["meta"]
            metas.append(m)
            caches.append(res["cache"])
            print(f"[{n}/{len(work)}] {'cached ' if res['cached'] else 'parsed '}"
                  f"{m['run']} TP-{m['rank']}: {m['n_kernels']:>8,} kernels, "
                  f"{fmt_us(m['kernel_time_us'])} kernel time, "
                  f"{fmt_us(m['gpu_busy_us'])} busy / {fmt_us(m['wall_us'])} wall"
                  + ("  [TRUNCATED]" if m.get("truncated") else ""))

    if not metas:
        print("every trace failed to parse", file=sys.stderr)
        return 1

    tdf = pd.DataFrame(metas).drop(columns=["phases"]).sort_values(
        by="run", key=lambda s: s.map(run_sort_key))
    for col in ("isl", "osl", "conc"):
        tdf[col] = tdf[col].astype("Int64")  # untagged runs have no shape
    tdf.to_csv(outdir / "traces.csv", index=False)

    print("=== aggregating")
    df = pd.concat([pd.read_parquet(c) for c in caches], ignore_index=True)
    if args.phase:
        keep = {p.strip() for p in args.phase.split(",") if p.strip()}
        df = df[df["phase"].isin(keep)]
        if df.empty:
            print(f"no events in phase(s) {sorted(keep)}; "
                  f"available: {sorted(pd.concat([pd.read_parquet(c, columns=['phase']) for c in caches])['phase'].unique())}",
                  file=sys.stderr)
            return 1

    tabs = aggregate(df)
    for name, tab in tabs.items():
        tab.to_csv(outdir / f"{name}.csv", index=False)

    text = write_report(outdir, tdf, tabs, args.top)
    print()
    print(text)
    print()
    print(f"=== wrote {outdir}/{{traces,kernels,kernels_by_run,categories,phases}}.csv "
          f"and report.txt")
    print(f"=== per-event rows cached in {outdir}/cache/*.parquet")
    if failed:
        trunc = [f for f in failed if f.get("kind") == "truncated"]
        if trunc:
            print(f"=== {len(trunc)} trace(s) skipped as truncated "
                  f"(re-collect them, or pass --allow-truncated):", file=sys.stderr)
            for f in trunc:
                print(f"      {f['path']}", file=sys.stderr)
        other = len(failed) - len(trunc)
        if other:
            print(f"=== {other} trace(s) failed to parse", file=sys.stderr)
        # 2 == the tables above are complete for every trace we could read, but
        # some were skipped. Distinct from 1 (nothing usable) so callers can
        # keep going while still noticing.
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
