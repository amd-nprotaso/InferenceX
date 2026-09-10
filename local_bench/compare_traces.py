#!/usr/bin/env python3
"""Pivot analyze_traces.py output into a run-vs-run kernel time comparison.

The profile sweep walks concurrency (isl32768_osl4_c1 .. _c16), so the question
is usually "which kernels grow with batch size, and which are flat". This turns
kernels_by_run.csv into that table.

  ./compare_traces.py trace_analysis
  ./compare_traces.py trace_analysis --phase EXTEND --top 30
  ./compare_traces.py trace_analysis --metric calls_per_rank

Outputs (under the same directory):
  compare_kernels.csv     kernel x run matrix, plus growth vs the baseline run
  compare_categories.csv  the same at category granularity
  compare.txt             the readable version

Times are per TP rank (mean across ranks), microseconds unless stated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sglang_trace_lib import TAG_RE, fmt_us, run_sort_key  # noqa: E402

METRICS = {
    "total_us_per_rank": "GPU time per rank (us)",
    "calls_per_rank": "kernel launches per rank",
    "mean_us": "mean kernel duration (us)",
}


def growth_pair(runs: list[str]) -> tuple[str, str] | None:
    """Pick the two runs the growth column compares.

    Prefer the ends of the concurrency sweep (isl..._cN tags). Untagged runs --
    traces dropped straight into PROFILE_DIR by a --profile benchmark rather
    than by the profile script -- are a different workload shape, so ratios
    against them are noise. Fall back to first/last only when no run is tagged.
    """
    tagged = [r for r in runs if TAG_RE.search(r)]
    pool = tagged if len(tagged) > 1 else runs
    return (pool[0], pool[-1]) if len(pool) > 1 else None


def pivot(df: pd.DataFrame, index: list[str], metric: str, runs: list[str]) -> pd.DataFrame:
    m = (
        df.pivot_table(index=index, columns="run", values=metric, aggfunc="sum")
        .reindex(columns=runs)
        .fillna(0.0)
    )
    m["total"] = m[runs].sum(axis=1)

    pair = growth_pair(runs)
    sort_by = "total"
    if pair:
        base, head = pair
        # growth over the sweep: how much more work the biggest run does per
        # rank than the smallest. inf means the kernel only shows up at scale.
        with np.errstate(divide="ignore", invalid="ignore"):
            m[f"x_{head}_over_{base}"] = np.where(m[base] > 0, m[head] / m[base], np.inf)
        sort_by = head
    return m.sort_values(sort_by, ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("analysis_dir", nargs="?", default="./trace_analysis",
                    help="directory written by analyze_traces.py")
    ap.add_argument("--metric", default="total_us_per_rank", choices=sorted(METRICS),
                    help="value to compare")
    ap.add_argument("--phase", default="", help="only this phase (e.g. EXTEND, TARGET_VERIFY)")
    ap.add_argument("--top", type=int, default=30, help="rows in compare.txt")
    ap.add_argument("--runs", default="", help="comma-separated subset of runs, in order")
    args = ap.parse_args()

    d = Path(args.analysis_dir)
    src = d / "kernels_by_run.csv"
    if not src.exists():
        print(f"{src} not found -- run analyze_traces.py first", file=sys.stderr)
        return 1

    df = pd.read_csv(src)
    if args.phase:
        keep = {p.strip() for p in args.phase.split(",") if p.strip()}
        df = df[df["phase"].isin(keep)]
        if df.empty:
            print(f"no rows for phase(s) {sorted(keep)}", file=sys.stderr)
            return 1

    if args.runs:
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]
        df = df[df["run"].isin(runs)]
    else:
        runs = sorted(df["run"].unique(), key=run_sort_key)
    if not runs:
        print("no runs to compare", file=sys.stderr)
        return 1
    if len(runs) == 1:
        print(f"note: only one run ({runs[0]}); comparison is a plain ranking",
              file=sys.stderr)

    kern = pivot(df, ["category", "kernel"], args.metric, runs)
    cats = pivot(df, ["category"], args.metric, runs)
    kern.to_csv(d / "compare_kernels.csv")
    cats.to_csv(d / "compare_categories.csv")

    lines: list[str] = []
    w = lines.append
    w("=" * 100)
    w(f"kernel time across runs -- metric: {METRICS[args.metric]}"
      + (f"   phase: {args.phase}" if args.phase else "   (all phases)"))
    w(f"runs: {', '.join(runs)}")
    w("=" * 100)

    def render(tab: pd.DataFrame, n: int | None) -> str:
        t = tab.head(n) if n else tab
        t = t.copy()
        if args.metric == "total_us_per_rank":
            for c in runs + ["total"]:
                t[c] = t[c].map(fmt_us)
        else:
            for c in runs + ["total"]:
                t[c] = t[c].round(1)
        growth = [c for c in t.columns if c.startswith("x_")]
        for c in growth:
            t[c] = t[c].map(lambda v: "inf" if not np.isfinite(v) else f"{v:.2f}x")
        return t.to_string(max_colwidth=64)

    w("")
    w("-- by category ---------------------------------------------------------")
    w(render(cats, None))
    w("")
    w(f"-- top {args.top} kernels ".ljust(72, "-"))
    w(render(kern, args.top))

    text = "\n".join(lines)
    (d / "compare.txt").write_text(text)
    print(text)
    print()
    print(f"=== wrote {d}/compare_kernels.csv, {d}/compare_categories.csv, {d}/compare.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
