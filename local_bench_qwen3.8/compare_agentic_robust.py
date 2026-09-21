#!/usr/bin/env python3
"""Percentile-based comparison of AgentX replay arms.

Why this exists alongside compare_agentic.py
--------------------------------------------
compare_agentic.py compares the *mean* of each per-request latency metric. On
this workload that is not robust, for two independent reasons:

1. Degenerate streaming records. The recipe runs --stream-interval 50, so the
   server emits 50 tokens per SSE chunk. When a whole response lands in
   effectively one chunk the derived inter-token latency collapses toward zero:
   the TP=1 conv_decode conc4 run contains a request reporting ITL=0.0016 ms
   over 87 output tokens, i.e. ~624,000 tok/s for that user. One such record
   moved mean per-user throughput from ~105 to ~1830 tok/s (+1966%). These are
   measurement artifacts of the streaming granularity, not model behaviour, and
   they are not symmetric between arms -- whichever arm happens to produce one
   wins or loses the mean outright.

2. A genuinely heavy-tailed workload. The AgentX trace has mean OSL ~950 but
   median OSL ~167 and max ~26,500. A mean over that distribution is dominated
   by a handful of very long responses, so it moves with *which* trajectories
   each time-bounded arm happened to finish rather than with kernel speed.

So this script reports p50 (typical request) and p90/p99 (tail) instead, and
counts the degenerate records rather than silently dropping them. Aggregate
rates -- output/request throughput -- are genuine run-level totals, not means of
per-request values, so those are reported unchanged.
"""

import argparse
import json
import re
from pathlib import Path

# Per-request latency metrics, compared at percentiles. (key, label, lower_is_better)
LATENCY = [
    ("time_to_first_token", "TTFT (ms)", True),
    ("inter_token_latency", "ITL (ms)", True),
    ("request_latency", "Request latency (ms)", True),
    ("full_decode_duration", "Decode duration (ms)", True),
]

# Run-level aggregate rates. (key, label, higher_is_better)
AGGREGATE = [
    ("output_token_throughput", "Output tput (tok/s)", True),
    ("request_throughput", "Request tput (req/s)", True),
]

WORKLOAD = [
    ("request_count", "requests completed"),
    ("input_sequence_length", "mean ISL"),
    ("output_sequence_length", "mean OSL"),
    ("benchmark_duration", "duration (s)"),
]

# An ITL below this is not a decode speed -- it is a whole response arriving in
# one streaming chunk. Decode at TP=1 on this model runs ~8-40 ms/token, so
# anything under a millisecond is an artifact by two orders of magnitude.
ITL_FLOOR_MS = 1.0


def val(m, k):
    v = m.get(k)
    return v["value"] if isinstance(v, dict) else v


def load_records(point: Path):
    rows = []
    f = point / "aiperf_artifacts" / "profile_export.jsonl"
    if not f.exists():
        return rows
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line)["metrics"])
        except Exception:
            pass
    return rows


def load_summary(point: Path):
    f = point / "aiperf_artifacts" / "profile_export_aiperf.json"
    return json.loads(f.read_text()) if f.exists() else None


def stat(rec, key, field="avg"):
    v = rec.get(key)
    return v.get(field) if isinstance(v, dict) else v


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def series(records, key):
    return sorted(v for v in (val(m, key) for m in records) if v is not None)


def degenerate_count(records):
    return sum(
        1
        for m in records
        if (val(m, "inter_token_latency") or ITL_FLOOR_MS) < ITL_FLOOR_MS
        and (val(m, "output_sequence_length") or 0) > 1
    )


def points_in(d: Path):
    """Map concurrency -> point dir for every complete conc<N> under d."""
    out = {}
    for p in sorted(d.glob("conc*")):
        m = re.match(r"conc(\d+)$", p.name)
        if m and (p / "aiperf_artifacts" / "profile_export_aiperf.json").exists():
            out[int(m.group(1))] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./agentic_results_tp1_ci")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--patched", default="conv_decode")
    ap.add_argument(
        "--baseline-dir",
        help="Directory holding the baseline conc<N>/ points directly, e.g. "
        "./baseline_reference_tp1. Use to compare a new patched run against a "
        "frozen baseline instead of re-measuring it. Overrides --baseline. "
        "Read VALIDITY.md there first: a frozen baseline is only comparable "
        "while the recipe, commits, trace and hardware are unchanged.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    base_dir = Path(args.baseline_dir) if args.baseline_dir else root / args.baseline
    base, new = points_in(base_dir), points_in(root / args.patched)
    shared = sorted(set(base) & set(new))
    if not shared:
        raise SystemExit(
            f"no overlapping concurrency points between {base_dir} and "
            f"{root / args.patched}\n"
            f"  baseline has: {sorted(base) or 'none'}\n"
            f"  patched  has: {sorted(new) or 'none'}"
        )
    if args.baseline_dir:
        print(f"baseline: {base_dir} (frozen reference, not measured in this run)")
        print(f"patched : {root / args.patched}")

    for conc in shared:
        bp, np_ = base[conc], new[conc]
        bs, ns = load_summary(bp), load_summary(np_)
        br, nr = load_records(bp), load_records(np_)

        print(f"\n{'=' * 84}")
        print(f"AgentX replay   concurrency {conc}")
        print("=" * 84)

        print("workload actually replayed (not a comparison):")
        for key, label in WORKLOAD:
            bv, nv = stat(bs, key), stat(ns, key)
            if bv is None or nv is None:
                continue
            print(f"  {label:<24}{bv:>14.1f}{nv:>16.1f}")

        bd, nd = degenerate_count(br), degenerate_count(nr)
        print(f"  {'sub-1ms ITL records':<24}{bd:>14d}{nd:>16d}"
              f"   <- streaming artifacts, excluded from p50/p90/p99")

        print(f"\n{'aggregate rate':<24}{'baseline':>14}{'aiter/flydsl':>16}{'delta':>12}")
        print("-" * 84)
        for key, label, higher in AGGREGATE:
            bv, nv = stat(bs, key), stat(ns, key)
            if bv is None or nv is None or bv == 0:
                continue
            d = (1 if higher else -1) * (nv - bv) / bv * 100.0
            mark = "+" if d > 0.5 else ("-" if d < -0.5 else "=")
            print(f"{label:<24}{bv:>14.2f}{nv:>16.2f}{d:>+11.2f}% {mark}")

        for q, qlabel in ((0.50, "p50"), (0.90, "p90"), (0.99, "p99")):
            print(f"\n{qlabel + ' latency':<24}{'baseline':>14}{'aiter/flydsl':>16}{'delta':>12}")
            print("-" * 84)
            for key, label, _ in LATENCY:
                bser, nser = series(br, key), series(nr, key)
                if key == "inter_token_latency":
                    bser = [v for v in bser if v >= ITL_FLOOR_MS]
                    nser = [v for v in nser if v >= ITL_FLOOR_MS]
                bv, nv = pct(bser, q), pct(nser, q)
                if bv is None or nv is None or bv == 0:
                    continue
                d = -(nv - bv) / bv * 100.0
                mark = "+" if d > 0.5 else ("-" if d < -0.5 else "=")
                print(f"{label:<24}{bv:>14.2f}{nv:>16.2f}{d:>+11.2f}% {mark}")

    print(f"\n{'=' * 84}")
    print("positive = patched better (throughput up / latency down)")
    print("Totals are omitted on purpose: the run is time-bounded, so a faster arm")
    print("completes more requests rather than the same work sooner.")


if __name__ == "__main__":
    main()
