#!/usr/bin/env python3
"""Aggregate the baseline-vs-all serving A/B into a paired comparison.

Reads every ``round_*/<arm>/trial_*/**.jsonl`` produced by
``qwen3.5_fp4_sglang_bench.sh`` and reports, per metric, each arm's mean and
the delta. Rounds alternate arm order, so the paired view (round-by-round) is
the one that is robust to clock drift across the run; the pooled view is shown
alongside it.

Usage:  python3 summarize.py [experiment_dir]
"""

import glob
import json
import os
import statistics
import sys

ARMS = ("baseline", "all")

# (key, label, higher_is_better)
METRICS = [
    ("output_throughput", "output tok/s", True),
    ("total_throughput", "total tok/s", True),
    ("duration", "duration s", False),
    ("mean_tpot_ms", "TPOT ms", False),
    ("median_itl_ms", "median ITL ms", False),
    ("mean_ttft_ms", "TTFT ms", False),
    ("mean_e2e_latency_ms", "E2E ms", False),
    ("accept_length", "accept len", None),
]


def load(experiment_dir):
    out = {}
    for arm in ARMS:
        for path in sorted(glob.glob(os.path.join(experiment_dir, "round_*", arm, "trial_*", "*.jsonl"))):
            rnd = int(path.split(os.sep)[-4].split("_")[1])
            trial = int(path.split(os.sep)[-2].split("_")[1])
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.setdefault(arm, []).append((rnd, trial, json.loads(line)))
    return out


def main():
    experiment_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    data = load(experiment_dir)
    if not all(data.get(a) for a in ARMS):
        raise SystemExit(f"no trials found for both arms under {experiment_dir}")

    print(f"experiment: {experiment_dir}")
    for arm in ARMS:
        rounds = sorted({r for r, _, _ in data[arm]})
        print(f"  {arm:9s}: {len(data[arm])} trials over rounds {rounds}")

    print("\n== per-trial ==")
    hdr = f"{'metric':<16}" + "".join(f"{a:>14}" for a in ARMS)
    for rnd in sorted({r for a in ARMS for r, _, _ in data[a]}):
        for trial in sorted({t for a in ARMS for r, t, _ in data[a] if r == rnd}):
            vals = {}
            for arm in ARMS:
                hit = [d for r, t, d in data[arm] if r == rnd and t == trial]
                if hit:
                    vals[arm] = hit[0]
            if len(vals) != len(ARMS):
                continue
            print(f"\n round {rnd} trial {trial}")
            print(" " + hdr)
            for key, label, _ in METRICS:
                row = f" {label:<16}"
                for arm in ARMS:
                    row += f"{vals[arm].get(key, float('nan')):>14.3f}"
                print(row)

    print("\n== pooled (all trials) ==")
    print(f"{'metric':<16}" + "".join(f"{a:>14}" for a in ARMS) + f"{'delta':>12}{'%':>9}")
    for key, label, higher_better in METRICS:
        means = {a: statistics.fmean(d.get(key, float('nan')) for _, _, d in data[a]) for a in ARMS}
        delta = means["all"] - means["baseline"]
        pct = 100.0 * delta / means["baseline"] if means["baseline"] else float("nan")
        mark = ""
        if higher_better is not None:
            good = (delta > 0) == higher_better
            mark = "  better" if abs(pct) > 0.5 and good else ("  WORSE" if abs(pct) > 0.5 else "")
        row = f"{label:<16}" + "".join(f"{means[a]:>14.3f}" for a in ARMS)
        print(row + f"{delta:>12.3f}{pct:>8.2f}%{mark}")

    print("\n== spread (stdev / mean, pooled) ==")
    for key, label, _ in METRICS:
        parts = []
        for arm in ARMS:
            xs = [d.get(key, float("nan")) for _, _, d in data[arm]]
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            m = statistics.fmean(xs)
            parts.append(f"{arm}={100 * sd / m:5.2f}%" if m else f"{arm}=  n/a")
        print(f"{label:<16}" + "  ".join(parts))

    print("\n== paired by round (robust to drift across the run) ==")
    print(f"{'metric':<16}" + "".join(f"{'round ' + str(r):>14}" for r in sorted({r for _, _, _ in [] } | {r for a in ARMS for r, _, _ in data[a]})) + f"{'mean':>12}")
    rounds = sorted({r for a in ARMS for r, _, _ in data[a]})
    for key, label, higher_better in METRICS:
        if higher_better is None:
            continue
        per_round = []
        for rnd in rounds:
            b = [d.get(key) for r, _, d in data["baseline"] if r == rnd]
            a = [d.get(key) for r, _, d in data["all"] if r == rnd]
            if not b or not a:
                per_round.append(float("nan"))
                continue
            per_round.append(100.0 * (statistics.fmean(a) - statistics.fmean(b)) / statistics.fmean(b))
        avg = statistics.fmean([x for x in per_round if x == x]) if any(x == x for x in per_round) else float("nan")
        row = f"{label:<16}" + "".join(f"{x:>13.2f}%" for x in per_round)
        print(row + f"{avg:>11.2f}%")


if __name__ == "__main__":
    main()
