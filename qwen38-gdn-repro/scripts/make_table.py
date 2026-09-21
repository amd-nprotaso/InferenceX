#!/usr/bin/env python3
"""Render the throughput / E2E-latency / TTFT markdown tables from a results dir.

    python3 scripts/make_table.py --root results
    python3 scripts/make_table.py --root reference_results > RESULTS_mine.md

Percentages are relative to the baseline arm and signed so that positive always
means better: throughput up, latency down.
"""

import argparse
import json
import re
from pathlib import Path

SECTIONS = [
    ("Throughput (total, tok/s)", "total_throughput", True, "{:.2f}"),
    ("E2E latency (mean, ms)", "mean_e2e_latency_ms", False, "{:.2f}"),
    ("TTFT (mean, ms)", "mean_ttft_ms", False, "{:.2f}"),
    ("TPOT (mean, ms)", "mean_tpot_ms", False, "{:.2f}"),
]


def load(arm_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(arm_dir.glob("*.jsonl")):
        m = re.search(r"_c(\d+)\.jsonl$", path.name)
        if not m:
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                out[int(m.group(1))] = json.loads(line)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--arms", nargs="*", default=["conv_only", "conv_decode"])
    ap.add_argument(
        "--labels", nargs="*", default=["conv", "both"],
        help="column label per --arms entry",
    )
    args = ap.parse_args()

    root = Path(args.root)
    base = load(root / args.baseline)
    arms = [(lbl, load(root / a)) for lbl, a in zip(args.labels, args.arms)]
    concs = sorted(set(base) & set.intersection(*[set(d) for _, d in arms]))
    if not concs:
        raise SystemExit(f"no overlapping concurrency points under {root}")

    any_rec = base[concs[0]]
    print(f"ISL {any_rec['random_input_len']} / OSL {any_rec['random_output_len']}. "
          "Percentages vs baseline; **positive = better**.\n")

    for title, key, higher, fmt in SECTIONS:
        print(f"### {title}\n")
        head = "| conc | baseline |"
        sep = "|---|---|"
        for lbl, _ in arms:
            head += f" {lbl} | {lbl} gain |"
            sep += "---|---|"
        print(head)
        print(sep)
        for c in concs:
            b = base[c][key]
            row = f"| {c} | {fmt.format(b)} |"
            for _, d in arms:
                v = d[c][key]
                sign = 1 if higher else -1
                pct = sign * (v - b) / b * 100.0
                row += f" {fmt.format(v)} | **{pct:+.2f}%** |"
            print(row)
        print()

    # Per-arm min/max across concurrency, so the headline claim is a range and
    # not a single cherry-picked point.
    print("### Summary (range across concurrency)\n")
    head = "| metric |" + "".join(f" {lbl} gain |" for lbl, _ in arms)
    print(head)
    print("|---|" + "---|" * len(arms))
    for title, key, higher, _ in SECTIONS:
        row = f"| {title.split(' (')[0]} |"
        for _, d in arms:
            sign = 1 if higher else -1
            gains = [sign * (d[c][key] - base[c][key]) / base[c][key] * 100.0 for c in concs]
            row += f" {min(gains):+.2f}% … {max(gains):+.2f}% |"
        print(row)


if __name__ == "__main__":
    main()
