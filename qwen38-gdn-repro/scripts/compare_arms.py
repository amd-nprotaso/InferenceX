#!/usr/bin/env python3
"""Side-by-side of two bench_results arms produced by run_client.sh.

Reads bench_results/<arm>/qwen38_<arm>_isl*_osl*_c*.jsonl and prints one row per
concurrency point. Deltas are patched-relative-to-baseline, signed so that a
positive percentage always means "patched is better" -- throughput up, latency
down.
"""

import argparse
import json
import re
from pathlib import Path

# (json key, label, higher_is_better)
METRICS = [
    ("input_throughput", "Input tput (tok/s)", True),
    ("output_throughput", "Output tput (tok/s)", True),
    ("total_throughput", "Total tput (tok/s)", True),
    ("request_throughput", "Req tput (req/s)", True),
    ("mean_ttft_ms", "Mean TTFT (ms)", False),
    ("median_ttft_ms", "Median TTFT (ms)", False),
    ("mean_tpot_ms", "Mean TPOT (ms)", False),
    ("median_tpot_ms", "Median TPOT (ms)", False),
    ("mean_itl_ms", "Mean ITL (ms)", False),
    ("median_e2e_latency_ms", "Median E2E (ms)", False),
    ("duration", "Wall clock (s)", False),
    ("accept_length", "MTP accept len", True),
]


def load(arm_dir: Path) -> dict[int, dict]:
    """Last record per concurrency; a rerun appends rather than truncating."""
    out: dict[int, dict] = {}
    for path in sorted(arm_dir.glob("*.jsonl")):
        m = re.search(r"_c(\d+)\.jsonl$", path.name)
        if not m:
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                out[int(m.group(1))] = json.loads(line)
    return out


def pct(base, new, higher_is_better):
    if base in (None, 0) or new is None:
        return None
    change = (new - base) / base * 100.0
    return change if higher_is_better else -change


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./bench_results")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--patched", default="aiter_gdn")
    args = ap.parse_args()

    root = Path(args.root)
    base = load(root / args.baseline)
    new = load(root / args.patched)

    shared = sorted(set(base) & set(new))
    if not shared:
        raise SystemExit(f"no overlapping concurrency points in {root}")

    for conc in shared:
        b, n = base[conc], new[conc]
        print(f"\n{'=' * 78}")
        print(
            f"concurrency {conc}   ISL {b['random_input_len']} / OSL "
            f"{b['random_output_len']}   completed {b['completed']}"
        )
        print("=" * 78)
        print(f"{'metric':<22}{'baseline':>14}{'aiter/flydsl':>16}{'delta':>12}{'':>4}")
        print("-" * 78)
        for key, label, higher in METRICS:
            if key not in b or key not in n:
                continue
            d = pct(b[key], n[key], higher)
            mark = "" if d is None else ("  +" if d > 0.5 else ("  -" if d < -0.5 else "  ="))
            ds = "   n/a" if d is None else f"{d:+.2f}%"
            print(f"{label:<22}{b[key]:>14.2f}{n[key]:>16.2f}{ds:>12}{mark:>4}")

    print(f"\n{'=' * 78}")
    print("delta sign convention: positive = patched better (tput up / latency down)")


if __name__ == "__main__":
    main()
