#!/usr/bin/env python3
"""Compare AgentX replay arms produced by run_agentic_sweep.sh.

Unlike the fixed-shape sweep, an AgentX replay is DURATION-bounded and driven by
recorded trajectories, so the two arms do not process the same requests. Totals
(total_output_tokens, request_count) are therefore not comparable between arms --
a faster arm simply gets through more work. Rates and per-request latencies are.

The script prints the workload each arm actually saw (request count, mean ISL /
OSL) next to the metrics, so a comparison distorted by the arms drawing
materially different work is visible rather than hidden.
"""

import argparse
import json
import re
from pathlib import Path

# (aiperf key, label, higher_is_better)
METRICS = [
    ("output_token_throughput", "Output tput (tok/s)", True),
    ("request_throughput", "Request tput (req/s)", True),
    ("output_token_throughput_per_user", "Per-user tput (tok/s)", True),
    ("time_to_first_token", "TTFT (ms)", False),
    ("inter_token_latency", "ITL (ms)", False),
    ("request_latency", "Request latency (ms)", False),
    ("full_decode_duration", "Decode duration (ms)", False),
]

# Workload descriptors: shown for sanity, never compared as a gain.
WORKLOAD = [
    ("request_count", "requests completed"),
    ("input_sequence_length", "mean ISL"),
    ("output_sequence_length", "mean OSL"),
    ("benchmark_duration", "duration (s)"),
]


def load_arm(arm_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for conc_dir in sorted(arm_dir.glob("conc*")):
        m = re.match(r"conc(\d+)$", conc_dir.name)
        if not m:
            continue
        f = conc_dir / "aiperf_artifacts" / "profile_export_aiperf.json"
        if f.exists():
            out[int(m.group(1))] = json.loads(f.read_text())
    return out


def stat(rec, key, field="avg"):
    v = rec.get(key)
    return v.get(field) if isinstance(v, dict) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./agentic_results")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--patched", default="conv_decode")
    ap.add_argument(
        "--baseline-dir",
        help="Directory holding the baseline conc<N>/ points directly, e.g. "
        "./baseline_reference_tp1. Overrides --baseline.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    base_dir = Path(args.baseline_dir) if args.baseline_dir else root / args.baseline
    base, new = load_arm(base_dir), load_arm(root / args.patched)
    shared = sorted(set(base) & set(new))
    if not shared:
        raise SystemExit(
            f"no overlapping concurrency points between {base_dir} and "
            f"{root / args.patched}"
        )

    for conc in shared:
        b, n = base[conc], new[conc]
        print(f"\n{'=' * 82}")
        print(f"AgentX replay   concurrency {conc}")
        print("=" * 82)

        print("workload actually replayed (not a comparison):")
        for key, label in WORKLOAD:
            bv, nv = stat(b, key), stat(n, key)
            if bv is None or nv is None:
                continue
            print(f"  {label:<22}{bv:>14.1f}{nv:>16.1f}")

        print(f"\n{'metric':<24}{'baseline':>14}{'aiter/flydsl':>16}{'delta':>12}")
        print("-" * 82)
        for key, label, higher in METRICS:
            bv, nv = stat(b, key), stat(n, key)
            if bv is None or nv is None or bv == 0:
                continue
            sign = 1 if higher else -1
            d = sign * (nv - bv) / bv * 100.0
            mark = "+" if d > 0.5 else ("-" if d < -0.5 else "=")
            print(f"{label:<24}{bv:>14.2f}{nv:>16.2f}{d:>+11.2f}% {mark}")

        # p99 tail, where a decode-side change tends to show up most.
        print()
        for key, label, higher in [("time_to_first_token", "TTFT p99 (ms)", False),
                                   ("inter_token_latency", "ITL p99 (ms)", False)]:
            bv, nv = stat(b, key, "p99"), stat(n, key, "p99")
            if bv is None or nv is None or bv == 0:
                continue
            d = -(nv - bv) / bv * 100.0
            print(f"{label:<24}{bv:>14.2f}{nv:>16.2f}{d:>+11.2f}%")

    print(f"\n{'=' * 82}")
    print("positive = patched better (throughput up / latency down)")
    print("Totals are omitted on purpose: the run is time-bounded, so a faster arm")
    print("completes more requests rather than the same work sooner.")


if __name__ == "__main__":
    main()
