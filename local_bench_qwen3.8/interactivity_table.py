#!/usr/bin/env python3
"""Latency + interactivity vs concurrency, for one arm (default: the AITER/FlyDSL
kernels), optionally with the stock-Triton baseline alongside.

Interactivity is the InferenceX 1/TPOT metric: output tokens per second seen by
a single user. At a given percentile it is exactly 1000 / ITL_ms -- verified
against aiperf's own output_token_throughput_per_user p50.

Direction matters and is easy to get wrong. aiperf reports per-user throughput
percentiles on the *throughput* distribution, so its p90/p99 are the FASTEST
users. A latency table wants the opposite tail, so the slow-end interactivity
here is derived from the ITL tail:

    interactivity p50       = 1000 / ITL p50      (median user)
    interactivity worst-10% = 1000 / ITL p90      (slowest decile)
    interactivity worst-1%  = 1000 / ITL p99      (slowest percentile)

Sub-1 ms ITL records are excluded: with --stream-interval 50 a response landing
in effectively one SSE chunk yields a near-zero derived ITL, which is streaming
granularity rather than decode speed.
"""

import argparse
import json
import re
from pathlib import Path

ITL_FLOOR_MS = 1.0


def val(m, k):
    v = m.get(k)
    return v["value"] if isinstance(v, dict) else v


def pct(vals, q):
    v = sorted(vals)
    if not v:
        return None
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def load_point(d: Path):
    a = d / "aiperf_artifacts"
    summary = json.loads((a / "profile_export_aiperf.json").read_text())
    recs = [
        json.loads(l)["metrics"]
        for l in (a / "profile_export.jsonl").read_text().splitlines()
        if l.strip()
    ]
    return summary, recs


def points_in(d: Path):
    out = {}
    for p in sorted(d.glob("conc*")):
        m = re.match(r"conc(\d+)$", p.name)
        if m and (p / "aiperf_artifacts" / "profile_export_aiperf.json").exists():
            out[int(m.group(1))] = p
    return out


def metrics(d: Path):
    s, recs = load_point(d)
    g = lambda k: (s.get(k) or {}).get("avg") if isinstance(s.get(k), dict) else s.get(k)
    ser = lambda k: [v for v in (val(m, k) for m in recs) if v is not None]
    itl = [v for v in ser("inter_token_latency") if v >= ITL_FLOOR_MS]
    ttft, rl = ser("time_to_first_token"), ser("request_latency")
    out = {
        "out_tput": g("output_token_throughput"),
        "req_tput": g("request_throughput"),
        "requests": len(recs),
        "out_tokens": sum(ser("output_sequence_length")),
        "excluded": len(ser("inter_token_latency")) - len(itl),
    }
    for q, tag in ((0.5, "p50"), (0.9, "p90"), (0.99, "p99")):
        out[f"ttft_{tag}"] = pct(ttft, q)
        out[f"itl_{tag}"] = pct(itl, q)
        out[f"rl_{tag}"] = pct(rl, q)
    out["intv_p50"] = 1000.0 / out["itl_p50"]
    out["intv_worst10"] = 1000.0 / out["itl_p90"]
    out["intv_worst1"] = 1000.0 / out["itl_p99"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./agentic_results_tp1_ci")
    ap.add_argument("--arm", default="conv_decode")
    ap.add_argument("--baseline-dir", default="./baseline_reference_tp1",
                    help="set to '' to omit the baseline columns")
    args = ap.parse_args()

    pts = points_in(Path(args.root) / args.arm)
    if not pts:
        raise SystemExit(f"no points under {Path(args.root) / args.arm}")
    base = points_in(Path(args.baseline_dir)) if args.baseline_dir else {}

    m = {c: metrics(d) for c, d in pts.items()}
    b = {c: metrics(d) for c, d in base.items() if c in m}
    concs = sorted(m)

    def row(label, key, fmt="{:.2f}", higher_better=None):
        cells = []
        for c in concs:
            v = m[c][key]
            s = fmt.format(v)
            if c in b and higher_better is not None:
                bv = b[c][key]
                d = (1 if higher_better else -1) * (v - bv) / bv * 100.0
                s += f" ({d:+.1f}%)"
            cells.append(s)
        return f"| {label} | " + " | ".join(cells) + " |"

    print(f"\n### AITER/FlyDSL GDN kernels — latency & interactivity vs concurrency")
    print(f"\nQwen3.8-Flash-Next MXFP4, 1x MI355X, TP=1, AgentX replay, 1800 s/point.")
    if b:
        print("Parenthesised = delta vs stock-Triton baseline; positive = better.\n")
    # One continuous table: a blank line would terminate it in markdown, so
    # groups are separated by a spanning label row instead.
    sep = lambda title: f"| **{title}** | " + " | ".join("" for _ in concs) + " |"

    print("| metric | " + " | ".join(f"conc {c}" for c in concs) + " |")
    print("|---|" + "---|" * len(concs))
    print(sep("Interactivity (tok/s/user, higher better)"))
    print(row("median user (p50)", "intv_p50", "{:.1f}", True))
    print(row("slowest decile", "intv_worst10", "{:.1f}", True))
    print(row("slowest percentile", "intv_worst1", "{:.1f}", True))
    print(sep("Inter-token latency (ms, lower better)"))
    print(row("ITL p50", "itl_p50", "{:.2f}", False))
    print(row("ITL p90", "itl_p90", "{:.2f}", False))
    print(row("ITL p99", "itl_p99", "{:.2f}", False))
    print(sep("Time to first token (ms, lower better)"))
    print(row("TTFT p50", "ttft_p50", "{:.0f}", False))
    print(row("TTFT p90", "ttft_p90", "{:.0f}", False))
    print(row("TTFT p99", "ttft_p99", "{:.0f}", False))
    print(sep("End-to-end request latency (ms, lower better)"))
    print(row("request latency p50", "rl_p50", "{:.0f}", False))
    print(row("request latency p99", "rl_p99", "{:.0f}", False))
    print(sep("Aggregate throughput (higher better)"))
    print(row("output tput (tok/s)", "out_tput", "{:.2f}", True))
    print(row("request tput (req/s)", "req_tput", "{:.3f}", True))
    print(sep("Workload actually drawn (context, not a result)"))
    print(row("requests", "requests", "{:.0f}"))
    print(row("output tokens", "out_tokens", "{:.0f}"))
    print(row("sub-1ms ITL records excluded", "excluded", "{:.0f}"))


if __name__ == "__main__":
    main()
