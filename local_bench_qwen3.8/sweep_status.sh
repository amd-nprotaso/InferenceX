#!/usr/bin/env bash
# Terse one-screen status for the TP=1 sweep: which points finished, what each
# server reported about the GDN kernels, and whether anything errored.
D=${1:-/var/home/my_InferenceX/InferenceX/local_bench_qwen3.8/agentic_results_tp1_ci}

echo "now $(date +%H:%M:%S)"
grep -E "^###|^SWEEP COMPLETE" /tmp/tp1_sweep_driver.log 2>/dev/null | tail -6

for arm in baseline conv_decode; do
  for c in 1 4 8; do
    p="$D/$arm/conc$c"
    [ -d "$p" ] || continue
    res="$p/aiperf_artifacts/profile_export_aiperf.json"
    if [ -f "$res" ]; then st="DONE"; else st="running/incomplete"; fi
    # One-shot per-process verdict; proves the kernel was reached, not that
    # every later batch stayed on it.
    k=$(awk 'length($0)<400' "$p/server.log" 2>/dev/null \
        | grep -oE "\[aiter-gdn\] (decode|prefill conv): (ACTIVE|OFF|FELL BACK)" \
        | sed 's/\[aiter-gdn\] //' | sort -u | paste -sd'; ')
    printf '%-12s conc%-2s %-18s %s\n' "$arm" "$c" "$st" "${k:-<no report yet>}"
  done
done

echo "-- errors --"
# Replay-side: aiperf reports err=N per progress line.
grep -hoE "err=[1-9][0-9]*" "$D"/logs/*.log 2>/dev/null | sort -u | head -5
# Server-side. Two exclusions, both verified noise rather than convenience:
#   - the server_args banner is one >100KB line that contains "watchdog_timeout"
#   - torch _dynamo's own telemetry raises "not JSON serializable" ~110x per
#     start; it is dynamo failing to log a metric, not a model-side failure
for f in "$D"/*/conc*/server.log; do
  [ -f "$f" ] || continue
  n=$(awk 'length($0)<400' "$f" \
      | grep -E "Traceback|CUDA out of memory|watchdog timeout|Watchdog" \
      | grep -vc "metrics_context.py")
  [ "$n" -gt 0 ] && echo "  $n in $f"
done
echo "(nothing above = clean)"
