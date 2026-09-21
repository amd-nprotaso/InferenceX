# AgentX replay — Qwen3.8-Flash-Next MXFP4, MI355X TP=1

Recipe: qwen3.8next_fp4_mi355x_sglang_mtp.sh (unmodified), branch feat/qwen38next-mxfp4-mi355x-agentic
Model: amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8 (cached; config names the non-PLEFP8 variant)
DURATION=1800s per point, TP=1, one run per arm. Date 2026-09-18.
Backends confirmed: decode=triton prefill=triton verify=triton (AITER hooks on path).


==================================================================================
AgentX replay   concurrency 1
==================================================================================
workload actually replayed (not a comparison):
  requests completed             149.0           149.0
  mean ISL                     97071.8         97072.0
  mean OSL                       865.5           865.5
  duration (s)                  1799.0          1796.6

metric                        baseline    aiter/flydsl       delta
----------------------------------------------------------------------------------
Output tput (tok/s)              71.64           71.64      +0.00% =
Request tput (req/s)              0.08            0.08      +0.00% =
Per-user tput (tok/s)            89.52           90.71      +1.33% +
TTFT (ms)                      2575.96         2651.67      -2.94% -
ITL (ms)                         15.14           15.62      -3.16% -
Request latency (ms)          13439.97        13706.23      -1.98% -
Decode duration (ms)          10864.05        11054.60      -1.75% -

TTFT p99 (ms)                 28318.68        32263.68     -13.93%
ITL p99 (ms)                     32.86           35.02      -6.57%

==================================================================================
AgentX replay   concurrency 4
==================================================================================
workload actually replayed (not a comparison):
  requests completed             352.0           358.0
  mean ISL                     75208.0         75735.7
  mean OSL                       966.9           940.3
  duration (s)                  1813.0          1807.1

metric                        baseline    aiter/flydsl       delta
----------------------------------------------------------------------------------
Output tput (tok/s)             185.98          182.96      -1.63% -
Request tput (req/s)              0.19            0.19      +1.15% +
Per-user tput (tok/s)            85.28           85.30      +0.02% =
TTFT (ms)                      1118.58         1099.76      +1.68% +
ITL (ms)                         15.05           16.67     -10.78% -
Request latency (ms)          13369.25        13905.90      -4.01% -
Decode duration (ms)          12250.71        12806.18      -4.53% -

TTFT p99 (ms)                  2644.15         2984.06     -12.86%
ITL p99 (ms)                     38.61           42.69     -10.56%

==================================================================================
positive = patched better (throughput up / latency down)
Totals are omitted on purpose: the run is time-bounded, so a faster arm
completes more requests rather than the same work sooner.
