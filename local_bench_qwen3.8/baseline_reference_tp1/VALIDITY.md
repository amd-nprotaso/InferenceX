# Frozen baseline — Qwen3.8-Flash-Next MXFP4, MI355X TP=1, stock Triton GDN

Measured **2026-09-20**. This is the `baseline` arm of the TP=1 AgentX sweep,
preserved verbatim so future kernel work can be compared without spending
~50 min re-measuring it per concurrency point.

**Stock Triton kernels.** Every point's `server.log` carries the self-check:

```
[aiter-gdn] decode: OFF (env gate closed or aiter import failed)
[aiter-gdn] prefill conv: OFF (env gate closed or aiter import failed)
```

## How to use it

```bash
# new patched run lands in agentic_results_<whatever>/conv_decode/conc*/
python3 compare_agentic_robust.py \
    --root agentic_results_<whatever> \
    --baseline-dir baseline_reference_tp1
```

Verify integrity first:

```bash
cd baseline_reference_tp1 && sha256sum -c SHA256SUMS --quiet && echo OK
```

The tree is `chmod -R a-w`, but note this shell runs as **root**, and root
ignores permission bits — a stray `cp`/`rm` here will still succeed. Treat the
mode as a speed bump and `SHA256SUMS` as the actual guarantee. Two limits on it:
it detects modified and deleted files, but **not added** ones, and it is only
meaningful if you run the check *before* trusting a comparison.

`baseline_summary.json` holds the headline numbers in plain text, so the
comparison survives even if the bulky artifacts are ever pruned.

## Exactly what was measured

| | |
|---|---|
| recipe | `benchmarks/single_node/agentic/qwen3.8next_fp4_mi355x_sglang_mtp.sh`, **unmodified** |
| scenario | agentic-coding AgentX replay, trace `semianalysis_cc_traces_weka_062126_256k` |
| model | `amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8` |
| hardware | 1× MI355X (gfx950) |
| TP / EP | 1 / 1 |
| duration | 1800 s per point |
| concurrency | 1, 4, 8 |
| runs per point | **1** |
| SGLang | `3003ddf157` (+ patches 0001–0005 applied, GDN gates closed) |
| AITER | `4ad998328` (+ patches 0001–0005) |
| gates | `SGLANG_AITER_GDN_CONV` and `SGLANG_AITER_GDN_DECODE` both unset |

Baseline numbers (from `baseline_summary.json`):

| conc | out tput (tok/s) | req tput | ITL p50 | ITL p99 | records drawn | output tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 70.47 | 0.08 | 8.76 | 32.99 | 160 | 128,965 |
| 4 | 188.04 | 0.20 | 10.16 | 36.40 | 404 | 346,034 |
| 8 | 295.11 | 0.40 | 11.74 | 51.19 | 812 | 540,134 |

---

## When this baseline stops being valid

Re-measure the baseline if **any** of these changes. Reusing it across one of
these turns an A/B into an uncontrolled comparison, and the difference will look
like a kernel result.

**Hard invalidation — the number is simply not comparable any more:**

- the recipe script, or any server flag it emits (`sglang_command.txt` is stored
  per point; diff against it)
- the SGLang or AITER commit, or the container image
- the model checkpoint or its revision
- the AgentX trace / `WEKA_LOADER_OVERRIDE` dataset
- TP, EP, duration, or the concurrency definition
- the GPU model, or anything that changes clocks/power caps

**Soft invalidation — still comparable, but widen your error bars:**

- a different physical machine of the same SKU
- other load on the host (this run had the box to itself; all 4 GPUs idle, only
  GPU 0 in use)

## The confound you must not forget

The AgentX replay is **time-bounded and trace-driven**. Each arm draws its own
workload — a faster arm completes *more* trajectories rather than the same ones
sooner. So a frozen baseline is not a fixed yardstick at every point:

| conc | records drawn | reuse is |
|---:|---:|---|
| 1 | 160 | **clean** — the draw was deterministic; the 2026-09-20 patched arm drew a byte-identical 160 records / 128,965 output tokens |
| 4 | 404 | **caution** — the two arms already diverged 2.7% in output tokens on the same day |
| 8 | 812 | **caution** — the patched arm drew 7.5% *more* output tokens |

At conc1 you can trust a stored-vs-new comparison directly. At conc4 and conc8,
always read the `workload_drawn` block that `compare_agentic_robust.py` prints
before reading the deltas: if the new run's output-token total differs from the
baseline's by more than a few percent, the latency percentiles are measuring
workload mix as much as kernel speed.

That confound does not shrink with a frozen baseline — it compounds, because the
two runs are now separated in time as well. If a future change produces a small
or ambiguous delta, do not settle it against this file. Use the repro package's
fixed-shape sweep (`qwen38-gdn-repro/scripts/run_all.sh`), where both arms
replay an identical request set and the confound disappears.

## Contents

```
conc{1,4,8}/                  full per-point artifacts, copied verbatim
  aiperf_artifacts/
    profile_export.jsonl        per-request records (robust comparison reads this)
    profile_export_aiperf.json  aggregates (both comparison scripts read this)
    *_timeslices.*              per-interval series
    server_metrics_export.json  scraped sglang Prometheus series
  server.log                   includes the [aiter-gdn] OFF/OFF verdict
  sglang_command.txt           exact server argv
  arm_env.json                 gates recorded at run time
  gpu_metrics*.csv, power_*    energy / power window
  workload_distribution_*      what the trace actually replayed
baseline_summary.json         headline numbers in plain text
source_run_metadata.json      provenance of the sweep this came from
SHA256SUMS                    integrity of all 79 files
```
