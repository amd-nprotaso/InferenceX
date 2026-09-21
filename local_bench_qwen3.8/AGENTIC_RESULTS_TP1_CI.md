# AgentX replay — Qwen3.8-Flash-Next MXFP4, MI355X **TP=1**, baseline vs AITER/FlyDSL GDN

Date 2026-09-20. Recipe `qwen3.8next_fp4_mi355x_sglang_mtp.sh`, **unmodified**.
Model `amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8`. 1× MI355X, TP=1, EP=1,
`DURATION=1800` per point, concurrency 1 / 4 / 8, **one run per point**.

The CI master config runs this recipe at `tp: 8, conc-list: [1, 4, 8, 12, 16]`.
This sweep is TP=1 over the concurrency subset that fits a single GPU.

Two arms, differing **only** in the two kernel gates:

| arm | `SGLANG_AITER_GDN_CONV` | `SGLANG_AITER_GDN_DECODE` | kernels |
|---|---|---|---|
| `baseline` | — | — | stock Triton GDN decode + Triton causal-conv1d prefill |
| `conv_decode` | 1 | 1 | AITER/FlyDSL GDN decode + AITER/FlyDSL prefill conv |

## The kernels were verified live, per point

The repro package warns that both kernels sit behind a `try/except` that falls
back to Triton silently, so a fully-gated-on server can quietly emit baseline
numbers. Patch `0006` (logging only, one report per process, identical code in
both arms) was left applied, so every server self-certifies:

```
baseline    conc1/4/8  ->  decode: OFF     prefill conv: OFF
conv_decode conc1/4/8  ->  decode: ACTIVE  prefill conv: ACTIVE
```

A standalone `verify_kernels.sh` run at TP=1 reported both ACTIVE beforehand.
All 6 points exited 0 with no server-side errors.

*Caveat on the method:* the report is one-shot per process. It proves a kernel
was reached and accepted a batch, not that every later batch stayed on it.

---

## Headline

**The kernels pay at concurrency 8 and are a wash below it.**

| conc | output tput | request tput | verdict |
|---:|---:|---:|---|
| 1 | +0.00% | +0.00% | wash — identical workload, no measurable difference |
| 4 | −2.17% | −0.29% | slightly negative, but the noisiest point (see below) |
| 8 | **+6.87%** | **+4.81%** | **clear win**, and the best-evidenced point |

At conc8 the patched arm also improved latency at nearly every percentile:

| conc8 | baseline | patched | delta |
|---|---:|---:|---|
| ITL p50 (ms) | 11.74 | 9.93 | **+15.4%** |
| ITL p90 (ms) | 23.52 | 20.98 | +10.8% |
| ITL p99 (ms) | 51.19 | 40.91 | **+20.1%** |
| TTFT p50 (ms) | 1074.48 | 981.70 | +8.6% |
| Decode duration p99 (ms) | 87236.57 | 76753.61 | +12.0% |

This is the regime the repro package predicts the kernels matter in: with the
GPU actually saturated, the GDN decode recurrence is a real share of each step.
At conc1 the replay is trace-paced and the server is underloaded, so kernel
speed has nothing to bite on.

---

## Read the workload before reading the deltas

The AgentX replay is **time-bounded and trace-driven**, so the two arms do not
process the same requests — a faster arm draws more work rather than finishing
the same work sooner. Totals are therefore not comparable, and latency
percentiles are only comparable to the extent the drawn work matched:

| conc | requests (base/patched) | output tokens (base/patched) | comparable? |
|---:|---|---|---|
| 1 | 160 / 160 | 128,965 / 128,965 | **exactly identical** |
| 4 | 404 / 401 | 346,034 / 336,682 (−2.7%) | approximately |
| 8 | 812 / **851** | 540,134 / **580,390** (+7.5%) | patched did *more* |

This matters in two directions:

- **conc1 is the clean comparison.** Both arms replayed a byte-identical
  trajectory list. The answer there is simply "no difference": output
  throughput matched to 2 decimals and ITL p50 moved +1.3%.
- **conc8 understates the win.** The patched arm processed 7.5% more output
  tokens *and* was faster at nearly every percentile. More work normally costs
  latency, so the +6.87% throughput and +15–20% ITL gains are conservative.

conc4 is the weakest point in the set: the arms drew different work, and some
per-request percentiles swing implausibly far for a kernel swap (p50 decode
duration −61% at an identical OSL p50 of 138). At 4 concurrent streams over a
trace whose OSL spans 2 → 26,571 tokens, a request's decode duration depends
heavily on which other requests happen to be batched alongside it. Treat conc4
as inconclusive rather than as a regression.

---

## Why this file reports percentiles, not means

`compare_agentic.py` compares the **mean** of each per-request metric. On this
workload that is not robust, and it produced one frankly broken number:

> conc4, per-user throughput: baseline 88.57 → patched **1829.72 tok/s (+1966%)**

That is not a speedup. The patched conc4 run contains a single request reporting
`inter_token_latency = 0.0016 ms` across 87 output tokens — ~624,000 tok/s for
that one user, which dragged the mean up by 20×. The recipe runs
`--stream-interval 50`, so the server emits 50 tokens per SSE chunk; when a
response effectively lands in one chunk the derived ITL collapses toward zero.
These are artifacts of streaming granularity, not model behaviour, and they are
not symmetric between arms — whichever arm happens to produce one wins the mean
outright.

Compounding it, the trace is heavy-tailed: mean OSL ≈ 950 but median OSL ≈ 167
and max ≈ 26,500. A mean over that distribution tracks *which* long trajectories
an arm happened to finish more than it tracks kernel speed.

`compare_agentic_robust.py` therefore reports p50/p90/p99, counts the degenerate
sub-1 ms ITL records instead of silently dropping them (0/0 at conc1, 0/1 at
conc4, 2/3 at conc8), and leaves aggregate rates alone — those are genuine
run-level totals, not means of per-request values.

Both tables are saved: `comparison_robust.txt` and `comparison_mean.txt`.

---

## Confidence and what would firm it up

**One run per point.** The conc8 win is internally consistent (throughput,
request rate, and every ITL percentile move the same way, while the arm carried
more load), so it is unlikely to be noise. conc4 is not trustworthy at n=1.

To firm this up: repeat conc4 and conc8 2–3× each and report spread. A cheaper
and sharper alternative is the repro package's own fixed-shape sweep
(`qwen38-gdn-repro/scripts/run_all.sh`), where both arms replay an identical
request set and the workload-draw confound disappears entirely.

## Layout

```
agentic_results_tp1_ci/
  run_metadata.json          provenance: commits, patches, gates, CI-config delta
  comparison_robust.txt      p50/p90/p99 tables (this file's source)
  comparison_mean.txt        original mean-based tables, kept for comparison
  {baseline,conv_decode}/conc{1,4,8}/
    arm_env.json             gates recorded per point
    server.log               includes the [aiter-gdn] ACTIVE/OFF verdict
    sglang_command.txt       exact server argv
    aiperf_artifacts/        profile_export.jsonl + aggregates
  logs/                      per-point driver logs
```

Reproduce with `./run_tp1_ci_sweep.sh` (drives `run_agentic_arm.sh`, which
invokes the unmodified recipe once per arm/concurrency).

## Reusing the baseline

The baseline arm is frozen verbatim in `baseline_reference_tp1/`, so future
kernel work does not have to re-measure it (~50 min per concurrency point):

```bash
ARMS=conv_decode BASE_RESULT_DIR=$PWD/agentic_results_next ./run_tp1_ci_sweep.sh
python3 compare_agentic_robust.py \
    --root agentic_results_next --baseline-dir baseline_reference_tp1
```

Read `baseline_reference_tp1/VALIDITY.md` before relying on it — it lists what
invalidates the stored numbers, and why conc4/conc8 reuse needs the drawn-
workload check that conc1 does not.
