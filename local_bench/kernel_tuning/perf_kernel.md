# FlyDSL kernels vs baseline — end-to-end performance

Workload: `CONC_LIST="4" OSL=1024 ./qwen3.5_fp4_sglang_bench.sh`
(ISL 32768, range-ratio 0.8, TP=4, CONC=4, MTP EAGLE, 40 prompts + 8 warmups per run)
Hardware: 4x MI350X (gfx950), `amd/Qwen3.5-397B-A17B-MXFP4`
Measured: 2026-09-10, both arms in one session, alternating server boots.

## Arms

- **both** — `kernel_tuning/run_server_both.sh`: FlyDSL `causal_conv1d_fn`
  (`conv/`) + FlyDSL varlen `chunk_gated_delta_rule_fwd_h`
  (`chunk_gated_delta_rule/`)
- **stock** — `qwen3.5_fp4_sglang_server.sh`, no FlyDSL hooks

> "stock" is not upstream-default SGLang. The server script already exports
> `SGLANG_GDN_CHUNK_H_BV=16 / NUM_WARPS=4 / NUM_STAGES=3`
> (`qwen3.5_fp4_sglang_server.sh:60-62`) — the tuned Triton chunk_h config,
> ~1.42x over SGLang's `BV=32` default. The baseline is already optimized, so
> these deltas are incremental on top of that, not against stock upstream.

## Raw runs

| run | both tok/s | both dur (s) | both TTFT (ms) | stock tok/s | stock dur (s) | stock TTFT (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 17570.25 | 69.52 | 1647.95 | 18089.86 | 67.52 | 1503.78 |
| 2 | 18126.97 | 67.38 | 1578.01 | 18260.80 | 66.89 | 1756.90 |
| 3 | 18312.42 | 66.70 | 1514.51 | 17831.56 | 68.50 | 1482.91 |
| 4 | 17101.44 | 71.42 | 1619.31 | 18045.73 | 67.69 | 1539.80 |
| 5 | 17848.43 | 68.43 | 1645.90 | 18182.55 | 67.18 | 1551.46 |

## Summary (n=5 per arm)

| metric | both | stock | delta |
|---|---:|---:|---:|
| Total throughput (tok/s) | **17791.90** | **18082.10** | **−1.60%** |
| std / range | 477.6 · 17101–18312 | 162.9 · 17832–18261 | 2.9x spread |
| Benchmark duration (s) | 68.69 | 67.56 | +1.68% |
| Mean TTFT (ms) | 1601.14 | 1566.97 | +2.18% |
| Accept length | 3.38–3.40 | 3.39–3.42 | — |

## Interpretation

**No statistically significant mean difference.** Welch's t on throughput:
t ≈ −1.29, df ≈ 4.9, p ≈ 0.25. The distributions overlap heavily — the best
`both` run (18312) beats every `stock` run, and the worst `stock` run (17832)
beats two `both` runs. The nominal −1.60% is not resolvable at n=5.

**The variance difference is the more real finding.** `both` shows ~2.9x the
run-to-run spread of `stock` (std 478 vs 163; F ≈ 8.6, df 4,4, p ≈ 0.03
one-tailed), driven by two slow runs (17101, 17570). A GDN-only arm measured
the same day was tight by comparison — 18338.26 / 18223.82 / 18143.69,
mean 18235.26, std 97.8 — which points at the conv kernel rather than chunk_h
as the source of the instability. Suggestive, not proven: those GDN-only runs
came from a different server boot, and between-session drift on the stock arm
alone was ~1% (18082 here vs 18261 earlier the same day).

**This workload cannot resolve these kernels.** Both patch prefill only. From
the rocprofv3 trace (`local_bench/trace_analysis/report.txt`, run
`isl32768_osl4_c1`), `chunk_h` is 9.21% of prefill GPU time, and prefill is
~34% of an OSL=1024 run — so chunk_h is ~3% of total wall time. Decode never
calls it. The expected end-to-end effect is well under 1%, below this
benchmark's noise floor.

## Supporting measurement: prefill-dominant shape

`MODE=prefill CONC_LIST="4" RANDOM_RANGE_RATIO=0.8` (OSL=4), **GDN kernel
only**, n=3 per arm — the shape where chunk_h actually shows up:

| metric | FlyDSL varlen | tuned Triton | delta |
|---|---:|---:|---:|
| Input throughput (tok/s) | 40012.12 | 39769.53 | **+0.61%** |
| Mean TTFT (ms) | 2443.59 | 2468.28 | **−1.00%** |

Ranges do not overlap (39904–40085 vs 39727–39802), so this one is a real
effect. **The equivalent both-kernels prefill measurement was not run** — that
is the gap to close if the conv kernel needs its own number.

## Kernel coverage (both arm)

- `chunk_gated_delta_rule_fwd_h`: **39780 / 39780 calls (100%)**, zero fallbacks
  (counters in `chunk_gated_delta_rule/bootstrap/sitecustomize.py`).
- `causal_conv1d_fn`: hook confirmed active in all 4 TP workers. It replaces the
  function unconditionally with no fallback and raises `ValueError` on
  unsupported geometry, so a run that completes implies 100% usage.

Note: the two kernels' bootstrap directories both contain a file named
`sitecustomize.py`, and Python imports that name only once. Concatenating the
two `run_server.sh` PYTHONPATHs silently activates one kernel and drops the
other, with no symptom beyond a missing banner. `bootstrap_both/` exists to load
both hook files explicitly; use `run_server_both.sh`, not a merged PYTHONPATH.

## Caveats

- `SIMULATE_ACC=1` fakes MTP acceptance, and generation is **nondeterministic**:
  the same server returned 3 different completions for an identical
  `temperature=0` request. Accept length still varied 3.38–3.42 across runs,
  which changes the number of decode passes for a fixed 1024 output tokens and
  is a real confounder at this effect size.
- Correctness of the GDN varlen kernel is established separately and is not in
  question: `check_varlen.py` reports 16/16 cases **bitwise identical**
  (maxerr 0.0) to the Triton reference for `h`, `v_new` and the state update.
- The conv kernel's correctness was not re-verified here; see `conv/check_conv.py`.

## Conclusion

On `CONC_LIST="4" OSL=1024`, enabling both FlyDSL kernels produces **no
measurable throughput change** versus the already-tuned baseline. The kernels
are genuinely faster where they run (+0.61% input throughput, −1.00% TTFT on
prefill-only), but that slice is too small a fraction of this decode-dominated
workload to surface. If a regression is suspected, the elevated variance in the
`both` arm — most likely from the conv kernel — is the thing to chase, not the
mean.
