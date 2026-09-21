# Results — Qwen3.8-Flash-Next MXFP4, MI355X

Reference run. Hardware: 1x AMD Instinct MI355X (gfx950), TP1.
Model: `amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8`.
SGLang `0.5.19.dev20260910+g908226fea2`, AITER `4ad998328`. Date: 2026-09-18.

`conv` = patches as originally authored (prefill conv only).
`both` = plus patch 0005 (prefill conv + GDN decode).

Single run per point — treat sub-1% differences as noise.

ISL 131072 / OSL 1024. Percentages vs baseline; **positive = better**.

### Throughput (total, tok/s)

| conc | baseline | conv | conv gain | both | both gain |
|---|---|---|---|---|---|
| 1 | 9698.54 | 9836.08 | **+1.42%** | 10151.91 | **+4.67%** |
| 4 | 18446.32 | 18580.78 | **+0.73%** | 19543.24 | **+5.95%** |
| 16 | 23728.38 | 23928.25 | **+0.84%** | 25035.61 | **+5.51%** |

### E2E latency (mean, ms)

| conc | baseline | conv | conv gain | both | both gain |
|---|---|---|---|---|---|
| 1 | 13618.02 | 13424.89 | **+1.42%** | 13009.61 | **+4.47%** |
| 4 | 28633.78 | 28418.89 | **+0.75%** | 27026.06 | **+5.61%** |
| 16 | 84847.62 | 84114.78 | **+0.86%** | 80406.09 | **+5.23%** |

### TTFT (mean, ms)

| conc | baseline | conv | conv gain | both | both gain |
|---|---|---|---|---|---|
| 1 | 4472.33 | 4438.91 | **+0.75%** | 4432.96 | **+0.88%** |
| 4 | 11517.85 | 11418.33 | **+0.86%** | 11360.77 | **+1.36%** |
| 16 | 44758.61 | 44314.98 | **+0.99%** | 43710.54 | **+2.34%** |

### TPOT (mean, ms)

| conc | baseline | conv | conv gain | both | both gain |
|---|---|---|---|---|---|
| 1 | 8.94 | 8.78 | **+1.75%** | 8.38 | **+6.22%** |
| 4 | 16.73 | 16.62 | **+0.67%** | 15.31 | **+8.48%** |
| 16 | 39.19 | 38.90 | **+0.72%** | 35.87 | **+8.46%** |

### Summary (range across concurrency)

| metric | conv gain | both gain |
|---|---|---|
| Throughput | +0.73% … +1.42% | +4.67% … +5.95% |
| E2E latency | +0.75% … +1.42% | +4.47% … +5.61% |
| TTFT | +0.75% … +0.99% | +0.88% … +2.34% |
| TPOT | +0.67% … +1.75% | +6.22% … +8.48% |

## Reading these numbers

**Throughput and E2E latency move together by construction.** ISL/OSL are fixed
and `ignore_eos` is on, so every request emits exactly 1024 tokens — total
throughput is a restatement of wall-clock. They are not independent
confirmations of each other.

**TPOT is where the decode kernel shows up** (+6.2% / +8.5% / +8.5%). That is
the metric to watch when tuning it.

**TTFT is the weak column, and that is the honest result.** The conv kernel is a
prefill kernel yet buys under 1% TTFT. `--chunked-prefill-size 16384` splits each
131072-token prompt into 8 chunks, so the conv never sees a long sequence, and at
this ISL prefill time is dominated by full-attention and MoE. The `both` TTFT
gain growing with concurrency (+0.88% → +2.34%) is not the conv kernel improving:
it is decode work from other in-flight requests interleaving with prefill, so a
faster decode shortens queueing ahead of the first token.

**The conc=16 point is partly queue-bound.** `max_total_num_tokens=1743360` fits
only ~13 requests of 131072 tokens, so the scheduler cannot hold all 16 in
flight. The same limit applies to every arm, so the comparison holds, but the
absolute numbers there are not a clean concurrency-16 measurement.

## Note on the "loses at high batch" assumption

The decode kernel's own docstring says it is ~1.2x Triton at concurrency 4 and
behind by concurrency 32, which is why it is opt-in. On Qwen3.8 it is still
+8.5% TPOT at concurrency 16. Qwen3.8 has 36 GDN layers at `linear_num_value_heads=48`,
so the decode recurrence is a much larger share of the step than on Qwen3.5 and
the K-split occupancy win keeps paying. Worth re-measuring the crossover on this
model before treating the opt-in default as settled.
