# FlyDSL chunk-state experiment, 2026-09-09

**English** | [中文](FLYDSL_zh.md)

The replacement is numerically correct for the supported specialization, but
**does not beat the best Triton configuration found in this experiment**.
FlyDSL is 0.719 ms; transposed-grid, four-stage Triton is 0.720 ms. That
difference is inside the measurement spread. The 0.65 ms and 0.50 ms targets
were not reached. The historical 0.850 ms bar is exceeded, but comparing only
against that older number would overstate the result.

## Repeated device measurements

TP4: B=1, T=32768, Hg=4, H=16, K=V=128, BT=64. Each cell is the median of
60 device dispatches, excluding reference and warmup launches. Trials alternate
the configuration order. `results_flydsl/summary.json` contains all results;
each trial directory contains its exact command, worker log, and raw CSV.

| Configuration | Trial 1 ms | Trial 2 ms | Trial 3 ms | Median of medians |
|---|---:|---:|---:|---:|
| Triton BV16/w4/s3/wpe4 | 0.797 | 0.813 | 0.812 | 0.812 |
| Same, transposed grid | 0.777 | 0.784 | 0.783 | 0.783 |
| Triton BV16/w4/s4/wpe4 | 0.732 | 0.734 | 0.729 | 0.732 |
| Same, transposed grid | 0.721 | 0.719 | 0.720 | 0.720 |
| FlyDSL, two-chunk prefetch, XOR LDS | 0.719 | 0.719 | 0.717 | 0.719 |

Per-trial dispatch standard deviations are 0.003–0.005 ms. These are device
microbenchmarks, not serving throughput. A separate SGLang server was resident
during this session; GPU exclusivity was not established. The alternating,
paired trials help control drift but do not prove exclusive-device performance.

**Change 1, XCD mapping:** positive but modest. Grid transposition improves the
three-stage configuration by about 3.6% and the four-stage configuration by
about 1.6%, using the median-of-medians. It does not deliver the large gain that
the L2-lockstep hypothesis might suggest. No L2 hit-rate claim is made: PMCs
remain unavailable. The experiment changes only the two program IDs and host
grid in an isolated copy of the installed source; all three shapes have exact
`h`, `v_new`, and updated-state comparisons.

**Change 2, deeper Triton prefetch:** works. Four stages improve the original
grid by about 9.9% and the transposed grid by about 8.0%. It compiles with
137,088 bytes LDS and zero spills. The proposed `34,432 * stages` extrapolation
was close, but not exact: stages=3 uses 103,296 bytes and stages=4 uses 137,088,
not 137,728. The production package is unchanged. Its existing stages env var
can select s4, while `wpe=4` still requires the coupled BV16/w4 configuration.

**Change 3, LDS restructuring:** works within FlyDSL, but is insufficient for
a win over newly tuned Triton. The final kernel uses 38 KiB LDS, 80 VGPRs, and
zero scratch. It retains the external `(V,K)` state layout. Direct accumulator
stores eliminate the separate `h`-store staging pass. State and corrected
values use wide LDS copies with XOR permutations; `k` uses the gfx950
`LDSReadTrans16_64b` copy atom and a transpose-compatible swizzle. Two alternating
`k` buffers protect reads from next-iteration writes.

The supplied traffic claim also needs an arithmetic correction: the harness
models 2,684,354,560 total bytes, of which 1,979,711,488 are duplicate w/k reads
(73.75%, not 94%). This is the no-cache traffic model, not measured HBM traffic.

## Development measurements

These are individual 60-dispatch experiments, not three-trial comparisons.
They explain the implementation choices, with intermediate traces retained in
the session's `/tmp/gdn_fly_*_trace` directories.

| Variant | Device median ms |
|---|---:|
| Initial exact recurrence, no prefetch | 3.062 |
| One-chunk prefetch, original staging prototype | 0.882 |
| Two-chunk prefetch, alternating k buffers | 0.867 |
| Add hardware-transpose LDS reads | 0.844 |
| Four-chunk prefetch before swizzling | 0.877 |
| Guarded host entry, unswizzled LDS | 0.871 |
| XOR state/value buffers | 0.802 |
| Also XOR k buffer | 0.720 |
| Fully swizzled, three-chunk prefetch | 0.746 |
| Fully swizzled, one-chunk prefetch | 0.726 |

The first one-chunk prototype reused k storage without alternating buffers and
is not a supported implementation. Passing its numerical checks did not prove
the absence of a cross-wave lifetime race. The delivered implementation uses
two k buffers. Two-chunk prefetch remains the selected configuration.

## ATT measurements

The container lacked its decoder library. Official
`ROCm/rocprof-trace-decoder` version `0.1.4` was downloaded to `/tmp`; ROCm itself
was not modified. Both accepted captures use CU0, GPU0, SE mask 0x1. One initial
capture had no target MFMA samples and was rejected. Accepted code object 20
contains eight MFMA instructions per iteration and 16,384 MFMA issues, or
2,048 wave-iterations. Source statistics and normalized JSON are under
`results_flydsl/att_{unswizzled,swizzled}.{csv,json}`.

| Bucket | Unswizzled cycles/wave/iteration | XOR cycles/wave/iteration | XOR share |
|---|---:|---:|---:|
| vmcnt waits | 79 | 161 | 5.2% |
| lgkmcnt waits | 940 | 170 | 5.5% |
| barriers | 467 | 713 | 23.0% |
| VALU | 628 | 669 | 21.6% |
| global loads | 746 | 567 | 18.3% |
| LDS instructions | 499 | 268 | 8.6% |
| global stores | 163 | 399 | 12.8% |
| MFMA | 40 | 63 | 2.0% |
| other | 82 | 95 | 3.1% |

This table sums ATT instruction `Latency`, divided by wave-iterations; it
includes the small prologue/epilogue contribution. It is **not** the median
wall-clock iteration measurement used in the original 4,048-cycle table, so
the two totals must not be compared as identical metrics. `summarize_att.py`
records the normalization and rejects zero-MFMA input. The clear movement is
LDS wait latency, 940→170 cycles, and LDS instruction latency, 499→268. Both
versions issue exactly two barriers per iteration, but fewer barriers do not
guarantee low barrier cost: the final barrier bucket remains large. Attribution
also moves between load/store issue and waits, so a low vmcnt bucket alone does
not establish that global-memory latency has disappeared.

## Interface and correctness

`chunk_delta_h_flydsl.py` exposes `chunk_gated_delta_rule_fwd_h` with the installed
host signature and returns `(h, v_new)`. Supported geometry is gfx950, contiguous
BF16 k/w/u, K=128, V divisible by 16, and one sequence covering all T tokens,
where T is a positive multiple of 64. `cu_seqlens`, when supplied, must describe
that full sequence `[0,T]`; `chunk_indices` must be the corresponding production
metadata. Metadata values are not read back to the CPU during dispatch.
Multiple packed sequences and tails remain outside this specialization; the
import hook routes those shapes to the installed implementation.

Optional scalar/per-channel gates are contiguous FP32. State has contiguous
inner `[H,V,K]` dimensions and an arbitrary slot pitch. Negative slots do not
read or write state. Missing state starts from zero and performs no state write.
The 64-bit slot pointer is rebased **before** constructing a buffer resource;
putting the large pitch into a buffer offset was incorrect despite using int64
arithmetic. A real pitch of `2**31 + 512` elements is tested.

Bitwise equivalence required following compiled Triton arithmetic: the first
four MFMAs share one accumulator, and state scaling/addition uses explicit FMA.
The source's separate dot expressions do not imply separate rounded dot sums.

`check_chunk_h.py` passes 48 full-shape combinations (three shapes × four gate
modes × state on/off × saved values on/off), plus padded slots, indirect and
envelope-strided slots, three graph replays per shape, an independent FP32
one-chunk reference, and the >int32-offset regression. Initial state is nonzero.
Triton comparisons are exact; independent FP32 output/state checks use BF16
`atol=rtol=0.016`. State is restored before every launch, including graph replay.
Results are in `results_flydsl/correctness.json`.

## Reproduce and integrate

From `local_bench`:

```bash
python3 kernel_tuning/chunk_gated_delta_rule/check_chunk_h.py
python3 kernel_tuning/chunk_gated_delta_rule/run_experiments.py
rocprofv3 --kernel-trace --output-format csv -d /tmp/gdn-check -- \
  python3 kernel_tuning/chunk_gated_delta_rule/profile_flydsl.py --launches 60
```

For ATT, follow the existing `trace_att.py` flags, use `--kernel-include-regex
kernel_0`, and point `--att-library-path` at the installed decoder. The generic
FlyDSL symbol must be attributed to its actual dumped code object; reject
captures containing no target MFMA. Do not use `--att-serialize-all`.

`bootstrap/sitecustomize.py` activates only with `SGLANG_FLYDSL_GDN_CHUNK_H=1`.
`run_server.sh` supplies the flag and Python paths to fresh server workers:

```bash
TP=4 bash kernel_tuning/chunk_gated_delta_rule/run_server.sh
MODE=prefill ISL=32768 OSL=4 bash qwen3.5_fp4_sglang_bench.sh
```

The hook was checked in a fresh process. Full model startup and paired serving
throughput/TTFT were **not run**. No end-to-end speedup is claimed, and this
experimental kernel should not replace newly tuned Triton on performance grounds.
Do not combine this bootstrap with the conv bootstrap without merging their
`sitecustomize` hooks. The installed SGLang package and running server are unchanged.

The required FlyDSL `bash scripts/check_python_style.sh --fix` gate passed with
no changes in that checkout. New local Python files also received explicit
Black (line length 120) and Ruff E/W/F/I checks; the formatters were installed
under `/tmp` because they were initially absent.

## Environment

| Component | Observed value |
|---|---|
| GPU | AMD Instinct MI355X, gfx950:sramecc+:xnack- |
| FlyDSL | 0.3.1 |
| FlyDSL reference checkout | ac227c35029d2008282cbf035caa5923b769d37e |
| SGLang | 0.5.18.dev20260829+g4d53767b09 |
| Torch package metadata | 2.9.1+rocm7.2.0.lw.git7e1940d4 |
| Triton | 3.7.0+amd.rocm7.2.0.git89002410 |
| HIP | 7.2.26015-fc0010cf6a |
| ATT decoder | official rocprof-trace-decoder 0.1.4, temporary install |

These version strings do not establish an exact container digest.
