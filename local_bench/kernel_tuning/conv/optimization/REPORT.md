# Causal convolution optimization results

**English** | [中文](REPORT_zh.md)

Validated on AMD Instinct MI350X (gfx950), 2026-09-10, with the requested
FlyDSL authoring, prefetch, trace capture, and trace analysis skills.
All measurements below are GPU kernel microbenchmarks, not serving throughput.

## Implemented changes

- Load groups of raw input values before converting/consuming them. The selected
  default is `prefetch=16`; 1/2/4/8/16 remain available for experiments.
- Clamp unused tail-load addresses to the sequence's last valid token instead
  of branching around each load. The outer guard excludes empty sequences and
  inactive lanes; output masks still exclude tail tokens. This preserves packed
  sequence boundaries and allows the compiler to schedule independent loads.
- Issue weights before the halo is consumed. State snapshot/update ordering is
  preserved, and only chunk zero accesses cached history.
- Use a vector of two adjacent channels per thread for even, channel-contiguous
  BF16/FP16 input with at least 8192 packed tokens. Other inputs use one channel
  per thread. `channels_per_thread=1|2` can override the choice; explicit pairing
  requires an even channel count. Vector construction infers the input type so
  FP32 bias is not rounded through BF16.
- Retain block 128 and 16 tokens per block. The scalar tuning sweep did not find
  a consistent improvement from 64/256 threads or 8/32 tokens at prefetch 4.
  This was a bounded search, not an exhaustive joint search of all pair layouts
  and tile settings. Automatic pairing starts at 8192 tokens as a conservative
  policy between the tested short and long cases, not a measured exact crossover.

Original source SHA256:
`db24d3d85a11d799ad8e6308b93ed73e9fcc8555db7c934b34a8d21824fd4d3b`.
Optimized source SHA256:
`54ed68638a7866e7a24bd6fa0849c9fc6629708db486de0ea28f4bf9e08d53b8`.
[causal_conv1d_baseline.py](causal_conv1d_baseline.py) preserves the original
implementation. Intermediate candidate sources and timing JSONs are retained
here for reproducibility; the runtime imports only `../causal_conv1d_flydsl.py`.

## Performance

[before_after.json](before_after.json) compares original and optimized FlyDSL
within one process, using identical inputs and five alternating trials with
100 samples per backend per trial and 10 warmups. State resets occur outside
event timing. Each sample measures one graph replay; summaries are medians of
trial medians. Both implementations must first pass output/state checks.

| TP=4 input | Original FlyDSL | Optimized FlyDSL | Original/optimized |
| --- | ---: | ---: | ---: |
| 1 × 32768 | 198.042 µs | 128.341 µs | 1.543× |
| 1 × 4096 | 30.920 µs | 28.120 µs | 1.100× |
| 4 × 8192 | 198.562 µs | 129.761 µs | 1.530× |

[sglang_final.json](sglang_final.json) separately compares against installed
SGLang using three alternating trials, otherwise the same timing methodology:

| TP=4 input | SGLang | Optimized FlyDSL | SGLang/FlyDSL |
| --- | ---: | ---: | ---: |
| 1 × 32768 | 207.981 µs | 128.661 µs | 1.616× |
| 1 × 4096 | 32.040 µs | 28.080 µs | 1.141× |
| 4 × 8192 | 209.802 µs | 128.961 µs | 1.627× |

Early masked-prefetch variants gave only modest long-sequence improvements.
Clamped prefetch plus earlier weights improved the scalar 32K case to about
155 µs at prefetch 4 and 141 µs at prefetch 16. Pairing improved it further.
GPU timing varied between processes, so use the final alternating comparison
for the before/after claim, not ratios between exploratory runs.

## Correctness and trace evidence

[correctness.json](correctness.json) and [correctness.log](correctness.log)
record 52 passing cases plus optional-state checks. Coverage includes widths
2–5, BF16/FP16/FP32, odd channels/scalar fallback, paired channels with a partial
last wave, short/empty sequences, padded and indirect cache slots, mixed initial
history, excess state capacity, strided layouts, non-power-of-two token tiles,
graph replay/nondefault stream, and BF16 input with FP32 bias. The latter also
uses an offset input and odd token stride. Width 5 uses the independent PyTorch
reference because installed SGLang arithmetic supports widths 2–4. State matches
exactly; output tolerances remain unchanged.

[Optimized ATT](att_32k/ui_output_agent_24517_dispatch_153/),
[hotspot report](att_32k/analysis.txt), and
[code-object metadata](att_32k/code_object_metadata.txt) verify the implementation.
The capture repeats the original setup: matched invocation 3, warmed eager
32K call, CU 1 across four shader engines, source debug information enabled.

| Evidence | Original | Optimized |
| --- | ---: | ---: |
| VMEM-wait share of sampled stalls | 84.4% | 8.7% |
| VGPRs from code object | 19 | 64 |
| SGPRs from code object | 29 | 52 |
| Register spills / LDS / scratch | 0 / 0 / 0 | 0 / 0 / 0 |
| Static `s_waitcnt` instructions | 34 | 18 |
| Main input / output ISA | `global_load_ushort` / `global_store_short` | `global_load_dword` / `global_store_dword` |
| Static instruction count | 1207 | 1749 |

The optimized wave processes twice as many channels, so the larger static
instruction count does not imply more instructions per output. All 1749
instructions carry source annotations, but many vector operations map to the
Python `functools.py` wrapper rather than the user's arithmetic line. Treat that
as a source attribution artifact, not Python execution on the GPU. The remaining
dominant sampled stalls are packed arithmetic classified as `other`.

The skill analyzer's gfx942 label is a detection heuristic error for this
non-MFMA kernel; code-object metadata confirms gfx950. Its occupancy value is
an estimated resource ceiling, not measured residency. Stall totals sample
different wave populations after pairing and must not be compared as elapsed
time. The speedup claim comes from unprofiled timing, not stall percentages.

## Reproduce

From the `conv` directory:

```bash
python check_conv.py --output optimization/correctness.json
python optimization/compare_versions.py
python compare_conv.py --output optimization/sglang_final.json
python compare_conv.py --prefetch 8 --channels-per-thread 1 --output /tmp/conv_scalar.json

FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  rocprofv3 -i traces_conv/att.yaml -d optimization/att_repeat -- \
  python compare_conv.py --shapes qwen35-397b-tp4 --warmup 2 --repeats 2 --trials 1
```

English/Chinese usage guides were updated. The required repository performance
changelog entry was appended with every existing byte preserved; schema and
referenced config keys were validated. See [validation](changelog_validation.json).
Its `pr-link: XXX` is a placeholder because no PR was created. Full-model
startup, eval accuracy, and serving throughput have not been validated for this
optimization.
