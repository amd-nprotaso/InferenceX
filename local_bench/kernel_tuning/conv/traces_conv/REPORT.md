# FlyDSL convolution trace findings

**English** | [中文](REPORT_zh.md)

Captured locally on 2026-09-10 using the requested FlyDSL
`capture-kernel-trace` and `kernel-trace-analysis` skills. Hardware: AMD Instinct
MI350X, gfx950. Kernel: `kernel_0` from `../causal_conv1d_flydsl.py`.
Source SHA256: `db24d3d85a11d799ad8e6308b93ed73e9fcc8555db7c934b34a8d21824fd4d3b`.
The kernel implementation was not changed.

## Capture and validation

All cases use Qwen3.5-397B TP=4 per-rank geometry: 3072 channels, BF16, width 4,
SiLU, no bias, fresh history, contiguous state `[1243,3072,3]`, block 128,
16 tokens per block. The packed case is synthetic. Each process validates eager
and graph outputs against SGLang, including exact state-pool equality.
All captures and the separate unprofiled comparison passed.

ATT selected matched invocation 3, a warmed eager call, with CU 1 selected in
each of four shader engines. Each decoded trace contains 1207 ISA instructions,
all source-mapped (1208 JSON rows including the kernel label). Durations below
come from dispatch 153 in each capture and include profiling conditions.

| Case | Workgroup grid | Traced duration | Sampled stall/total cycles | VMEM-wait/all stalls | Line 84/all stalls |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 × 32768 | `(2048,1,24)` | 193.082 µs | 82.4% | 84.4% | 76.13% |
| 1 × 4096 | `(256,1,24)` | 25.320 µs | 75.0% | 70.2% | 62.93% |
| 4 × 8192 | `(512,4,24)` | 194.282 µs | 82.4% | 84.8% | 76.24% |

rocprofv3 CSV grid sizes are in work-items; the table converts X by the
128-thread workgroup size. Do not interpret summed wave stall cycles as
device elapsed time or attainable speedup.

Decoded traces and detailed analyzer reports:

- [32K](tp4_32k/ui_output_agent_20469_dispatch_153/) and [analysis](tp4_32k/analysis.txt).
- [4K](tp4_4k/ui_output_agent_3884_dispatch_153/) and [analysis](tp4_4k/analysis.txt).
- [Packed 4](tp4_packed4/ui_output_agent_24805_dispatch_153/) and [analysis](tp4_packed4/analysis.txt).
- [L2 counters](pmc_32k/pass_1/l2_counter_collection.csv) and [analysis](pmc_32k/analysis.txt).
- [Code-object metadata](tp4_32k/code_object_metadata.txt).

The analyzer's gfx942 label is a heuristic error: it detects gfx950 only from
certain matrix instructions, which this depthwise kernel does not use. The
code object explicitly targets gfx950 and reports 19 VGPRs, 29 SGPRs, zero AGPRs,
zero LDS, zero scratch, and zero register spills. The dispatch CSV instead
reports VGPR_Count=12 and SGPR_Count=32; use code-object/ISA evidence for register
usage. The analyzer's occupancy figure is a resource ceiling estimate, not
measured residency. Its references to MFMA accumulator form are inapplicable
here: there are no MFMA instructions.

Separate single-pass PMC capture measured 1,623,271 `TCC_HIT_sum` and
3,442,362 `TCC_MISS_sum`: 32.0% aggregate L2 hit rate for one dispatch. This is
not input-only hit rate. No HBM byte counters were captured; bandwidth
saturation and over-fetch remain unmeasured. The analyzer's generic streaming
decode expectations do not apply to this prefill convolution.

## Prioritized optimization experiments

1. **Prefetch a small window of input tokens.** Line 84 loads a BF16 value
   immediately before use. The generated ISA has repeated `global_load_ushort`
   followed by `s_waitcnt vmcnt(0)`; this source line accounts for 76.13% of
   32K stalls. Start by issuing 2 or 4 independent future-token loads before
   computing the current convolution and SiLU. The loop is already unrolled,
   so a compile-time prefetch list/window is a natural first experiment.
   Preserve masks for short/tail sequences and preserve chunk-zero ownership
   of state reads/writes. Verify that generated ISA actually separates loads
   from consumption; a source reorder alone is not proof. Low current register
   usage permits experimentation, but check allocation and spills afterward.
   This follows the `prefetch-data-load` skill's latency-hiding principle;
   this kernel has less arithmetic to hide latency than its GEMM examples.

2. **Hoist weights and halo loads before consumption.** Lines 61 and 63
   contribute another 6.15% and 4.45% of 32K stalls. Issue independent weight,
   halo, and first-window loads before converting/using them, overlapping with
   metadata and address work where possible. The compiler already loads all
   four BF16 weights with a `global_load_dwordx2`; widening weight loads is
   therefore not the main opportunity. Maintain the short-sequence snapshot
   semantics and avoid inter-block state races.

3. **Try multiple adjacent channels per lane.** Current input/output operations
   are `global_load_ushort` / `global_store_short`. A 2-channel mapping could
   use packed 32-bit transactions per lane, reducing instruction count and
   exposing independent arithmetic. Current accesses are already coalesced
   across lanes; this is an instruction/parallelism experiment, not evidence
   of uncoalesced traffic. It requires channel-tail handling and different
   address/history bookkeeping; profile before accepting the extra complexity.

4. **Retune tile size after fixing load scheduling.** Test tokens per block
   8/16/32 and block sizes 64/128/256 with the comparison script. Larger token
   tiles amortize prologue and reduce halo duplication (3/16=18.75% extra
   input values per interior tile, versus 3/32=9.375%); these are logical loads,
   not measured HBM over-fetch. Larger tiles can also increase register use or
   lengthen serialized work, so changing `--tokens` alone is not a proven fix.

SiLU accounts for 4.52% of 32K stalls but 10.45% at 4K; consider its instruction
sequence only after the load experiments. LDS tiling and matrix instructions
are low priority: this kernel has no cross-channel reduction, already keeps
the rolling window in registers, and shows no LDS/barrier bottleneck.

Accept changes only when eager/graph correctness and exact state checks pass,
unprofiled alternating trials improve across relevant shapes, and a new ATT
capture confirms reduced waits without spills. Also run existing `check_conv.py`
coverage for short/empty sequences, history, padding, and supported layouts.
No speedup is predicted from stall percentages alone.

## Unprofiled baseline and reproduction

[unprofiled_baseline.json](unprofiled_baseline.json) records 3 alternating trials,
100 samples per backend per trial, and 10 warmups. State resets are excluded
from event timing but warm the cache.

| Case | SGLang | FlyDSL | SGLang/FlyDSL |
| --- | ---: | ---: | ---: |
| 1 × 32768 | 201.902 µs | 190.722 µs | 1.059× |
| 1 × 4096 | 32.001 µs | 31.000 µs | 1.032× |
| 4 × 8192 | 201.961 µs | 191.281 µs | 1.056× |

From the `conv` directory, choose a new output directory for a repeat capture:

```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  rocprofv3 -i traces_conv/att.yaml -d traces_conv/repeat_32k -- \
  python compare_conv.py --shapes qwen35-397b-tp4 --warmup 2 --repeats 2 --trials 1

python /var/home/conv_FlyDSL/FlyDSL/.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py \
  traces_conv/repeat_32k/ui_output_agent_* --kernel kernel_0 \
  --mode both --topk 12 --detail --context 2

rocprofv3 -i traces_conv/pmc_l2.yaml -d traces_conv/repeat_pmc -- \
  python compare_conv.py --shapes qwen35-397b-tp4 --warmup 2 --repeats 2 --trials 1
```

Use `qwen35-397b-tp4-4k` or `qwen35-397b-tp4-packed4` for the other captures.
Raw ATT, code objects, source snapshots, wave JSON, CSVs, and reports are retained
locally (about 453 MB in total). Kernel timings do not establish serving throughput.
