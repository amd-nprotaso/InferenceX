# ATT capture — `chunk_delta_h_flydsl.py`, Qwen3.5-397B-A17B-MXFP4 TP4 prefill

Instruction-level Advanced Thread Trace of the FlyDSL chunk-state kernel at the
production shape (`shapes.QWEN35_397B_TP4`: B=1, T=32768, Hg=4, H=16, K=128,
V=128, BT=64), captured on MI350X / gfx950 with ROCm 7.2, rocprofv3 1.1.0.

Unlike `trace_att.py` (CLI flags, per-instruction stats CSV), this capture uses
the `-i input.yaml` path so the decoder emits a `ui_output_agent_*_dispatch_*`
directory containing `code.json` — per-instruction stall/cycle data already
attributed to `chunk_delta_h_flydsl.py` source lines. That is the input format
the `kernel-trace-analysis` skill's `hotspot_analyzer.py` reads.

## Reproduce

```bash
mkdir -p /tmp/att_flydsl/run1 && cd /tmp/att_flydsl/run1
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i trace_input.yaml -- \
    python .../profile_flydsl.py --backend flydsl --shape qwen35-397b-tp4 --launches 5

python .../kernel-trace-analysis/scripts/hotspot_analyzer.py \
    ui_output_agent_<pid>_dispatch_<n> --kernel kernel_0 --topk 15 --mode both
python .../hotspot_analyzer.py ui_output_agent_<pid>_dispatch_<n> \
    --kernel kernel_0 --topk 6 --mode src --detail --context 3
```

`trace_input.yaml` (kept next to this file) pins CU 0, SE mask 0x1, GPU 0 — the
same sample the Triton captures in the sibling directories use, and for the same
reason: `--att-target-cu` defaults to 1, where the decoder resolves PCs into a
different code object and emits a well-formed but entirely false profile.
`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` is what makes `code.json`'s source column
non-empty; without it every hotspot lands on line 0.

The FlyDSL symbol is `kernel_0`. `kernel_iteration_range: "[4]"` traces one
dispatch past the warmup and the correctness comparison that
`profile_flydsl.py` runs first.

## Validity of this sample

* Decoded code object 20 contains the 8 `v_mfma_f32_16x16x32_bf16` of the loop
  body with 16,384 issues = **2,048 wave-iterations**. A capture with no MFMA in
  the target object is a mis-decode and must be rejected (see FLYDSL.md).
* 4 waves sampled — one per SIMD of one workgroup, `num_stitched == num_insts`
  (116,490) on all four, i.e. **no truncation**; each wave covers all 512 chunk
  iterations, and the traced duration (1.615 M cycles) matches the untraced
  761 us dispatch.
* 2048 wave-iterations / 4 waves = 512 = `T/BT`. Per-iteration figures below are
  `total / 2048`.

## Two things the tooling gets wrong here

1. `hotspot_analyzer.py` reports **gfx942 (CDNA3)**. This part is gfx950. Auto
   detection keys on gfx950-only MFMA opcodes and this kernel uses
   `v_mfma_f32_16x16x32_bf16`, which exists on both. The consequence is the LDS
   limit: the analyzer applies the CDNA3 64 KB budget to the kernel's 38,912 B
   workgroup and reports "1 wave/SIMD, bound by LDS". On gfx950's 160 KB LDS the
   real limiter is VGPR: `512/(159+0) -> 3 waves/SIMD` (LDS allows 4, SGPR 7).
2. Occupancy limits are moot at this shape anyway. The grid is
   `(H, V/16) = (16, 8) = 128 workgroups` on 256 CUs, so **one workgroup per CU
   and 1 wave/SIMD in practice** — which is exactly what the trace sampled.
   Register pressure is not what caps waves in flight here; workgroup count is.

## Results

Per wave-iteration (`total / 2048`), 3,052 cycles total of which 2,162 (70.8%)
are stall. Class table in `../../../att_flydsl_tp4.csv`.

| Stall bucket | cycles/wave-iter | share of stall |
|---|---:|---:|
| barrier | 713 | 32.9% |
| VMEM-load | 490 | 22.7% |
| VMEM-store | 332 | 15.4% |
| VMEM-wait (`s_waitcnt vmcnt`) | 170 | 7.9% |
| LDS/SMEM-wait (`s_waitcnt lgkmcnt`) | 169 | 7.8% |
| other (VALU/SALU) | 170 | 7.9% |
| LDS | 89 | 4.1% |
| MFMA | 29 | 1.4% |

MFMA latency is 61 cycles/wave-iteration out of 3,052 — 2% of the iteration.
Nothing about this kernel is compute-bound.

`load_vec` / `store_vec` / `mma` are shared `@flyc.jit` helpers, so every call
site collapses onto lines 24, 34 and 64. Splitting those by buffer resource
(`s[16:19]` carries the `j*32`-element immediates that only `w` produces;
`s[12:15]` recomputes its address per `j`, which only `k` needs; the
1-element BF16 `ushort` form is `u`) recovers the attribution:

| Tensor / op | ISA form | insts | stall | cycles/wave-iter |
|---|---|---:|---:|---:|
| `w` load (line 140) | `buffer_load_dwordx4` s[16:19] | 12 | 440.7K | 215 |
| `u` load (line 146) | `buffer_load_ushort` s[20:23] | 12 | 362.3K | 177 |
| `k` load (line 137) | `buffer_load_dwordx4` s[12:15] | 12 | 126.8K | 62 |
| `vn` store (line 198) | `buffer_store_short` s[28:31] | 4 | 447.1K | 218 |
| `h` store (lines 170-171) | `buffer_store_dwordx2` s[24:27] | 2 | 233.1K | 114 |
| state load/store (129-130, 247-248) | s[8:11] | 4 | 0 | 0 |
| barrier, line 179 | `s_barrier` | 1 | 1.03M | 503 |
| barrier, line 203 | `s_barrier` | 1 | 428.5K | 209 |

Each prefetch call site appears 3x in the ISA (two pre-loop stages plus the
in-loop one), hence 12 instructions for 4 loads.

`u` and `vn` touch `[.., H, V]` at `vb + col`: 16 lanes x 2 B = 32 B segments.
`h` is `[.., V, K]` at `(vb+col)*128 + kr`: 4 lanes x 4 elements = 32 B
segments. Every global access in the loop is a quarter of a 128 B line.

## Hardware counters are still unavailable

The 32 B-granularity question raised below (is the partial-line write traffic
actually reaching HBM, or does L2 combine it?) needs `--pmc` TCC counters.
`CapEff` in this container is `0xa80425fb`; CAP_PERFMON is not held, so PMC
collection is denied — the same limitation the README records. No PMC run was
attempted.
