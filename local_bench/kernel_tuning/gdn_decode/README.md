# GDN decode recurrence — benchmark harness and BV findings

The prefill half of the GDN kernel family has a full bench/compare/ATT pipeline
under `../chunk_gated_delta_rule/`. The decode half had none. This directory is
its twin, targeting
`sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update`.

Motivation, measured from a serving trace (ISL 32768 / OSL 1024 / conc 4, TP=4):

| | GPU time | share |
|---|---:|---:|
| GDN **decode** (6 kernels, 73.5k launches) | 505.4 ms | 6.1 % |
| GDN **prefill** (4 kernels, 900 launches) | 242.2 ms | 2.9 % |

Decode is the larger of the two at OSL 1024 and, unlike prefill, had never been
tuned or ported.

## Files

| file | what it does |
|---|---|
| `decode_shapes.py` | Production decode geometries (TP=4: Hg=4, HV=16, K=V=128) at several concurrencies, plus grid/wave-slot arithmetic |
| `bench_decode.py` | Correctness-gated sweep of `BV` / `num_warps` / `num_stages` using CUDA events |
| `device_timing.py` | The same sweep timed **device-side** via `rocprofv3 --kernel-trace` |

SGLang is never edited. Both tools load a source-patched copy of the installed
wrapper (the trick `../chunk_gated_delta_rule/profile_flydsl.py` uses for its
"permuted" arm) with `BV` / `num_warps` / `num_stages` read from the
environment; everything else is stock.

## Use `device_timing.py` for timing, `bench_decode.py` for correctness

This matters enough to state plainly. The kernel is ~10 µs device-side, while
one Python-side invocation costs ~33 µs of CPU (`o = q.new_empty(...)`, argument
marshalling, Triton launch). CUDA events around the call therefore measure the
**launch path**, and the GPU starves. Bursting does not rescue it — the CPU
still cannot feed the GPU. The symptom is unmistakable and worth recognising:

```
# bench_decode.py, idle GPUs — every config identical, BV apparently irrelevant
  BV  warps    WGs        ms  speedup
  32      1    256    0.0337    1.000   <- stock
   8      1   1024    0.0339    0.995
   4      4   2048    0.0349    0.966
```

Under `rocprofv3 --kernel-trace`, which reports device begin/end timestamps and
excludes launch overhead, the same configs separate cleanly — and stock lands at
10.24 µs, matching the 10.72 µs seen in the serving trace:

```
  BV  warps    WGs  wave slot%       us  speedup     GB/s
  32      1    256        3.1%    10.24    1.000    409.6   <- stock
  16      1    512        6.2%     8.04    1.274    521.7
   8      1   1024       12.5%     6.76    1.515    620.5   <- best
   4      1   2048       25.0%     7.56    1.354    554.8
```

`bench_decode.py` is still the right tool for correctness: it checks each
candidate's output **and** its in-place write-back to the state pool against
stock.

## Finding: `BV` should track the batch, not be a constant

The wrapper hardcodes

```python
BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
num_stages = 3
num_warps  = 1
grid       = (NK, NV, N * HV)      # NV = cdiv(V, BV)
```

`BV` is the only knob that changes the grid, and the grid is the whole story:
these are single-wave workgroups, so at conc 4 the launch is 256 waves against
a 256-CU × 4-SIMD × 8-slot machine — about 3 % of the wave slots.

Device-side optimum by concurrency (`num_warps=1` everywhere; raising it always
hurt, a sequential scan gains nothing from more waves per workgroup):

| shape | N·HV | stock BV=32 | best BV | best WGs | speedup |
|---|---:|---:|---:|---:|---:|
| conc 4, 4 draft tokens | 64 | 10.24 µs | **8** | 1024 | **1.52×** |
| conc 4, no spec | 64 | 4.72 µs | **8** | 1024 | 1.16× |
| conc 8, 4 draft tokens | 128 | 10.34 µs | **16** | 1024 | 1.27× |
| conc 32, 4 draft tokens | 512 | 16.72 µs | **64** | 1024 | 1.05× |

The best config at every shape is the one whose grid lands near **1024
workgroups** — 4 single-wave workgroups per CU. That suggests the fix is not a
new constant but a policy:

```python
BV = clamp(V / (TARGET_WGS / (N * HV)), 4, 64)   # TARGET_WGS ~ 1024, power of 2
```

A fixed `BV=32` happens to be near-optimal at conc 32 and costs 1.5× at conc 4,
which is why this never showed up in high-concurrency tuning.

Changing `BV` is **bit-exact**: it re-blocks an independent axis of `V`.
Verified at `--rtol 0 --atol 0` for both the returned output and the state pool.

## The FlyDSL decode kernel

`gdn_decode_flydsl.py` is a from-scratch FlyDSL port of the recurrence, checked
by `check_decode.py` and timed by `compare_decode.py`.

It splits **K across lanes** instead of keeping it inside one lane:

    lane owns  v = vb + tid // KSPLIT,  k in [ (tid % KSPLIT) * KLOCAL, +KLOCAL )

so `KSPLIT` lanes cooperate on one v column and the two reductions over K (the
delta-rule projection and the output projection) plus the two l2-norms become
`log2(KSPLIT)` `shuffle_xor` steps. No LDS, no barriers. At the default
KSPLIT=8 the launch is 256 workgroups x 256 threads = 1024 waves at concurrency
4, 4x Triton's 256.

Correctness (`check_decode.py`, output **and** state write-back, 6/6):
errors are at bf16 rounding — 3e-5 output / 4e-3 state — not bit-exact, because
the transcendentals and the reduction order differ from Triton.

Device-side, vs stock Triton:

| shape | Triton | FlyDSL | speedup | Triton waves | FlyDSL waves |
|---|---:|---:|---:|---:|---:|
| **conc 4, 4 draft** | 9.46 µs | **7.76 µs** | **1.22x** | 256 | 1024 |
| conc 8, 4 draft | 10.44 µs | 9.64 µs | **1.08x** | 512 | 2048 |
| conc 4, no spec | 4.74 µs | 4.80 µs | 0.99x | 256 | 1024 |
| conc 32, 4 draft | 16.72 µs | 23.74 µs | **0.70x** | 2048 | 8192 |

VGPR 40, zero LDS, zero scratch.

**It wins at low concurrency and loses at high.** The reason is the same lever
in both directions: the decomposition spends 4x the lanes on the same output.
When the GPU is half idle (conc 4) those lanes are free and the extra
parallelism is the whole point. When it is already full (conc 32, 8192 waves)
they are not free — the per-token gate math (`sqrt` x2, `exp` x3, `log`) is
wave-uniform and gets computed 4x over, and each v group re-loads the same q/k
row. Sweeping KSPLIT over 4/8/16 does not rescue conc 32 (0.71x / 0.70x /
0.54x): it trades which axis is oversubscribed, not the total redundancy.

The fix is to give each lane **several v columns** instead of one, so wave count
stays near 1024 as batch grows. That is the next change, and it would also
amortize the q/k load and the gate math across those columns.

Until then the useful policy is conditional: FlyDSL below roughly concurrency 8,
Triton above. `compare_decode.py` is the tool for re-deciding that threshold.

### FlyDSL constraints discovered here

Both cost real debugging time and are worth not rediscovering:

1. **A global store inside a runtime `range(...)` carry loop segfaults the
   compiler.** Not the conditional, not the operand — any store. The decode
   kernel sidesteps it by baking `T` (the draft-token count, a small constant)
   into `build` and unrolling with `range_constexpr`, which also schedules
   better. The wrapper therefore requires a uniform sequence length.
2. **Predicate stores by address, not control flow.** Steer non-writing lanes
   past the buffer's `num_records`; the hardware drops them. This *requires*
   `make_buffer_tensor(..., max_size=False, num_records_bytes=...)` — with the
   default unbounded descriptor the steered stores are not dropped, they
   scribble far past the tensor and fault.
3. Statement-level `for` in a kernel body is rewritten into a runtime `scf.for`
   (making the index a DSL value); list comprehensions and `functools.reduce`
   are not. Use those, or `range_constexpr`, for compile-time unrolling.

## Headroom beyond BV

Even the best config moves 620 GB/s of SSM state against MI355X's ~8000 GB/s
peak — under 8 %. The BV fix buys occupancy, not efficiency. The structural
target is the five-kernel decode chain, which round-trips HBM between each stage:

| kernel | ms (OSL 1024) | mean |
|---|---:|---:|
| `fused_sigmoid_gating_delta_rule_update` | 146.1 | 10.72 µs |
| `_fused_gate_sigmoid_mul_add` | 132.1 | 7.17 µs |
| `fused_qkvzba_split_reshape_cat_contiguous` | 87.5 | 6.33 µs |
| `fused_qkv_split_gdn_prefill` | 74.4 | 5.38 µs |
| `_causal_conv1d_update` | 62.0 | 4.55 µs |

240 launches per decode step across 45 layers. Fusing the chain so the state
stays in registers/LDS across gating → conv-update → recurrence → gate is the
next real change.

## Reproduce

```bash
# correctness-gated sweep (CUDA events; use only for correctness)
python3 bench_decode.py --shapes qwen35-397b-tp4-c4 --rtol 0 --atol 0

# device-side timing (the number to trust)
python3 device_timing.py --shape qwen35-397b-tp4-c4 --launches 100
python3 device_timing.py --shape qwen35-397b-tp4-c32 --bv 128 64 32 16
```

Run with the GPUs idle. A live SGLang server holding 74 % of VRAM inflated every
config by ~3× and compressed the differences between them.
