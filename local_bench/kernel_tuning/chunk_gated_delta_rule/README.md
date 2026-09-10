# chunk_gated_delta_rule_fwd_kernel_h_blockdim64 — tuning harness

## FlyDSL rewrite results (2026-09-09)

The validated FlyDSL specialization reaches **0.719 ms**, but ties the newly
tuned **0.720 ms** Triton configuration (transposed grid, BV16/w4/s4/wpe4).
The <=0.65 ms target was not reached. Grid transposition helps modestly; four
Triton stages help consistently; XOR LDS staging is essential to the rewrite.

See [the complete FlyDSL report](FLYDSL.md) / [中文报告](FLYDSL_zh.md) for
independent before/after measurements, the regenerated ATT budget, environment
versions, all correctness checks, supported shapes, and opt-in integration.
The raw paired measurements are in `results_flydsl/summary.json`. These are
kernel microbenchmarks; end-to-end serving validation remains unperformed.


Standalone bench + config sweep for the GDN chunk-state Triton kernel, at the
exact shapes Qwen3.5-397B-A17B runs during a 32768-token prefill.

## Why this kernel

Measured on matched TP4 traces (`isl32768_osl100_c4_amd` vs `..._nv`), per
prefill forward:

| | MI350X | GB200 |
|---|---:|---:|
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 69.16 ms / 45 calls | 46.96 ms / 45 calls |

Same Triton source, same call count, same input shapes — 1.47x slower on AMD.
It is not a code-quality gap. It is launch geometry:

| | MI350X | GB200 |
|---|---|---|
| grid | `[4, 16, 1]` = 64 WGs | `[4, 16, 1]` = 64 blocks |
| block | `[256, 1, 1]` (4 warps x wave64) | `[128, 1, 1]` (4 warps x warp32) |
| compute units | 256 CUs | 148 SMs |
| **device fill** | **25%** | **43%** |

43/25 = 1.72x utilization ratio vs 1.47x measured performance ratio. The gap is
device fill, essentially in full. The kernel writes 268 MB of `h` in ~1.4 ms =
~300 GB/s on an ~8 TB/s part, so it is nowhere near bandwidth-bound — occupancy
is the only thing holding it back.

Grid is `(cdiv(V, BV), N*H)` and nothing else (`chunk_delta_h.py:355`). The T
axis carries recurrent state and cannot be parallelized. With V=128 and H=16,
**BV is the only knob that changes workgroup count.**

## Results (MI350X, 15 reps, production shape)

```
config                 ms      speedup   WGs   blk  fill%    GB/s   maxerr
BV=16 w=4 s=3 wpe=4  0.880      1.53x    128   256   50.0     458  0.00e+00
BV=16 w=4 s=3 wpe=3  0.927      1.45x    128   256   50.0     435  0.00e+00
BV=16 w=4 s=3        0.951      1.42x    128   256   50.0     423  0.00e+00
BV=16 w=2 s=3        1.023      1.32x    128   128   50.0     394  0.00e+00
BV=16 w=4 s=2        1.102      1.23x    128   256   50.0     365  0.00e+00
BV=32 w=2 s=3        1.111      1.22x     64   128   25.0     362  0.00e+00
BV=32 w=4 s=2        1.351      1.00x     64   256   25.0     298  <- production default
BV=16 w=4 s=3 wpe=6  1.697      0.79x    128   256   50.0     250  0.00e+00
BV=8  w=2 s=3        1.604      0.84x    256   128  100.0     251  1.88e-01
```

Every winning config is **bitwise identical** to the baseline
(`maxerr 0.00e+00`) — expected, since BV only partitions the V axis and each V
row accumulates independently.

**Shippable today, 1.42x**, env only:

```bash
SGLANG_GDN_CHUNK_H_BV=16 SGLANG_GDN_CHUNK_H_NUM_WARPS=4 SGLANG_GDN_CHUNK_H_NUM_STAGES=3
```

`num_stages=3` is worth almost as much as BV here (BV=32 s=3 alone is 1.09x, and
every top config uses s=3). The sglang default is 2.

### waves_per_eu: +1.08x, but do not ship it loose

`waves_per_eu=4` takes the winner from 0.951 to 0.880 ms (1.53x total). It is an
AMD backend compile option with no sglang env knob, so shipping it means editing
the `triton.Config` at `chunk_delta_h.py:41`.

**Confirmed at 60 reps x 3 independent trials** (2026-09-07), because a 15-rep
run had suggested only 1.02x and that turned out to be a sampling artifact:

| | trial 1 | trial 2 | trial 3 |
|---|---:|---:|---:|
| `wpe=4` | 0.882 | 0.891 | 0.885 |
| `wpe=2` | 0.951 | 0.970 | 0.972 |
| `wpe=1` | 0.959 | 0.972 | 0.958 |

0.960 -> 0.885 ms = **1.085x**, std +/-0.010 or better on every row. The original
1.08x number is correct. Do not re-litigate it at low rep counts: below ~30 reps
the wpe=4 / wpe=1 gap is inside the noise and the sign is not even stable.

**It is only valid coupled to `BV=16, num_warps=4`.** Everywhere else it is
neutral or catastrophic:

| config | no wpe | wpe=4 | |
|---|---:|---:|---|
| BV=16 w=4 | 0.970 ms | **0.898 ms** | 1.08x better |
| BV=16 w=2 | 0.995 ms | 1.261 ms | 1.27x worse |
| BV=32 w=4 | 1.235 ms | **4.407 ms** | **3.6x worse** |
| BV=32 w=2 | 1.117 ms | **4.867 ms** | **4.4x worse** |

So a hardcoded `waves_per_eu=4` in the Config is a live footgun: any deployment
where BV falls back to the 32 default — an unset env var, a different arch
branch, someone reverting the BV change alone — silently gets a kernel **3.6x
slower than today's baseline**. If it ships, it has to ship as one atomic
(BV=16, warps=4, wpe=4) triple with an assert, not as three independent knobs.

The value curve is also sharp and non-monotonic even at BV=16/w=4: 3 -> 0.927,
**4 -> 0.880**, 6 -> 1.697, and **8 aborts the Triton compiler outright** (LLVM
assertion in `Sequence.h:275`, SIGABRT, core dump — which is why `--isolate`
exists).

Given 1.08x upside against that blast radius, the env-only `BV=16 / s=3` at
1.42x is the change to land first.

### The important negative result

**`BV=8` reaches 100% fill and is 1.7x *slower* than `BV=16` at 50% fill.**

So "more workgroups" is not the objective on its own — the `[BV, 64]`
accumulator tile gets too narrow for the MFMA path below BV=16. See "What
actually limits this kernel" for the per-workgroup numbers that quantify it.

## What actually limits this kernel

Measured across three shapes x three BV values, with the full traffic model
(`shapes.py: total_bytes`, which counts the `grid.x`-fold re-read of `w` and `k`
that the old `h + v_new` figure ignored):

| shape | BV | WGs | MB moved | ms | TB/s | **GB/s per WG** |
|---|---:|---:|---:|---:|---:|---:|
| TP4 | 16 | 128 | 2684 | 0.936 | 2.87 | **22.4** |
| TP4 | 32 | 64 | 1611 | 1.247 | 1.29 | **20.2** |
| TP4 | 8 | 256 | 4832 | 1.717 | 2.81 | **11.0** |
| TP2 | 16 | 256 | 5369 | 1.011 | 5.31 | **20.7** |
| TP2 | 32 | 128 | 3221 | 1.320 | 2.44 | **19.1** |
| TP2 | 8 | 512 | 9664 | 3.417 | 2.83 | **5.5** |
| TP4-4k | 16 | 128 | 336 | 0.144 | 2.33 | **18.2** |
| TP4-4k | 32 | 64 | 201 | 0.194 | 1.04 | **16.2** |

Two things fall out, and the last column is the one that matters:

**1. Every workgroup delivers ~20 GB/s at BV>=16, and aggregate throughput is
just that times the workgroup count.** This holds across a 16x range of problem
size and both TP configurations. The kernel is per-workgroup latency-bound;
device fill is the binding constraint, exactly as this README originally said.

**2. There is no global bandwidth wall.** TP4 at BV=16 tops out at 2.87 TB/s,
which looks like a ceiling until TP2 at BV=16 — same tile, same 256-thread
block, twice the workgroups — reaches **5.31 TB/s**. Nothing saturates at 2.87;
TP4 simply runs out of workgroups at 128 of 256 CUs.

That also settles BV=8: its per-workgroup rate is **half** everyone else's
(11.0 vs 22.4 GB/s). It is not losing to memory traffic, and 512 workgroups at
TP2 does not rescue it (5.5 GB/s per WG, same 2.8 TB/s aggregate). The tile is
the problem, as originally suspected.

The practical consequence is encouraging: **TP2 proves the kernel scales
~linearly to 256 workgroups at BV=16.** If TP4 could be given 2x the workgroups
without shrinking the tile, ~1.8x is on the table on top of the current 1.46x.

### Why the K-split cannot work

It is not unbuilt, it is architecturally impossible, and it should not be
attempted:

- `b_v` is accumulated over **every** K block —
  `b_v = Σᵢ b_wᵢ @ b_hᵢᵀ` (`chunk_delta_h.py:178-196`)
- that same `b_v` then updates **every** `b_hᵢ` (`:275-293`)

The K blocks are fully coupled in both directions once per chunk, so splitting K
across workgroups needs a grid-wide reduction on all NT=512 iterations. K=128
here anyway, so the ceiling would have been 2x even if it were free.

V is exhausted (BV=8 halves per-WG rate) and `N*H = 16` is fixed by the
deployment. **The only remaining axis is T** — a two-pass chunked scan, splitting
the 512 chunks into S segments computed from a zero state and then combined
through the cumulative decay. The recurrence `h_{t+1} = a_t·h_t + b_t` is
associative, so this is mathematically sound; it is a real rewrite and it costs
an extra pass over `h`, but it is the only thing left that adds workgroups
without shrinking the tile.

### A note on the traffic model

`shapes.py` now models full traffic and `sweep.py` reports it as `TB/s`, with
the old `h + v_new` figure kept as the `h+v` column. This was worth fixing —
the old column made BV=8 look 1.8x worse on bandwidth when it moves 1.8x more
bytes at a similar aggregate rate — but note that **total bytes moved does not
predict runtime here**. TP2 at BV=16 moves exactly 2x TP4's bytes in 1.08x the
time. Workgroup count predicts runtime; byte count does not.

## Layout

| file | what |
|---|---|
| `shapes.py` | Model shapes read from the trace; builds inputs via the production preamble; full traffic model |
| `bench.py` | Times and verifies one Triton config against the production kernel |
| `sweep.py` | CLI driver, comparison table, CSV |
| `trace_kernel.py` | rocprofv3 `--kernel-trace` capture — device-side dispatch time, launch geometry, LDS/register footprint |
| `trace_att.py` | rocprofv3 `--att` capture — per-instruction stall profile on one CU |
| `profile_traffic.py` | rocprofv3 counter collection — **currently blocked, see below** |

## rocprofv3 kernel traces

`./trace_kernel.py` wraps the same production call path in
`rocprofv3 --kernel-trace`, one process per config, and reports the *device-side*
dispatch duration rather than sweep.py's CUDA-event timing. Traces land under
`traces/<shape>/<config>/` as CSV and, with `--pftrace`, as Perfetto files.

Captured 2026-09-08 on an idle machine, 20 launches per config (first dropped as
warmup):

| shape | config | median ms | ± | speedup | WGs | fill% |
|---|---|---:|---:|---:|---:|---:|
| TP4 | BV=32 w=4 s=2 | 1.327 | 0.006 | 1.00x | 64 | 25 |
| TP4 | BV=16 w=4 s=3 | 0.905 | 0.007 | **1.47x** | 128 | 50 |
| TP4 | BV=16 w=4 s=3 wpe=4 | 0.850 | 0.005 | **1.56x** | 128 | 50 |
| TP4 | BV=8 w=2 s=3 | 1.565 | 0.005 | 0.85x | 256 | 100 |
| TP2 | BV=32 w=4 s=2 | 1.394 | 0.006 | 1.00x | 128 | 50 |
| TP2 | BV=16 w=4 s=3 | 0.982 | 0.036 | 1.42x | 256 | 100 |
| TP2 | BV=16 w=4 s=3 wpe=4 | 0.953 | 0.035 | 1.46x | 256 | 100 |
| TP4-4k | BV=32 w=4 s=2 | 0.143 | 0.001 | 1.00x | 64 | 25 |
| TP4-4k | BV=16 w=4 s=3 | 0.096 | 0.001 | 1.49x | 128 | 50 |
| TP4-4k | BV=16 w=4 s=3 wpe=4 | 0.090 | 0.001 | **1.60x** | 128 | 50 |

Every number above corroborates the event-timed sweep, at roughly a quarter the
run-to-run spread — the trace excludes launch overhead, so ±0.005 ms is typical
instead of ±0.03. The 1.42x / 1.53x headline figures hold, and they hold at all
three shapes.

### What the trace adds that the sweep could not see

The resource footprint is identical across all three shapes, so this is a
property of the compiled config, not the problem size:

| config | VGPR | SGPR | regs | spills | LDS | **WG/CU allowed** | scratch |
|---|---:|---:|---:|---:|---:|---:|---:|
| BV=32 w=4 s=2 | 108 | 96 | 214 | 0 | 74.4 KB | **2** | 0 |
| BV=16 w=4 s=3 | 64 | 112 | 124 | 0 | 100.9 KB | **1** | 0 |
| BV=16 w=4 s=3 wpe=4 | 56 | 112 | 110 | 0 | 100.9 KB | **1** | 0 |
| BV=8 w=2 s=3 | 104 | 112 | 201 | 0 | 98.0 KB | **1** | 0 |

Three things fall out:

**1. Nothing spills.** Zero scratch, zero spills, in every config. So the whole
BV story really is geometry, and no part of it is register pressure — which had
been an open possibility for BV=16's narrower tile.

**2. LDS caps BV=16 at one workgroup per CU.** 100.9 KB against 160 KB per CU
means two will never fit. At 128 workgroups on 256 CUs that is invisible today
(one per CU either way), but it forecloses "raise occupancy by co-residency" as
an alternative to raising workgroup count. The T-split remains the only path.

**3. `waves_per_eu=4` works by shrinking the register allocation** — 64 -> 56
VGPRs, 124 -> 110 total — and it buys the 1.08x for free, with no spill. That
also explains its blast radius at BV=32: 214 regs has no comparable slack, so
forcing more waves there has to come out of something else.

## ATT: what the workgroup is actually waiting on

`./trace_att.py` runs the same call path under

```
rocprofv3 --att=true --att-library-path /opt/rocm/lib
```

Advanced Thread Trace samples the instruction stream on **one CU** and reports,
per instruction, issue count, latency and stall cycles. Unlike `--pmc` it needs
no elevated capability, so it works here — and it answers the question the whole
directory has been circling: this README asserts the kernel is *per-workgroup
latency-bound*, inferred from scaling behaviour. ATT shows it directly.

Stall cycles by instruction class, production TP4 shape:

| class | BV=32 w=4 s=2 | BV=16 w=4 s=3 | BV=16 w=4 s=3 wpe=4 | BV=8 w=2 s=3 |
|---|---:|---:|---:|---:|
| mfma | 2.9% | 0.9% | 1.4% | 1.0% |
| vmem_ld | 3.0% | 6.3% | 2.2% | 0.9% |
| lds | 19.6% | 6.0% | 5.8% | 0.7% |
| **wait** | **60.7%** | **77.0%** | **82.6%** | **86.0%** |
| valu | 10.3% | 9.3% | 7.8% | 7.8% |

And it reproduces at the other two shapes — TP2 gives 63.5% / 79.7% `wait` for
the two main configs, TP4-4k gives 60.5% / 70.7%. The profile is a property of
the config, not the problem size.

**The kernel does essentially nothing but wait.** MFMA stall never exceeds 3%.
The single largest stall site in every config is one `s_waitcnt vmcnt(N)`:

| config | top stall site | cycles | share of all stall |
|---|---|---:|---:|
| BV=32 w=4 s=2 | `s_waitcnt vmcnt(4)` | 1,183,912 | 16% |
| BV=16 w=4 s=3 | `s_waitcnt vmcnt(5)` | 1,771,820 | 31% |
| BV=16 w=4 s=3 wpe=4 | `s_waitcnt vmcnt(3)` | 1,972,692 | 41% |
| BV=8 w=2 s=3 | `s_waitcnt lgkmcnt(0)` | 260,544 | 11% |

This is the direct confirmation the traffic-model work could only reach by
inference: it is global-load *latency*, not bandwidth and not compute. It also
explains why `num_stages=3` is worth nearly as much as BV — more stages is more
outstanding loads to hide that wait behind.

Two further readings:

**BV=32's LDS stall is 19.6%, and BV=16 cuts it to 6%.** That is a third of
BV=32's non-wait stall disappearing, and it is a second, independent reason
BV=16 wins beyond the workgroup count. The narrower accumulator tile puts less
pressure on LDS.

**BV=8 pushes `wait` to 86% with LDS stall at 0.7%.** It is not losing to LDS or
to memory traffic; it is a 128-thread block with only 2 waves per CU and nothing
left to hide latency behind. That is the per-workgroup-rate halving from the
traffic table, seen from the instruction side.

Compare stall *shares* across configs, never stall *totals*: the totals sum over
whatever waves the traced CU held, and that count changes with the config
(BV=32 puts 2 workgroups x 4 waves on a CU, BV=8 puts 1 x 2).

### Two ATT traps, both of which produce output that looks fine

**1. `--att-target-cu` defaults to 1, and CU 1 is wrong here.** At CU 1 the
decoder resolves the sampled PCs into a *different code object* — torch's, not
Triton's — and emits a complete, well-formed, entirely false profile: 927
instructions, no MFMA, no LDS, 99.9% of stall on a single `s_waitcnt vmcnt(0)`.
It does not look like garbage; it looks like a memory-bound kernel, which is
roughly the conclusion you were hoping for. `trace_att.py` defaults to CU 0,
filters the stats to the code object that actually holds the kernel symbol, and
rejects any decode containing zero MFMA, since this kernel provably has some.

**2. ATT arms all four agents unless told otherwise.** The app runs on one, and
the decoder writes its stats CSV for whichever agent it enumerated last — an
empty trace. The tell is three 424-byte `.att` files next to one real one.
`--att-gpu-index` pins it.

Also: whether the traced window lands on the target dispatch or the one before
it is **not deterministic** — roughly one run in three misses, and more often on
the short 4k shape. `trace_att.py` retries. And `--att-serialize-all` aborts the
profiler on this workload; do not enable it.

### A note on the LDS column

`rocprofv3`'s `LDS_Block_Size` column reads **0** for every dispatch of this
kernel, which is wrong — Triton requests LDS as a dynamic group segment and that
column only counts the static block. `trace_kernel.py` therefore takes LDS,
`n_regs` and `n_spills` from the compiled Triton kernel's metadata and only the
timing and grid geometry from the trace. Do not read the raw CSV's LDS column.

### Hardware counters still do not work in this container

Re-confirmed 2026-09-08: `rocprofv3 --pmc FETCH_SIZE WRITE_SIZE` on a bare
matmul hung until killed at 120 s and wrote no CSV, while `--kernel-trace` on
the same script finished in 3.8 s.

`profile_traffic.py` would settle whether the redundant `w`/`k` re-reads reach
DRAM or are absorbed by the 256 MB Infinity Cache, by comparing rocprofv3
`FETCH_SIZE` against the model. It cannot run here:

```
perf_event_paranoid = 4
CapEff = 0xa80425fb     # no CAP_SYS_ADMIN (bit 21), no CAP_PERFMON (bit 38)
```

`rocprofv3 --pmc ...` hangs indefinitely — reproducible on a bare matmul, so it
is not this kernel. `--kernel-trace` works fine, so rocprofv3 itself is
functional; it is counter collection specifically that is denied. Unblocking it
needs the container relaunched with `--cap-add=CAP_PERFMON` (or `SYS_ADMIN`) and
`perf_event_paranoid` lowered — a host-side change.

The script is left in place and ready. In the meantime the question was answered
indirectly, by the TP2/TP4/4k scaling comparison above, which shows the answer
does not change the conclusion: workgroup count predicts runtime, byte count
does not.

## Usage

```bash
./sweep.py                                   # 48-config default sweep (~30 s)
./sweep.py --quick                           # BV only, other knobs at default
./sweep.py --bv 16,32 --warps 2,4 --stages 2,3
./sweep.py --shape qwen35-397b-tp2           # TP2 control: 32 heads -> 128 WGs for free
./sweep.py --waves-per-eu 1,2,4              # AMD occupancy hint
./sweep.py --isolate --waves-per-eu 1,2,4,8  # one process per config; survives compiler aborts
./sweep.py --csv out.csv --reps 30

./trace_kernel.py                            # rocprofv3 kernel trace, 3 configs
./trace_kernel.py --configs 32/4/2,16/4/3    # BV/warps/stages[/waves_per_eu]
./trace_kernel.py --shape qwen35-397b-tp2 --launches 20 --pftrace --csv t.csv

./trace_att.py                               # ATT stall profile, baseline + winner
./trace_att.py --configs 16/4/3/4 --csv a.csv
./trace_att.py --shape qwen35-397b-tp4-4k --kernels 4   # short shape needs a wider window
```

Use `--isolate` whenever you sweep `waves_per_eu` or `matrix_instr_nonkdim`.
Those reach the AMD backend's compile options, and bad values abort the process
rather than raising — `wpe=8` core-dumps. In-process the sweep dies with it; with
`--isolate` it comes back as `rc=-6` on that row and the sweep continues.
`--isolate` implies `--no-verify`, since outputs cannot be compared across
processes, so do a normal in-process run afterwards to confirm the winner's
`maxerr`.

Read the `WGs` and `fill%` columns before the `ms` column. If a winning config
did not raise workgroup count, occupancy was not the binding constraint and the
whole premise above needs re-checking.

## Two things this harness gets right that a naive one does not

**1. `w` and `u` are not free inputs.** They come from
`chunk_gated_delta_rule_fwd_intra` (fused kkt + solve_tril + recompute_w_u), and
the delta rule is only stable for `w` that actually solves that triangular
system. Feeding random `w` diverges — over NT=512 chunk iterations >90% of `h`
comes back NaN, which invalidates every cross-config comparison and perturbs
timing through denormal handling. `build_inputs()` therefore generates
`q/k/v/beta/g` at the traced shapes and runs the same two preamble steps
`chunk.py:47-59` runs, including the L2-norm on `k` (the trace shows
`l2norm_fwd_kernel` 90x = 45 layers x {q,k}).

**2. The kernel writes its output state back over its input.** It runs with
`INPLACE_UPDATE=True` and stores `ht` into `initial_state`
(`chunk_delta_h.py:130`). Re-running without restoring feeds its own output back
in. `Inputs.restore_state()` is called before *every* launch, warmups included,
outside the timing window. This is the same hazard that made sglang disable
autotune on this kernel — see the comment at `chunk_delta_h.py:29`.

## How configs are applied

The production config is baked at import time from three env vars
(`chunk_delta_h.py:23-26`), so sweeping via env would need a fresh process per
config. Instead `bench.py` swaps `Autotuner.configs` in place. That is exact,
not a hack: `Autotuner.run` only takes the benchmarking branch when
`len(self.configs) > 1`, and otherwise goes straight to `self.configs[0]`
(`triton/runtime/autotuner.py:220-262`). With one config there is no autotune
pass and no cache lookup, so nothing stale leaks between configs.

`waves_per_eu` and `matrix_instr_nonkdim` are AMD backend compile options
(`triton/backends/amd/compiler.py:47,64`), not kernel parameters — they ride in
`Config.kwargs` and are parsed out by the backend at launch. **They have no
sglang env knob**, so if one of them wins, shipping it needs a code change in
`chunk_delta_h.py`, not a server flag.

## Scope

This kernel is worth **+22.2 ms of a 638 ms prefill step (3.5%)** to reach NV
parity, and prefill is ~23% of E2E at OSL=1024. Do not expect a benchmark to
show it: on `MODE=full` (OSL 1024) decode is 77% of the request and the run-to-
run noise floor is ±1.1%. Use `MODE=prefill` (ISL 32768 / OSL 4) and read
`Input token throughput` / `Mean TTFT`.

The bigger prefill target is the TP all-reduce at +114.2 ms (2.28x) — see the
`ROCM_QUICK_REDUCE_*` notes. Nothing in this directory touches that.

## Verifying it landed on the server

A benchmark is the wrong instrument for a 3.5% change. Check the grid directly:

```python
import json
ev = json.load(open('<trace>.json'))['traceEvents']
k = [e for e in ev if e.get('cat') == 'kernel'
     and 'chunk_gated_delta_rule_fwd_kernel_h' in e['name']][0]
print(k['args']['grid'], k['args']['block'])   # want [8, 16, 1] [128 or 256, 1, 1]
```

If the grid did not change, the env did not reach the TP workers — nothing else
matters until it does.
