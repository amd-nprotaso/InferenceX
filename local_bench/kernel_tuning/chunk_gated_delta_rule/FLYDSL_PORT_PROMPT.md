# Prompt: FlyDSL rewrite of `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`

Copy everything below the line into a fresh agent session running in
`/var/home/InferenceX/local_bench`.

---

## Task

Write a FlyDSL replacement for SGLang's Triton GDN chunk-state kernel
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, targeting AMD MI355X
(gfx950), and beat the best tuned Triton configuration on the production
Qwen3.5-397B-A17B TP4 prefill shape.

This is not a blind port. An ATT-based analysis has already identified where the
Triton kernel loses its time and which three structural changes are available.
Your job is to realize those in FlyDSL, where the layout and scheduling control
exists to do it — Triton cannot express them.

## Read these first — do not invent API

Grounding, in priority order. FlyDSL has a large surface and a hallucinated API
call costs more than the reading does.

| What | Where |
|---|---|
| Project conventions, layout, env vars, style gate | `/var/home/main_FlyDSL/FlyDSL/CLAUDE.md` |
| `@flyc.kernel` / `@flyc.jit`, LDS, tiled copy/MMA | `docs/kernel_authoring_guide.md` |
| Tiling, LDS double-buffer/swizzle, prefetch, MFMA scheduling, **XCD swizzle** | `docs/kernel_tuning_guide.md` |
| Layout algebra (`zipped_divide`, `slice`, `make_layout`) | `docs/layout_system_guide.md` |
| Closest MFMA reference: bf16 GEMM on our exact arch | `kernels/gemm/gemm_a16w16_gfx950.py` |
| Minimal tiled-MMA skeleton | `examples/03-tiledMma.py` |
| Pipelining / LDS-stage helpers | `kernels/common/mma/`, `kernels/pipeline_utils.py` |
| Working precedent: FlyDSL kernel replacing an SGLang Triton kernel in this same server, incl. the integration hook | `kernel_tuning/conv/causal_conv1d_flydsl.py` + `kernel_tuning/conv/README.md` |

Project-local skills exist and are worth invoking rather than reimplementing:
`/flydsl-kernel-authoring`, `/flydsl-tile-programming`, `/gemm-optimization`,
`/lds-optimization`, `/prefetch-data-load`, `/kernel-trace-analysis`,
`/debug-flydsl-kernel`.

Full background on the kernel, the shapes, the traffic model, and every dead end
already explored: `kernel_tuning/chunk_gated_delta_rule/README.md`. Read it.

## The kernel

- Source: `sglang/kernels/ops/attention/fla/chunk_delta_h.py`, host entry
  `chunk_gated_delta_rule_fwd_h` (a verbatim copy of the traced source is at
  `traces_att/qwen35-397b-tp4/BV16_w4_s3/ui_output_agent_37568_dispatch_40/source_2_chunk_delta_h.py`).
- Production shape (TP4): `B=1, T=32768, Hg=4, H=16, K=128, V=128, BT=64`, so
  `NT=512`. Defined in `shapes.py`.
- Grid today: `(cdiv(V, BV), N*H)`. The `T` axis carries the recurrence and is
  serial.
- Per `i_t` iteration the kernel: stores `b_h` to `h`, computes
  `b_v = Σ b_w @ b_hᵀ`, subtracts from `v`, applies the gate, then
  `b_h += (b_k @ b_v)ᵀ`. `K=128` means two 64-wide halves (`b_h1`, `b_h2`).
- `INPLACE_UPDATE=True`: the epilogue writes `ht` back over `initial_state`.

## Baselines to beat

Device-side dispatch time, MI355X, from `trace_kernel.py` (median of 20, ±0.007):

| config | ms | note |
|---|---:|---|
| `BV=32 w=4 s=2` | 1.327 | old sglang default |
| `BV=16 w=4 s=3` | 0.905 | shipped default (AMD), env-only |
| `BV=16 w=4 s=3 wpe=4` | **0.850** | best known; **this is your bar** |

For reference, the same Triton source on GB200 is ~1.043 ms/call, so the tuned
AMD config already beats NV. You are pushing past that, not chasing it.

Targets: **≤0.65 ms is a clear win** (1.3x over best Triton, 2.0x over the old
default). **≤0.50 ms** would mean the LDS restructuring worked. Report honestly
if you land between or above.

## What the ATT traces measured

Reproduce or trust these; the raw data is in `traces_att/`, and
`trace_att.py` regenerates it. Per loop iteration per wave, median iteration
**4048 cycles**, for the current best config (`BV=16 w=4 s=3`):

| bucket | cyc/iter | % of iteration |
|---|---:|---:|
| `s_waitcnt vmcnt` — global-load latency | 1263 | 31.1% |
| `s_waitcnt lgkmcnt` — LDS latency | 509 | 12.6% |
| `s_barrier` — wave sync (**10.1 barriers/iter**) | 413 | 10.2% |
| VALU | 264 | 6.5% |
| global load issue | 180 | 4.4% |
| LDS issue | 171 | 4.2% |
| **MFMA** | **27** | **0.7%** |

The kernel spends 0.7% of its time doing math. 70% is stall. Note that the
README's headline "77% wait" splits into 31% global-load and 23%
LDS-plus-barrier — the second block is a separate problem and it is the one
Triton cannot fix.

Only 4 waves are resident per CU out of a hardware max of 32 (LDS is 103,296 B,
so one workgroup per CU). There is nothing to hide latency behind.

## The three changes, in the order to attempt them

### 1. Fix the XCD mapping — highest value, lowest cost

**Verified on the machine, not assumed.** Probing `HW_REG_XCC_ID` on this
MI355X gives 8 XCDs with strict round-robin, `xcd = linear_wg_id % 8`, and
`hipDeviceProp.l2CacheSize` = 4 MB per XCD.

`w` and `k` do not depend on `i_v` (`chunk_delta_h.py:238-241`, `:335-336`).
Grid is `(cdiv(V,BV), N*H)` with `i_v` fastest, so `linear_id = i_v + 8*i_nh`
and therefore **`xcd = i_v`**: the 8 workgroups that read byte-identical `w`/`k`
are scattered across all 8 XCDs into 8 separate L2s, while each XCD holds 16
workgroups with 16 *different* `i_nh` that share nothing. It is the worst of the
8 possible assignments.

Resulting traffic at TP4:

| tensor | unique | loaded | amplification |
|---|---:|---:|---:|
| `k` (Hg=4, so 4 v-heads share a k-head, times 8 `i_v`) | 32 MB | 1024 MB | **32x** |
| `w` | 128 MB | 1024 MB | **8x** |
| `v` | 128 MB | 128 MB | 1x |

94% of the 2684 MB the kernel moves is redundant re-reads that the current
mapping guarantees cannot hit in a local L2.

The fix is to make `xcd = i_nh % 8`, so all 8 `i_v` siblings share one XCD's L2
while running the same `i_t` in near-lockstep over a shared 32 KB block. In
FlyDSL either launch `(N*H, cdiv(V,BV))` and swap the block indices, or use the
built-in `xcd_swizzle` remap (see `docs/prebuilt_kernels_guide.md` and
`compile_preshuffle_gemm`; `tests/kernels/test_preshuffle_gemm.py` shows it in
use). Prefer the built-in.

Validate this one in Triton first — it is a two-line change there
(`chunk_delta_h.py:355` plus swapping `tl.program_id(0)`/`(1)`) and output must
stay bitwise identical, since it is a pure permutation of work onto workgroups.
If it does not help in Triton, understand why before building on it.

**The open question, which only measurement answers:** the 8 siblings must stay
roughly in lockstep across 512 iterations for the L2 hits to materialize. They
start together and do identical work, so drift should be bounded — but verify,
do not assume.

### 2. Deeper prefetch — LDS is free budget here

`sweep_tp4.csv` only ever tested `num_stages ∈ {1,2,3}`. Since `vmcnt` is the
largest single bucket and 2→3 was worth 1.23x→1.42x, 4 is the obvious untested
point.

The arithmetic: measured LDS is 103,296 B at `s=3`, which is exactly
`34,432 × 3`, so `s=4` needs 137,728 B. `hipDeviceProp.sharedMemPerBlock` on
this part is 163,840 B. **It fits, with 26 KB spare.** `s=5` (172,160 B) does
not.

It also costs nothing. The grid never exceeds 256 workgroups on 256 CUs, so
there is never more than one workgroup per CU regardless of LDS — capacity is
not competing with occupancy at any point. The usual "more stages costs
occupancy" tradeoff does not exist for this kernel, which also means the
existing `s=2` vs `s=3` sweep never actually tested co-residency.

Test `SGLANG_GDN_CHUNK_H_NUM_STAGES=4` on the Triton kernel first — it is one
env var — then carry the answer into the FlyDSL pipeline depth.

### 3. Collapse the LDS layout round-trips — the real reason to use FlyDSL

The LDS instruction mix is lopsided:

| op | issues | width |
|---|---:|---|
| `ds_write_b16` | 41,008 | **2 B** |
| `ds_read_b128` | 20,480 | 16 B |
| `ds_read_u16` | 16,448 | **2 B** |
| `ds_read_b64_tr_b16` | 16,384 | 8 B (HW transpose) |

There are **five separate staging round-trips per iteration**, four of them
writing at 2-byte granularity:

| source line | what | why |
|---|---|---|
| `:221`, `:226` | `b_h1`/`b_h2` → LDS → global | accumulator layout is not coalescable for the store |
| `:242`, `:248` | `tl.trans(b_h1)`/`(b_h2)` | accumulator → MFMA B-operand layout |
| `:333` | `b_v` fp32→bf16 relayout | operand layout for the second dot |

That is what drives 10.1 barriers and 509 cycles of `lgkmcnt` per iteration.
Note `b_h1` goes through LDS **twice per iteration** from the same accumulator
registers — once for the store, once transposed for the dot. The `k` loads
already use the efficient gfx950 `ds_read_b64_tr_b16` path; the `b_h` paths do
not.

In FlyDSL you control the fragment layouts directly, so the targets are:

- Choose an MFMA atom and accumulator layout such that `b_h`'s accumulator can
  feed the next `tl.dot` as a B-operand **without** a round-trip, or with one
  wide (`b128`) round-trip instead of 2-byte scatter.
- Serve the `h` store and the transpose from **one** staging buffer rather than
  two. This may mean storing `h` as `(K, V)` rather than `(V, K)`; if so, check
  the downstream consumer (`chunk_o` / the second-stage kernel) and say so
  explicitly before changing a layout that crosses a kernel boundary.
- Apply XOR swizzle on the staging buffer (`/lds-optimization`) and cut the
  barrier count.

Budget available: ~23% of the iteration. This is the largest single block after
`vmcnt` and it is entirely inaccessible from Triton.

## Correctness constraints — these have already bitten people

1. **`INPLACE_UPDATE=True` writes `ht` over `initial_state`.** Re-running
   without restoring feeds the kernel its own output. `Inputs.restore_state()`
   in `bench.py` must be called before *every* launch including warmups, outside
   the timing window. This is why sglang disabled autotune on this kernel
   (`chunk_delta_h.py:29`).
2. **`w` and `u` are not free inputs.** They come from
   `chunk_gated_delta_rule_fwd_intra` (fused kkt + solve_tril + recompute_w_u),
   and the delta rule is only stable for `w` that actually solves that
   triangular system. Random `w` returns >90% NaN over 512 iterations and
   silently invalidates every comparison. Use `shapes.py: build_inputs()`, which
   runs the real preamble including the L2-norm on `k`.
3. **`k` carries `Hg=4` grouped heads, not `H=16`** — `k` is `[B,T,Hg,K]` and
   head `i_h` reads slice `i_h // (H // Hg)`. Getting this wrong changes the
   launch grid silently.
4. **Padded slots carry a `-1` sentinel** in `initial_state_indices`
   (`:185-188`); guard on it.
5. **int64 offsets.** The state pool may be an envelope-strided view whose
   per-slot pitch spans all layers; int32 index products overflow (`:181-184`).
6. **BV-only changes must be bitwise identical** (`maxerr 0.00e+00`), because
   `BV` only partitions `V` and each row accumulates independently. If your
   FlyDSL version is not bitwise identical to the Triton reference, you have
   changed the arithmetic — justify it or fix it. `BV=8` is the known exception
   and it is a losing config anyway.

## FlyDSL conventions to follow

From `CLAUDE.md` — these prevent the common failure modes:

- `@flyc.kernel` for the device kernel, `@flyc.jit` for the launch wrapper.
- `range_constexpr` for compile-time unrolled loops; `range(..., init=[...])`
  for `scf.for` with loop-carried values. **The `i_t` recurrence is a real
  `scf.for` with `b_h1`/`b_h2` carried** — this is the crux of the kernel, get
  the loop-carried state explicit and compact.
- Prefer `fx.rocdl.make_buffer_tensor()` + layout ops + `fx.copy_atom_call`.
  Raw `buffer_ops.create_buffer_resource()` and manual byte offsets are legacy.
- Allocate LDS with `SharedAllocator` (`fx.SharedAllocator`) over an
  `@fx.struct` storage layout, not the legacy `SmemAllocator`.
- Clear `SmemPtr._view_cache = None` after exiting `scf.for` if you recreate
  shared-memory views, or you get MLIR dominance errors.
- No early `return`, no branch-local `return`/`yield` in traced functions; do
  not define a value only inside one branch and use it after.
- Cache compiled launchers per concrete tensor signature (dtype, shape, stride)
  — see the `_build` / `lru_cache` / `launch._compiled` pattern in
  `causal_conv1d_flydsl.py`.
- Run `bash scripts/check_python_style.sh --fix` before finishing.

## Validation protocol

1. **Correctness** against the Triton kernel via the existing harness:
   `bench.py` already times and verifies one config against production. Extend
   it, or write `check_chunk_h.py` modelled on `kernel_tuning/conv/check_conv.py`
   (39 cases, independent FP32 reference, exact state comparison,
   dtype-dependent output tolerances, graph capture/replay).
   Cover: all three shapes in `shapes.py` (TP4, TP2, TP4-4k), padded/`-1` slots,
   `USE_INITIAL_STATE` on and off, `SAVE_NEW_VALUE` on and off, and the
   `use_exp2` gk path.
2. **Timing** with `trace_kernel.py` (device-side dispatch, ±0.005 ms), not
   wall clock. `sweep.py`'s CUDA-event timing is the cross-check, not the
   primary number.
3. **Stalls** with `trace_att.py`. Regenerate the budget table above for your
   kernel and show which bucket you actually moved. Two traps, both of which
   produce output that looks fine: `--att-target-cu` defaults to 1 and CU 1
   decodes a *different code object* into a plausible but entirely false
   profile; and ATT arms all four agents unless `--att-gpu-index` pins one. The
   script already defaults to CU 0 and rejects any decode with zero MFMA. Do not
   enable `--att-serialize-all`; it aborts the profiler on this workload.
4. **Hardware counters do not work in this container** — no `CAP_PERFMON`,
   `perf_event_paranoid = 4`, and `rocprofv3 --pmc` hangs indefinitely
   (reproducible on a bare matmul, so it is not this kernel). `profile_traffic.py`
   is written and ready if the container is ever relaunched with the capability.
   Until then you cannot directly measure L2 hit rate, which matters for
   change #1 — so infer it from runtime and say that you are inferring it.
   Note this contradicts the tuning guide's "always confirm with PMCs"; the
   guide is right, the container is the blocker.
5. **End-to-end**, only after the kernel is green: follow the conv precedent —
   `sitecustomize.py` import hook behind an env flag, wrapper script,
   `MODE=prefill` (ISL 32768 / OSL 4), read `Input token throughput` and
   `Mean TTFT`. Do not use `MODE=full`; decode is 77% of the request there and
   the noise floor is ±1.1%, which is wider than the whole effect.

## Known dead ends — do not spend time here

- **Splitting `K` is architecturally impossible**, not merely unbuilt. `b_v` is
  accumulated over every K block (`:178-196`) and that same `b_v` then updates
  every `b_h` (`:275-293`). The blocks are fully coupled in both directions once
  per chunk, so a K-split needs a grid-wide reduction on all 512 iterations.
  `K=128` caps the ceiling at 2x anyway.
- **`BV=8` reaches 100% device fill and is 1.7x slower than `BV=16` at 50%.**
  The `[BV,64]` accumulator tile gets too narrow for the MFMA path; per-workgroup
  rate halves. More workgroups is not the objective on its own.
- **`waves_per_eu=4` is coupled to `(BV=16, num_warps=4)`.** At `BV=32` it is
  3.6-4.4x *worse*, and `wpe=8` aborts the Triton compiler with an LLVM
  assertion. If you expose it, ship it as one atomic triple with an assert.
- **Raising occupancy by co-residency is foreclosed** at `BV=16`: 103 KB of
  160 KB means two workgroups will never fit on a CU. Change #2 exploits this
  rather than fighting it.
- Do not re-litigate the `wpe` measurement at low rep counts — below ~30 reps
  the sign is not stable. Use ≥60.

## Deliverables

1. `kernel_tuning/chunk_gated_delta_rule/chunk_delta_h_flydsl.py` — the kernel
   plus a host entry matching `chunk_gated_delta_rule_fwd_h`'s signature.
2. `check_chunk_h.py` — the correctness suite.
3. `bootstrap/sitecustomize.py` + `run_server.sh` — opt-in integration behind
   `SGLANG_FLYDSL_GDN_CHUNK_H=1`, following `kernel_tuning/conv/`.
4. `README.md` — what changed, measured before/after for each of the three
   changes *independently*, the regenerated ATT budget table, and the
   environment/versions table.
5. The Triton-side results for changes #1 and #2, which are cheap and land
   sooner than the rewrite.

## Reporting

State plainly which of the three changes worked, which did not, and by how
much. A negative result on the XCD swizzle is a genuinely useful finding and
should be reported as such, not buried. Keep kernel microbenchmarks separate
from end-to-end serving numbers — the README is explicit that this kernel is
worth 3.5% of a prefill step, and a serving benchmark cannot resolve that.

If you find that the analysis above is wrong somewhere, say so and show the
measurement. It was derived from the traces in `traces_att/`, and two of its
hardware claims (XCD round-robin, 160 KB LDS cap) were probed directly on this
machine, but the L2-lockstep argument in change #1 is a hypothesis that has not
been measured.
