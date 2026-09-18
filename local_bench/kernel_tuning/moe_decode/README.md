# FlyDSL fused decode-MoE front-end (Qwen3.5-397B-A17B MXFP4, MI355X)

The decode MoE for Qwen3.5-397B-A17B MXFP4 on gfx950 costs **70.3 µs per layer
across 7 kernel launches**. The two GEMMs are 41.7 µs of that and sit at 68–71 %
of HBM peak; the other five kernels move ~1.5 MB and cost as much as stage 1.
This directory collapses the front-end of that chain.

Status: **changes #1 and #2 are landed and measured. Change #3 (folding the MX
quant + scatter into the fused kernel) is not implemented.**

---

## Running the server

```bash
# both changes on (the default)
TP=4 CONC=4 bash kernel_tuning/moe_decode/run_server.sh

# one arm at a time
MOE_ATOMIC_EPILOGUE=0 bash kernel_tuning/moe_decode/run_server.sh   # change #2 only
MOE_FUSED_SORT=0      bash kernel_tuning/moe_decode/run_server.sh   # change #1 only
MOE_FUSED_SORT=0 MOE_ATOMIC_EPILOGUE=0 \
                      bash kernel_tuning/moe_decode/run_server.sh   # baseline
```

`run_server.sh` sets the hook path and env, then execs the unmodified
`qwen3.5_fp4_sglang_server.sh`, so **every flag of that script still applies**
(`MODEL TP CONC PORT MEM_FRAC_STATIC SIMULATE_ACC DRY_RUN …`). Use `DRY_RUN=1`
to print the resolved launch command without starting anything.

Script-specific env:

| var | default | meaning |
|---|---|---|
| `MOE_FUSED_SORT` | 1 | change #2: FlyDSL fused router+sort |
| `MOE_ATOMIC_EPILOGUE` | 1 | change #1: atomic stage-2 epilogue via tuned CSV |
| `MOE_MAX_TOKENS` | 16 | token capacity the fused kernel is compiled for; 32 is the LDS ceiling at E=512 |
| `MOE_COUNT_COMPILES` | 0 | log one line per FlyDSL JIT compile |

**Check the arm is actually live.** Each TP worker prints

```
[FlyDSL-moe] fused router+sort enabled (max_tokens=16)
```

once, at the first `import aiter.fused_moe`. If that line is absent the fused
path is *not* running — the most likely cause is a `PYTHONPATH` that puts some
other `sitecustomize.py` first (see "One hook module" below). A worker that
cannot import the kernel logs `decode-MoE patch NOT installed` and continues on
the stock path rather than taking the server down.

`MOE_MAX_TOKENS` bounds which decode batches are accelerated. Under EAGLE MTP a
verify step carries `bs × --speculative-num-draft-tokens` rows, so the default
16 covers `CONC ≤ 4` at `--speculative-num-draft-tokens 4`. Larger batches fall
back to the stock router+sort; they are not mis-routed.

### One hook module

`kernel_tuning/bootstrap_both/sitecustomize.py` is the single `sitecustomize`
on the path and loads each per-kernel hook file by path. Python imports the name
`sitecustomize` exactly once, from whichever `sys.path` entry comes first, so
putting `conv/bootstrap` and `moe_decode/bootstrap` both on `PYTHONPATH` would
silently activate one and drop the other. Adding a kernel means adding an entry
to `_HOOKS` there, not a new file named `sitecustomize.py`. Each hook stays
gated by its own env flag, so the combined module activates nothing on its own
and one launcher can serve every arm.

---

## Measured results

`compare_moe_decode.py`, Qwen3.5-397B-A17B MXFP4 **TP=4** per-rank shapes
(hidden 4096, `E=512`, `topk=10`, `inter_dim=256`), 30 iters × 2 profile runs
per point, per-kernel median, GPU 0. Raw output in
`results_tp4_20260911.txt` / `.json`.

**Front-end µs per layer per rank** — router + sort + quant/scatter + reduce,
i.e. everything these changes touch:

| tokens | sglang | +atomic (#1) | flydsl (#2) | flydsl+atomic (#1+#2) | launches |
|---:|---:|---:|---:|---:|---:|
| 4  | 21.96 | +0.33 | −6.27 | **−6.68** | 6 → 4 |
| 8  | 25.85 | −0.60 | −7.29 | **−7.34** | 7 → 5 |
| 16 | 25.95 | −2.09 | −4.97 | **−7.90** | 7 → 4 |
| 32 | 24.09 | +0.35 | −0.84 | −0.81 | 6 → 4 |

At the production decode shape (`tokens=16`, i.e. `CONC=4` × 4 draft tokens)
the whole MoE goes **72.68 → 65.52 µs/layer, 7 → 4 launches**. The bench runs
~3 % hot versus the production trace, so in trace terms that is roughly
70.3 → 63.5 µs — about the 62.8 µs ASM bar, not below it. The ≤50 µs / ≤3
launch target needs change #3.

`+atomic` is a no-op at tokens 4/8/32 because only the `token=16` CSV row was
patched; the other tiers already use non-`reduce` epilogues. The fused kernel
loses its advantage by tokens=32 (−0.8 µs): it runs the whole front-end in one
workgroup, so its cost grows with tokens while the three kernels it replaces
spread across the device.

### Per-change detail at tokens=16

| stage | sglang | +atomic | flydsl | flydsl+atomic |
|---|---:|---:|---:|---:|
| router (`topkGatingSoftmax`) | 9.30 | 9.64 | — | — |
| sort (`opus_moe_sorting_entry` ×2) | 8.08 | 8.52 | — | — |
| **route+sort (fused, 1 launch)** | — | — | 12.59 | 12.50 |
| quant+scatter | 5.58 | 5.71 | 5.69 | 5.54 |
| reduce (`moe_reduction_kernel_0`) | 3.00 | — | 2.70 | — |
| stage 1 | 33.22 | 34.30 | 32.62 | 34.16 |
| stage 2 | 13.52 | 13.38 | 13.53 | 13.32 |
| **front-end** | **25.95** | **23.86** | **20.98** | **18.04** |
| launches | 7 | 6 | 5 | 4 |

---

## Change #1 — atomic stage-2 epilogue

A tuned-CSV row, no code. Row `token=16, inter_dim=256, expert=512, topk=10` of
`aiter/configs/model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv` switches
`kernelName2` from
`flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_reduce_nt_sbm32` to
`..._t32x128x256_atomic_sbm32`. That deletes `moe_reduction_kernel_0`.

The patched file is checked in as `qwen3_5_397b_fp4_tuned_fmoe.atomic.csv`
(full copy, one field changed) plus `..._atomic_persist.csv` for the variant.
Regenerate from the current aiter tree with the snippet in
"Regenerating the CSVs" below.

Measured (`atomic_persist` is the noisier of the two and was not adopted):

| stage-2 config | front-end | launches |
|---|---:|---:|
| `reduce_nt_sbm32` (shipped) | 25.95 | 7 |
| `atomic_sbm32` | 23.86 | 6 |

**This is decode-only and must stay that way.** The atomic epilogue is
1.6–2.8× *slower* at prefill scale (327 680 rows: `reduce` 1065+615 µs vs
`atomic` 2098 µs) because 1.34 G packed-bf16 atomic adds into a 268 MB region
serialise on shared cache lines. The gate is structural: the CSV is a token-tier
table and only the `token=16` row is patched.

**It also makes stage 2 order-dependent.** `moe_reduction` is deterministic;
`global_atomic_pk_add_bf16` is not. Correctness has to be checked with
tolerances, not bitwise equality, once this is on. Generation is already
nondeterministic under `SIMULATE_ACC=1`, so this is acceptable — but it is a
decision, not an accident.

### Regenerating the CSVs

```python
import csv
src = "/sgl-workspace/aiter/aiter/configs/model_configs/qwen3_5_397b_fp4_tuned_fmoe.csv"
rows = list(csv.reader(open(src)))
K2 = 20
for name, k2 in [("atomic", "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_sbm32"),
                 ("atomic_persist", "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_persist_sbm32")]:
    out = [rows[0]]
    for r in rows[1:]:
        r = list(r)
        if (r[2], r[4], r[5], r[6]) == ("16", "256", "512", "10"):
            r[K2] = k2
        out.append(r)
    csv.writer(open(f"qwen3_5_397b_fp4_tuned_fmoe.{name}.csv", "w", newline="")).writerows(out)
```

> **Do not just prepend the patched CSV to `AITER_CONFIG_FMOE`.** aiter merges
> every `model_configs/*tuned_fmoe*.csv` into one table and, on a duplicate
> shape key, **rewrites the source files in place** and raises. Doing that once
> stripped row 14 out of aiter's shipped CSV (48 → 47 rows); it was restored
> with `git checkout`. `fmoe_chain.py` builds a chain with the stock qwen3.5
> file *dropped* and the patched one in its place, which is what
> `run_server.sh` and `compare_moe_decode.py` use.

---

## Change #2 — fused router + sort

`moe_fused_route_sort_flydsl.py` ports
`compile_moe_sorting_oneshot_fused` from upstream FlyDSL
(`kernels/moe/moe_sorting_kernel.py:696`) plus its gating helpers
(`kernels/moe/topk_gating_softmax_kernel.py`) into aiter's vendored FlyDSL
namespace, which only carries the *unfused* oneshot sorter. One launch turns
`gating_logits[M, E]` into the full `moe_sorting` contract
(`sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf`),
replacing `topk_softmax` **and** both `opus_moe_sorting_entry` launches, with
every per-token-K intermediate staged in LDS.

### The premise that turned out to be wrong

The brief attributed the satellite cost to the ~4.2 µs gfx950 dispatch floor and
expected the router's 9.8 µs to go to zero. It does not. Measured in isolation,
one block at a time:

| | µs (device) |
|---|---:|
| `aiter.topk_softmax` (E=512, topk=10, T=16) | 9.08 |
| the same gating body as a standalone FlyDSL kernel, grid=1 | 7.18 |
| aiter's vendored unfused FlyDSL oneshot sort | 7.02 |
| `opus` sort (2 launches) | 7.58 |

The router is ~5 µs of *real serial work* above the floor — 10 sequential
argmax passes over 512 experts — and fusing it into the sort cannot remove it,
only overlap it. The honest ceiling for change #2 is "one kernel doing gating
plus sorting", ~12.5 µs against the 17.5 µs of the three it replaces.

### The straight port was a regression

Ported verbatim it ran **20.05 µs** — worse than the 18.4 µs baseline. Getting
to 12.5 took four changes, all documented inline:

1. **Block size 512** (upstream hard-codes 256). Everything but the `moe_buf`
   zeroing runs in block 0, so this is the parallelism of the whole front-end.
   Swept: 256 → 15.18 µs, **512 → 12.60 µs**, 1024 → 14.54 µs.
2. **Gating layout picks the *smallest* valid VPT**, not the largest. The top-k
   loop rescans all VPT register slots on each of `topk` iterations, so VPT
   trades against butterfly depth; at block 512 that selects VPT=16/TPT=32.
   Upstream's unconditional "largest VPT" also caps at 16, which for E=512
   forces `TOKENS_PER_BLOCK=8` — **half the 16 tokens of a CONC=4 verify step
   would never have been routed**, since gating runs only in block 0. The
   compiler now raises if `TOKENS_PER_BLOCK < max_tokens` instead.
3. **Coalesced sentinel fill.** Upstream maps one thread per expert and loops
   `unit_size` slots, so consecutive lanes write addresses `unit_size` dwords
   apart and every store fans out to one cache line per lane. Mapping
   `unit_size` consecutive lanes to one expert makes each store a few
   contiguous runs. 17.18 → 13.76 µs.
4. **One thread per expert in the count phase**, walking its own mesh column.
   Upstream spreads each expert over an 8-lane group and cross-reduces with
   three `shuffle_xor` steps per mesh cell — 48 cross-lane ops per thread to
   save 14 LDS reads, on a column read that is bank-conflict-free anyway.
   13.76 → 12.62 µs.

Plus a bound fix: the `sorted_expert_ids` write loop ran `max_tokens` times per
expert when an expert spans at most `ceil(max_tokens / unit_size)` GEMM tiles
(1, at `unit_size=32`) — 16 predicated stores to write one.

### Tried and rejected

A **block-wide prefix sum** (every wave scans `WARP_SIZE` experts, wave totals
combined through an LDS scratch row) replacing upstream's wave-0-only serial
chunk chain: worth **0.09 µs** (12.62 → 12.53) and it **faulted at block size
256 and 512** while passing at 1024. Reverted to upstream; the prefix sum is not
on the critical path.

### Integration

`moe_decode_patch.py` patches two seams in `aiter.fused_moe`:

- **`fused_topk`** — when the decode gate matches, the router is *not run*; the
  gating logits are stashed and cached zero-filled `topk_weights`/`topk_ids` of
  the right shape and device are returned. (Zeroing per call instead of caching
  cost two `FillFunctor` launches, 2.8 µs/layer — over half the win.)
- **`moe_sorting`** — consumes the stash and produces the whole 5-tensor
  contract in one launch.

SGLang runs the router in `select_experts` and only later hands `topk_ids` to
`fused_moe`; the stash is the only way to make the router's *output* disappear,
because those two calls are the only place both the logits and the sort are
visible.

Every gate miss is **fail-safe**: if the stash does not match the sort call
(different M/E/topk, `num_local_tokens`, EP dispatch policy, `flat`,
`output_aux`) the original router is run into the caller's buffers first and
the stock sorter takes over. A miss costs a launch, never correctness. EP
(`expert_mask`) is carried through to the kernel's `has_mask` path and not
dropped.

CUDA graphs: the first call per shape JIT-compiles, which happens during
SGLang's pre-capture warmup replays. On the cached path nothing is allocated
beyond the output tensors — the same as the stock `moe_sorting`.

### Correctness

Verified against `aiter.topk_softmax` + `aiter.moe_sorting_opus_fwd` at
`E=512, topk=10, T=16, block_size ∈ {256, 512, 1024}`: identical
`num_valid_ids`, identical expert for every one of the 160 `(token, k)` pairs,
weights within 2e-3. A full harness (`check_moe_decode.py`, sweeping
`token ∈ {1,2,4,8,16,32,64}`, graph capture/replay, `expert_mask`, and an
independent FP32 reference) is **not yet written**.

---

## Change #3 — fold MX quant + scatter — NOT DONE

`aiter::fused_mx_quant_moe_sort_kernel` (5.5 µs/layer) quantises activations to
fp4 and scatters them into the sorted layout after the sort has already written
the route map — the same warps touching the same routes twice.
`aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py` already does this
fusion for the grouped/masked contiguous-M layout; the intra-warp mapping, the
butterfly `shuffle_xor` amax reduction and the scale-preshuffle derivation all
transfer, and only the destination row computation changes
(`expert_row_base[e] + slot` → `sorted_token_ids` order).

Expected ≈ −5.5 µs/layer and 4 → 3 launches, which would put the shape at
~60 µs/layer in bench terms (~58 µs in trace terms).

---

## Known issue: OOB sentinel stores

The ported kernel builds its buffer descriptors with `max_size=True`, which sets
`num_records = 0xFFFFFFFF` — i.e. **hardware OOB checking is off**. Upstream's
idiom of steering inactive lanes to element `0x7FFFFFFF` therefore does not
discard the store; it writes ~8 GiB past the base. This faults deterministically
on the 128-bit `moe_buf` zeroing store, which is why that path now folds
inactive lanes onto element 0 (writing an extra zero into a buffer being zeroed
is idempotent).

The same sentinel is still used on the scatter and padding stores inherited from
upstream. Those have not faulted in any run here, but they are a latent
correctness hazard and should be fixed by passing real `num_records_bytes` for
`sorted_ids` / `sorted_weights` / `sorted_expert_ids`.

---

## Files

| file | what |
|---|---|
| `moe_fused_route_sort_flydsl.py` | the fused kernel + host entry `flydsl_fused_route_sort` |
| `moe_decode_patch.py` | runtime patch of `aiter.fused_moe.{fused_topk, moe_sorting}` |
| `bootstrap/sitecustomize.py` | import hook, gated on `SGLANG_FLYDSL_MOE_DECODE=1` |
| `run_server.sh` | launcher; wraps `qwen3.5_fp4_sglang_server.sh` |
| `compare_moe_decode.py` | the 4-arm A/B at TP=4 shapes |
| `bench_moe_decode.py` | single-arm per-kernel breakdown incl. the router |
| `bench_route_sort.py` | isolated router / sort / fused timing |
| `fmoe_chain.py` | builds a safe `AITER_CONFIG_FMOE` chain |
| `qwen3_5_397b_fp4_tuned_fmoe.atomic{,_persist}.csv` | change #1 |
| `results_tp4_20260911.{txt,json}` | the sweep above |

---

## Benchmarking notes

`compare_moe_decode.py` **interleaves repeats across arms** rather than
finishing one arm at a time. Arms run back to back for minutes and the GPU
clocks drift down as it heats; measured the naive way, stage 1 — byte-for-byte
the same kernel on the same inputs in every arm — climbed monotonically by 3 µs
from the first arm to the last, which is larger than the effect under test.

Read the **FRONT-END** row, not TOTAL. The two GEMMs do identical work in every
arm and carry all the residual noise (±1.5 µs on stage 1 even interleaved).

No end-to-end serving numbers have been taken yet. The brief's bar is that a
kernel-level win below ~8 µs/layer cannot be resolved end-to-end; at tokens=16
this is −7.9 µs/layer, i.e. right at that bar, so a serving A/B should use
`SIMULATE_ACC=0`, n ≥ 5 paired runs with alternating server boots, and report
accept-length alongside throughput. Do **not** compare against
`bench_results/qwen35_mxfp4_mtp_isl32768_osl1024_c4.log` — that run predates
this build and used a different all-reduce configuration.

---

## Environment

| component | version |
|---|---|
| GPU | AMD Instinct MI355X, `gfx950` |
| sglang | `0.5.19.dev20260908+g554f817948` (git `5097f9ac95`) |
| aiter | `4ad998328` |
| FlyDSL | `0.3.2` (wheel; `kernels/` is not shipped, hence the port) |
| upstream FlyDSL source | `/var/home/main_FlyDSL/FlyDSL` @ `ac227c35` |
| torch | `2.9.1+rocm7.2.0.git7e1940d4` |
| triton | `3.7.0` |

Hardware counters do not work in this container (no `CAP_PERFMON`,
`rocprofv3 --pmc` hangs), so L2 hit rate and dispatch occupancy are not
measurable here; every attribution above is inferred from runtime.
