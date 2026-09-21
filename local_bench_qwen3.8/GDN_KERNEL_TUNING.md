# Qwen3.8 GDN chunk-state kernel: profiling + tuning notes

Working notes for replacing / tuning SGLang's Triton
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` with the AITER FlyDSL
equivalent, on 1x MI355X (gfx950, 256 CUs).

Date: 2026-09-21. All numbers measured on this machine unless marked *projected*.

---

## 0. TL;DR / where this left off

- The AITER FlyDSL kernel **exists, is correct, and is at parity** with SGLang
  Triton at the production shape (0.997x). It wins only at larger per-pass work
  (1.05x at T=32768, 1.14x at 4x16384).
- **Accuracy is a solved non-issue**: `bf16_convert_trunc=False` makes it
  bit-identical to SGLang, for free.
- **Scheduling tuning is exhausted.** BV=32 (+10%) and the gfx942 sched_barrier
  (+0.8% at BV=32) are the only wins; the prefetch-split knob does nothing.
- ATT says the kernel is **occupancy-starved, not compute-bound**: MFMA is 3.5%
  of latency, SIMDs are 31-77% *empty*.
- The two things actually worth doing next are in section 9: kill the
  head-major transpose, then prototype the `fwd_h` + `chunk_fwd_o` fusion.
- Keep it in proportion: this kernel is **6% of prefill GPU time**, so a 25%
  win on it is ~1.5% of prefill and invisible in TPOT.

---

## 1. Setup

Model `amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8`.

**GDN shapes (from `config.json.text_config`, not from aiter test defaults):**

| | |
|---|---|
| `linear_num_key_heads` | Hg = **16** |
| `linear_num_value_heads` | H = **48** |
| `linear_key_head_dim` / `linear_value_head_dim` | K = V = **128** |
| chunk size | BT = **64** |
| dtype / state dtype | bf16 / bf16 (`--mamba-ssm-dtype bfloat16`) |

Note H/Hg = **3**. Every shape in aiter's own tests and tuned CSVs has ratio 4
(64/16, 32/8, 16/4, 8/2), so Qwen3.8's GQA ratio is uncovered upstream.

`T = 16384` is one `--chunked-prefill-size` chunk; ISL=32768 is two of them.

**Gotchas that cost time:**

- `HF_HOME=/data/hf_cache` is required. `/root/.cache/huggingface` holds a
  partial download (12 of 131 shards) and HF resolves there first.
  `qwen38-gdn-repro/scripts/env.sh` does not set it; `run_tp1_ci_sweep.sh` does.
- MXFP4 is the **MoE/GEMM** format only. The GDN recurrence is bf16 and the
  kernels hard-reject anything else. Evidence from the prefill trace: the
  quantizer is `dynamic_per_group_scaled_quant_kernel<bfloat16_t, fp4_t, ...>`
  (bf16 -> fp4 at the MoE boundary) and the MoE GEMMs are `afp4_wfp4_**bf16**`.
  There is **no fp4 GDN kernel in aiter at all** — every fp4 file there is
  GEMM/MoE. fp32 appears only as the accuracy *reference*, never as an input.

---

## 2. Artifacts

### Scripts (in `local_bench_qwen3.8/`)

| file | what |
|---|---|
| `bench_gdn_chunk_h.py` | correctness + perf, sglang vs flydsl vs hip. `--sweep-bv`, `--bf16-convert-trunc`, N-sequence support |
| `check_gdn_chunk_h_accuracy.py` | scores **both** against an independent fp32 reference (transcribed from the Triton kernel body) + a bf16-emulated reference |
| `sweep_gdn_chunk_h_conc.py` | concurrency sweep, `packed` and `per-seq` modes |
| `att_flydsl_gdn_h.py` | standalone single-kernel driver for ATT capture |
| `run_profile.sh` | torch-profiler capture against a live server |

### Traces

`/var/home/sglang_profiling/isl32768_osl32768_c1/`

| dir | notes |
|---|---|
| `1789994300.335755/` | graph-enabled server. **DECODE trace is nearly empty** (`hipGraphLaunch` x40, only 12.6% of span attributed) — decode ran as a CUDA-graph replay and ROCm does not name graph-internal kernels |
| `1789995467.6245127/` | server with `--disable-decode-cuda-graph`. DECODE populated (94 MB, 52,603 kernel events). **Use this one.** |

ATT output: `att_out/` (regenerate, see section 6).

### Repro commands

```bash
cd /var/home/my_InferenceX/InferenceX/local_bench_qwen3.8

# correctness + perf at the production shape
FLYDSL_K5_OPT_BV=32 python3 bench_gdn_chunk_h.py --T 16384

# accuracy vs fp32 ground truth
python3 check_gdn_chunk_h_accuracy.py --T 2048 8192 16384

# concurrency
FLYDSL_K5_OPT_BV=32 SCHED=1 python3 sweep_gdn_chunk_h_conc.py --conc 1 4 8 16

# server (patched arm)
cd ../qwen38-gdn-repro && HF_HOME=/data/hf_cache ARM=conv_decode ./scripts/run_server.sh
```

`AITER_DIR` defaults to `/var/home/my_aiter/aiter` (the checkout with the
qwen3.5/3.8 GDN work). It must win over `/sgl-workspace/aiter` on `sys.path`.

---

## 3. Where prefill time actually goes

One 32768-token prompt = 2 chunks, 910.7 ms total GPU kernel time, 72 launches
per GDN kernel (36 GDN layers x 2 chunks).

| area | ms | % |
|---|---:|---:|
| Dense GEMM + quant | 252.75 | 27.8% |
| MoE | 205.41 | 22.6% |
| Attention (full/sparse GQA) | 146.89 | 16.1% |
| Elementwise / fused / misc | 124.67 | 13.7% |
| **GDN, stock Triton** | **112.23** | **12.3%** |
| Copy/cat | 26.09 | 2.9% |
| QSA indexer (TileLang) | 16.89 | 1.9% |
| **GDN conv (AITER, your patch)** | 12.88 | 1.4% |
| Norm | 12.84 | 1.4% |

GDN Triton breakdown (72 launches each):
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` **54.49 ms (6.0%, 757 us/launch)**,
`chunk_fwd_kernel_o` 27.48, `recompute_w_u_fwd_kernel` 13.70,
`chunk_gated_delta_rule_fwd_kkt_solve_kernel` 13.28, `fused_gdn_gating_kernel` 1.85,
`chunk_local_cumsum_scalar_kernel` 1.44.

Harness fidelity check: 0.714 ms x 72 = 51.4 ms vs 54.5 ms in the live trace.

---

## 4. Accuracy — settled

Scored against an independent fp32 reference, relative Frobenius error, T=16384:

| tensor | bf16 format floor | sglang | flydsl default | flydsl `trunc=False` |
|---|---:|---:|---:|---:|
| `h` | 2.266e-3 | 2.075e-3 | 5.493e-3 | 2.075e-3 |
| `v_new` | 1.688e-3 | 1.683e-3 | 3.474e-3 | 1.683e-3 |
| `state` | 2.582e-3 | 2.349e-3 | 6.273e-3 | 2.349e-3 |

- SGLang sits **at the bf16 storage floor** — as accurate as the output format allows.
- FlyDSL's default was 2.6x worse *and above the floor* (i.e. losing precision internally).
- Cause: `bf16_convert_trunc=True` — truncating fp32->bf16 (biased, mean 0.5 ULP)
  instead of RNE (unbiased).
- With `bf16_convert_trunc=False`: `v_new` and `state` **bit-identical** to
  sglang, `h` differs by 9.3e-10. **Zero speed cost** (0.780 vs 0.780 ms).
- Error ratio is stable across T (2.65x at 2048 / 8192 / 16384) — degraded, not divergent.

**Important nuance:** the truncating default is *deliberate*, not a bug.
`chunk_gated_delta_h.py`:

```python
# Truncation keeps outputs bit-identical to the HIP/C++ K5 kernel; RNE is closer
# to the fp32 reference but does NOT bit-match HIP.
_BF16_CONVERT_TRUNC_DEFAULT = True
```

So RNE belongs at the **SGLang call site**, not as a changed global default —
flipping it globally would break HIP/C++ K5 equivalence.

---

## 5. Performance and tuning

### What was tried

| lever | effect |
|---|---|
| `BV` 16 -> 32 | **+10%** — the only substantial win |
| `sched_barrier(mask_mfma)` enabled on gfx950 | +4% at BV=16, **+0.8% at BV=32** |
| `GEMM1_PF_SPLIT` in {0,1,2} (prefetch placement) | **nothing** — +/-1% at every BV |

**BV**: `_tuned_bv(...)` returns `None` for every Qwen3.8 shape (no
gfx950 / H=48 / Hg=16 row exists; runtime key is `('gfx950', 256)`), so it falls
back to `_hipeq_select_bv`, which returns **16** at every length. BV=32 is better
everywhere. Independently confirmed by rocprof geometry: Triton's autotuner
launches 192 WGs (BV=32), FlyDSL 384 (BV=16).

**sched_barrier**: gated to gfx942 only via `sched_gfx942=_IS_GFX942` at
`linear_attention_prefill_kernels.py:815`. gfx950 gets no scheduling hints at
all. Looks like an oversight given the stall profile. `_IS_GFX942` is a module
global, so A/B it with a runtime monkeypatch — no repo edit needed.

**GEMM1_PF_SPLIT**: hardcoded `= 1` at `chunk_gated_delta_h.py:477`. Decides how
many GEMM1 K-blocks run before the next chunk's `w` prefetch. Tested 0/1/2 at
BV=16/32/64 — all within noise. I made it env-tunable to test, then **reverted**;
the aiter tree is clean. Not part of the compile cache key, so vary only across
fresh processes (safe today: `aiter/jit/flydsl_cache` is empty, in-process only).

### Final config: `BV=32` + sched_barrier ON + RNE

Median of 3, spread <= 0.011 ms:

| shape | sglang | flydsl | ratio |
|---|---:|---:|---:|
| T=16384, N=1 — **production chunk** | 0.714 | 0.716 | **0.997x** |
| T=32768, N=1 | 1.422 | 1.352 | 1.051x |
| 4 x 16384 tokens | 1.775 | 1.560 | 1.138x |

### Blocker: layout

SGLang emits `w`/`u` token-major `[B,T,H,*]`; both AITER entries require
head-major `[B,H,T,*]`. That transpose is **0.26 ms on a 0.72 ms kernel (+36%)**
and is not part of the kernel. Adopting AITER while paying it is a net
regression (0.67x). `v_new` comes back head-major too.

---

## 6. ATT: why it's slow

```bash
rocprofv3 --att=true --att-library-path /opt/rocm/lib \
  --kernel-include-regex 'chunk_gdn_fwd_h_flydsl_opt' --att-target-cu 1 \
  -d att_out -o att -- python3 att_flydsl_gdn_h.py --T 2048
```

The `--kernel-include-regex` matters — without it the trace drowns in torch init
kernels. Real dispatch name is **`chunk_gdn_fwd_h_flydsl_opt`** (`kernel_0` is
only the torch-profiler label; both FlyDSL kernels get that generic name, so in
the EXTEND trace `kernel_0` is the conv and in DECODE it's the GDN decode).

**Stalls (296k cycles) / latency (449k):**

| class | % latency | % stalls |
|---|---:|---:|
| `s_waitcnt` | 40.7% | **61.8%** |
| `s_barrier` | 13.7% | **20.8%** |
| VALU | 17.8% | 6.1% |
| LDS write | 11.6% | 5.3% |
| **MFMA** | **3.5%** | **2.4%** |

Worst sites: `s_waitcnt vmcnt(12)` 45.8k cyc / 128 hits (~358 cyc each),
`s_waitcnt vmcnt(7)` (~301 cyc), `s_barrier` (~277 cyc).

**Wave states — SIMDs are empty, not busy:** SIMD1 30.8% empty, SIMD2 30.7%,
**SIMD3 76.7%**.

**Resources (rocprofv3 `--kernel-trace`):**

| | Triton | FlyDSL |
|---|---:|---:|
| grid | 1024x48 -> **192 WGs** (BV=32) | 2048x48 -> **384 WGs** (BV=16) |
| workgroup | 256 thr (4 waves) | 256 thr (4 waves) |
| VGPR / SGPR | 108 / 96 | **60** / 112 |
| LDS | 76160 B | **40960 B** |
| scratch | 0 | 0 |

Root cause is structural: grid is `(V/BV) x H x N` = 192-384 WGs on **256 CUs**,
4 waves each. The chunk loop is sequential, so parallelism is capped at
`(V/BV) x H x N`, and at conc=1 prefill N=1. Resources are not the limit
(scratch=0, no spills) — there simply aren't enough workgroups.

`NUM_WARPS = 4` is baked in; the kernel comment says *"BT=64 is baked into the
wave mapping"*, so 8 waves/WG means redoing the warp partition.

---

## 7. Concurrency

`N` = sequences packed into one prefill pass. Two readings, both measured:

**packed — fixed 16384-token budget split N ways (total work constant):**

| N | tok/seq | WGs | WG/CU | sglang | flydsl | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16384 | 192 | 0.8 | 0.724 | 0.718 | 1.01x |
| 4 | 4096 | 768 | 3.0 | 0.474 | **0.434** | 1.09x |
| 8 | 2048 | 1536 | 6.0 | 0.432 | 0.431 | 1.00x |
| 16 | 1024 | 3072 | 12.0 | 0.445 | 0.452 | 0.98x |

**per-seq — N x 16384 tokens:** N=1 1.00x, N=4 1.16x, N=8 1.05x.

- **Same work costs 40% less at N=8 than N=1** (0.724 -> 0.432 ms). Confirms the
  occupancy diagnosis quantitatively. Crossover is ~N=4 / 768 WGs / 3 per CU.
- **But flydsl's advantage does not grow with N** — it peaks at N=4 and converges
  to parity by N=8-16. Concurrency is not an argument for adopting AITER.
- **The 40% is probably unreachable today.** Server log shows
  `Prefill batch, #new-seq: 1, #new-token: 16384` — at ISL=32768 one request's
  chunk fills the budget exactly, so production sits in the N=1 row even at
  higher client concurrency. Reaching N>1 means lowering
  `--chunked-prefill-size`, which changes prefill batching for attention and MoE
  too. **Verify with a real conc=16 server trace before acting on this.**

---

## 8. Bugs found

1. **SGLang Triton int32 overflow (upstream, latent).** At 262144 tokens in one
   pass the Triton kernel takes `Memory access fault by GPU node-2`; FlyDSL
   completes fine. Consistent with
   `tl.make_block_ptr(h + i_t * stride_h, ...)` where `stride_h = H*V*K = 786432`
   overflows 2^31 at `i_t > 2730`. The base offset *is* cast to int64
   (`((boh * H + i_h) * V * K).to(tl.int64)`) but this per-chunk term isn't.
   Measurements bracket the threshold: NT=2048 ok, NT=4096 faults, 2048 < 2730 < 4096.
   **Not reachable at `--chunked-prefill-size 16384`** (256 chunks, 10x margin).

2. **aiter `hip` backend unusable in this checkout.** `module_chunk_gdr_fwd_h.so`
   has a pybind11 ABI split (v11) vs `module_aiter_core.so` (v12); every call
   raises. Fix: `AITER_REBUILD=1` or `rm -rf /var/home/my_aiter/aiter/aiter/jit`.
   It also ignores `_Hg`, so its GQA support is doubtful anyway — FlyDSL K5 is
   the GQA-aware one and the right target for Qwen3.8.

3. **Eager decode GPU fault (server).** With `--disable-decode-cuda-graph` the
   server died after ~4000 generated tokens with
   `Memory access fault ... address 0x758645e55000`. Traces were already written.
   Unclear whether it's the AITER GDN decode kernel outside graph capture;
   the clean A/B is `ARM=baseline` with the same flag.

---

## 9. Next steps, in the order I'd do them

1. **Emit head-major `w`/`u` from `chunk_gated_delta_rule_fwd_intra`.**
   Kills the 0.26 ms transpose (+36% on a 0.72 ms kernel) — larger than every
   tuning win combined, and a prerequisite for adoption being net-positive.

2. **Prototype fusing `fwd_h` + `chunk_fwd_o`.** *(projected 20-30% on the pair)*
   - `h` is `[1,256,48,128,128]` bf16 = **384 MiB** at T=16384, written by
     `fwd_h` and read straight back by `chunk_fwd_o`: a **768 MiB round-trip,
     ~101 us** at 8 TB/s, per launch.
   - `chunk_fwd_o` is mostly that read: 25.8 GFLOP in 382 us is ~2.7% of bf16
     peak, while 384 MiB in 382 us is ~1.0 TB/s. Its grid is
     `(cdiv(V,64), NT, B*H)` = **24,576 WGs** (96/CU) — already saturated.
   - Fusion is algorithmically clean: `o[t] = q[t] @ h[t]`, `h[t]` is already in
     `fwd_h`'s registers/LDS, and **both partition V** so `o[:, v_slice]` needs
     only `h[v_slice, :]` — no cross-workgroup reduction.
   - This is where the MFMA argument lands: `fwd_h` has MFMA at 3.5% with 61.8%
     of stalls on `s_waitcnt`; `o`'s work is pure MFMA and can fill that stall
     shadow nearly free.
   - **Check first:** `o`'s work drops from 24,576 WGs to 192 (0.75 waves/SIMD).
     That's only acceptable because `o` is memory-bound today and the fusion
     deletes the memory it's bound on. If `q` tiles + `o` accumulators push
     `fwd_h` (40 KB LDS, 60 VGPR) into spills, the win evaporates.
   - **Do not** fuse `fwd_intra` — it's already chunk-parallel and well-occupied;
     folding it into the sequential chunk loop serializes it for only 375 us of
     upside, and the `A` tril-solve has an intra-chunk dependency.
   - **Fusion will not fix the empty SIMDs.** It adds work per workgroup, not
     workgroups. MFMA-idle and SIMD-empty are different problems.

3. **Add gfx950 / H=48 / Hg=16 tuned rows** so BV=32 is selected automatically.
   Untuned CSV format is `dtype,K,V,BT,H,Hg,is_varlen,use_h0,store_fs`; the row
   is `torch.bfloat16,128,128,64,48,16,True,True,True`. Tuner:
   `csrc/gdn_k5/chunk_gdn_h_opt_tune.py`.

4. **Widen the sched_gfx942 gate to gfx950** (`linear_attention_prefill_kernels.py:815`).

5. **Upstream the int32 overflow fix** (section 8.1) — cheap, prevents a future
   footgun if `--chunked-prefill-size` is ever raised above ~174k.

6. **Verify the N>1 headroom** against a real conc=16 server trace before
   changing `--chunked-prefill-size`.

### Structural ideas not yet tried

- `NUM_WARPS` 4 -> 8 to attack the 31-77% empty SIMDs directly (kernel rewrite,
  the warp partition is baked around BT=64).
- Deeper LDS double-buffering to cut the 20.8% `s_barrier` stall.
- Parallel scan over chunks: `h_{t+1} = a_t * h_t + b_t` is a first-order linear
  recurrence, so a Blelloch scan would give log(NT) depth instead of NT and break
  the `(V/BV) x H x N` parallelism ceiling. Big change; the real fix if this
  kernel ever matters more than 6%.

---

## 10. `_sparse_gqa_chunk_prefill` — verified, NO easy win

Same treatment applied to the QSA sparse-attention kernels, which are a
**bigger** target than GDN: `_sparse_gqa_chunk_prefill` 78.44 ms (8.6%, 13
launches, 6.03 ms each — the most expensive per-launch kernel in the trace) plus
`_sparse_gqa_prefill` 62.29 ms (6.8%). Together **15.5% of prefill**, ~2.5x the
GDN chunk-state kernel.

They split the prefill by chunk, which is why each runs 13x not 26x:
chunk 1 (prefix=0) -> `_sparse_gqa_prefill`, kv_len 16384;
chunk 2 (prefix=16384) -> `_sparse_gqa_chunk_prefill`, kv_len 32768.

Harness: `bench_sparse_gqa.py`, ATT driver `att_sparse_gqa.py`.
Shapes: q[16384,24,256], kv[32768,2,256], topk=2048 (`indexer_budget`, after the
`indexer_compress_ratio=4` block expansion), grid = (16384, 2) = 32768 WGs.

**Result: three candidate optimizations, all verified dead.**

### 10.1 The NVIDIA config table is a real bug with zero impact

`sparse_attn.py:_get_best_config` selects its tuning tuple by NVIDIA GPU name:

```python
table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS
```

On `AMD Instinct MI355X` that is False, so it uses the **NVIDIA L20** table and
at prefill-scale `total_q` falls through to the catch-all
`(float("inf"), (16, 1, 2))` -> BLOCK_N=16, **num_warps=1**, num_stages=2
(a 64-thread workgroup on AMD; the table was tuned for 32-thread warps).

Looks damning, **measures as nothing.** Full sweep of BLOCK_N {16,32,64,128} x
warps {1,2,4,8} x stages {1,2,3}: best 7.478 ms vs stock 7.512 ms = **1.00x**.
BLOCK_N >= 128 fails with `OutOfResources: shared memory`. num_warps is
irrelevant because the grid is 32768 workgroups — occupancy was never the
problem here (the opposite of the GDN kernel). Still worth fixing upstream as a
correctness/portability matter; just don't expect speed.

### 10.2 The kernel is at the memory roofline

Time scales **perfectly linearly with topk** (ms/topk flat at 3.64 across
topk 256..4096) — a throughput-limit signature, not latency:

| topk | ms | gather GB | effective TB/s |
|---:|---:|---:|---:|
| 256 | 1.019 | 8.6 | 8.43 |
| 1024 | 3.794 | 34.4 | 9.06 |
| 2048 | 7.463 | 68.7 | 9.21 |
| 4096 | 14.921 | 137.4 | 9.21 |

With realistic indices (below) it's 5.800 ms = **11.85 TB/s effective**, above
MI355X HBM3E peak (~8 TB/s) — the 64 MiB KV working set fits the 256 MB LLC, so
much of it is cache-served. Either way the kernel is bandwidth-saturated and
there is no tuning headroom; time = gather volume / achieved bandwidth.

ATT (`--kernel-include-regex 'sparse_gqa'`) agrees but looks different from GDN:
**VALU 45.5% of latency** (online-softmax + index math), `s_waitcnt` 26.8%
latency / **60.0% of stalls**, MFMA only 5.4% / 2.8%. Worst site
`s_waitcnt vmcnt(2)`, 322 cyc per hit.

### 10.3 Sorting the topk indices — break-even, dropped

Gather-pattern sensitivity, **synthetic** indices: scattered 7.516 ms,
sorted 6.477 (1.16x), contiguous 4.117 (1.83x). Looked like a cheap 16% win.

**It isn't.** `fast_topk` does not emit score-scrambled indices — measured
output is **91.4% ascending-adjacent**, arriving as sorted runs
(`[656,661,665,682,686,842,845,846,893,894, 192,218,...]`). After the 4-token
block expansion the token indices are **99.0% ascending**. Re-measured against
that realistic structure:

| pattern | % ascending | ms | vs realistic |
|---|---:|---:|---:|
| realistic (fast_topk-like, 45 runs) | 99.0% | 5.800 | 1.00x |
| fully sorted | 100.0% | 5.550 | **1.05x** |
| fully scattered | 88.6% | 5.932 | 0.98x |

Full sort saves 0.25 ms and `torch.sort` on the 512 block indices costs
**0.23 ms** — break-even, and worse if sorting the 2048 expanded token indices
(0.607 ms). Dropped.

Block granularity alone also buys nothing (block-scattered 1.01x vs
token-scattered): each token's 512 B per-head slice already coalesces, and
consecutive tokens are strided by the other kv head.

**Methodology note that mattered:** the synthetic fully-scattered pattern
inflated the apparent opportunity ~3x. The realistic model also matches the live
trace far better (5.800 ms vs 6.034 ms, against 7.5 ms synthetic). Any future
work on this kernel must use realistic index structure.

### 10.4 What's actually left

Only algorithmic levers, all needing real captured indices to evaluate:

- **Query tiling.** One program currently handles ONE query token; gathered KV
  is reused only across that token's 12 q-heads. If adjacent query tokens select
  overlapping blocks — plausible for sparse attention — tiling T queries would
  divide gather volume by up to T. This is the only large lever, and it cannot
  be sized without dumping real `topk_indices` from a live forward and measuring
  inter-query block overlap. **Do that first.**
- **Both kv heads per program** (halves index-read traffic, modest).
- **Lower `indexer_budget`** — a model hyperparameter; accuracy tradeoff, not an
  engineering one.

---

## 11. Side findings (not GDN)

- **QSAIndexer "22 ms" is not a copy problem.** Median 22.82 ms CPU wall per
  call, 26 calls, 561 ms total — 91% of it one `hipMemcpyWithStream`. But that
  copy is **8 bytes**, device->host: a sync point, not a transfer. Source is
  `qsa_indexer.py:431`,
  `self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))`
  (same pattern at `:186`). **The GPU is 99.9% busy during it** — across all 32
  blocking copies (524.9 ms of CPU wall) total GPU idle is **0.4 ms**, and
  prefill overall is 95.5% GPU-busy. So it costs nothing today, but it caps CPU
  run-ahead at one QSA-layer interval and would turn into bubbles if the GPU got
  faster. Avoidable: `positions.max()` is known host-side, or pre-size the
  cos/sin cache to 262144 once at init.
- **Decode traces need `--disable-decode-cuda-graph`** to contain anything: the
  48-layer target forward is a graph replay and ROCm doesn't name graph-internal
  kernels. Costs 4x decode speed (29 vs 116 tok/s) so timings are not
  production-representative, but kernel shares are.
- **Proof the GDN decode patch is live** (from the eager decode trace):
  `flydsl_gdn_decode_varlen` x720 = 36 GDN layers x 20 steps, zero declines,
  no Triton fallback kernel anywhere. 7.69 ms, 1.9% of decode GPU time.
