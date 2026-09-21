# AITER/FlyDSL GDN kernels for Qwen3.8-Flash-Next — reproduction package

Two hand-written AMD kernels for the Gated Delta Net (GDN) linear-attention
layers, plus the SGLang routing to reach them, benchmarked against stock Triton
on Qwen3.8-Flash-Next MXFP4:

| kernel | replaces | patches |
|---|---|---|
| FlyDSL packed GDN decode recurrence | Triton `fused_sigmoid_gating_delta_rule_update` | 0001, 0003, 0005 |
| FlyDSL prefill causal-conv1d | Triton `causal_conv1d_fn` | 0002, 0004 |

Headline on 1x MI355X at ISL 131072 / OSL 1024: **+4.7…+6.0% throughput,
+6.2…+8.5% TPOT**. Full tables and caveats in [RESULTS.md](RESULTS.md).

---

## Read this first: the kernels fail silently

Both kernels are dispatched behind a `try/except ValueError` that falls back to
Triton when the AITER wrapper declines a batch. **At default log level that
fallback is invisible.** A server with every gate turned on can quietly produce
baseline numbers.

This is not hypothetical. On Qwen3.8 the GDN decode kernel declined *100% of
calls* and the "patched" arm was measuring stock Triton. The wrapper required
`q.is_contiguous()`, but Qwen3.8's GDN layer hands SGLang three **views into one
fused QKV buffer** — token stride `10240 = (16+16+48)*128`, not each tensor's own
width — so `is_contiguous()` is False. The kernel itself takes
`q/k/v_token_stride` as parameters and indexes `tok*token_stride + head*D + d`;
it only needs head-major contiguity *within* a token. The guard was stricter than
the kernel. **Patch 0005** relaxes it to `stride[-2:] == (dim, 1)`.

So: **always run `verify_kernels.sh` before trusting a number.**

```
$ ./scripts/verify_kernels.sh
================ VERDICT ================
  [aiter-gdn] decode: ACTIVE (AITER FlyDSL kernel accepted the batch)
  [aiter-gdn] prefill conv: ACTIVE (AITER FlyDSL kernel accepted the batch)
=========================================
```

Anything reading `FELL BACK to Triton` means that kernel is not being measured,
and the printed reason is the shape or dtype the wrapper rejected.

---

## Requirements

- AMD MI355X / gfx950 (the kernels are `gfx950` FlyDSL; `hsaco` paths are arch-specific)
- **One** GPU is enough — TP1, ~80 GB used
- Source checkouts of SGLang and AITER (both are imported from source, so **no
  rebuild is needed** after patching — just restart the server)
- The model cached locally: `amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8`

Validated against SGLang `0.5.19.dev20260910+g908226fea2` and AITER `4ad998328`.

Point the scripts at your checkouts if they are not in the default location:

```bash
export SGLANG_DIR=/sgl-workspace/sglang    # default
export AITER_DIR=/sgl-workspace/aiter      # default
```

---

## Quickstart

```bash
./scripts/apply_patches.sh --check    # dry run: do the patches apply?
./scripts/apply_patches.sh            # apply 0001-0005
./scripts/run_tests.sh                # kernel accuracy tests (~10 s, no server)
./scripts/verify_kernels.sh           # PROVE both kernels are live
./scripts/run_all.sh                  # all 3 arms + comparison table
```

`run_all.sh` takes roughly 60–70 min: three arms, each ~4 min of server start
plus a ~15 min sweep. For a faster smoke test:

```bash
CONC_LIST="1" NUM_PROMPTS_MULT=2 ./scripts/run_all.sh     # ~15 min
```

That short form validates the plumbing, not the numbers: one or two unwarmed
requests per point swing by a couple of percent. Use the full sweep for results.

Results land in `results/<arm>/`. Render the tables:

```bash
python3 scripts/make_table.py --root results
```

and compare against the reference run shipped here:

```bash
python3 scripts/compare_arms.py --root reference_results \
    --baseline baseline --patched conv_decode
```

Undo everything with `./scripts/apply_patches.sh --revert`.

---

## The three arms

An arm is nothing but a set of kernel gates on an otherwise byte-identical
server, so any delta is attributable to the kernels. The mapping lives in one
place — `arm_env()` in `scripts/env.sh`.

| arm | gates | what runs |
|---|---|---|
| `baseline` | *(none)* | stock Triton decode + Triton prefill conv |
| `conv_only` | `SGLANG_AITER_GDN_CONV=1` | AITER prefill conv (= patches as originally authored, since decode declined) |
| `conv_decode` | `+ SGLANG_AITER_GDN_DECODE=1` | both AITER kernels |

Gates are read at **import** time, so they cannot be flipped on a live process —
each arm needs its own server. `run_all.sh` handles that.

---

## Correctness

```bash
./scripts/run_tests.sh          # 50 tests + 1 documented xfail, ~10 s, no server needed
```

`tests/` is a pytest suite checking both kernels for accuracy. What it compares
differs per kernel, deliberately:

| kernel | reference | why |
|---|---|---|
| prefill conv | **independent fp32 PyTorch reference**, plus "no worse than Triton" | a causal depthwise conv is short enough to reimplement without the reference itself becoming the bug |
| GDN decode | **the Triton kernel** | the gated delta-rule recurrence is intricate enough that a hand-written fp32 reference would more likely encode a misreading than catch a kernel bug; "drop-in for what ships today" is the property that matters |

Measured: the AITER conv is *more* accurate than Triton against fp32
(1.4e-3 vs 2.1e-3 mean relative error); the decode kernel agrees with Triton to
~1e-8 mean.

Where an exact answer exists the tests assert it instead of a tolerance:

- fused-QKV **views** vs contiguous **copies** — must be **bit-identical**
  (only addressing differs; this is what patch 0005 turns on)
- `disable_state_update=True` — state pool must be **byte-unchanged**
- slots no request owns, and `pad_slot_id` slots — **byte-unchanged**

Coverage: conv widths 2–5, bias on/off, silu/no activation, varlen batches from
`[1]` to `[64]*16`, mixed `has_initial_state`, scattered and padded cache slots,
transposed input; decode batch 1–33, 1–4 draft tokens, fp32/bf16 gating inputs,
and the guard behaviour (unsupported input must **raise**, never compute
silently-wrong results).

Two tests exist to keep the suite honest:

- `test_tolerances_would_catch_a_regression` fails the suite if the tolerances
  are ever loosened enough to accept a 0.5% drift or one corrupted element.
- `test_ragged_but_divisible_is_rejected` is a **strict xfail** documenting a
  real gap (below). If it ever passes, the suite fails and tells you to update it.

The suite was mutation-tested: reverting patch 0005 turns 25 decode tests red
while every conv test stays green.

### Known gap: ragged batches that happen to divide evenly

`flydsl_gdn_decode_varlen` only checks that the token total is divisible by the
sequence count. A genuinely ragged batch that passes that check — `cu_seqlens =
[0, 3, 7, 12]`, lengths 3/4/5, total 12, 3 sequences — is **accepted and computed
wrong** (~60x relative error). Uniform lengths are a caller precondition, and
speculative decode always satisfies it, so this does not fire in production.

It is not trivially fixable: detecting it needs a host read of `cu_seqlens`,
which raises `hipErrorStreamCaptureUnsupported` under CUDA-graph capture. If you
want a real check, the sibling conv wrapper's `validate_data=False` opt-in flag
is the pattern to copy — off during capture, on in tests and bring-up.

---

## Constraints worth knowing before you re-tune

The decode kernel is narrower than it looks. It declines, and you silently get
Triton, unless all of these hold:

- `--speculative-eagle-topk 1`. A linear draft chain only; a tree verify needs a
  parent-state reload the kernel does not implement.
- `linear_key_head_dim == 128`. Hard-coded `KDIM`.
- `--mamba-ssm-dtype bfloat16`. The state pool must be bf16.
- Uniform sequence length in the batch — the token loop is unrolled on it.
  (Speculative decode always satisfies this.)
- `--linear-attn-backend triton` (the default). The patches wrap `TritonGDNKernel`
  and the Triton conv call site; they are invisible to the CuteDSL/FlashInfer
  backends. Confirm with the `Linear attention kernel backend: decode=triton,
  prefill=triton, verify=triton` line `run_all.sh` echoes from the server log.

`--chunked-prefill-size` caps what the prefill conv ever sees. At 16384 a
131072-token prompt is split into 8 chunks, which is a large part of why the conv
gain is ~1% end-to-end rather than its standalone speedup.

---

## Layout

```
patches/
  0001  aiter    FlyDSL packed GDN decode recurrence (new kernel)
  0002  aiter    FlyDSL prefill causal-conv1d (new kernel)
  0003  sglang   route GDN decode + MTP verify to AITER
  0004  sglang   route GDN prefill conv to AITER
  0005  aiter    accept fused-QKV views  <-- required for Qwen3.8
  optional/
    0006  sglang  one-shot ACTIVE/FELL-BACK logging (SGLANG_AITER_GDN_DEBUG=1)
scripts/
  env.sh                     shared config + the arm -> gates mapping
  apply_patches.sh           apply | --debug | --check | --revert
  verify_kernels.sh          proves which kernels actually ran
  run_server.sh              one server for one arm
  run_client.sh              concurrency sweep against a running server
  run_all.sh                 all arms end to end, then the comparison
  compare_arms.py            per-metric two-arm diff
  make_table.py              markdown tables (used to generate RESULTS.md)
  run_tests.sh               pytest wrapper for tests/
  test_gdn_decode_strided.py standalone decode check (superseded by tests/)
tests/
  conftest.py                fp32 conv reference + comparison helpers
  test_prefill_conv.py       conv kernel vs fp32 and vs Triton
  test_gdn_decode.py         decode kernel vs Triton, layout and guard behaviour
reference_results/           the MI355X run behind RESULTS.md
RESULTS.md                   tables + how to read them
```

Patch 0006 is optional and applied only by `verify_kernels.sh`, which reverts it
afterwards — so measured runs use exactly 0001–0005. Its logging is gated on an
env var resolved once at import, so it costs nothing when off if you prefer to
keep it applied permanently (`./scripts/apply_patches.sh --debug`).
