# Qwen3.5 TP=4 convolution comparison

**English** | [中文](COMPARE_zh.md)

Run from this directory:

```bash
python compare_conv.py --output compare_tp4.json
python compare_conv.py --lengths 1 7 17 --history mixed --warmup 2 --repeats 5 --trials 2
python compare_conv.py --shapes qwen35-397b-tp4 --block 256 --tokens 16 --output compare_block256.json
```

The script compares `causal_conv1d_flydsl.py` with installed SGLang's
`sglang.kernels.ops.mamba.causal_conv1d_triton.causal_conv1d_fn`, following
`../chunk_gated_delta_rule/compare_chunk_h.py`.

Defaults target Qwen3.5-397B-A17B-MXFP4 with BF16 GDN tensors at TP=4:
3072 channels (`2*4*128 + 16*128`), width 4, no bias, and SiLU. The per-rank
head geometry comes from `../chunk_gated_delta_rule/shapes.py`. Presets are
`qwen35-397b-tp4` (one 32768-token sequence), `qwen35-397b-tp4-4k` (one 4096-token
sequence), and `qwen35-397b-tp4-packed4` (four 8192-token sequences). The packed
case is a synthetic split of the 32K token budget, not a traced scheduling claim.
TP=4 describes one rank's geometry; this script uses one GPU (`--device`).

`--lengths` supplies custom packed sequence lengths. `--tokens` controls FlyDSL
tokens per block, not sequence length. `--history fresh|cached|mixed` controls
cached history. Defaults use contiguous state `[1243,3072,3]`; `--pool-slots`
changes capacity and `--state-layout channel-last` tests channel-contiguous state.
Input is a token-major allocation viewed as `[channels,total_tokens]`.
`--flydsl-file` allows comparing an alternate implementation with the same API.

Eager and graph replay outputs must match SGLang within `--rtol`/`--atol`
(both 0.016 by default); the entire updated state pool must match exactly.
Nonfinite results fail validation. State is reset before every invocation,
outside timing. GPU events measure one graph replay at a time, excluding
compilation and Python dispatch. Backend order alternates across trials, and
the summary uses the median of trial medians. This is a warm-cache benchmark:
the state reset itself touches the cache before each measured invocation.

JSON saves all timing samples, correctness checks, tensor strides, package
versions, source paths and hashes, and options after each completed shape.
Speedup is SGLang time divided by FlyDSL time; values above 1 favor FlyDSL.
These are kernel measurements, not full-model throughput results.

The optimized kernel also accepts `--prefetch 1|2|4|8|16` (kernel default 16)
and `--channels-per-thread 1|2` (omitted uses automatic mapping). Automatic
pairing requires even channels, channel-contiguous BF16/FP16 input, and at least
8192 packed tokens. Keep these flags omitted when loading an older kernel that
does not expose them. See [optimization results](optimization/REPORT.md).
