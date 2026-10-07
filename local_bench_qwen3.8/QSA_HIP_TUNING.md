# QSA HIP/CK tuning experiment

**English** | [中文](QSA_HIP_TUNING_zh.md)

This experiment recompiles the existing CK `FmhaFwdKernel` for the Qwen3.8 TP=1 QSA attention call. It uses BF16 Q/K/V, 24 query heads, 2 KV heads, D=256, one query per packed sequence, and 2048–2051 selected keys. ISL=32678, OSL=32678 and concurrency 1,4,8,16 match the requested microbenchmark. Both ordinary decode and four-token verification are measured across six context samples.

## Reproduction

```bash
python3 tune_qsa_hip.py
python3 summarize_qsa_hip.py
```

The driver depends on the baseline `.qsa_aiter_jit` build from `bench_qsa_ck.py`. It reads the actual Ninja compile/link arguments, recompiles only the matched D256/gfx950 translation unit, and links the result with unchanged baseline objects. Each variant has a separate library directory. Original AITER source, baseline objects and baseline library are preserved.

Each variant folder under `qsa_hip_tuning` contains the modified `kernel.cpp`, `kernel.patch`, exact commands and baseline library hash in `build.json`, build logs, benchmark logs, per-case JSONL and profiler traces. Compiler-only variants have an empty source patch; their change is recorded in `build.json`.

```bash
# Reuse an already built variant, with a fresh measurement directory:
python3 tune_qsa_hip.py --variants baseline,post_sched --reuse \
  --run-label repeat --repeats 30
python3 summarize_qsa_hip.py --label repeat
```

## Measurement and scope

Every case checks the output against independent FP32 softmax attention before timing. Timing uses 20 calls per HIP graph replay, 15 measured replays for the exploratory sweep, and separate profiling to confirm the executing kernel. Inputs and seeds are identical across variants. Tables compare matched context cases; aggregate speedup is the geometric mean of baseline/variant latency, not the ratio of unrelated timing extrema. Full raw samples are retained.

Only the post-compaction attention call is measured. Index selection, compaction, other layers, and full serving are excluded. Repeated input reuse produces a warm-cache microbenchmark. Correctness validation covers these synthetic inputs, not model-quality evaluation.

The modified kernel retains its original external dispatch trait to let the existing AITER wrapper invoke the experimental specialization. Therefore these libraries are scoped to this length-one, D256 benchmark; they are not general production AITER replacements. Promoting a winner requires updating CK codegen/dispatch correctly and validating the serving path.

## Confirmed results

MI355X / gfx950, ROCm PyTorch 2.11. Each candidate was measured in the exploratory sweep and again with 30 graph timing samples per case. All 48 cases passed for every candidate. The largest observed absolute error against the FP32 reference was 0.001054 (rounded up).

| Variant | Change | Geometric mean speedup |
| --- | --- | --- |
| post_sched | HIP `-mllvm -enable-post-misched=1` instead of `=0` | 1.040× |
| n128 | KV tile 64 → 128; otherwise original tile | 1.289× |
| m16_w1 | Query tile 128 → 16; 4 waves → 1; 16×16×16 MFMA tile | 1.456× |
| m16_n128 | Query/KV tile 16×128; 1 wave; 16×16×16 MFMA tile | 1.571× |

For the fixed `m16_n128` configuration, median latency across the six context cases (microseconds):

| Mode | CONC | Baseline | Tuned | Matched-case speedup |
| --- | --- | --- | --- | --- |
| decode | 1 | 130.40 | 130.53 | 1.011× |
| decode | 4 | 148.04 | 132.08 | 1.133× |
| decode | 8 | 148.88 | 132.96 | 1.132× |
| decode | 16 | 294.06 | 141.37 | 2.103× |
| verify | 1 | 147.91 | 132.24 | 1.119× |
| verify | 4 | 293.92 | 141.36 | 2.079× |
| verify | 8 | 442.85 | 168.87 | 2.622× |
| verify | 16 | 882.27 | 396.15 | 2.228× |

`n128` wins at small row counts; `m16_w1` is slightly faster than `m16_n128` at verification C=16. There is no single configuration that is best for all cases. The 1.57× aggregate is an equally weighted microbenchmark geometric mean, not a serving speedup. No production dispatch policy was changed.

Other explored changes: removing kernarg preload was neutral; fast math improved only about 0.5%; occupancy=2 regressed overall; combining post scheduling with N=128 was slower than N=128 alone. Larger query-tile and one-wave combinations were also measured; see the full exploratory CSV.

```bash
python3 bench_qsa_ck.py --jit-dir qsa_hip_tuning/m16_n128/jit \
  --isl 32678 --osl 32678 --conc 1,4,8,16 --profile \
  --output qsa_hip_tuned_rerun
```

[Confirmed comparison](qsa_hip_tuning/comparison_confirm.csv) · [All exploratory variants](qsa_hip_tuning/comparison_results.csv) · [Winning source patch](qsa_hip_tuning/m16_n128/kernel.patch) · [Exact build commands](qsa_hip_tuning/m16_n128/build.json)

The final baseline rerun showed median latency drift of 0.02% and maximum absolute per-case drift of 1.03%. The original baseline library hash is unchanged. See [validation](qsa_hip_tuning/validation.json).
