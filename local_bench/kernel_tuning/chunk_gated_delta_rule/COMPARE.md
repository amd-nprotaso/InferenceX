# Compare FlyDSL and SGLang on Qwen3.5

**English** | [中文](COMPARE_zh.md)

`compare_chunk_h.py` compares `chunk_gated_delta_rule_fwd_h` in the local
`chunk_delta_h_flydsl.py` against the installed SGLang implementation. It uses
the existing Qwen3.5-397B-A17B-MXFP4 geometries from `shapes.py`; the GDN tensors
are BF16 even though the model weights use MXFP4. The FlyDSL specialization
requires a gfx950 GPU and the same Python environment as SGLang and FlyDSL.

From this directory:

```bash
python3 compare_chunk_h.py --output /tmp/qwen35_chunk_h.json
```

The defaults cover:

| Shape | Tokens | Grouped key heads | Value heads | K / V |
|---|---:|---:|---:|---:|
| `qwen35-397b-tp4` | 32768 | 4 | 16 | 128 / 128 |
| `qwen35-397b-tp2` | 32768 | 8 | 32 | 128 / 128 |
| `qwen35-397b-tp4-4k` | 4096 | 4 | 16 | 128 / 128 |

Every case has one sequence and chunk size 64. These are shapes for the
397B model; this script does not infer other Qwen3.5 model geometries.

Select shapes or override sequence lengths:

```bash
python3 compare_chunk_h.py --shapes qwen35-397b-tp4 --tokens 64 4096 8192 32768
python3 compare_chunk_h.py --shapes qwen35-397b-tp4 qwen35-397b-tp2 --repeats 200 --trials 5
```

`--tokens` accepts positive multiples of 64. Repeated head/length combinations
are deduplicated. Use `--device 1` for another logical GPU. `--pool-slots 2`
reduces state-pool allocation; the default preserves the traced pool size.

Compare an experimental kernel with the same host signature:

```bash
python3 compare_chunk_h.py \
  --flydsl-file /tmp/gdn_vn_delayed_stores/delayed/chunk_delta_h_flydsl.py \
  --shapes qwen35-397b-tp4 --output /tmp/qwen35_delayed_stores.json
```

## What is measured

The script uses SGLang's installed configuration without overriding its Triton
settings. Its configuration and relevant GDN environment overrides are recorded
in JSON. Disable the FlyDSL bootstrap before running: the script rejects a
patched SGLang entry point to prevent comparing FlyDSL against itself.

Inputs are built with SGLang's production preparation functions, rather than
random independent w/u tensors. Before timing, the script compares `h`,
`v_new`, and the updated state in eager execution and graph replay. Nonfinite
values or failed comparisons stop the run. BF16 comparison defaults are
`--rtol 0.016 --atol 0.016`; use `--rtol 0 --atol 0` to require bitwise equality.
JSON also reports exact equality and maximum absolute error for each output.

Each backend is warmed and captured in a separate GPU graph. The script restores
state before every replay, outside the timed GPU-event interval. Timing excludes
Python dispatch, output allocation, input preparation, and state reset. Default
settings are 10 warmup launches, 100 measured replays per trial, and 3 trials,
alternating backend order. The table reports the median of trial medians.
`Speedup = SGLang time / FlyDSL time`; values above 1 favor FlyDSL.

Use an idle GPU for performance conclusions. These are kernel microbenchmarks,
not full-model throughput or TTFT measurements. Graph replay and cache state
can differ from serving. A small speedup should be checked against the raw
trial variation stored in JSON.

The optional JSON report includes all timing samples, per-trial medians and
standard deviations, correctness results, tensor geometries, source paths and
SHA256 hashes, package versions, GPU architecture, and SGLang configuration.
It is updated after each completed shape, so earlier results survive a later
failure.
