# FlyDSL causal depthwise convolution

**English** | [中文](README_zh.md)

Standalone replacement for SGLang GDN's packed prefill `causal_conv1d_fn` on
ROCm. It uses the cached compiler/dispatch and explicit tensor-view conventions
from `/var/home/conv_FlyDSL/FlyDSL/kernels/conv/conv3d_implicit.py`. The arithmetic
is a dedicated depthwise kernel: one thread owns a channel, reuses a short
register window, and applies SiLU before the output conversion. No GEMM channel
padding or layout-transpose kernels are needed.

## Run with the existing server

From `/var/home/InferenceX/local_bench`, start a fresh server:

```bash
TP=4 CONC=4 bash kernel_tuning/conv/run_server.sh
```

Run the existing client in another terminal:

```bash
ISL=32768 OSL=1024 CONC_LIST="4" ./qwen3.5_fp4_sglang_bench.sh
```

The wrapper sets `PYTHONPATH` and `SGLANG_FLYDSL_CAUSAL_CONV=1` for the server and
its children. An opt-in `sitecustomize.py` import hook replaces the GDN backend's
`causal_conv1d_fn` after that module loads. Look for
`[FlyDSL] GDN causal prefill convolution enabled` in each importing worker.
This verifies selection; use a GPU profile to verify execution. Existing server
processes are not patched. For baseline measurements, stop this server and use
the original `qwen3.5_fp4_sglang_server.sh` command in a fresh shell.

The installed SGLang package and original launcher are not edited. Decode and
EAGLE verification continue using their existing convolution-update paths.
The hook's module path is specific to the inspected SGLang installation; check
it again after upgrading SGLang. The bootstrap directory intentionally supplies
Python's `sitecustomize` module, so do not combine it with another required
`sitecustomize` without merging the startup hooks.

## Share with colleagues

Send the `conv` directory together with the two original server/client scripts.
The wrapper resolves the server script relative to its own location, so preserve
the `kernel_tuning/conv` directory structure. From this workspace, create a bundle:

```bash
cd /var/home/InferenceX/local_bench
tar -czf /tmp/qwen35-flydsl-conv.tar.gz \
  --exclude='__pycache__' --exclude='*.pyc' \
  kernel_tuning/conv \
  qwen3.5_fp4_sglang_server.sh qwen3.5_fp4_sglang_bench.sh
```

Share `/tmp/qwen35-flydsl-conv.tar.gz` through your usual file-sharing channel.
It includes the kernel, bootstrap hook, wrapper, tests, bilingual documentation,
and local reference results. The original `conv_FlyDSL` checkout is not needed
at runtime; the new kernel imports the installed `flydsl` package directly.
Model weights and the Python/ROCm environment are not included.

### Recipient environment

Use the same ROCm/SGLang container or a matching environment where possible.
The local validation used AMD Instinct MI355X with these installed versions:

| Component | Version |
| --- | --- |
| FlyDSL | `0.3.1` |
| SGLang | `0.5.18.dev20260829+g4d53767b09` |
| PyTorch | `2.9.1+rocm7.2.0.git7e1940d4` |
| HIP reported by PyTorch | `7.2.26015-fc0010cf6a` |

Share the actual container image tag/digest or matching build instructions along
with the archive; version labels alone may not capture local SGLang changes.
The smoke test needs ROCm PyTorch and FlyDSL. The comparison suite also needs
SGLang and Triton. Full serving additionally needs the original launcher's AITER
dependencies, model access, and sufficient GPU memory; the TP4 command requires
four GPUs. Recipients configure their own `MODEL_PATH` or Hugging Face access.
Other GPU architectures and package versions require fresh validation.

### Unpack, check, and run

Replace `/path/to/qwen35-flydsl-conv.tar.gz` with the downloaded archive path.
The destination can be anywhere; it does not need the sender's absolute paths.

```bash
mkdir -p ~/qwen35-flydsl
cd ~/qwen35-flydsl
tar -xzf /path/to/qwen35-flydsl-conv.tar.gz
python3 kernel_tuning/conv/check_conv.py --smoke
python3 kernel_tuning/conv/check_conv.py --bench \
  --channels 3072 --length 32768 --batch 1 \
  --output colleague_results.json
TP=4 CONC=4 bash kernel_tuning/conv/run_server.sh
```

After the server is healthy, run the client from another terminal in the same
directory. Using `bash` also works if a file-sharing tool drops executable bits:

```bash
cd ~/qwen35-flydsl
ISL=32768 OSL=1024 CONC_LIST="4" bash qwen3.5_fp4_sglang_bench.sh
```

Confirm the FlyDSL enablement log described above. If correctness checks fail,
resolve the environment or compatibility issue before enabling it for serving.
For comparisons, stop the FlyDSL server and start the original launcher with
the same settings in a fresh shell. Ask colleagues to return
`colleague_results.json`, their GPU and package/container versions, the exact
server/client commands, and paired baseline/FlyDSL serving results. Keep kernel
timings separate from end-to-end throughput and quality conclusions.

## Interface and constraints

`causal_conv1d_flydsl.causal_conv1d_fn` accepts the inspected SGLang prefill API:

- Input `[channels, total_tokens]`; weights `[channels, width]`.
- BF16, FP16, or FP32; filter widths 2–5; optional bias and SiLU/Swish.
- Packed sequence offsets, zero-length sequences, optional cached history,
  indirect cache slots, and padded slots (`-1` by default).
- State `[slots, channels, history_capacity]` with capacity at least `width-1`.
  Only its first `width-1` values are updated, matching SGLang prefill.
- Strided tensors with at least one unit-stride axis; contiguous metadata.
- Active cache slots must be unique and in range. CPU lengths must agree with
  GPU offsets. `validate_data=True` checks these by reading metadata back to the
  CPU; use it for debugging, outside graph capture.

It returns a separate output allocation and mutates state in place. Empty and
padded slots leave state unchanged; padded output values are unspecified, as in
SGLang. Only the first time tile accesses state, so other blocks cannot race
with the history update. Short sequences use a register snapshot of old state.
Accumulation and SiLU use FP32, with explicit output/state dtype conversion.

The default launch is 128 threads and 16 tokens per tile. Standalone callers can
set `block` (64/128/256) and `tokens` (`width-1` through 64). Kernels specialize on
geometry, strides, dtype, and optional inputs. Warm each signature before graph
capture; compilation executes the first call once and updates state once.

## Validation and timings

```bash
python3 kernel_tuning/conv/check_conv.py --smoke
python3 kernel_tuning/conv/check_conv.py --bench \
  --channels 3072 --length 32768 --batch 1 \
  --output kernel_tuning/conv/results_tp4.json
```

The suite checks 39 cases against an independent FP32 PyTorch reference and,
for widths 2–4, installed SGLang Triton. It covers channel tails, short and empty
sequences, mixed initial history, padded/indirect cache slots, extra state
capacity, three layouts and dtypes, bias/activation options, a nondefault stream,
and graph capture/replay. Additional checks cover missing state, repeated
dispatch, all-empty input, and inconsistent metadata. State comparison is exact;
outputs use dtype-dependent tolerances (BF16: `atol=rtol=0.016`, FP16: `0.002`,
FP32: `2e-6`). The no-state grouped-convolution check uses BF16 `atol=0.03125`.
Width 5 uses only the independent reference because the installed Triton forward
arithmetic handles widths 2–4.

The existing `../chunk_gated_delta_rule/shapes.py` trace notes contain TP4 Q/K
heads `[4,128]` and V heads `[16,128]`, giving `2*4*128 + 16*128 = 3072` convolution
channels. The server limits packed prefill to 32768 tokens, so client concurrency
4 does not imply a single convolution launch with four 32768-token sequences.
To benchmark a split packed batch with the same total token budget:

```bash
python3 kernel_tuning/conv/check_conv.py --bench-only \
  --channels 3072 --length 8192 --batch 4
```

Initial MI355X graph timings at 3072 channels and 32768 tokens were approximately
171 µs FlyDSL versus 178 µs Triton (1.04×). The earlier 2048-channel measurement
in `results.json` was 80 µs versus 95 µs (1.18×); that is a different shape.
`tuning_results.json` records six alternative geometries, which did not justify
changing the default. Timings use `triton.testing.do_bench_cudagraph`, including
state writes and fused activation, excluding compilation and Python dispatch.
The result files are local microbenchmarks, not end-to-end serving speedups.
The final validation recorded 175.4 µs versus 180.1 µs (1.027×) for one 32K
sequence, and 172.6 µs versus 179.1 µs (1.038×) for four packed 8K sequences.
These small differences should be confirmed with repeated serving measurements.

The worker import hook and GPU kernel have been tested locally. Full model
startup, generated-text quality, and the user's complete client workload have
not been tested with this override. The original server defaults to simulated
MTP acceptance; use `SIMULATE_ACC=0` when measuring real acceptance and quality.
