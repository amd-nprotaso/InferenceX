# Qwen3.8 32768/32768 benchmark attempt

[中文](report_zh.md)

Status: blocked before measurement. No valid performance comparison was produced.

SGLang was installed editable from `/var/home/my_sglang/sglang` at `9699a8c65`, including working-tree changes. `/opt/venv/lib/python3.12/site-packages/00_sglang_local.pth` gives this source precedence over the old container PYTHONPATH. The exact requested AITER `hc_mix.py` was copied into `/sgl-workspace/aiter/aiter/ops/flydsl/hc_mix.py`. FlyDSL was upgraded from 0.3.1 to the AITER repository requirement, 0.3.4.1.

Intended experiment: amd/Qwen3.8-Flash-Next-Quark-MXFP4, GPU 0, TP=1, concurrency=1, real NEXTN speculative decoding, identical 32768-token chat-formatted input and exactly 32768 generated tokens. Existing GDN and HC optimizations enabled in both arms. Baseline unsets SGLANG_AITER_QSA_PA_DECODE; second arm sets it to 1.

Baseline initialization failed compiling `_hc_up_silu` in the requested HC kernel. FlyDSL 0.3.1 raised UnicodeDecodeError; 0.3.4.1 segfaulted in `flydsl.expr.primitive._infer_int_tuple_type`, reached via `_load` at hc_mix.py:72 from hc_mix.py:188. The existing repository GPU correctness test independently reproduces the segfault. A temporary index-cast experiment also failed and was reverted; installed hc_mix.py is byte-identical to the requested source.

No measured request completed. The flag-enabled arm was not started because the baseline kernel blocks initialization. Logs, source provenance, installation logs and the sequential benchmark driver are retained here. This requires a working HC kernel/FlyDSL combination before the performance comparison can run.
