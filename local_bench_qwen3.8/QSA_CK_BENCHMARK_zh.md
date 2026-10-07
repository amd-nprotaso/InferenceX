# Qwen3.8 CK QSA 微基准

[English](QSA_CK_BENCHMARK.md) | **中文**

`bench_qsa_ck.py` 测量 SGLang 将选中的 KV 压紧后调用的 AITER attention。脚本使用本地 `/var/home/my_aiter/aiter`，并在 `.qsa_aiter_jit` 中为当前 ROCm PyTorch 编译所需模块。原 checkout 中的 attention 二进制与 PyTorch 2.11 不兼容，加载时报告 `c10::hip::getCurrentHIPStream` 未定义符号；独立缓存解决了此问题，无需替换原有二进制。

```bash
python3 bench_qsa_ck.py --isl 32678 --osl 32678 --conc 1,4,8,16 \
  --profile --output qsa_ck_results
```

请在本目录执行。`--dry-run` 只输出形状矩阵，不加载 PyTorch 或使用 GPU。`--device` 选择可见 GPU。`--modes decode` 或 `--modes verify` 可以限制测量范围。`--aiter-path`、`--jit-dir`、`--trace-dir` 可覆盖本地路径。使用不同输出目录可保留历史结果。

## 证据与形状

提供的 `/var/home/qwen_trace_analysis/report.md` 记录 TP=1、concurrency=1、ISL=32768、OSL=32768、三个 NEXTN steps 和四个 verification tokens。脚本保留本次请求的 **32678**，没有改成 trace 中的 **32768**。报告依据缓存的模型配置确定模型维度；AMD kernel events 未记录输入张量形状。并发 4、8、16 是依据源码推导的测试场景，不是 trace 中捕获的场景。

重建依据来自本地 SGLang 源码：

- `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py` 的 `_forward_paged_attention`：压紧 KV，使用 `max_seqlen_q=1`、`max_seqlen_k=topk`、`causal=True` 和层内 scale 调用 `flash_attn_varlen_func`。
- `python/sglang/srt/layers/attention/qsa/qsa_indexer.py`：token budget 为 2048，compression ratio 为 4，展开后的索引宽度为 2051。
- `python/sglang/srt/layers/attention/qsa/kernel.py`：在选中的完整压缩组之后追加未满一组的尾部 token。
- AITER `aiter/ops/mha.py`：BF16 D=256 走 CK varlen 路径，而非 assembly v3 路径。

TP=1 时使用 BF16、24 个 query heads、2 个 KV heads、D=256、scale=1/16。Decode 的行数为 `R=CONC`，四 token verification 的行数为 `R=4*CONC`。每个 verification token 都对应独立的长度为 1 的 query sequence 和独立的 packed KV segment，并非四个 query 共享一个 KV segment。

| 张量或参数 | 取值 |
| --- | --- |
| Q | `[R,24,256]` |
| K、V 容量 | 各为 `[R*2051,2,256]` |
| `cu_seqlens_q` | `[0,1,...,R]`，int32 |
| `cu_seqlens_k` | 有效选中长度的前缀和，int32 |
| 长上下文下有效选中长度 | `2048 + visible_tokens % 4` |
| `max_seqlen_q`、`max_seqlen_k` | 1、2051 |
| Causal | True；单 query 可以关注其全部压紧后的 keys |

矩阵采样前四个 decode context lengths、中点和终点。Verification 从各采样位置展开四个连续位置，属于合成 verification windows，可能包含超出最终输出位置的 speculative tokens。稀疏预算饱和后，上下文增长只改变尾部余数，不会让该 kernel 读取 32K–65K 个 KV tokens。OSL 用于选取 context samples，不会执行对应次数的完整 decode。Prefill 使用其他 sparse kernels，不在本基准范围内。

## 验证与测量

每个 case 在计时前，均与独立计算的 FP32 softmax attention 比较。脚本使用随机 packed Q/K/V；原始 trace 没有记录选中索引和输入张量。输入已完成压紧，因此不包含 indexer、top-k、KV extraction、模型其他层或请求调度。

计时通过 GPU events 测量包含 20 次 attention 调用的 graph，在 warmup 后重复 30 次，报告折算到单次调用的 median、minimum 和 p95。计时包含 AITER API 发出的 kernels 及 graph 执行开销，不等于 profiler 中单个 kernel 的耗时，也不是端到端 serving throughput。重复使用输入意味着这是 warm-cache 微基准。

`--profile` 为每个 case 单独记录一次调用，要求存在 CK `FmhaFwdKernel` event，保存其耗时，并检查完整 symbol 是否与提供的 trace 一致。Profiler 耗时与 graph 计时分开记录。`metadata.json` 保存环境、参数和原 trace 的 phase statistics；`results.jsonl` 保存形状、计时样本、数值误差和 symbol 检查；`profile_*.json` 可在 Perfetto 中查看。

原 trace 中 target-verification kernel 的 median 为 149.279 microseconds，每 cycle 调用 12 次。Draft 和 draft-extension 的 median 分别为 147.360 和 151.2795 microseconds。这些结果来自完整模型负载下的 C=1 profiler 记录，不能直接等同于独立 graph 计时。

## 本地结果（2026-09-28）

全部 48 个 case 均通过 FP32 reference 检查，且完整 kernel symbol 与原 trace 一致。下表范围覆盖六个 context samples，数值为单次调用的 graph median，单位为 microseconds。

| CONC | Decode | Verify (4 tokens/request) |
| --- | --- | --- |
| 1 | 126.2–130.9 | 147.8–148.0 |
| 4 | 143.4–148.0 | 293.5–294.1 |
| 8 | 144.2–148.9 | 441.6–443.0 |
| 16 | 285.0–294.4 | 879.2–883.1 |

[Raw results](qsa_ck_results/results.jsonl) · [Summary CSV](qsa_ck_results/summary.csv) · [Environment](qsa_ck_results/metadata.json)
