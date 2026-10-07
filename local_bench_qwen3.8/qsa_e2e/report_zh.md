# Qwen3.8 无 profiling 的端到端 QSA 对比

[English](report.md) | **中文**

在本次单轮 A/B 对比中，N=128 CK QSA specialization **没有提高实测端到端吞吐量**：输出吞吐下降 2.06%，从 152.55 降至 149.40 tokens/s。两次生成文本及 speculative acceptance 不同，因此不能将差异直接归因于 kernel 修改。

## 配置与流程

两组均使用提供的 `/var/home/qwen3_8_profiling/server-start.sh`，设置 `HF_HOME=/data/hf_cache/`。模型为 `amd/Qwen3.8-Flash-Next-Quark-MXFP4`，snapshot 为 `b2091c3809eb51902f46a163973362d8176f9e58`。硬件为单张 MI355X，TP=1、concurrency=1。ISL=32768、OSL=32768，NEXTN 三个 steps、四个 draft tokens，chunked prefill=16384，stream interval=50。客户端精确复现了原请求 payload，使用 `temperature=0` 和 `ignore_eos=True`。

Launcher 实际使用 `/workspace/sglang-qwen-next/python/sglang` 和 `/sgl-workspace/aiter`。Tuned library 基于此 serving stack 重新编译，没有直接使用此前本地 checkout 的微基准二进制。已经验证 server build 与 isolated build 的 baseline kernel 源码逐字节一致。

每组均启动新 server，完成启动与 graph capture，执行 32768-input/32-output warmup，清空 prefix cache，再测量一个完整请求。两个实测请求均未启用 profiling，均完成精确的 32768 input/output tokens，cached prompt tokens 为零，没有 retractions，以长度限制结束。结果不包含模型加载、编译或 warmup 时间。

Baseline 使用原安装目录中的 QSA library。Tuned 通过 `AITER_JIT_DIR` 指向独立 overlay，仅替换 QSA library，其余已安装 libraries 通过 symlink 复用。通过 `/proc` mappings 确认两组 scheduler 加载了预期 library。原 launcher 和已安装源码未修改。

QSA 修改为 HIP template tile `sequence<128,64,32,256,32,256>` → `sequence<128,128,32,256,32,256>`。Query tile、wave layout、datatype、scale 和 masking 保持不变。选择 N=128，是因为它在此前微基准中对 launcher 的 C=1 shape 最快。两组 library 在 serving 比较前均单独通过 FP32 attention 数值对比。

## 结果

| 指标 | Baseline | Tuned N=128 | 变化 |
| --- | ---: | ---: | ---: |
| 客户端观测 TTFT | 1.0326 s | 1.0304 s | −0.22% |
| 端到端延迟 | 214.805 s | 219.332 s | +2.11% |
| TPOT | 6.5240 ms | 6.6622 ms | +2.12% |
| 输出吞吐 | 152.547 tok/s | 149.399 tok/s | −2.06% |
| Decode interactivity | 153.280 tok/s | 150.100 tok/s | −2.07% |
| 平均 speculative acceptance length | 3.2511 | 3.1348 | −3.58% |
| Speculative acceptance rate | 75.04% | 71.16% | −3.88 个百分点 |
| Verification steps | 10079 | 10453 | +374 |

`(e2e − TTFT) / spec_verify_ct` 从 baseline 的 21.210 ms 降至 tuned 的 20.884 ms，下降 1.54%。这是包含 draft 工作与其他开销的平均 verification cycle 时间近似值，不是 QSA kernel 的直接计时。不同输出与 context trajectories 限制了这一指标的解释范围。

## 可比性与限制

短 warmup 输出一致，但完整生成文本不同。虽然使用 greedy decoding，launcher 没有固定 server random seed，baseline 为 933737487，tuned 为 995034812。除随机 seed 外，保存的 serving settings 一致；启动耗时自然存在差异。目前没有证据确认 seed 差异或 kernel 修改是输出分歧的原因，数值舍入和其他非确定性执行也可能影响结果。

每组仅测量一个完整请求。Tuned 的 acceptance 降低、verification 工作增多，与吞吐下降同时出现。这些结果不能证明端到端加速，也不能证明单独修改 N 会普遍导致性能退化。要区分 kernel 节省的时间与生成波动，需要固定 server seeds 并在合适的 workload suite 上重复实验。强制生成长输出的合成 prompt 不属于模型质量评估。

TTFT 和 TPOT 沿用原客户端的 streaming 计算方式，stream interval=50 保持不变。所有原始 stream timestamps 与最终 server metadata 已保存。

第一次 baseline 启动到达 ready 状态后，在 `/health` 触发的单 token generation 上发生 GPU fault，当时尚未开始实测请求。日志保存在 `baseline/server_attempt1_health_fault.log`。随后 harness 改为检查 ready 日志标记并读取 `/get_server_info`，再次执行未修改的 launcher。报告中的两组结果都来自采用相同修正后 readiness 流程的成功运行。

## 产物与复现

- [对比 CSV](comparison.csv)、[对比检查](comparison_checks.json)、[完整对比 JSON](comparison.json)。
- [Baseline 结果](baseline/client-exact-result.json)、[tuned 结果](tuned/client-exact-result.json)。
- [Baseline QSA 加载记录](baseline/loaded_qsa.json)、[tuned QSA 加载记录](tuned/loaded_qsa.json)。
- [Kernel patch](compiled/n128/kernel.patch)、[构建命令](compiled/n128/build.json)、[源码一致性检查](source_match.json)。
- [Client driver](run_client.py)、[受控切换 driver](run_tuned.py)、[对比脚本](compare.py)。

重新运行前请保留已有输出目录。Baseline 使用原 launcher，再从 workspace 启动 client driver。Tuned 在调用相同 launcher 前设置 overlay 的绝对路径：

```bash
export HF_HOME=/data/hf_cache/
export AITER_JIT_DIR=/var/home/my_InferenceX/InferenceX/local_bench_qwen3.8/qsa_e2e/tuned_jit
bash /var/home/qwen3_8_profiling/server-start.sh
```

Overlay 是针对当前 QSA workload 的本地实验构建，不是已安装的生产修改。比较结束后，已停止本次基准启动的 servers。
