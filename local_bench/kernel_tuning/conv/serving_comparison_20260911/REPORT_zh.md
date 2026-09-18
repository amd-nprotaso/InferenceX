# TP4 服务性能对比：修复缓存后的 FlyDSL 卷积

[English](REPORT.md) | **中文**

测试日期为 2026-09-11。输出吞吐量中位数从 **578.32** 提升至 **580.83 tok/s**，增幅 **0.43%**。两组测量范围重叠，这一小幅差异尚不能证明具有统计可靠性的加速。本次对比未出现此前明显变慢的情况。

| Metric | Triton baseline | FlyDSL convolution | Change |
|---|---:|---:|---:|
| Output throughput (tok/s) | 578.324 | 580.826 | +0.433% |
| Duration (s) | 63.489 | 63.215 | -0.431% |
| Mean TTFT (ms) | 1421.255 | 1422.779 | +0.107% |
| Mean TPOT (ms) | 5.275 | 5.222 | -1.006% |

表中数值为三轮测试的中位数；延迟指标取各轮平均值的中位数。

| Trial | Triton (tok/s) | FlyDSL (tok/s) |
|---|---:|---:|
| 1 | 576.753 | 580.826 |
| 2 | 578.324 | 580.553 |
| 3 | 581.691 | 581.150 |

先运行三轮基线，再启动新服务器运行三轮 FlyDSL。每个后端只启动一次服务器，三轮是在同一服务进程内重复测试，并非交错重启的独立实验，未控制启动及顺序效应。

TP=4、并发 4，标称输入/输出长度为 32768/1024，随机长度比例 0.8，seed 42。每轮测量 40 个请求，并预热 8 个请求。240 个测量请求全部成功。每轮输入 1,184,722 token、输出 36,717 token；实际请求长度随随机比例变化。缓存命中率为零，耗时不包含模型加载及预热。

两组均采用服务器脚本默认的模拟 MTP 接受长度 3.39，因此结果是模拟接受率下的吞吐投影，不代表真实接受率或模型质量。两组均关闭 FlyDSL GDN chunk_h 替换，唯一替换开关差异为 SGLANG_FLYDSL_CAUSAL_CONV=0/1。FlyDSL 日志包含四条 worker 启用消息，基线没有。已安装的 prefill 路径会调用被替换的模块函数；本实验未采集 GPU profile。

两个运行脚本均成功退出并释放 GPU 显存。服务器清理阶段出现取消及 SystemExit 堆栈，发生在测量结束之后，不是测量请求失败。

[元数据及源码哈希](metadata.json)、[结构化汇总](summary.json) 和后端子目录保留原始 JSONL、服务器日志及最终服务器命令。现有客户端脚本会覆盖相同并发的 .log 文件；driver 日志保留全部三轮输出，JSONL 以追加方式保留三条记录。

复现命令如下；使用新的 RESULT_DIR，避免向已有测量文件追加记录：

```bash
# Run from local_bench. Use backend=baseline or backend=flydsl.
backend=baseline
override=0
if [ "$backend" = flydsl ]; then override=1; fi
env HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 SKIP_DOWNLOAD=1 \
  TP=4 CONC=4 CONC_LIST='4 4 4' ISL=32768 OSL=1024 \
  RANDOM_RANGE_RATIO=0.8 NUM_PROMPTS_MULT=10 WARMUP_MULT=2 \
  SIMULATE_ACC=1 GPU_MONITOR=0 SERVER_READY_TIMEOUT=5 \
  SGLANG_FLYDSL_GDN_CHUNK_H=0 SGLANG_FLYDSL_CAUSAL_CONV="$override" \
  PYTHONPATH="$PWD/kernel_tuning/conv/bootstrap:$PWD/kernel_tuning/conv:${PYTHONPATH:-}" \
  RESULT_DIR="$PWD/kernel_tuning/conv/serving_comparison_20260911/$backend" \
  bash qwen3.5_fp4_sglang_run.sh
```
