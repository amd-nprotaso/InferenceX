# Qwen3.5 TP=4 卷积对比

[English](COMPARE.md) | **中文**

在此目录运行：

```bash
python compare_conv.py --output compare_tp4.json
python compare_conv.py --lengths 1 7 17 --history mixed --warmup 2 --repeats 5 --trials 2
python compare_conv.py --shapes qwen35-397b-tp4 --block 256 --tokens 16 --output compare_block256.json
```

脚本参照 `../chunk_gated_delta_rule/compare_chunk_h.py`，将
`causal_conv1d_flydsl.py` 与已安装 SGLang 的
`sglang.kernels.ops.mamba.causal_conv1d_triton.causal_conv1d_fn` 进行对比。

默认采用 Qwen3.5-397B-A17B-MXFP4 在 TP=4 下的 BF16 GDN 张量：
3072 个通道（`2*4*128 + 16*128`）、宽度 4、无 bias、SiLU 激活。
每个 rank 的 head 形状来自 `../chunk_gated_delta_rule/shapes.py`。
预设包括 `qwen35-397b-tp4`（单条 32768-token 序列）、
`qwen35-397b-tp4-4k`（单条 4096-token 序列）以及
`qwen35-397b-tp4-packed4`（四条 8192-token 序列）。打包案例是将 32K token
预算拆分的合成输入，不代表 trace 中的实际调度。TP=4 描述单个 rank 的形状；
脚本只使用一张 GPU（`--device`）。

`--lengths` 指定自定义打包序列长度。`--tokens` 控制 FlyDSL 每个 block
处理的 token 数，不是序列长度。`--history fresh|cached|mixed` 控制历史缓存。
默认 state 为连续的 `[1243,3072,3]`；`--pool-slots` 修改容量，
`--state-layout channel-last` 测试通道连续的 state。输入按 token-major 分配，
再以 `[channels,total_tokens]` 视图传入。`--flydsl-file` 可指定接口相同的其他实现。

Eager 和图重放的输出必须在 `--rtol`/`--atol` 容差内与 SGLang 一致
（默认均为 0.016）；更新后的整个 state pool 必须完全一致。非有限结果会导致
验证失败。每次调用前恢复 state，恢复操作不计时。GPU event 每次测量一次图重放，
不包含编译和 Python dispatch。各轮交替执行后端，最终取各轮中位数的中位数。
这是热缓存基准：每次计时前恢复 state 的操作本身会访问缓存。

每完成一个形状，JSON 保存全部计时样本、正确性检查、张量 stride、依赖版本、
源码路径及哈希和运行选项。加速比为 SGLang 耗时除以 FlyDSL 耗时，大于 1 表示
FlyDSL 更快。这些是内核测量结果，不代表完整模型吞吐量。

优化后的内核还支持 `--prefetch 1|2|4|8|16`（内核默认 16）和
`--channels-per-thread 1|2`（不指定则自动选择）。自动双通道要求通道数为偶数、
BF16/FP16 输入通道连续且打包 token 总数至少为 8192。加载不支持这些选项的旧版
内核时请省略这两个参数。参见[优化结果](optimization/REPORT_zh.md)。
