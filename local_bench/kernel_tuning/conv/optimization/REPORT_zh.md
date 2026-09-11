# 因果卷积优化结果

[English](REPORT.md) | **中文**

2026-09-10 在 AMD Instinct MI350X（gfx950）验证，使用用户指定的 FlyDSL
内核编写、预取、trace 采集和分析 skills。以下均为 GPU 内核微基准，不代表服务吞吐量。

## 已实现改动

- 先分组读取原始输入，再转换和使用。选定默认 `prefetch=16`，仍可选择
  1/2/4/8/16 进行实验。
- 将尾部无效读取地址钳制为序列最后一个有效 token，替代逐次读取前的分支。
  外层条件排除空序列和无效 lane，输出 mask 继续排除尾部 token。
  因此不会越过打包序列边界，并允许编译器调度独立读取。
- 在使用 halo 之前发出权重读取。保留 state 快照和更新顺序，仍然只有
  chunk 0 访问历史缓存。
- 对通道数为偶数、通道连续的 BF16/FP16 输入，打包 token 总数至少为 8192
  时，每线程使用两个相邻通道的向量；其他输入每线程处理一个通道。
  可通过 `channels_per_thread=1|2` 覆盖选择，显式双通道要求通道数为偶数。
  构造向量时推断输入类型，避免 FP32 bias 先被舍入为 BF16。
- 保留 block 128 和每 block 16 token。单通道调优中，在 prefetch 4 下，
  64/256 线程或 8/32 token 未带来一致改善。这是有界搜索，并未穷举所有
  双通道布局与分块组合。8192-token 自动切换阈值是基于已测试短/长输入的
  保守策略，并非实测的精确性能交叉点。

原始源码 SHA256：
`db24d3d85a11d799ad8e6308b93ed73e9fcc8555db7c934b34a8d21824fd4d3b`。
优化源码 SHA256：
`54ed68638a7866e7a24bd6fa0849c9fc6629708db486de0ea28f4bf9e08d53b8`。
[causal_conv1d_baseline.py](causal_conv1d_baseline.py) 保存原始实现。
本目录也保留中间候选源码和计时 JSON 供复现；运行时只导入
`../causal_conv1d_flydsl.py`。

## 性能

[before_after.json](before_after.json) 在同一进程、相同输入上交替比较原始和
优化 FlyDSL，共五轮，每轮每后端 100 个样本、10 次预热。state 恢复不计入
event 时间。每个样本测量一次图重放，结果取各轮中位数的中位数。
两个实现均先通过输出及 state 检查。

| TP=4 输入 | 原始 FlyDSL | 优化 FlyDSL | 原始/优化 |
| --- | ---: | ---: | ---: |
| 1 × 32768 | 198.042 µs | 128.341 µs | 1.543× |
| 1 × 4096 | 30.920 µs | 28.120 µs | 1.100× |
| 4 × 8192 | 198.562 µs | 129.761 µs | 1.530× |

[sglang_final.json](sglang_final.json) 采用三轮交替测试与已安装 SGLang
独立比较，其他计时方法相同：

| TP=4 输入 | SGLang | 优化 FlyDSL | SGLang/FlyDSL |
| --- | ---: | ---: | ---: |
| 1 × 32768 | 207.981 µs | 128.661 µs | 1.616× |
| 1 × 4096 | 32.040 µs | 28.080 µs | 1.141× |
| 4 × 8192 | 209.802 µs | 128.961 µs | 1.627× |

早期带 mask 分支的预取方案只小幅改善长序列。采用钳制地址和提前读取权重后，
单通道 32K 在 prefetch 4 下约为 155 µs，prefetch 16 下约为 141 µs，
双通道继续改善。不同进程之间 GPU 耗时有波动，因此优化前后加速比以最终
交替比较为准，不使用探索阶段不同进程的耗时相除。

## 正确性与 trace 证据

[correctness.json](correctness.json) 和 [日志](correctness.log) 记录 52 个
用例通过，另有可选 state 检查。覆盖宽度 2–5、BF16/FP16/FP32、奇数通道的
单通道回退、最后一个 wave 不完整的双通道、短/空序列、填充及间接缓存槽位、
混合初始历史、额外 state 容量、非连续布局、非 2 的幂次 token 分块、
图重放/非默认 stream、BF16 输入配合 FP32 bias。最后一个案例还使用带偏移、
token 步长为奇数的输入。宽度 5 使用独立 PyTorch 参考实现，因为已安装 SGLang
仅支持宽度 2–4 的算术。state 精确一致，输出容差不变。

[优化 ATT](att_32k/ui_output_agent_24517_dispatch_153/)、
[热点报告](att_32k/analysis.txt) 和
[code-object 元数据](att_32k/code_object_metadata.txt) 验证了实现。
采集配置与原始版本相同：第 3 次匹配调用、预热后的 eager 32K、四个
shader engine 各选 CU 1，并启用源码调试信息。

| 证据 | 原始 | 优化 |
| --- | ---: | ---: |
| VMEM-wait 占采样 stall 比例 | 84.4% | 8.7% |
| Code object 的 VGPR 数 | 19 | 64 |
| Code object 的 SGPR 数 | 29 | 52 |
| 寄存器 spill / LDS / scratch | 0 / 0 / 0 | 0 / 0 / 0 |
| 静态 `s_waitcnt` 指令数 | 34 | 18 |
| 主要输入 / 输出 ISA | `global_load_ushort` / `global_store_short` | `global_load_dword` / `global_store_dword` |
| 静态指令数 | 1207 | 1749 |

优化后的 wave 处理两倍通道，静态指令数增加不等于每个输出的指令数增加。
全部 1749 条指令都有源码标注，但许多向量操作被归到 Python `functools.py`
包装器，而不是用户的算术行。这是源码归因问题，并非 GPU 执行 Python。
目前剩余主要 stall 来自被分析器归类为 `other` 的 packed 算术。

分析器对非 MFMA 内核误标 gfx942；code object 确认 gfx950。
occupancy 数值只是资源上限估算，不是实际驻留测量。双通道改变了采样 wave
数量，不能将 stall 总数当作耗时直接比较。加速比来自无 profiler 的计时。

## 复现

在 `conv` 目录运行：

```bash
python check_conv.py --output optimization/correctness.json
python optimization/compare_versions.py
python compare_conv.py --output optimization/sglang_final.json
python compare_conv.py --prefetch 8 --channels-per-thread 1 --output /tmp/conv_scalar.json

FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  rocprofv3 -i traces_conv/att.yaml -d optimization/att_repeat -- \
  python compare_conv.py --shapes qwen35-397b-tp4 --warmup 2 --repeats 2 --trials 1
```

已同步更新中英文使用文档。按仓库要求追加性能 changelog，保留所有既有字节，
并验证 schema 和引用的配置键，见[验证记录](changelog_validation.json)。
因未创建 PR，`pr-link: XXX` 保留占位符。本次优化尚未验证完整模型启动、
eval 准确率或服务吞吐量。
