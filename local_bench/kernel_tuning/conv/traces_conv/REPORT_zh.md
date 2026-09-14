# FlyDSL 卷积 trace 分析

[English](REPORT.md) | **中文**

2026-09-10 按用户指定的 FlyDSL `capture-kernel-trace` 和
`kernel-trace-analysis` skills 在本地采集。硬件为 AMD Instinct MI350X，
gfx950。目标为 `../causal_conv1d_flydsl.py` 的 `kernel_0`。
源码 SHA256：`db24d3d85a11d799ad8e6308b93ed73e9fcc8555db7c934b34a8d21824fd4d3b`。
此次未修改内核实现。

## 采集与验证

采用 Qwen3.5-397B TP=4 单个 rank 的形状：3072 通道、BF16、宽度 4、
SiLU、无 bias、全新历史、连续 state `[1243,3072,3]`、block 128、
每个 block 16 token。打包案例为合成输入。所有采集和独立无 profiler
对比均通过 eager、图重放输出及整个 state pool 的精确一致性检查。

ATT 选择第 3 次匹配调用，即预热后的 eager 调用，在四个 shader engine
各选择 CU 1。每个 trace 有 1207 条 ISA 指令，全部带源码映射；JSON 中
另有一行内核标签，共 1208 行。下表耗时来自各次采集的 dispatch 153，
处于 profiler 环境中。

| 案例 | Workgroup grid | 采集耗时 | 采样 stall/总周期 | VMEM-wait/全部 stall | 第 84 行/全部 stall |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 × 32768 | `(2048,1,24)` | 193.082 µs | 82.4% | 84.4% | 76.13% |
| 1 × 4096 | `(256,1,24)` | 25.320 µs | 75.0% | 70.2% | 62.93% |
| 4 × 8192 | `(512,4,24)` | 194.282 µs | 82.4% | 84.8% | 76.24% |

CSV 的 grid 以 work-item 为单位，表中 X 已除以每组 128 个线程。
各 wave 的 stall 周期之和不等于设备耗时，也不能直接换算成可实现的加速比。

数据与分析报告：

- [32K trace](tp4_32k/ui_output_agent_20469_dispatch_153/) 和 [分析](tp4_32k/analysis.txt)。
- [4K trace](tp4_4k/ui_output_agent_3884_dispatch_153/) 和 [分析](tp4_4k/analysis.txt)。
- [四序列 trace](tp4_packed4/ui_output_agent_24805_dispatch_153/) 和 [分析](tp4_packed4/analysis.txt)。
- [L2 counters](pmc_32k/pass_1/l2_counter_collection.csv) 和 [分析](pmc_32k/analysis.txt)。
- [Code-object 元数据](tp4_32k/code_object_metadata.txt)。

分析器根据特定矩阵指令推断架构，因本内核没有这些指令而误标为 gfx942。
Code object 明确标注 gfx950，使用 19 VGPR、29 SGPR、0 AGPR，无 LDS、
scratch 或寄存器 spill。CSV 报告 VGPR_Count=12、SGPR_Count=32；应依据
code object 和 ISA 判断寄存器使用。分析器 occupancy 是资源上限估算，
并非实际驻留测量；其 MFMA 累加器说明也不适用于此处，因为没有 MFMA 指令。

独立单 pass PMC 采集得到 `TCC_HIT_sum`=1,623,271，
`TCC_MISS_sum`=3,442,362，单次 dispatch 的总体 L2 命中率为 32.0%。
这不是单独针对输入的命中率。未采集 HBM 字节数，因此带宽是否饱和、
是否过量读取仍未测定。分析器针对 streaming decode 的通用说明不适用于此 prefill 卷积。

## 按优先级排序的优化实验

1. **预取一个小的输入 token 窗口。** 第 84 行在使用前立即读取 BF16，
   ISA 反复出现 `global_load_ushort` 后跟 `s_waitcnt vmcnt(0)`，占 32K
   全部 stall 的 76.13%。先试提前加载 2 或 4 个未来 token，再计算当前
   卷积和 SiLU。循环已经展开，适合先用编译期预取列表或窗口。
   保留短序列和尾部 mask，以及 chunk 0 独占 state 读写的约束。
   必须检查 ISA 是否真正拉开 load 和使用位置；源码重排本身不构成证明。
   目前寄存器使用较少，但改动后仍需检查分配和 spill。此建议应用
   `prefetch-data-load` skill 的延迟隐藏原则；本内核可用于隐藏延迟的
   算术量少于该 skill 的 GEMM 示例。

2. **提前发出权重和 halo 读取。** 第 61、63 行分别贡献 32K stall 的
   6.15%、4.45%。在转换和使用前，先发出相互独立的权重、halo 和首个
   窗口读取，尽可能与元数据及地址计算重叠。编译器已经通过一条
   `global_load_dwordx2` 读取四个 BF16 权重，因此优先考虑读取位置。
   必须保留短序列旧历史快照语义，避免 block 间 state 竞争。

3. **尝试每个 lane 处理多个相邻通道。** 当前输入输出为
   `global_load_ushort` / `global_store_short`。双通道映射可能使用
   每 lane 32-bit 读写，减少指令数并增加独立算术。当前跨 lane 访问已经
   合并；此实验针对指令开销和并行度，并非已发现访问不合并的问题。
   需要处理通道尾部并调整地址及历史窗口，收益须通过测量确认。

4. **修复读取调度后重新调 tile。** 用对比脚本测试每 block token 数
   8/16/32 及 block 64/128/256。更大 token tile 可摊薄前置开销和
   halo 重复读取：内部 tile 的额外输入数量从 3/16=18.75% 降至
   3/32=9.375%。这是逻辑读取量，不是实测 HBM 过量读取。
   更大 tile 也可能增加寄存器使用或串行工作，因此单独修改 `--tokens`
   尚不能视为已验证的优化。

SiLU 占 32K stall 的 4.52%，4K 的 10.45%；应先实验读取调度再处理它。
LDS tiling 和矩阵指令优先级较低：本内核无跨通道归约，历史窗口已保存在
寄存器中，trace 也没有 LDS/barrier 瓶颈。

只有在 eager/图重放和精确 state 检查通过、无 profiler 的交替试验在相关
形状上改善、且新 ATT 显示 wait 减少而无 spill 后，才接受改动。
同时运行现有 `check_conv.py`，覆盖短/空序列、历史、padding 和支持的布局。
不能仅凭 stall 占比预测加速比。

## 无 profiler 基线与复现

[unprofiled_baseline.json](unprofiled_baseline.json) 保存 3 轮交替测试，
每轮每后端 100 个样本、10 次预热。state 恢复不计入 event 时间，但会预热缓存。

| 案例 | SGLang | FlyDSL | SGLang/FlyDSL |
| --- | ---: | ---: | ---: |
| 1 × 32768 | 201.902 µs | 190.722 µs | 1.059× |
| 1 × 4096 | 32.001 µs | 31.000 µs | 1.032× |
| 4 × 8192 | 201.961 µs | 191.281 µs | 1.056× |

在 `conv` 目录运行，为重复采集使用新输出目录：

```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  rocprofv3 -i traces_conv/att.yaml -d traces_conv/repeat_32k -- \
  python compare_conv.py --shapes qwen35-397b-tp4 --warmup 2 --repeats 2 --trials 1

python /var/home/conv_FlyDSL/FlyDSL/.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py \
  traces_conv/repeat_32k/ui_output_agent_* --kernel kernel_0 \
  --mode both --topk 12 --detail --context 2

rocprofv3 -i traces_conv/pmc_l2.yaml -d traces_conv/repeat_pmc -- \
  python compare_conv.py --shapes qwen35-397b-tp4 --warmup 2 --repeats 2 --trials 1
```

其他形状使用 `qwen35-397b-tp4-4k` 或 `qwen35-397b-tp4-packed4`。
原始 ATT、code object、源码快照、wave JSON、CSV 和报告均保存在本地，
总计约 453 MB。内核耗时不能证明完整服务吞吐量。
