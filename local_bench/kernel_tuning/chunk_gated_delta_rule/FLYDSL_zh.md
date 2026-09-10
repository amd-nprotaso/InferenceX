# FlyDSL chunk-state 实验，2026-09-09

[English](FLYDSL.md) | **中文**

在支持的特化范围内，替代内核的数值结果正确，但**没有超过本次实验找到的最佳
Triton 配置**。FlyDSL 为 0.719 ms，交换网格维度并使用四级流水线的 Triton 为
0.720 ms。差异处于测量波动范围内，尚未达到 0.65 ms 或 0.50 ms 的目标。
虽然超过了历史上的 0.850 ms 基准，但仅与这个旧结果比较会夸大收益。

## 重复设备端测量

TP4：B=1、T=32768、Hg=4、H=16、K=V=128、BT=64。每个单元格都是 60 次设备端
dispatch 的中位数，不包含参考实现和预热。各轮交替执行配置顺序。
`results_flydsl/summary.json` 保存全部结果；各轮目录保存完整命令、日志和原始 CSV。

| 配置 | 第 1 轮 ms | 第 2 轮 ms | 第 3 轮 ms | 各轮中位数的中位数 |
|---|---:|---:|---:|---:|
| Triton BV16/w4/s3/wpe4 | 0.797 | 0.813 | 0.812 | 0.812 |
| 上述配置，交换网格维度 | 0.777 | 0.784 | 0.783 | 0.783 |
| Triton BV16/w4/s4/wpe4 | 0.732 | 0.734 | 0.729 | 0.732 |
| 上述配置，交换网格维度 | 0.721 | 0.719 | 0.720 | 0.720 |
| FlyDSL，预取两个 chunk，XOR LDS | 0.719 | 0.719 | 0.717 | 0.719 |

各轮 dispatch 的标准差为 0.003–0.005 ms。这些是内核微基准，不是服务吞吐量。
实验期间另一个 SGLang 服务仍驻留在 GPU 上，没有确认 GPU 独占。交替配对测量
有助于控制漂移，但不能证明独占设备时的性能。

**改动 1，XCD 映射：有小幅正收益。** 按各轮中位数比较，交换网格维度使三级配置
改善约 3.6%，四级配置改善约 1.6%。没有出现 L2 同步推进假设可能暗示的大幅收益。
PMC 仍不可用，因此不声称测得 L2 命中率。实验只在已安装源码的隔离副本中交换
两个 program ID 和启动网格；三种形状的 `h`、`v_new` 和更新后状态均逐位一致。

**改动 2，增加 Triton 预取深度：有效。** 四级流水线使原网格改善约 9.9%，交换后的
网格改善约 8.0%。编译结果使用 137,088 字节 LDS，没有溢出。原先按
`34,432 * stages` 推算接近实际，但不完全准确：三级为 103,296 字节，四级为
137,088 字节，而不是 137,728 字节。已安装的生产包没有修改。现有环境变量可选择
s4；`wpe=4` 仍需与 BV16/w4 配套使用。

**改动 3，重构 LDS：对 FlyDSL 有效，但不足以超过新调优的 Triton。** 最终内核
使用 38 KiB LDS、80 个 VGPR，没有 scratch，保持对外 `(V,K)` 状态布局。
直接从累加寄存器写出 `h`，取消了单独的输出暂存过程。状态和修正值通过带 XOR
置换的宽 LDS 拷贝传递；`k` 使用 gfx950 的 `LDSReadTrans16_64b` 拷贝原语和兼容
转置的 swizzle。两个交替使用的 `k` 缓冲区避免下一轮写入覆盖上一轮尚未完成的读取。

原任务中的流量比例也需要算术修正：harness 模型的总流量为 2,684,354,560 字节，
其中重复 w/k 读取为 1,979,711,488 字节，即 73.75%，不是 94%。这是完全没有缓存
复用时的流量模型，不是实测 HBM 流量。

## 开发阶段测量

下表是各自 60 次 dispatch 的单轮实验，不是三轮配对结果。中间 trace 保存在本次
会话的 `/tmp/gdn_fly_*_trace` 目录中。

| 版本 | 设备端中位数 ms |
|---|---:|
| 首个逐位一致的递推实现，无预取 | 3.062 |
| 预取一个 chunk，初始暂存原型 | 0.882 |
| 预取两个 chunk，交替 k 缓冲区 | 0.867 |
| 增加硬件转置 LDS 读取 | 0.844 |
| 未 swizzle 时预取四个 chunk | 0.877 |
| 带状态保护的主机入口，未 swizzle | 0.871 |
| 对状态和值缓冲区应用 XOR | 0.802 |
| 进一步对 k 缓冲区应用 XOR | 0.720 |
| 全部 swizzle，预取三个 chunk | 0.746 |
| 全部 swizzle，预取一个 chunk | 0.726 |

首个单 chunk 预取原型没有交替使用 k 缓冲区，不是受支持的实现。数值测试通过不能
证明不存在跨 wave 的生命周期竞争。交付版本使用两个 k 缓冲区，并选择双 chunk 预取。

## ATT 测量

容器缺少解码库，因此将官方 `ROCm/rocprof-trace-decoder` 的 `0.1.4` 版本下载到
`/tmp`，没有修改 ROCm 安装。两份有效采样均使用 CU0、GPU0 和 SE mask 0x1。
首次采样没有目标 MFMA，已丢弃。有效的 code object 20 每轮有 8 条 MFMA，累计
16,384 次 MFMA issue，对应 2,048 个 wave-iteration。原始统计表和归一化 JSON
位于 `results_flydsl/att_{unswizzled,swizzled}.{csv,json}`。

| 类别 | 未 swizzle cycles/wave/iteration | XOR cycles/wave/iteration | XOR 占比 |
|---|---:|---:|---:|
| vmcnt 等待 | 79 | 161 | 5.2% |
| lgkmcnt 等待 | 940 | 170 | 5.5% |
| barrier | 467 | 713 | 23.0% |
| VALU | 628 | 669 | 21.6% |
| 全局读取 | 746 | 567 | 18.3% |
| LDS 指令 | 499 | 268 | 8.6% |
| 全局写入 | 163 | 399 | 12.8% |
| MFMA | 40 | 63 | 2.0% |
| 其他 | 82 | 95 | 3.1% |

计算方法是 ATT 指令 `Latency` 总和除以 wave-iteration 数，包含少量序言和尾声开销。
它**不是**原始 4,048-cycle 表所用的循环实际时长中位数，不能把两个总数当成相同
指标直接比较。`summarize_att.py` 记录归一化方法，并拒绝没有 MFMA 的输入。
最明确的改善是 LDS 等待 940→170 cycles，以及 LDS 指令延迟 499→268 cycles。
两个版本每轮都只有两个 barrier，但 barrier 数量少不代表等待成本低：最终 barrier
仍占很大比例。延迟也会在读取、写入 issue 和 wait 之间转移，因此 vmcnt 占比低
不能单独证明全局内存延迟已经消失。

## 接口与正确性

`chunk_delta_h_flydsl.py` 提供与已安装主机入口签名相同的
`chunk_gated_delta_rule_fwd_h`，返回 `(h, v_new)`。支持 gfx950、连续 BF16 k/w/u、
K=128、V 可被 16 整除，以及覆盖全部 T 个 token 的单序列；T 必须是 64 的正整数倍。
如果提供 `cu_seqlens`，它必须描述完整序列 `[0,T]`，`chunk_indices` 必须是对应的
生产元数据。dispatch 不会把元数据读回 CPU。多条打包序列和尾块不在此特化范围内；
导入钩子会把这些形状交给已安装的实现。

可选标量门控和逐通道门控使用连续 FP32。状态的 `[H,V,K]` 内部维度必须连续，slot
间距可以任意。负 slot 不读取或写入状态。没有初始状态时从零开始，不写回状态。
必须在构造 buffer resource **之前**重定位 64 位 slot 指针；即使前面的算术使用
int64，把大间距直接放进 buffer offset 仍会出错。测试使用了真实的
`2**31 + 512` 元素 slot 间距。

逐位一致要求遵循 Triton 编译后的算术顺序：前四条 MFMA 共用一个累加器，状态缩放
和加法使用显式 FMA。源码中分开的 dot 表达式不意味着分别舍入两个点积再相加。

`check_chunk_h.py` 通过了 48 个完整形状组合（三种形状 × 四种门控模式 × 初始状态
开关 × 保存新值开关），以及 padded slot、间接和 envelope-strided slot、每种形状
三次 graph replay、独立 FP32 单 chunk 参考，以及超出 int32 offset 的回归测试。
初始状态非零。Triton 比较逐位一致；独立 FP32 的输出和状态比较使用 BF16
`atol=rtol=0.016`。每次启动（包括 graph replay）前都恢复状态。结果位于
`results_flydsl/correctness.json`。

## 复现与集成

在 `local_bench` 中运行：

```bash
python3 kernel_tuning/chunk_gated_delta_rule/check_chunk_h.py
python3 kernel_tuning/chunk_gated_delta_rule/run_experiments.py
rocprofv3 --kernel-trace --output-format csv -d /tmp/gdn-check -- \
  python3 kernel_tuning/chunk_gated_delta_rule/profile_flydsl.py --launches 60
```

ATT 使用现有 `trace_att.py` 中的参数，加上 `--kernel-include-regex kernel_0`，
并让 `--att-library-path` 指向已安装的解码库。FlyDSL 的通用符号必须归属到正确的
code object；没有目标 MFMA 的采样必须拒绝。不要使用 `--att-serialize-all`。

`bootstrap/sitecustomize.py` 仅在 `SGLANG_FLYDSL_GDN_CHUNK_H=1` 时生效。
`run_server.sh` 为新启动的服务 worker 设置该变量和 Python 路径：

```bash
TP=4 bash kernel_tuning/chunk_gated_delta_rule/run_server.sh
MODE=prefill ISL=32768 OSL=4 bash qwen3.5_fp4_sglang_bench.sh
```

已在新 Python 进程中验证导入钩子。**尚未运行**完整模型启动或配对的服务吞吐量、TTFT
测试，不声称端到端加速。没有性能依据用此实验内核替换新调优的 Triton。若要同时使用
conv bootstrap，必须合并两个 `sitecustomize` 钩子。已安装的 SGLang 包和现有服务
进程没有修改。

要求执行的 FlyDSL `bash scripts/check_python_style.sh --fix` 检查已通过；该 checkout
没有待检查的改动。新增本地 Python 文件还单独通过了 Black（行宽 120）和
Ruff E/W/F/I 检查。格式工具原先未安装，因此安装在 `/tmp` 下。

## 环境

| 组件 | 观测值 |
|---|---|
| GPU | AMD Instinct MI355X, gfx950:sramecc+:xnack- |
| FlyDSL | 0.3.1 |
| FlyDSL 参考 checkout | ac227c35029d2008282cbf035caa5923b769d37e |
| SGLang | 0.5.18.dev20260829+g4d53767b09 |
| Torch 包元数据 | 2.9.1+rocm7.2.0.lw.git7e1940d4 |
| Triton | 3.7.0+amd.rocm7.2.0.git89002410 |
| HIP | 7.2.26015-fc0010cf6a |
| ATT 解码器 | 官方 rocprof-trace-decoder 0.1.4，临时安装 |

仅凭这些版本字符串无法确定完整容器 digest。
