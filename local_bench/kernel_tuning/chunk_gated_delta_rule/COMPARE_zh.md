# 比较 Qwen3.5 形状下的 FlyDSL 与 SGLang 性能

[English](COMPARE.md) | **中文**

`compare_chunk_h.py` 比较本地 `chunk_delta_h_flydsl.py` 中的
`chunk_gated_delta_rule_fwd_h` 与当前安装的 SGLang 实现。脚本复用 `shapes.py`
中已有的 Qwen3.5-397B-A17B-MXFP4 形状。虽然模型权重采用 MXFP4，GDN 张量仍为
BF16。该 FlyDSL 特化实现需要 gfx950 GPU，并应在安装了 SGLang 和 FlyDSL 的
同一 Python 环境中运行。

在本目录运行：

```bash
python3 compare_chunk_h.py --output /tmp/qwen35_chunk_h.json
```

默认覆盖以下形状：

| 形状 | Token 数 | 分组 key head 数 | value head 数 | K / V |
|---|---:|---:|---:|---:|
| `qwen35-397b-tp4` | 32768 | 4 | 16 | 128 / 128 |
| `qwen35-397b-tp2` | 32768 | 8 | 32 | 128 / 128 |
| `qwen35-397b-tp4-4k` | 4096 | 4 | 16 | 128 / 128 |

所有用例均为单序列，chunk size 为 64。这些形状属于 397B 模型；脚本不会推断
其他 Qwen3.5 模型的形状。

选择形状或覆盖序列长度：

```bash
python3 compare_chunk_h.py --shapes qwen35-397b-tp4 --tokens 64 4096 8192 32768
python3 compare_chunk_h.py --shapes qwen35-397b-tp4 qwen35-397b-tp2 --repeats 200 --trials 5
```

`--tokens` 只接受 64 的正整数倍。重复的 head 数与长度组合会被去重。
使用 `--device 1` 可选择其他逻辑 GPU。`--pool-slots 2` 可减少状态池的内存分配；
默认保留原始 trace 中的状态池大小。

比较具有相同 host 函数签名的实验内核：

```bash
python3 compare_chunk_h.py \
  --flydsl-file /tmp/gdn_vn_delayed_stores/delayed/chunk_delta_h_flydsl.py \
  --shapes qwen35-397b-tp4 --output /tmp/qwen35_delayed_stores.json
```

## 测量内容

脚本使用当前安装的 SGLang 配置，不覆盖其 Triton 设置。配置及相关 GDN 环境变量
会记录到 JSON。运行前请禁用 FlyDSL bootstrap：若 SGLang 入口已被替换，脚本会
拒绝执行，避免实际比较的是 FlyDSL 与它自身。

输入通过 SGLang 的生产准备函数构建，不使用相互独立的随机 w/u 张量。计时前，
脚本会比较 eager 执行与 graph replay 的 `h`、`v_new` 和更新后的状态。出现非有限值
或比较失败时立即停止。BF16 默认容差为 `--rtol 0.016 --atol 0.016`；若要求逐位一致，
可使用 `--rtol 0 --atol 0`。JSON 也会报告每项输出是否完全相等及其最大绝对误差。

两个后端分别预热并捕获独立 GPU graph。每次 replay 前都会恢复状态，恢复操作位于
GPU event 计时区间之外。计时不包含 Python dispatch、输出分配、输入准备和状态恢复。
默认执行 10 次预热，每轮测量 100 次 replay，共 3 轮，并交替后端顺序。表格报告各轮
中位数的中位数。`Speedup = SGLang time / FlyDSL time`，大于 1 表示 FlyDSL 更快。

请在空闲 GPU 上评估性能。这是内核微基准，不代表完整模型的吞吐量或 TTFT。
Graph replay 和缓存状态可能与实际 serving 不同。对于较小的加速幅度，应检查 JSON
中保存的各轮原始数据及其波动。

可选 JSON 报告包含所有计时样本、各轮中位数和标准差、正确性结果、张量形状、
源码路径及 SHA256、软件包版本、GPU 架构和 SGLang 配置。每完成一个形状便更新一次，
因此即使后续用例失败，先前结果仍会保留。
