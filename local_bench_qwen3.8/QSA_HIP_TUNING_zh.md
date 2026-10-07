# QSA HIP/CK 调优实验

[English](QSA_HIP_TUNING.md) | **中文**

本实验为 Qwen3.8 TP=1 QSA attention 调用重新编译现有 CK `FmhaFwdKernel`。输入为 BF16 Q/K/V，24 个 query heads、2 个 KV heads、D=256，每个 packed sequence 包含一个 query 和 2048–2051 个选中的 keys。ISL=32678、OSL=32678，并发为 1、4、8、16，与请求的微基准一致。普通 decode 和四 token verification 均覆盖六个 context samples。

## 复现

```bash
python3 tune_qsa_hip.py
python3 summarize_qsa_hip.py
```

驱动依赖 `bench_qsa_ck.py` 生成的 `.qsa_aiter_jit` baseline 构建。它读取实际 Ninja 编译和链接参数，仅重新编译匹配的 D256/gfx950 translation unit，并与其余未修改的 baseline objects 链接。每个 variant 使用独立库目录，保留原 AITER 源码、baseline objects 和 baseline library。

`qsa_hip_tuning` 下每个 variant 目录包含修改后的 `kernel.cpp`、`kernel.patch`、记录精确命令和 baseline library hash 的 `build.json`、构建日志、基准日志、逐 case JSONL 和 profiler traces。仅修改编译参数的 variant，其源码 patch 为空，修改内容记录在 `build.json`。

```bash
# 复用已构建的 variant，在新目录中重新测量：
python3 tune_qsa_hip.py --variants baseline,post_sched --reuse \
  --run-label repeat --repeats 30
python3 summarize_qsa_hip.py --label repeat
```

## 测量方法与范围

每个 case 在计时前均与独立 FP32 softmax attention 对比。每次 HIP graph replay 包含 20 次调用，探索性 sweep 测量 15 次 replay，并单独通过 profiling 确认实际执行的 kernel。各 variant 使用相同输入和 seed。表格逐 context case 对齐比较；总体 speedup 为 baseline/variant latency 的几何平均值，不使用不对应的计时极值相除。原始计时样本均保留。

实验仅测量 KV 压紧之后的 attention 调用，不包含索引选择、KV 压紧、其他层或完整 serving。重复使用输入意味着这是 warm-cache 微基准。正确性验证覆盖这些合成输入，不等于模型质量评估。

修改后的 kernel 保留原 external dispatch trait，使现有 AITER wrapper 能调用实验性 specialization。因此这些库仅适用于当前 length-one、D256 基准，不是通用生产 AITER 替代品。将获胜配置用于生产需要正确更新 CK codegen/dispatch，并验证 serving 路径。

## 复测结果

使用 MI355X / gfx950、ROCm PyTorch 2.11。每个候选配置先完成探索性 sweep，再以每 case 30 次 graph 计时样本复测。所有候选配置的 48 个 case 均通过验证。相对 FP32 reference 的最大绝对误差为 0.001054（向上取整）。

| Variant | Change | Geometric mean speedup |
| --- | --- | --- |
| post_sched | HIP `-mllvm -enable-post-misched=1` instead of `=0` | 1.040× |
| n128 | KV tile 64 → 128; otherwise original tile | 1.289× |
| m16_w1 | Query tile 128 → 16; 4 waves → 1; 16×16×16 MFMA tile | 1.456× |
| m16_n128 | Query/KV tile 16×128; 1 wave; 16×16×16 MFMA tile | 1.571× |

固定使用 `m16_n128` 时，六个 context cases 的 latency 中位数如下（microseconds）：

| Mode | CONC | Baseline | Tuned | Matched-case speedup |
| --- | --- | --- | --- | --- |
| decode | 1 | 130.40 | 130.53 | 1.011× |
| decode | 4 | 148.04 | 132.08 | 1.133× |
| decode | 8 | 148.88 | 132.96 | 1.132× |
| decode | 16 | 294.06 | 141.37 | 2.103× |
| verify | 1 | 147.91 | 132.24 | 1.119× |
| verify | 4 | 293.92 | 141.36 | 2.079× |
| verify | 8 | 442.85 | 168.87 | 2.622× |
| verify | 16 | 882.27 | 396.15 | 2.228× |

较少 query rows 时 `n128` 更快；verification C=16 时，`m16_w1` 略快于 `m16_n128`。不存在对所有 case 都最优的单一配置。总体 1.57× 是等权微基准 speedup 的几何平均值，不代表 serving 加速比。本实验未修改生产 dispatch policy。

其他实验结果：移除 kernarg preload 基本无收益；fast math 仅改善约 0.5%；occupancy=2 总体退化；post scheduling 与 N=128 组合后，比单独 N=128 更慢。还测量了其他 query tile 和 one-wave 组合，详见完整探索性 CSV。

```bash
python3 bench_qsa_ck.py --jit-dir qsa_hip_tuning/m16_n128/jit \
  --isl 32678 --osl 32678 --conc 1,4,8,16 --profile \
  --output qsa_hip_tuned_rerun
```

[Confirmed comparison](qsa_hip_tuning/comparison_confirm.csv) · [All exploratory variants](qsa_hip_tuning/comparison_results.csv) · [Winning source patch](qsa_hip_tuning/m16_n128/kernel.patch) · [Exact build commands](qsa_hip_tuning/m16_n128/build.json)

最后一次 baseline 复测的 latency 漂移中位数为 0.02%，逐 case 最大绝对漂移为 1.03%。原 baseline library hash 未改变，详见 [验证记录](qsa_hip_tuning/validation.json)。
