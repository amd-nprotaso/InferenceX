# gfx950 大 KV tile FMHA 实验

[English](report.md)

较大的 KV tile 降低了独立内核延迟，但两次端到端测试的吞吐量均低于基线。建议继续使用基线服务配置。

硬件为 MI355X/gfx950，模型为 amd/Qwen3.8-Flash-Next-Quark-MXFP4，TP=1、并发=1、ISL=32768、OSL=32768。复制原 server-start.sh，仅为两组添加相同的 --random-seed 20260928，设置 HF_HOME=/data/hf_cache/。端到端测试关闭 profiling；每组顺序执行两次完整请求，每次先运行32768/32预热并清空缓存。先测基线，再测调优配置。

## 内核修改

block tile 从 sequence<128,64,32,256,32,256> 改为 sequence<64,256,32,256,32,256>；保留四个 wave，两处 MFMA shape 从32x32x16改为16x16x16。使用 gfx950 的 BlockFmhaPipelineQRKSVS BF16 D256特化，仅重新编译一个翻译单元，并链接到独立 AITER 库。为实验保留原 dispatch trait，不是通用生产调度修改。

微基准使用 Hq=24、Hkv=2、D=256，每个 varlen 行查询长度为1，选中 KV 长度2048–2051，decode/verification 行数为1/4。七种配置各十二个测试均通过独立 FP32参考校验（atol0.003，rtol0.03），最大绝对误差0.000937313。使用 HIP graph计时、20个样本，下表为各上下文位置的中位数。微基准在端到端计时开始前结束。

| Variant | Decode (µs) | Verification (µs) |
|---|---:|---:|
| baseline | 128.184 | 147.982 |
| n128 | 107.058 | 113.885 |
| n256 | 158.820 | 171.083 |
| n512 | 187.702 | 238.600 |
| m64_n256 | 111.823 | 119.617 |
| m32_n256 | 226.080 | 246.720 |
| m16_n256 | 127.297 | 134.874 |

M64 N256相对基线将 decode延迟降低12.8%，verification延迟降低19.2%。仅将 N增至256或512反而更慢。M128 N128的微基准仍优于本次选中的较大 KV tile。

## 端到端结果

| Trial | E2E (s) | TTFT (s) | TPOT (ms) | Output tok/s | Acceptance length | Verification steps |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 201.372 | 1.037 | 6.114 | 162.724 | 3.4863 | 9399 |
| baseline_repeat | 209.080 | 1.033 | 6.349 | 156.724 | 3.3450 | 9796 |
| tuned | 567.928 | 1.032 | 17.301 | 57.697 | 1.1939 | 27446 |
| tuned_repeat | 233.952 | 1.029 | 7.108 | 140.063 | 2.9208 | 11219 |

每次请求吞吐量的算术平均值：基线159.724 tok/s，调优98.880 tok/s（-38.09%）；平均延迟205.226s与400.940s（+95.37%）。这是每组两次测试的描述性统计，不能据此确定可重复的内核因果效应。

四次请求 payload相同，实际输入/输出均为32768/32768，cached_tokens=0、num_retractions=0；排除启动时间和内部状态动态信息后，启动参数相同。进程映射确认加载了各组对应的库。四次输出文本哈希均不同，即使相同种子下的基线重复也不同。未进行模型准确率评估。

调优首次接受长度为1.194，需要27446次验证；基线为3.345–3.486，需要9399–9796次。decode总耗时除以验证次数：调优20.65–20.76ms，基线21.24–21.31ms；此值仅为服务层代理指标，不是内核计时。额外验证次数抵消了内核加速，但这些数据不能证明生成轨迹分歧的原因，也不能证明 tile修改本身导致接受率骤降。

两个实验服务均已停止，原安装库与启动脚本保留。本次未证明端到端收益，仅覆盖并发1，不覆盖4/8/16。

## 文件与复现

内核修改见 [patch](micro/m64_n256/kernel.patch) 和 [源码](micro/m64_n256/kernel.cpp)；[微基准脚本](sweep.py)、[结果](micro_summary.json)、[端到端脚本](e2e.py)、[启动参数修改](launcher.patch)、[比较脚本](compare.py)、[完整比较](comparison.json)、[CSV](trials.csv)。

在上级实验目录依次执行 `python3 qsa_largekv/sweep.py`、`python3 qsa_largekv/e2e.py`、`python3 qsa_largekv/compare.py`。重跑前保留现有结果并使用新的结果目录；脚本不会覆盖已有 repeat符号链接。
