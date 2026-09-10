# FlyDSL 因果逐通道卷积

[English](README.md) | **中文**

这是适用于 ROCm 的独立实现，用于替换 SGLang GDN 打包 prefill 路径中的
`causal_conv1d_fn`。它沿用了
`/var/home/conv_FlyDSL/FlyDSL/kernels/conv/conv3d_implicit.py` 的编译缓存、调用和
显式张量视图方式。计算部分采用专用逐通道内核：每个线程处理一个通道，在寄存器中
复用短卷积窗口，并在输出类型转换前计算 SiLU，无需 GEMM 通道补齐或布局转置内核。

## 配合现有服务器运行

在 `/var/home/InferenceX/local_bench` 中启动新服务器：

```bash
TP=4 CONC=4 bash kernel_tuning/conv/run_server.sh
```

在另一个终端运行现有客户端：

```bash
ISL=32768 OSL=1024 CONC_LIST="4" ./qwen3.5_fp4_sglang_bench.sh
```

包装脚本为服务器及子进程设置 `PYTHONPATH` 和 `SGLANG_FLYDSL_CAUSAL_CONV=1`。
显式启用的 `sitecustomize.py` 导入钩子会在 GDN backend 模块加载后替换其
`causal_conv1d_fn`。各个导入该模块的 worker 应输出
`[FlyDSL] GDN causal prefill convolution enabled`。这条日志证明选择了新实现；
实际执行情况需通过 GPU profile 确认。已运行的服务器不会被修改。测量基线时，
停止这个服务器，并在新 shell 中使用原来的 `qwen3.5_fp4_sglang_server.sh` 命令。

安装的 SGLang 包及原始启动脚本保持原样。Decode 和 EAGLE 验证仍使用原来的卷积
更新路径。导入钩子的模块路径针对当前检查过的 SGLang 版本，升级后需要重新确认。
bootstrap 目录提供 Python 的 `sitecustomize` 模块；如果环境依赖另一个
`sitecustomize`，应先合并启动钩子。

## 与同事共享

将 `conv` 目录与原来的服务器、客户端两个脚本一起发送。包装脚本根据自身位置查找
服务器脚本，因此需要保留 `kernel_tuning/conv` 目录结构。在当前工作目录中打包：

```bash
cd /var/home/InferenceX/local_bench
tar -czf /tmp/qwen35-flydsl-conv.tar.gz \
  --exclude='__pycache__' --exclude='*.pyc' \
  kernel_tuning/conv \
  qwen3.5_fp4_sglang_server.sh qwen3.5_fp4_sglang_bench.sh
```

通过常用的文件共享渠道发送 `/tmp/qwen35-flydsl-conv.tar.gz`。压缩包包含内核、
bootstrap 钩子、包装脚本、测试、双语文档和本地参考结果。运行时无需原来的
`conv_FlyDSL` 源码目录，新内核直接导入已安装的 `flydsl` 包。压缩包不包含模型权重
或 Python/ROCm 环境。

### 接收方环境

尽量使用相同的 ROCm/SGLang 容器或匹配的环境。本地验证使用 AMD Instinct MI355X，
安装版本如下：

| 组件 | 版本 |
| --- | --- |
| FlyDSL | `0.3.1` |
| SGLang | `0.5.18.dev20260829+g4d53767b09` |
| PyTorch | `2.9.1+rocm7.2.0.git7e1940d4` |
| PyTorch 报告的 HIP 版本 | `7.2.26015-fc0010cf6a` |

请随压缩包提供实际的容器镜像 tag/digest 或对应的构建说明；仅凭版本号可能无法
复现本地 SGLang 修改。冒烟测试需要 ROCm PyTorch 和 FlyDSL；对照测试还需要
SGLang 和 Triton。完整服务还需要原始启动脚本使用的 AITER 依赖、模型访问权限及
足够的 GPU 显存，TP4 命令需要四张 GPU。接收方自行配置 `MODEL_PATH` 或
Hugging Face 访问方式。其他 GPU 架构或软件版本需要重新验证。

### 解压、检查和运行

将 `/path/to/qwen35-flydsl-conv.tar.gz` 替换为下载的压缩包路径。解压目录可以自行
选择，不需要复用发送方的绝对路径。

```bash
mkdir -p ~/qwen35-flydsl
cd ~/qwen35-flydsl
tar -xzf /path/to/qwen35-flydsl-conv.tar.gz
python3 kernel_tuning/conv/check_conv.py --smoke
python3 kernel_tuning/conv/check_conv.py --bench \
  --channels 3072 --length 32768 --batch 1 \
  --output colleague_results.json
TP=4 CONC=4 bash kernel_tuning/conv/run_server.sh
```

服务器就绪后，在另一个终端进入同一目录运行客户端。使用 `bash` 调用，即使文件
共享工具没有保留可执行权限也可以运行：

```bash
cd ~/qwen35-flydsl
ISL=32768 OSL=1024 CONC_LIST="4" bash qwen3.5_fp4_sglang_bench.sh
```

确认出现前文说明的 FlyDSL 启用日志。如果正确性检查失败，应先解决环境或兼容性
问题，再启用服务。进行对照测量时，停止 FlyDSL 服务器，在新 shell 中用相同参数
启动原始脚本。请同事回传 `colleague_results.json`、GPU 和软件包/容器版本、完整的
服务器及客户端命令，以及配对的基线/FlyDSL 服务结果。内核耗时应与端到端吞吐和
质量结论分别报告。

## 接口与约束

`causal_conv1d_flydsl.causal_conv1d_fn` 接受当前 SGLang prefill 接口：

- 输入为 `[channels, total_tokens]`，权重为 `[channels, width]`。
- 支持 BF16、FP16、FP32，卷积宽度 2–5，可选 bias 和 SiLU/Swish。
- 支持打包序列偏移、空序列、可选历史缓存、间接缓存索引及填充槽位（默认 `-1`）。
- 状态形状为 `[slots, channels, history_capacity]`，容量至少为 `width-1`。
  只更新前 `width-1` 个值，与 SGLang prefill 一致。
- 支持至少一个轴步长为 1 的非连续张量；元数据必须连续。
- 活跃缓存槽位必须唯一且合法，CPU 长度必须与 GPU 偏移一致。
  `validate_data=True` 会将元数据读回 CPU 进行检查，仅用于调试，不应在图捕获中使用。

函数分配独立输出并原地更新状态。空序列及填充槽位不改变状态；填充位置的输出值
未定义，与 SGLang 一致。只有第一个时间分块访问状态，因此其他线程块不会与历史
更新竞争。短序列使用旧状态的寄存器快照。累加和 SiLU 使用 FP32，输出及状态存储
显式转换为相应类型。

默认每个线程块包含 128 个线程，处理 16 个 token。独立调用时可设置 `block`
（64/128/256）及 `tokens`（从 `width-1` 到 64）。内核按几何参数、步长、类型及可选
输入进行特化。图捕获前需预热每种签名；首次编译会执行一次计算并更新一次状态。

## 验证与耗时

```bash
python3 kernel_tuning/conv/check_conv.py --smoke
python3 kernel_tuning/conv/check_conv.py --bench \
  --channels 3072 --length 32768 --batch 1 \
  --output kernel_tuning/conv/results_tp4.json
```

测试包含 39 个用例，对照独立 FP32 PyTorch 参考实现；宽度 2–4 还与已安装的
SGLang Triton 比较。覆盖通道尾部、短序列和空序列、混合初始历史、填充及间接槽位、
额外状态容量、三种布局和类型、bias/激活选项、非默认 stream 以及图捕获和重放。
另外检查无状态、重复调用、全空输入及元数据不一致的情况。状态要求完全一致；
输出采用按类型设置的容差（BF16：`atol=rtol=0.016`，FP16：`0.002`，FP32：`2e-6`）。
无状态分组卷积检查采用 BF16 `atol=0.03125`。宽度 5 仅与独立参考实现比较，因为
当前 Triton forward 的计算逻辑只处理宽度 2–4。

现有 `../chunk_gated_delta_rule/shapes.py` 的 TP4 trace 记录中，Q/K heads 为
`[4,128]`，V heads 为 `[16,128]`，因此卷积通道数是
`2*4*128 + 16*128 = 3072`。服务器将打包 prefill 限制为 32768 个 token，客户端并发
4 不代表一次卷积会同时处理四个 32768-token 序列。相同总 token 预算的分组测试：

```bash
python3 kernel_tuning/conv/check_conv.py --bench-only \
  --channels 3072 --length 8192 --batch 4
```

MI355X 上首次测得 3072 通道、32768 token 的图执行耗时约为 FlyDSL 171 µs、
Triton 178 µs（1.04×）。此前 `results.json` 中的 2048 通道测量为 80 µs 和
95 µs（1.18×），形状不同。`tuning_results.json` 记录了另外六种启动配置，结果不足以
支持修改默认参数。测量使用 `triton.testing.do_bench_cudagraph`，包含状态写入和融合
激活，不包含编译和 Python 调用开销。这些结果是本地微基准，不是端到端服务加速比。
最终验证中，单个 32K 序列耗时为 175.4 µs 和 180.1 µs（1.027×）；四个打包的
8K 序列耗时为 172.6 µs 和 179.1 µs（1.038×）。这些差异较小，应通过重复的服务
性能测量进一步确认。

GPU 内核及 worker 导入钩子已在本地验证。尚未使用该替换实现验证完整模型启动、
生成文本质量或用户的完整客户端负载。原始服务器默认模拟 MTP 接受率；测量真实
接受率和质量时应使用 `SIMULATE_ACC=0`。
