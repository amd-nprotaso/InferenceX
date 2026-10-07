# 基线 QSA：HC combine split 端到端比较

[English](report.md)

已安装 SGLang 定义 `SGLANG_HC_COMBINE_SPLIT = EnvBool(True)`，因此不设置和 `=1` 启用相同设置。独立的 /var/home/my_sglang/sglang checkout 在 HIP 上默认关闭，但原启动脚本实际未使用该栈。本实验增加显式 `=0`，用于比较关闭与开启。

主表 enabled 平均154.048 output tok/s，disabled 平均137.940（关闭后低10.46%）。每次验证的 decode 耗时：开启21.068–21.073ms，关闭22.119–22.164ms，关闭后约高5.1%。建议在当前已安装栈中保持开启。每组两次的均值受生成轨迹与接受率影响，不能确定精确可重复收益。六次输出哈希均不同，所有服务均已停止。

## 设置

MI355X/gfx950，模型 amd/Qwen3.8-Flash-Next-Quark-MXFP4，TP1、并发1、ISL32768、OSL32768。所有组均使用 /sgl-workspace/aiter/aiter/jit 中已安装的基线 QSA 库；启动环境移除 AITER_JIT_DIR，不加载此前调优库。SGLang 从 /workspace/sglang-qwen-next/python/sglang 导入。复制原启动脚本，统一添加 --random-seed20260928，HF_HOME=/data/hf_cache/，关闭 profiling；其余参数相同。

按 unset、enabled、disabled 顺序，每组启动新服务并顺序执行两次测量；每次先运行32768/32预热并清空缓存。unset 显式移除变量，enabled 设为1，disabled 设为0。运行时探针和 scheduler 环境检查确认参数，每次库映射确认基线 QSA。

已安装 hyperconnection.py 在构造时读取参数。hc_count4、hidden_size2560 满足形状要求；满足类型条件的 BF16/FP16、行数不超过32的 batch 在开启时选择 hc_combine_split，关闭时选择 hc_combine；更大的 batch 无论开关均使用非 split 路径。这是源码路由说明，不是内核 trace，端到端测量未开启 profiling。

## 结果

| Setting | Trial | E2E seconds | TTFT seconds | TPOT ms | Output tok/s | Acceptance length | Verification count |
|---|---:|---:|---:|---:|---:|---:|---:|
| unset | 1 | 255.215 | 1.030 | 7.757 | 128.394 | 2.7241 | 12029 |
| unset | 2 | 196.990 | 1.032 | 5.980 | 166.343 | 3.5410 | 9254 |
| enabled | 1 | 200.648 | 1.032 | 6.092 | 163.311 | 3.4584 | 9475 |
| enabled | 2 | 226.322 | 1.029 | 6.876 | 144.785 | 3.0650 | 10691 |
| disabled | 1 | 219.527 | 1.035 | 6.668 | 149.266 | 3.3173 | 9878 |
| disabled | 2 | 258.804 | 1.033 | 7.867 | 126.613 | 2.8175 | 11630 |

| Setting | Mean per-request output tok/s |
|---|---:|
| unset | 147.368 |
| enabled | 154.048 |
| disabled | 137.940 |

Enabled versus unset: +4.53%. Disabled versus enabled: -10.46%.

均值为两次请求吞吐量的算术平均。unset 与 enabled 均为 True，不能将其差异解释为启用参数的效果。每组仅两次且投机接受率可能不同，限制了对开关差异的因果解释。单次接受长度、验证次数、输出哈希及每次验证 decode 耗时代理值见 comparison.json；代理值包含所有服务开销，不是 HC 内核计时。固定种子不保证确定性输出。

验证实际输入/输出均为32768/32768、请求成功、cached_tokens0、无 retraction、payload 相同、排除动态信息后启动参数相同，并检查参数值及基线库映射。未进行模型准确率评估，仅覆盖并发1。

## 审计恢复

首次 enabled 请求成功完成，244.079s、134.251 tok/s，但测量后 harness 断言失败：setproctitle 清除了操作系统环境快照中的变量，Python 仍正确读取原值。CPU 小复现确认了此行为。该结果保留在 enabled_audit_interrupted，不纳入主表的两次测量。重新运行 enabled 两次；enabled/disabled 设置 SPT_NOENV=1，在修改进程名称时保留环境快照，仅影响进程名称记录，不改变 HC 参数或模型计算。原 unset 两次保留，初次与恢复执行日志均保存。

## 文件与复现

[启动脚本](server-start.sh)、[脚本修改](launcher.patch)、[执行脚本](e2e.py)、[比较脚本](compare.py)、[完整比较](comparison.json)、[CSV](trials.csv)。各组目录保存服务及客户端日志、flag_probe.json、server_info.json、预热、请求、输出文本、结果和库映射；repeat 目录共享对应服务日志。

依次执行 e2e.py、compare.py、report.py。重跑时保留本次结果并使用新目录；已有 repeat 符号链接不会被覆盖。保留原启动脚本及已安装 QSA 库。
