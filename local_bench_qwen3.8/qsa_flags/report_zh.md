# gfx950 QSA 编译参数调优

[English](report.md)

未证明可重复的端到端收益：调优58.98/154.36 tok/s，基线148.49/151.08。较快的调优重复也有更高投机接受率。建议保留原安装基线。两个服务均已停止；相同种子下四次输出哈希仍不同。

六种配置的微基准选择了 `-enable-post-misched=1`，verification 延迟降低3.8%，decode 延迟基本不变。内核源码与 M128 N64 tile 算术保持不变。更高 occupancy hint 与更少参数预加载均未改善性能。本次不测试 fast math 或 tile 修改。

## 微基准

MI355X/gfx950，使用服务端已安装的 AITER 栈；BF16、Hq24、Hkv2、D256、每行查询长度1、选中 KV2048–2051，decode/verify 行数1/4。每种配置十二个测试，HIP graph 重放结合 GPU event 计时，每项20个样本；表中为各上下文位置的中位数。所有配置均通过独立 FP32 校验（atol0.003、rtol0.03），最大绝对误差0.000937313；固定种子的相同合成输入产生与基线相同的输出哈希。有限样本检查不能保证所有模型激活均相同。

post_sched 将 LLVM post-machine scheduling 从0改为1；occupancy2/3 设置 CK 每 CU block 数提示（D256 自动值为1），实际驻留量仍取决于资源；preload0/16 将 --amdgpu-kernarg-preload-count 从32改为0/16。FAST_EXP2 保持1。选择评分为3×decode+12×verification 延迟。

| Variant | Decode µs | Verification µs | Bitwise equal to baseline |
|---|---:|---:|---|
| baseline | 125.672 | 147.665 | True |
| post_sched | 125.282 | 142.026 | True |
| occupancy2 | 164.714 | 171.109 | True |
| occupancy3 | 388.166 | 397.097 | True |
| preload0 | 128.498 | 148.141 | True |
| preload16 | 126.652 | 147.982 | True |

## 端到端比较

模型 amd/Qwen3.8-Flash-Next-Quark-MXFP4，TP1、并发1、ISL32768、OSL32768，关闭 profiling。复制原启动脚本，为两组添加相同 --random-seed20260928，HF_HOME=/data/hf_cache/。每组顺序测两次，先基线再调优；每次先运行32768/32预热并清空缓存。微基准在端到端计时前完成。

| Trial | E2E seconds | TTFT seconds | TPOT ms | Output tok/s | Acceptance length | Verification count |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 220.674 | 1.035 | 6.703 | 148.491 | 3.1502 | 10402 |
| baseline_repeat | 216.895 | 1.032 | 6.588 | 151.078 | 3.2069 | 10218 |
| tuned | 555.619 | 1.033 | 16.925 | 58.976 | 1.2464 | 26291 |
| tuned_repeat | 212.279 | 1.030 | 6.447 | 154.363 | 3.2628 | 10043 |

基线平均吞吐量：149.784 output tok/s；调优：106.669；变化：-28.78%。

这些是两次试验的描述性均值，不是置信区间或参数收益的因果估计。解释平均值时需结合单次数据与投机接受率。四次 payload 相同，实际 token 数32768/32768、cached_tokens0、无 retraction；排除动态信息后启动参数相同。每次库映射均已保存。输出哈希及每次验证耗时代理值见 comparison.json，代理值包含其他服务开销，不是 QSA 内核计时。固定种子不保证服务输出确定性。

## 文件

[编译参数修改](compiler_flags.patch)、[微基准](sweep.py)、[微基准结果](micro_summary.json)、[端到端脚本](e2e.py)、[比较](comparison.json)、[CSV](trials.csv)。编译命令位于 micro/<variant>/build.json，仅重新编译选定翻译单元并链接到独立缓存，保留原安装库和启动脚本。

依次执行 sweep.py、e2e.py、compare.py、report.py。重跑时保留本次结果并使用新目录；端到端脚本不会覆盖 repeat 符号链接。结果仅适用于并发1。
