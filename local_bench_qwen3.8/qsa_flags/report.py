#!/usr/bin/env python3
"""Render measured compiler-flag tuning results in English and Chinese."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def main() -> None:
    data = json.loads((ROOT/'comparison.json').read_text())
    micro = json.loads((ROOT/'micro_summary.json').read_text())
    assert not data['checks']['server_argument_differences']
    mt = '| Variant | Decode µs | Verification µs | Bitwise equal to baseline |\n|---|---:|---:|---|\n'
    for x in micro:
        if x['status']=='pass':
            mt += f"| {x['name']} | {x['decode']:.3f} | {x['verify']:.3f} | {x['bitwise_equal_baseline']} |\n"
        else:
            mt += f"| {x['name']} | failed | failed | — |\n"
    et = '| Trial | E2E seconds | TTFT seconds | TPOT ms | Output tok/s | Acceptance length | Verification count |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for x in data['trials']:
        et += f"| {x['trial']} | {x['e2e_s']:.3f} | {x['ttft_s']:.3f} | {x['tpot_ms']:.3f} | {x['output_throughput_tok_s']:.3f} | {x['spec_accept_length']:.4f} | {x['spec_verify_ct']} |\n"
    m = data['summary']['output_throughput_tok_s']
    stat = f"Baseline mean: {m['baseline']['mean']:.3f} output tok/s; tuned mean: {m['tuned']['mean']:.3f}; change: {m['change_percent']:+.2f}%."
    zhstat = f"基线平均吞吐量：{m['baseline']['mean']:.3f} output tok/s；调优：{m['tuned']['mean']:.3f}；变化：{m['change_percent']:+.2f}%。"
    (ROOT/'report.md').write_text('''# gfx950 QSA compiler flag tuning

[中文](report_zh.md)

No repeatable E2E gain demonstrated: tuned throughput was58.98/154.36 tok/s versus baseline148.49/151.08. The faster repeat also had higher speculative acceptance. Keep the installed baseline. Both servers have been stopped; all four output hashes differ despite equal seeds.

Selected `-enable-post-misched=1` after a six-configuration microbenchmark sweep. It reduced verification latency by3.8%, with decode latency essentially unchanged. Kernel source and M128 N64 tile arithmetic were unchanged. Higher occupancy hints and reduced argument preload were slower. This experiment excludes fast math and tile changes.

## Microbenchmark

MI355X/gfx950; installed serving AITER stack; BF16, Hq24, Hkv2, D256, length-one varlen queries, selected KV2048–2051, decode/verify rows1/4. Twelve cases per variant, 20 GPU-event samples around HIP graph replay. Values below are medians across context positions. All variants passed independent FP32 checks (atol0.003, rtol0.03), max absolute error0.000937313, and their output hashes matched baseline on identical seeded synthetic inputs. This finite input check is not a guarantee for every model activation.

`post_sched` changes LLVM post-machine scheduling from0 to1. `occupancy2/3` set CK's block-per-CU hint (automatic is1 for D256); actual occupancy remains resource-dependent. `preload0/16` change `--amdgpu-kernarg-preload-count` from32. FAST_EXP2 stays1. Candidate selection uses3×decode+12×verification latency.

'''+mt+'''
## E2E comparison

Model amd/Qwen3.8-Flash-Next-Quark-MXFP4, TP1, concurrency1, ISL32768, OSL32768; profiling disabled. Original launcher copied locally with equal --random-seed20260928 added for both arms; HF_HOME=/data/hf_cache/. Two sequential requests per arm, baseline first. Each request follows a32768/32 warmup and cache flush. Microbenchmark execution ended before measured E2E requests.

'''+et+'\n'+stat+'''

These are descriptive means of two trials, not a confidence interval or a causal estimate of the flag's performance effect. Check acceptance and individual request values before interpreting the average. All measured requests used identical payloads, actual32768/32768 token counts, cached_tokens0, no retractions, and matching launch arguments excluding dynamic telemetry. Loaded-library mappings are saved per trial. Output hashes and serving verification-time proxies are in comparison.json; the latter include all serving work, not just the QSA kernel. Fixed seed does not enforce deterministic serving.

## Artifacts

[Compiler flag diff](compiler_flags.patch), [micro sweep](sweep.py), [micro results](micro_summary.json), [E2E harness](e2e.py), [comparison](comparison.json), [CSV](trials.csv). Builds and compiler commands are under micro/<variant>/build.json. Only the selected translation unit is rebuilt and linked into a private cache. Original installed libraries and launcher are preserved.

Run sweep.py, e2e.py, compare.py, report.py in that order. Preserve this completed directory and use a fresh artifact directory for another run; the E2E harness does not overwrite repeat symlinks. Results apply to concurrency1 only.
''')
    (ROOT/'report_zh.md').write_text('''# gfx950 QSA 编译参数调优

[English](report.md)

未证明可重复的端到端收益：调优58.98/154.36 tok/s，基线148.49/151.08。较快的调优重复也有更高投机接受率。建议保留原安装基线。两个服务均已停止；相同种子下四次输出哈希仍不同。

六种配置的微基准选择了 `-enable-post-misched=1`，verification 延迟降低3.8%，decode 延迟基本不变。内核源码与 M128 N64 tile 算术保持不变。更高 occupancy hint 与更少参数预加载均未改善性能。本次不测试 fast math 或 tile 修改。

## 微基准

MI355X/gfx950，使用服务端已安装的 AITER 栈；BF16、Hq24、Hkv2、D256、每行查询长度1、选中 KV2048–2051，decode/verify 行数1/4。每种配置十二个测试，HIP graph 重放结合 GPU event 计时，每项20个样本；表中为各上下文位置的中位数。所有配置均通过独立 FP32 校验（atol0.003、rtol0.03），最大绝对误差0.000937313；固定种子的相同合成输入产生与基线相同的输出哈希。有限样本检查不能保证所有模型激活均相同。

post_sched 将 LLVM post-machine scheduling 从0改为1；occupancy2/3 设置 CK 每 CU block 数提示（D256 自动值为1），实际驻留量仍取决于资源；preload0/16 将 --amdgpu-kernarg-preload-count 从32改为0/16。FAST_EXP2 保持1。选择评分为3×decode+12×verification 延迟。

'''+mt+'''
## 端到端比较

模型 amd/Qwen3.8-Flash-Next-Quark-MXFP4，TP1、并发1、ISL32768、OSL32768，关闭 profiling。复制原启动脚本，为两组添加相同 --random-seed20260928，HF_HOME=/data/hf_cache/。每组顺序测两次，先基线再调优；每次先运行32768/32预热并清空缓存。微基准在端到端计时前完成。

'''+et+'\n'+zhstat+'''

这些是两次试验的描述性均值，不是置信区间或参数收益的因果估计。解释平均值时需结合单次数据与投机接受率。四次 payload 相同，实际 token 数32768/32768、cached_tokens0、无 retraction；排除动态信息后启动参数相同。每次库映射均已保存。输出哈希及每次验证耗时代理值见 comparison.json，代理值包含其他服务开销，不是 QSA 内核计时。固定种子不保证服务输出确定性。

## 文件

[编译参数修改](compiler_flags.patch)、[微基准](sweep.py)、[微基准结果](micro_summary.json)、[端到端脚本](e2e.py)、[比较](comparison.json)、[CSV](trials.csv)。编译命令位于 micro/<variant>/build.json，仅重新编译选定翻译单元并链接到独立缓存，保留原安装库和启动脚本。

依次执行 sweep.py、e2e.py、compare.py、report.py。重跑时保留本次结果并使用新目录；端到端脚本不会覆盖 repeat 符号链接。结果仅适用于并发1。
''')
    print(stat)

if __name__=='__main__':
    main()
