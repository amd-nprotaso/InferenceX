#!/usr/bin/env python3
"""Render the completed HC combine split comparison."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main() -> None:
    data=json.loads((ROOT/'comparison.json').read_text())
    assert not data['checks']['server_argument_differences']
    table='| Setting | Trial | E2E seconds | TTFT seconds | TPOT ms | Output tok/s | Acceptance length | Verification count |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for x in data['trials']:
        arm=x['trial'].removesuffix('_repeat');trial=2 if x['trial'].endswith('_repeat') else 1
        table+=f"| {arm} | {trial} | {x['e2e_s']:.3f} | {x['ttft_s']:.3f} | {x['tpot_ms']:.3f} | {x['output_throughput_tok_s']:.3f} | {x['spec_accept_length']:.4f} | {x['spec_verify_ct']} |\n"
    m=data['summary']['output_throughput_tok_s']
    means='| Setting | Mean per-request output tok/s |\n|---|---:|\n'
    for arm in ['unset','enabled','disabled']:means+=f"| {arm} | {m[arm]['mean']:.3f} |\n"
    delta=f"Enabled versus unset: {m['enabled_vs_unset_percent']:+.2f}%. Disabled versus enabled: {m['disabled_vs_enabled_percent']:+.2f}%."
    text='''# Baseline QSA: HC combine split E2E comparison

[中文](report_zh.md)

The installed SGLang defines `SGLANG_HC_COMBINE_SPLIT = EnvBool(True)`: unset and `=1` enable the same setting. The separate /var/home/my_sglang/sglang checkout defaults to disabled on HIP, but it is not the serving stack used by the supplied launcher. An explicit `=0` arm is included to measure disabled versus enabled.

The primary enabled trials averaged154.048 output tok/s versus137.940 disabled (disabled10.46% lower). Decode elapsed time per verification was21.068–21.073ms enabled versus22.119–22.164ms disabled, about5.1% higher when disabled. Keep split combine enabled on this installed stack. These two-trial E2E means are affected by generated trajectories and acceptance; they do not establish a precise repeatable speedup. All six output hashes differ. All servers are stopped.

## Setup

MI355X/gfx950, model amd/Qwen3.8-Flash-Next-Quark-MXFP4, TP1, concurrency1, ISL32768, OSL32768. All arms use the installed baseline QSA module under /sgl-workspace/aiter/aiter/jit; AITER_JIT_DIR is removed from the launch environment, so none of the tuned experimental libraries are used. SGLang imports from /workspace/sglang-qwen-next/python/sglang. The original launcher is copied with equal --random-seed20260928 added, HF_HOME=/data/hf_cache/; no profiling. All other launch settings remain equal.

Two sequential requests per setting, in order unset, enabled, disabled; each setting starts a fresh server. Each measured request follows a32768/32 warmup and cache flush. The flag is explicitly removed for unset, set to1 for enabled, and set to0 for disabled. Runtime probes and scheduler environment checks confirm these values; per-trial library mappings verify the baseline QSA library.

The installed hyperconnection.py reads this flag at construction. hc_count4 and hidden_size2560 satisfy its split-shape checks. Eligible BF16/FP16 batches with at most32 rows select hc_combine_split when enabled; disabled selects hc_combine. Larger batches use the unsplit path regardless. This source-based routing explanation is not a kernel trace; measured E2E requests were unprofiled.

## Results

'''+table+'\n'+means+'\n'+delta+'''

Means are arithmetic means of two per-request throughput values. The unset-versus-enabled difference cannot be attributed to enabling this flag because both settings resolve to True. Two trials per setting and differing speculative acceptance limit causal interpretation of enabled-versus-disabled differences. Individual acceptance lengths, verification counts, output hashes, and decode-time-per-verification proxies are preserved in comparison.json; the proxy includes all serving work and is not an HC kernel timer. A fixed seed does not guarantee deterministic serving outputs.

Validation checks exact32768/32768 actual token counts, successful completion, cached_tokens0, no retractions, matching request payloads, equal server arguments excluding dynamic telemetry, flag values, and baseline library mappings. No model accuracy evaluation is included. This comparison covers concurrency1 only.

## Harness audit recovery

The first enabled request completed in244.079s at134.251 tok/s, but a post-run harness assertion failed: setproctitle erased the flag from the OS environment snapshot while Python still read its value correctly. A CPU-only reproduction confirmed this. The run is preserved in enabled_audit_interrupted and excluded from the primary two-trial table. The enabled pair was restarted; enabled/disabled runs set SPT_NOENV=1 to preserve environment snapshots during process renaming. This changes process-title bookkeeping, not the HC flag or model computation. Initial unset trials are retained. Both original and resumed orchestration logs are preserved.

## Artifacts and reproduction

[Launcher](server-start.sh), [launcher patch](launcher.patch), [harness](e2e.py), [comparison script](compare.py), [full comparison](comparison.json), [CSV](trials.csv). Each setting directory contains server/client logs, flag_probe.json, server_info.json, warmup, request, output text, results, and library mappings. Repeat directories share their setting's server log.

Run e2e.py, compare.py, report.py in order from this directory or via their paths. Use a fresh artifact directory for another run and preserve these results; existing repeat symlinks are not overwritten. Original launcher and installed QSA libraries are preserved.
'''
    (ROOT/'report.md').write_text(text)
    zh='''# 基线 QSA：HC combine split 端到端比较

[English](report.md)

已安装 SGLang 定义 `SGLANG_HC_COMBINE_SPLIT = EnvBool(True)`，因此不设置和 `=1` 启用相同设置。独立的 /var/home/my_sglang/sglang checkout 在 HIP 上默认关闭，但原启动脚本实际未使用该栈。本实验增加显式 `=0`，用于比较关闭与开启。

主表 enabled 平均154.048 output tok/s，disabled 平均137.940（关闭后低10.46%）。每次验证的 decode 耗时：开启21.068–21.073ms，关闭22.119–22.164ms，关闭后约高5.1%。建议在当前已安装栈中保持开启。每组两次的均值受生成轨迹与接受率影响，不能确定精确可重复收益。六次输出哈希均不同，所有服务均已停止。

## 设置

MI355X/gfx950，模型 amd/Qwen3.8-Flash-Next-Quark-MXFP4，TP1、并发1、ISL32768、OSL32768。所有组均使用 /sgl-workspace/aiter/aiter/jit 中已安装的基线 QSA 库；启动环境移除 AITER_JIT_DIR，不加载此前调优库。SGLang 从 /workspace/sglang-qwen-next/python/sglang 导入。复制原启动脚本，统一添加 --random-seed20260928，HF_HOME=/data/hf_cache/，关闭 profiling；其余参数相同。

按 unset、enabled、disabled 顺序，每组启动新服务并顺序执行两次测量；每次先运行32768/32预热并清空缓存。unset 显式移除变量，enabled 设为1，disabled 设为0。运行时探针和 scheduler 环境检查确认参数，每次库映射确认基线 QSA。

已安装 hyperconnection.py 在构造时读取参数。hc_count4、hidden_size2560 满足形状要求；满足类型条件的 BF16/FP16、行数不超过32的 batch 在开启时选择 hc_combine_split，关闭时选择 hc_combine；更大的 batch 无论开关均使用非 split 路径。这是源码路由说明，不是内核 trace，端到端测量未开启 profiling。

## 结果

'''+table+'\n'+means+'\n'+delta+'''

均值为两次请求吞吐量的算术平均。unset 与 enabled 均为 True，不能将其差异解释为启用参数的效果。每组仅两次且投机接受率可能不同，限制了对开关差异的因果解释。单次接受长度、验证次数、输出哈希及每次验证 decode 耗时代理值见 comparison.json；代理值包含所有服务开销，不是 HC 内核计时。固定种子不保证确定性输出。

验证实际输入/输出均为32768/32768、请求成功、cached_tokens0、无 retraction、payload 相同、排除动态信息后启动参数相同，并检查参数值及基线库映射。未进行模型准确率评估，仅覆盖并发1。

## 审计恢复

首次 enabled 请求成功完成，244.079s、134.251 tok/s，但测量后 harness 断言失败：setproctitle 清除了操作系统环境快照中的变量，Python 仍正确读取原值。CPU 小复现确认了此行为。该结果保留在 enabled_audit_interrupted，不纳入主表的两次测量。重新运行 enabled 两次；enabled/disabled 设置 SPT_NOENV=1，在修改进程名称时保留环境快照，仅影响进程名称记录，不改变 HC 参数或模型计算。原 unset 两次保留，初次与恢复执行日志均保存。

## 文件与复现

[启动脚本](server-start.sh)、[脚本修改](launcher.patch)、[执行脚本](e2e.py)、[比较脚本](compare.py)、[完整比较](comparison.json)、[CSV](trials.csv)。各组目录保存服务及客户端日志、flag_probe.json、server_info.json、预热、请求、输出文本、结果和库映射；repeat 目录共享对应服务日志。

依次执行 e2e.py、compare.py、report.py。重跑时保留本次结果并使用新目录；已有 repeat 符号链接不会被覆盖。保留原启动脚本及已安装 QSA 库。
'''
    (ROOT/'report_zh.md').write_text(zh)
    print(means+'\n'+delta)

if __name__=='__main__':main()
