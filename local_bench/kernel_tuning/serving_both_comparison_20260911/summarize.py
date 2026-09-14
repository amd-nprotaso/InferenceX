"""Summarize the saved default-versus-two-kernel serving experiment."""
import json
import re
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
fields = ['output_throughput', 'duration', 'mean_ttft_ms', 'mean_tpot_ms', 'accept_length']
summary = {}
for backend in ('baseline', 'both'):
    rows = []
    for trial in range(1, 4):
        files = list((root / backend / f'trial_{trial}').glob('*.jsonl'))
        assert len(files) == 1, files
        records = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert len(records) == 1
        row = records[0]
        assert row['completed'] == 40
        assert (row['total_input_tokens'], row['total_output_tokens']) == (1184722, 36717)
        assert row['cache_report']['cache_hit_rate_pct'] == 0
        rows.append({key: row[key] for key in fields})
    log = (root / backend / 'server.log').read_text(errors='replace')
    summary[backend] = {
        'trials': rows,
        'median': {key: statistics.median(row[key] for row in rows) for key in fields},
        'compile_calls': log.count('[FlyDSL-compile]'),
        'conv_enablement_messages': log.count('GDN causal prefill convolution enabled'),
        'gdn_enablement_messages': log.count('GDN chunk_h hook installed'),
        'gdn_stats': [json.loads(f.read_text()) for f in sorted((root / backend / 'gdn_stats').glob('*.json'))],
    }
baseline = summary['baseline']['median']
both = summary['both']['median']
summary['change_percent'] = {key: 100 * (both[key] / baseline[key] - 1) for key in fields}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
table = '| Metric | Default | Both FlyDSL kernels | Change |\n|---|---:|---:|---:|\n'
for label, key in [('Output throughput (tok/s)', 'output_throughput'), ('Duration (s)', 'duration'), ('Mean TTFT (ms)', 'mean_ttft_ms'), ('Mean TPOT (ms)', 'mean_tpot_ms'), ('Observed acceptance length', 'accept_length')]:
    table += f'| {label} | {baseline[key]:.3f} | {both[key]:.3f} | {summary["change_percent"][key]:+.3f}% |\n'
trials = '| Trial | Default (tok/s) | Both kernels (tok/s) |\n|---|---:|---:|\n'
for i in range(3):
    trials += f'| {i+1} | {summary["baseline"]["trials"][i]["output_throughput"]:.3f} | {summary["both"]["trials"][i]["output_throughput"]:.3f} |\n'
delta = summary['change_percent']['output_throughput']
en = f'''# Default versus both FlyDSL kernels

**English** | [中文](REPORT_zh.md)

Median output throughput changed **{delta:+.2f}%**, from **{baseline['output_throughput']:.2f}** to **{both['output_throughput']:.2f} tok/s**.

{table}
Values are medians across three trials; latency rows use the median of each trial's mean.

{trials}
Measured on 2026-09-11 using `kernel_tuning/run_server_both.sh` for both arms: FLYDSL_CONV=FLYDSL_GDN=0 for default and both flags 1 for the two-kernel version. TP=4, concurrency=4, nominal input/output lengths 32768/1024, random range ratio 0.8, seed 42. Each trial used 40 measured and 8 warmup requests. All 240 measured requests succeeded, with exactly 1,184,722 input and 36,717 output tokens per trial and zero cache hits.

Both arms use simulated MTP acceptance length 3.39, so these are projected throughput measurements, not real-acceptance or quality results. The table also records observed acceptance, which can fluctuate under simulation. All three baseline trials ran first, then three trials with both kernels in a fresh server. They are repeated trials within one server session per backend, not interleaved independent restarts; small differences must be interpreted with that limitation. Loading and warmup are excluded from timed duration.

Compilation counting was enabled for both arms. GDN usage was logged every 500 calls. [summary.json](summary.json) retains compile counts, hook enablement counts, and saved GDN usage/fallback counters. [metadata.json](metadata.json) records source hashes and package versions. Each backend/trial directory contains its raw JSONL and client log; each backend also has the server log and resolved command.

[run.sh](run.sh) records the exact experiment. Use a new output directory before rerunning to avoid appending to existing JSONL results.
'''
zh = f'''# 默认版本与两个 FlyDSL 内核的对比

[English](REPORT.md) | **中文**

输出吞吐量中位数从 **{baseline['output_throughput']:.2f}** 变为 **{both['output_throughput']:.2f} tok/s**，变化 **{delta:+.2f}%**。

{table}
表中数值为三轮测试的中位数；延迟指标取各轮平均值的中位数。

{trials}
测试日期为 2026-09-11。两组均使用 `kernel_tuning/run_server_both.sh`：默认版本设置 FLYDSL_CONV=FLYDSL_GDN=0，双内核版本将两者设为 1。TP=4，并发 4，标称输入/输出长度 32768/1024，随机长度比例 0.8，seed 42。每轮测量 40 个请求并预热 8 个请求。240 个测量请求全部成功，每轮均输入 1,184,722 token、输出 36,717 token，缓存命中率为零。

两组均采用模拟 MTP 接受长度 3.39，结果属于模拟接受率下的吞吐投影，不代表真实接受率或模型质量。表中同时记录实际观测接受长度，模拟模式下该值可能波动。先运行三轮默认版本，再启动新服务器运行三轮双内核版本。每组在同一服务进程中重复测试，并非交错重启的独立实验，解读小幅差异时应考虑这一限制。计时不包含模型加载及预热。

两组均启用编译计数，GDN 每 500 次调用记录一次使用情况。[summary.json](summary.json) 保存编译次数、hook 启用次数及 GDN 使用/回退计数；[metadata.json](metadata.json) 保存源码哈希及软件版本。各后端/轮次目录包含原始 JSONL 和客户端日志，各后端还保存服务器日志及最终命令。

[run.sh](run.sh) 保存完整实验命令。再次运行前使用新输出目录，避免向已有 JSONL 追加结果。
'''
(root / 'REPORT.md').write_text(en)
(root / 'REPORT_zh.md').write_text(zh)
print(table)
print(trials)
print(json.dumps({b: {k: v for k, v in summary[b].items() if k not in ('trials', 'median')} for b in ('baseline', 'both')}, indent=2))
