"""Validate all throughput trials and save the four-way comparison."""
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
fields = ('output_throughput', 'duration', 'mean_e2e_latency_ms', 'mean_ttft_ms',
          'mean_tpot_ms', 'accept_length')
summary = {}
tokens = set()
for backend in ('baseline', 'conv', 'gdn', 'both'):
    rows = []
    sessions = []
    for round_number in (1, 2):
        session = root / f'round_{round_number}' / backend
        for trial in (1, 2):
            files = list((session / f'trial_{trial}').glob('*.jsonl'))
            assert len(files) == 1, (session, trial, files)
            records = [json.loads(line) for line in files[0].read_text().splitlines()]
            assert len(records) == 1
            record = records[0]
            assert record['completed'] == 40
            assert record['cache_report']['cache_hit_rate_pct'] == 0
            tokens.add((record['total_input_tokens'], record['total_output_tokens']))
            rows.append({'round': round_number, 'trial': trial,
                         **{key: record[key] for key in fields}})
        log = (session / 'server.log').read_text(errors='replace')
        compile_counts = {}
        for trial in (1, 2):
            begin = f'[experiment] trial {trial} begin'
            end = f'[experiment] trial {trial} end'
            assert begin in log and end in log
            window = log.split(begin, 1)[1].split(end, 1)[0]
            compile_counts[str(trial)] = window.count('[FlyDSL-compile]')
        stats = [json.loads(f.read_text()) for f in (session / 'gdn_stats').glob('*.json')]
        sessions.append({'round': round_number,
                         'compile_calls': log.count('[FlyDSL-compile]'),
                         'trial_including_warmup_compile_calls': compile_counts,
                         'conv_banners': log.count('GDN causal prefill convolution enabled'),
                         'gdn_banners': log.count('GDN chunk_h hook installed'),
                         'gdn_calls': sum(s['calls'] for s in stats),
                         'gdn_fallbacks': sum(s['fallback'] for s in stats)})
    summary[backend] = {'trials': rows, 'sessions': sessions,
                        'median': {key: statistics.median(r[key] for r in rows) for key in fields},
                        'round_mean_throughput': {str(n): statistics.mean(r['output_throughput'] for r in rows if r['round'] == n) for n in (1, 2)}}
assert len(tokens) == 1, tokens
baseline = summary['baseline']
for backend, result in summary.items():
    result['median_throughput_change_pct'] = 100 * (result['median']['output_throughput'] / baseline['median']['output_throughput'] - 1)
    result['round_throughput_change_pct'] = {n: 100 * (v / baseline['round_mean_throughput'][n] - 1) for n, v in result['round_mean_throughput'].items()}
(root / 'summary.json').write_text(json.dumps({'tokens_per_trial': list(tokens)[0], 'backends': summary}, indent=2) + '\n')
table = '| Version | Median tok/s | Change | Trial range (tok/s) | TTFT (ms) | TPOT (ms) | Acceptance |\n|---|---:|---:|---:|---:|---:|---:|\n'
for backend, result in summary.items():
    m = result['median']
    values = [r['output_throughput'] for r in result['trials']]
    table += f"| {backend} | {m['output_throughput']:.2f} | {result['median_throughput_change_pct']:+.2f}% | {min(values):.2f}–{max(values):.2f} | {m['mean_ttft_ms']:.2f} | {m['mean_tpot_ms']:.3f} | {m['accept_length']:.4f} |\n"
round_table = '| Version | Round 1 vs default | Round 2 vs default |\n|---|---:|---:|\n'
for backend, result in summary.items():
    d = result['round_throughput_change_pct']
    round_table += f"| {backend} | {d['1']:+.2f}% | {d['2']:+.2f}% |\n"
en = f'''# Four-way serving experiment at 32K/1K

**English** | [中文](REPORT_zh.md)

{table}
Each entry summarizes four throughput trials from two fresh server sessions. Latency and acceptance columns are medians of the per-trial mean metrics. Trial ranges describe the observed samples, not confidence intervals.

{round_table}
Round comparisons use the mean throughput of each version's two trials divided by that round's default mean.

TP=4, concurrency=4, ISL=32768, OSL=1024, random range ratio 0.8, client seed 42. Each throughput trial contains 40 measured requests and 8 warmup requests, with cache flushing enabled. All 640 measured requests succeeded and had identical aggregate input/output token counts and zero cache hits. Model loading and warmup are excluded from measured duration.

Round 1 runs default, convolution only, GDN only, then both. Round 2 reverses the order. Each session starts a fresh server through `run_server_both.sh`, setting FLYDSL_CONV and FLYDSL_GDN for the selected arm. The default keeps the original server's existing tuning and disables the two custom overrides. Mirroring the order reduces a linear time trend but does not eliminate all scheduling, clock, or acceptance variation. Four trials within two sessions are not four independent server replications.

The scripts' simulated MTP acceptance target remains 3.39. Actual sampled acceptance is reported above; these are simulated-acceptance throughput results, not a quality evaluation. Neither TTFT nor TPOT is a direct kernel timer.

After each round-2 session's throughput trials, a separate GPU-only profile runs four measured requests plus four warmup requests, still at ISL=32768 and OSL=1024. Capture is bounded to 12 forward steps. Profile timings must not be treated as throughput measurements.

See [summary.json](summary.json), [metadata.json](metadata.json), and [run.sh](run.sh). Raw results and logs are organized by round, version, and trial. Compilation counts covering a trial include its warmup; startup and profiler compilations are separate. Do not append new measurements to these result directories when reproducing.
'''
zh = f'''# 32K/1K 四版本服务性能实验

[English](REPORT.md) | **中文**

{table}
每个版本汇总四轮吞吐测试，来自两个独立启动的服务进程。延迟及接受长度列为各轮平均指标的中位数。测量范围表示观测样本范围，不是置信区间。

{round_table}
每轮对比取该版本两次测试的平均吞吐量，除以同一轮默认版本的平均吞吐量。

TP=4、并发 4、ISL=32768、OSL=1024、随机长度比例 0.8、客户端 seed 42。每次吞吐测试测量 40 个请求并预热 8 个请求，启用缓存清理。640 个测量请求全部成功，各次输入/输出 token 总数完全一致，缓存命中率为零。测量耗时不包含模型加载及预热。

第一轮依次运行默认、仅卷积、仅 GDN、双内核，第二轮顺序相反。每组均通过 `run_server_both.sh` 启动新服务器，并设置相应的 FLYDSL_CONV、FLYDSL_GDN。默认版本保留原服务器已有调优，关闭两个自定义替换。反向顺序可减弱线性时间漂移影响，但无法消除所有调度、频率及接受长度波动。两个服务进程内的四次测试不能视为四次独立的服务启动实验。

保留脚本的模拟 MTP 接受长度目标 3.39，表中报告实际采样接受长度。结果是模拟接受率下的吞吐测量，不是质量评估。TTFT、TPOT 都不是内核直接计时。

第二轮每个版本完成吞吐测试后，另运行 GPU-only profile，测量四个请求并预热四个请求，保持 ISL=32768、OSL=1024，将采集限制为 12 个 forward step。profile 耗时不能作为吞吐测量。

参见 [summary.json](summary.json)、[metadata.json](metadata.json) 和 [run.sh](run.sh)。原始结果及日志按轮次、版本、测试编号保存。每次测试的编译计数包含其预热阶段，启动和 profile 阶段另计。复现时使用新目录，不要向已有结果追加记录。
'''
(root / 'REPORT.md').write_text(en)
(root / 'REPORT_zh.md').write_text(zh)
print(table)
print(round_table)
