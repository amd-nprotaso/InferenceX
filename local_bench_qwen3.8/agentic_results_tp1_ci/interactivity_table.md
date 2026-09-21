
### AITER/FlyDSL GDN kernels — latency & interactivity vs concurrency

Qwen3.8-Flash-Next MXFP4, 1x MI355X, TP=1, AgentX replay, 1800 s/point.
Parenthesised = delta vs stock-Triton baseline; positive = better.

| metric | conc 1 | conc 4 | conc 8 |
|---|---|---|---|
| **Interactivity (tok/s/user, higher better)** |  |  |  |
| median user (p50) | 115.6 (+1.3%) | 105.6 (+7.3%) | 100.7 (+18.2%) |
| slowest decile | 29.5 (-8.0%) | 27.3 (-28.1%) | 47.7 (+12.1%) |
| slowest percentile | 28.2 (-6.8%) | 25.1 (-8.7%) | 24.4 (+25.1%) |
| **Inter-token latency (ms, lower better)** |  |  |  |
| ITL p50 | 8.65 (+1.3%) | 9.47 (+6.8%) | 9.93 (+15.4%) |
| ITL p90 | 33.93 (-8.6%) | 36.58 (-39.1%) | 20.98 (+10.8%) |
| ITL p99 | 35.40 (-7.3%) | 39.87 (-9.5%) | 40.91 (+20.1%) |
| **Time to first token (ms, lower better)** |  |  |  |
| TTFT p50 | 692 (+23.0%) | 1024 (-14.3%) | 982 (+8.6%) |
| TTFT p90 | 3820 (+8.1%) | 2569 (-20.3%) | 2842 (+3.9%) |
| TTFT p99 | 18213 (-2.3%) | 5136 (-1.8%) | 11501 (+3.8%) |
| **End-to-end request latency (ms, lower better)** |  |  |  |
| request latency p50 | 5068 (+1.9%) | 4539 (-38.4%) | 4559 (+13.7%) |
| request latency p99 | 125822 (-3.3%) | 148069 (-33.6%) | 77703 (+11.9%) |
| **Aggregate throughput (higher better)** |  |  |  |
| output tput (tok/s) | 70.47 (+0.0%) | 183.96 (-2.2%) | 315.38 (+6.9%) |
| request tput (req/s) | 0.081 (+0.0%) | 0.196 (-0.3%) | 0.415 (+4.8%) |
| **Workload actually drawn (context, not a result)** |  |  |  |
| requests | 160 | 401 | 851 |
| output tokens | 128965 | 336682 | 580390 |
| sub-1ms ITL records excluded | 0 | 1 | 3 |
