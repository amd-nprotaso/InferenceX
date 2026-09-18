# Experiment interrupted by concurrent GPU work

**English** | [中文](REPORT_zh.md)

The ISL=32768 / OSL=1024 four-way comparison is incomplete. Do not use the overlapping results to attribute performance to either custom kernel.

A separate process, PID 220478, started `python3 op_tests/test_moe_2stage.py -q 4 -dim 4096,256 -t 32768 -e 512 -k 10 -p t --kernel --no-legacy` at 2026-09-11 09:38:20 UTC. This overlaps round_1/conv/trial_2 and the GDN-only trials. GPU 3 reached about 305.6 GB allocated out of 309.2 GB while the other GPUs were around 228.5 GB. GDN compilation counts stayed stable and its counters showed no fallback, so those observations do not explain away the concurrent resource contention.

The comparison driver, its benchmark client, and its server were stopped; the separate MoE benchmark was left untouched. Logs and any completed raw results are preserved. No GPU profiles or reversed round were completed. A clean comparison requires idle GPUs and new output directories; do not append to the interrupted data.

See [interruption.json](interruption.json) for process identity, start time, and GPU-memory evidence, and [metadata.json](metadata.json) for the intended experiment settings.
