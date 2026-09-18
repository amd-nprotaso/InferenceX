# GPU 并发任务导致实验中断

[English](REPORT.md) | **中文**

ISL=32768 / OSL=1024 四版本对比尚未完成。不能将受重叠任务影响的结果用于判断任一自定义内核的性能。

另一个进程 PID 220478 于 2026-09-11 09:38:20 UTC 启动 `python3 op_tests/test_moe_2stage.py -q 4 -dim 4096,256 -t 32768 -e 512 -k 10 -p t --kernel --no-legacy`，与 round_1/conv/trial_2 及 GDN-only 测试重叠。GPU 3 在总计 309.2 GB 的显存中占用约 305.6 GB，其他 GPU 约占用 228.5 GB。GDN 编译计数保持不变且没有回退，但这不能排除已观测到的并发资源争用。

已停止本实验的驱动脚本、客户端和服务器，未操作另一个 MoE 测试进程。日志及已完成的原始结果均已保留。尚未完成 GPU profile 或反向轮次。有效对比需要空闲 GPU 和新输出目录，不应向中断数据追加记录。

进程身份、启动时间和显存证据见 [interruption.json](interruption.json)，实验设置见 [metadata.json](metadata.json)。
