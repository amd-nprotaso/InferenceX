# Qwen3.8 32768/32768 基准测试尝试

[English](report.md)

状态：在正式测量前受阻，未生成有效的性能对比结果。

已从 `/var/home/my_sglang/sglang` 安装可编辑版本的 SGLang（9699a8c65，包含工作区修改），并通过 `00_sglang_local.pth` 确保新版本优先加载。已将指定的 `hc_mix.py` 安装到当前 AITER 中。FlyDSL 已从 0.3.1 更新到 AITER 要求的 0.3.4.1。

计划使用 GPU 0、TP=1、并发数 1 和真实 NEXTN 推测解码，输入和输出均为 32768 tokens。两组均启用现有 GDN 和 HC 优化，仅第二组设置 `SGLANG_AITER_QSA_PA_DECODE=1`。

基线在编译 HC 内核 `_hc_up_silu` 时失败：FlyDSL 0.3.1 抛出 UnicodeDecodeError，0.3.4.1 在 `_infer_int_tuple_type` 中发生段错误。仓库已有的 GPU 正确性测试可独立复现。临时索引类型转换实验也失败，已撤销；当前安装的 hc_mix.py 与指定源文件逐字节一致。

没有完成任何正式测量请求。由于基线无法初始化，尚未启动启用该 flag 的测试组。日志、源代码版本信息、安装记录和测试脚本均已保存。需要先修复 HC 内核与 FlyDSL 的兼容性问题，才能继续性能比较。
