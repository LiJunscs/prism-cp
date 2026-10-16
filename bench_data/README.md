# 节点内 CP 扩展性原始数据

这些小表是研究证据，随源码迁移，不能只留下拟合系数。时间单位为 ms，显存单位以 CSV 列名为准。H100 使用独立新数据，保留这些历史值解释旧结论。

|文件|数据与用途|
|---|---|
|`bench_merged_cp1.csv` / `cp2` / `cp4` / `cp8`|分别 45/48/54/54 行。包含 attention（GQA）与 gated_delta_net 的 F、B、F+B 测量、TFLOPS、MFU、峰值显存。B=1 的长度扩展性来自这些表；形状见各文件头：H4096、GQA Q32/KV8、GDN 32 heads。|
|`fixed_overhead.csv`|32 行；world=8、固定 global batch=8，microbatch 随 CP 改变。保留 F/B/both 时间与显存；不能与固定 B=1 混作同一种实验口径。|
|`bench_mfu_mem_log.csv`|58 行历史吞吐与显存记录，保留作测量背景；不是当前二维评分器的直接输入。|
|`cp_response_observations.csv`|当前 CP 响应拟合实际使用的 99 条去重 F+B 观测。`source,data_row` 指向原 CSV，data_row 从去掉注释和表头后的第 1 行起计数。`sequence_ki` 为 tokens/1024。|

六份原始表从本仓库已提交版本恢复（最后修改它们的提交为 `74e384f`）。`cp_response_observations.csv` 来自保留的 [fit_validation.csv](../experiments/sequence_dynamic_cp/results/historical_cp_response_20261012/fit_validation.csv)，其中的观测值与恢复后的源表逐条一致，包括 `fixed_overhead.csv` 的全部 32 条 F+B 时间。它保存观测值，不是模型预测。

拟合结果在 [historical_cp_response_20261012/profiles](../experiments/sequence_dynamic_cp/results/historical_cp_response_20261012/profiles/)；该目录外的 `summary.csv` 是历史调度候选摘要，不是原始 CP 测量表。`fit_statistics.csv` 和 `fit_validation.csv` 保存误差及长度留出检查。当前二维响应由 `joint_cp/response.py` 加载，关联见 [接手文档](../progress/CODEX_HANDOFF.md)。

更早的 GDN/CSA/HCA 对照在 [analysis](../analysis/README.md)，跨节点 CP8/16 与 CP1 stack 数据见 [CALIBRATION.md](../experiments/sequence_dynamic_cp/results/CALIBRATION.md)。
