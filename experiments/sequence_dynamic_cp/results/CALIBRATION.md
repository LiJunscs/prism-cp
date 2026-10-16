# 保留的 H800 模型证据

这些数据用于解释既有候选和结果，不是 H100 的校准值。主快照为 `../configs/joint_16gpu_measured.json`。数据到模型/调度器的完整关系见 [接手文档](../../../progress/CODEX_HANDOFF.md)。

- [bench_data](../../../bench_data/README.md)：节点内 GQA/GDN 的 CP1/2/4/8 原始长度扫描、固定 global batch=8 扫描及吞吐/显存表。`cp_response_observations.csv` 保留当前 fit 使用的 99 条去重 F+B 观测，与原始表逐项一致。
- [analysis](../../../analysis/README.md)：早期 GDN/CSA/HCA 扩展性两表，共 20 条记录；属于算子差异的历史证据，不作为当前 GQA 响应。
- `historical_cp_response_20261012/profiles/`：节点内 GQA/GDN 的三个拟合口径；同目录保留 fit_statistics、fit_validation CSV。`summary.csv` 是历史调度候选摘要，不是原始 CP 测量。`joint_cp/response.py` 直接读取 fit。
- `cross_node_cp_evidence_20261012/`：同系列 CP8/16 的原始逐 rank 测量、配对响应和输入锚点。GQA 64Ki 的 CP8/16 完整模块 F+B 为 39.752/184.915ms，256Ki 为 468.353/712.500ms；两点均是 CP8 更快。
- `nsys_gqa_cross_node_overlap_20261012/`：只保留已有 overlap 的 summary、layer_summary、unit_coverage 与核对摘要。大 sqlite、逐 kernel CSV、图片及源代码快照已移除。
- `../../performance_model/results/cp1_20261009_p1_v2/`：CP1 stack coefficients、测量表、拟合残差、costs 与输入。保留的 `module_bench.py`、`fit_costs.py`、`cost_model.py` 可用于后续经授权的模块测量与拟合。

旧 H800 网络锚点：节点内 8 卡 1GiB/rank AllReduce 为 11.479ms / 163.70GB/s busbw，跨机 16 卡为 74.189ms / 27.14GB/s；16-rank ring 的 64/128MiB 每轮约 18.0/36.0ms。AllReduce busbw 不能替代 ring 带宽。原网络 sweep 和诊断执行脚本已退休。

历史 GDN 响应来自 32 heads，当前模型为 16 heads；CP16 GDN、packed 响应、MLP/路由及 overlap 仍有未校准部分。H100 需独立记录硬件、拓扑与响应，不能把旧模型预估当成新硬件实测。
