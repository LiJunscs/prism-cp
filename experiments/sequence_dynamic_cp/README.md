# 序列与层联合动态 CP

当前范围为 `GGG*GGG*`，原生 N 的固定/可变容量与二维联合方案。[项目归档](../../progress/ARCHIVE-20261016.md) 保存唯一当前进度；[执行合同](JOINT_CP.md) 说明算法与训练边界。

|入口|用途|
|---|---|
|`joint_cp/`|独立计划 IR、求解、完整显存/成本模型、分层执行与路由|
|`joint_cp_search.py`、`joint_cp_scale_search.py`、`joint_cp_scale_audit.py`|完整输入搜索与给定候选库计数 DP|
|`native.py`、`planner.py`、`mixed_plan.py`|原生函数提取、计划合法性和原生混合 CP 组批|
|`joint_advantage_search.py`、`random_batch_capacity_analysis.py`|有限候选与随机完整 batch 的容量分析|
|`random_batch_capacity_gpu.py`|三输入八项矩阵的准备与逐项执行；节点和输出必须显式指定|
|`gpu_smoke.py`、`runtime.py`|原生 N/二维冻结计划的完整训练入口|
|`results_io.py`|读取新 JSONL 或无损压缩的历史逐 rank 记录|

H100 CPU 搜索显式使用 `joint_cp_search.py --hardware H100 --manifest <inputs.json> --output <新目录> --world 16 --pattern 'GGG*GGG*'`，world=32 自动选 `configs/joint_32gpu_h100.json`。16/32 卡配置分别保留 CP8/16 与 CP8/16/32 实测响应，GDN32 用全部 3+7 次计量中位数；其余形状/CP 的外推和训练成本明确标为未校准。H800 配置 `configs/joint_16gpu_measured.json` 与全部历史证据保持不变。模型输出完整计划的 GPU·时间和预测每卡有效 TFLOP/s；实测收益按同一资源口径判断。

结果入口：

- `results/random_batch_capacity_gpu_20261016_r1/`：224Ki 已完成；不规则输入被外部占用打断；当前暂停。
- `results/native_capacity_ablation_20261016_r1/`：256Ki、72 条的 K16/K24/K32 对照。
- `results/joint_cp_memory_validation_20261016_r2/`：568 条短批重打包显存优化。
- `results/native_n_remeasure_20261016/`、`results/joint_cp_two_20261015/`：原生 N 和二维原始测量。
- `results/random_batch_capacity_20261016_r1/`：当前 3:1 随机完整输入的 CPU 比较。
- `results/joint_advantage_search_20261016_r1/`：原生/布局核对、候选评分缓存及当前三输入计划。
- `results/h100_capacity_analysis_20261016_r2/`：H100 完整训练迁移反例及执行器控制。
- `results/h100_module_cp_20261016_r1/`、`h100_module_cp_analysis_20261016_r1/`：H100 GQA/GDN CP8/16 原始记录及每卡效率分析。
- `results/h100_model_validation_20261016_r1/`：新响应的四锚点、13 候选回溯、独立求解/replay 和真实 CLI 的 CPU 验证。
- `results/h100_module_cp32_analysis_20261016_r1/combined_analysis.json`：CP32 两模块、GDN 方差复核的最终统计及源路径。
- `results/h100_joint_opportunity_20261016_r1/final_summary.json`、`dense_native_summary.json`：33 个 CPU 场景组合与弱正例的加密 N 容量检查。
- `results/CALIBRATION.md`：分开保留的 H800/H100 响应、模型边界与来源。

历史数据经过精简：输入、唯一计划、比较/监督摘要与原始完整更新字段保留；重复 jobs/replay、joint config、逐 kernel profiling、源码快照和成功运行的重复日志已删除。报告中的旧路径是原始来源，压缩逐 rank 数据位于对应 run 的 `records/steps_by_rank.json.gz`。当前脚本可以在新目录重建三输入八项矩阵，不覆盖历史证据。

迁移和 Git 范围见 [H100_MIGRATION.md](../../progress/H100_MIGRATION.md)。H100 已授权实验的结果见归档；阅读入口和 CPU 模型检查不要求执行 GPU `run`。
