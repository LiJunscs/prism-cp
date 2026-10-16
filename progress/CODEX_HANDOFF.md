# 新集群 Codex 接手：数据 → 模型 → 调度 → 执行

研究目标是 GDN:GQA=3:1 模型的序列与层联合动态 CP。比较三个方案：全局固定统一容量的原生 N、每个 global batch 可变统一容量的原生 N，以及我们的二维调度器。batch 内和 batch 间长度分布都允许随机；当前 GPU 证据是固定长度集合的短运行。两个仓库一起迁移，GPU 仍按用户要求暂停。

先读 [当前事实与结果](ARCHIVE-20261016.md)，再读本页实现地图和 [执行合同](../experiments/sequence_dynamic_cp/JOINT_CP.md)。下一步由 [roadmap](HETEROGENEOUS_CP_MASTER_ROADMAP_20260918.md) 定义，路径和 Git 范围见 [H100_MIGRATION.md](H100_MIGRATION.md)。

本次 prism-cp 迁移提交配套 Megatron-LM `ce909a8ca5d9e1d7c1ada33e7ec6d1adc7e08312`（`prism-cp` 分支）。两个仓库均需带到新集群；本次只创建本地提交，尚未 push。

## 必须带走的数据

下面路径从 prism-cp 根目录起算。原始小表和压缩完整训练记录均应提交；不能仅迁移结论或拟合系数。

|证据|路径|连接到后续工作|
|---|---|---|
|早期 GDN/CSA/HCA CP1/2/4/8 扩展性|`analysis/gdn_csa_hca_scaling.csv`、`analysis/csa_hca(csa128)_scaling.csv`|解释算子 CP 响应差异；[数据说明](../analysis/README.md)。|
|GQA/GDN 节点内长度扫描和固定 batch 扫描|`bench_data/bench_merged_cp{1,2,4,8}.csv`、`fixed_overhead.csv`、`bench_mfu_mem_log.csv`|F/B/F+B、吞吐、显存原始表；[来源和口径](../bench_data/README.md)。|
|二维响应使用的原始观测、fit 与误差|`bench_data/cp_response_observations.csv`；`experiments/sequence_dynamic_cp/results/historical_cp_response_20261012/`|99 条源观测与三个 profile（joint/single_b1/fixed_g8）。|
|跨节点 GQA CP8/16 配对|`experiments/sequence_dynamic_cp/results/cross_node_cp_evidence_20261012/`|逐 rank 原始值、配对响应、64/256Ki 锚点；供 CP16 响应政策使用。|
|已有通信 overlap 证据|`experiments/sequence_dynamic_cp/results/nsys_gqa_cross_node_overlap_20261012/`|summary、layer_summary、unit_coverage；已删除大 profiling 数据库。|
|CP1 模块/stack 校准|`experiments/performance_model/results/cp1_20261009_p1_v2/`|measurements、inputs、coefficients、costs、residuals；`module_bench.py`、`fit_costs.py`、`cost_model.py` 仍在。|
|完整训练结果|`experiments/sequence_dynamic_cp/results/` 下保留的 N、joint、容量消融、显存修复及随机 batch 实验目录|输入、计划、协议、每步时间/显存、拒绝或失败状态；逐 rank 记录压缩为 `steps_by_rank.json.gz`，由 `results_io.py` 读取。|

[CALIBRATION.md](../experiments/sequence_dynamic_cp/results/CALIBRATION.md) 补充历史拓扑与测量口径。CSV 与 fit 都保留，不能把 `historical_cp_response.../summary.csv` 误认作原始扩展性数据。历史 GDN 为 32 heads，当前训练为 16 heads；旧响应是迁移先验，H100 需要自己的配置与同硬件对照。

## 实现地图

以下模块均位于 `experiments/sequence_dynamic_cp/`，Megatron 路径另列。

|环节|入口|负责什么|
|---|---|---|
|输入|`joint_cp/inputs.py`、`two_dimensional_inputs.py`、`two_dimensional_packet.py`|显式长度 manifest、稳定 sample ID、确定性 token/label/mask、packet 供数。|
|计划表示|`joint_cp/ir.py`|Group/Layout/Microbatch/Plan、rank/sample 归属、布局及完整输入覆盖校验。|
|时间与显存模型|`joint_cp/model.py`、`response.py`、`analysis.py`|算子 CP 响应、拓扑搬运、组间最大耗时、跨 F/B 存活显存、整步成本与分项解释。|
|独立二维调度器|`joint_cp/solver.py`；命令 `joint_cp_search.py`|从完整输入池选微批、packing、各层 CP、rank 组和 MLP token 放置；不依赖 N 的计划。exact 与 bounded 均保留。|
|大输入搜索和显存选择|`joint_cp_scale_search.py`、`joint_cp_scale_audit.py`、`joint_cp/memory_selection.py`|微批库/DP、CP1 短批重打包，保留完整执行计划。|
|固定/可变容量 N 对照|`native.py`、`native_n_generalization_theory.py`、`joint_advantage_search.py`、`random_batch_capacity_analysis.py`|调用 Megatron 原函数生成 N，逐输入与跨输入分布选择统一 K；有限候选比较和对称显存偏好。|
|计划回放与训练|`joint_cp/replay.py`、`adapter.py`、`transport.py`；`runtime.py`、`gpu_smoke.py`|冻结计划编译，层边界与 MLP 搬运，输入到优化器更新的完整训练。|
|实验准备与启动|`random_batch_capacity_gpu.py`、根目录 `scripts/run_manual_gdn_cp_case.sh`、`scripts/torchrun_launcher.py`|重建保留矩阵；显式指定单项启动；主机、Python、输出路径可配置。|

组内 S=ΣL、Q=ΣL²：GQA 计算项随 Q 和 S，GDN/MLP 随 S；各阶段取并行组最大时间，累加微批、路由、输出头和更新成本。显存模型保存跨层 F/B 激活，同时计入模型状态、workspace、搬运暂存及余量。`analysis.measure` 可对任意合法计划复算，`analysis.describe` 给出阶段归因。

当前联合模型配置为 `configs/joint_16gpu_measured.json`。CP1 计算主体仍是解析 FLOP/吞吐系数；`response.py` 将历史 CP 响应转为有效残差，并用 CP8/16 配对处理跨机 GQA。已包含的算子通信不再重复计费，布局搬运单独计费。`performance_model` 的旧 CP1 stack 系数也保留，但不能误称为当前联合模型直接照搬的绝对 CP1 时间。

N 必须使用 Megatron `next_hdp_group_packing_aware`，beta=0、原 n²/CP 排序与 packing/tie-break，生成 K 与执行 K 一致。固定 N 只固定 K，仍按输入重新组批；可变 N 每个 batch 只选一个统一 K，候选可为任意合法对齐值，例如已有 K15.5Ki。连续 GDN 段及其 MLP 保持 contiguous，转换仅在 GQA/GDN 边界及模型 I/O。这个约束针对 baseline；二维保留 GQA MLP 重分布等全部自由度。

Megatron 的关键实现是 `megatron/core/mixed_context_parallel.py`、`context_parallel_layout/utils.py`、`datasets/data_schedule.py`、`packed_seq_params.py`、`transformer/attention.py`、`ssm/gated_delta_net/gdn.py`、`extensions/transformer_engine.py`、`transformer/transformer_block.py` 与 `transformer_config.py`。精确保留清单为 [MEGATRON_GIT_PATHS.txt](MEGATRON_GIT_PATHS.txt)。修改它之前读该仓库的 `AGENTS.md`。

## 新集群先做一个 CPU 接手检查

两个仓库放在同一父目录，或显式设置 `MEGATRON_ROOT`。用已有 Python 环境，从 prism-cp 根目录运行：

```bash
export PYTHON=/path/to/existing/environment/bin/python
export MEGATRON_ROOT=/shared/repos/Megatron-LM
"$PYTHON" experiments/sequence_dynamic_cp/handoff_check.py
```

该命令只读明确列出的数据和源码：核对 99 条拟合观测与原始 CSV，加载三个响应 profile，复算保留的 224Ki 计划及显存，完成一个 3:1 小规模独立求解与 replay 校验，并核对 5 份 N 原函数计划和 4 份 joint 布局。还检查交接文件是否列入 Git 路径清单；不启动 GPU，不遍历工作区，不生成快照或 hash。

需要准备迁移实验时，按 [H100_MIGRATION.md](H100_MIGRATION.md) 的 `prepare --output ...` 命令建立新目录。要搜索新的输入，使用 `joint_cp_search.py --manifest <inputs.json> --output <新目录> --world 16 --pattern 'GGG*GGG*' --model-profile <模型.json>`。结果包括评分/计划 `result.json`、`replay.json` 和 `executor.json`。它是 CPU 求解；GPU 启动需用户恢复实验。

## 接下来做什么

最强已测正例是 8×224Ki＋64×4Ki：H800 固定 N-K24 为 17.954s，可变 N-K14 为 16.089s，二维为 12.444s；二维在容量适配之后再提高约 29.3% 吞吐，allocated 增加约 3.4%。这说明二维收益并非只是选了更好的统一容量。

H100 先建立该正例的同硬件三方案对照；随后补不规则长度正例与 192Ki 选择 N 的对照。保留矩阵共有三个输入八项，但不应把旧 H800 最优 K 或排序预设为 H100 结论。后续再做固定候选集合的模型评分消融、实际在线 N/二维选择并计入决策/切换成本。当前在线选择未接入；旧 online 原型已退休，独立二维模型、求解器和冻结执行链均保留。

上轮代码精简后的 GPU 完整训练路径尚未复跑，恢复 GPU 后由首个授权项验证；CPU 检查不能替代这一点。不要重建大 sweep，不把未完成/外部占用失败的项当成有效性能数据。
