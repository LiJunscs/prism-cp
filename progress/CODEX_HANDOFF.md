# 新集群 Codex 接手：数据 → 模型 → 调度 → 执行

研究目标是 GDN:GQA=3:1 模型的序列与层联合动态 CP。比较三个方案：全局固定统一容量的原生 N、每个 global batch 可变统一容量的原生 N，以及我们的二维调度器。batch 内和 batch 间长度分布都允许随机；当前证据是冻结计划短运行。两个仓库已迁入 H100 集群，用户已明确恢复 GPU 实验。首轮被 FLA/Triton 兼容性检查拦截；用户安装 TileLang 后第二轮已完成：14 次启动、12 项有效、2 项显存 guard 拒绝，62 个完整更新。三个输入的当前冻结二维计划均慢于已测 N，且各有更快、显存更低的 N 候选。四节点该轮进程均已退出。随后用户另授权的 16 卡模块实验也已完成：256Ki 下 CP8→16 的 GQA/GDN 单组延迟分别缩短至原来的约 65%–67% / 54%–55%，但每卡有效 TFLOP/s／MFU 分别降至 CP8 的 75.10%–76.54% / 90.28%–93.34%，均不标记为性能收益；两节点任务进程均已退出。用户仍要求环境问题立即报告、不得自行修复；结果以归档为准。

最新补充：四节点 CP32 及 GDN 方差复核完成，最终 CP32 GQA/GDN 为 221.344/10.643ms，CP16→32 每卡效率保持率为 69.02%/67.50%；四节点已于 23:45 清理。CPU 搜索 33 个输入/卡数组合，最好的完整 N 候选比较仅约 +1.08% 预测效率，尚无可靠的大幅二维收益。详见归档的新 CP32 与场景搜索段。

先读 [当前事实与结果](ARCHIVE-20261016.md)，再读本页实现地图和 [执行合同](../experiments/sequence_dynamic_cp/JOINT_CP.md)。下一步由 [roadmap](HETEROGENEOUS_CP_MASTER_ROADMAP_20260918.md) 定义，路径和 Git 范围见 [H100_MIGRATION.md](H100_MIGRATION.md)。

本次 prism-cp 迁移提交配套 Megatron-LM `ce909a8ca5d9e1d7c1ada33e7ec6d1adc7e08312`（`prism-cp` 分支）。两个仓库均需带到新集群；本次只创建本地提交，尚未 push。

## 数据与性能口径

H800 原始表、fit、模型与完整训练证据继续保留。后续 H100 按 [硬件证据索引](../experiments/sequence_dynamic_cp/configs/cp_evidence_sources.json) 使用独立配置 `configs/joint_16gpu_h100.json`（CP8/16）或 `joint_32gpu_h100.json`（新增 CP32）。`--hardware H100` 按 world 选择配置，未测范围明确标为未校准；不覆盖 H800。H100 的 256Ki、B=1、CP8/16/32 模块锚点已接入求解器；其他形状/CP 的外推和其余训练项仍未校准，不能称为完整实测模型。

收益主指标是按分配 GPU 数归一的每卡有效 TFLOP/s 或 MFU。相同有效工作量和硬件下，效率比为 `(p_old*T_old)/(p_new*T_new)`；卡数翻倍但时间未减半就是效率下降。延迟比仅辅助报告，不能据此宣称收益。完整计划计入所有分配卡及等待；固定相同卡数、输入与训练工作时，时间比才等价于每卡效率比。

## 必须带走的数据

下面路径从 prism-cp 根目录起算。原始小表和压缩完整训练记录均应提交；不能仅迁移结论或拟合系数。

|证据|路径|连接到后续工作|
|---|---|---|
|早期 GDN/CSA/HCA CP1/2/4/8 扩展性|`analysis/gdn_csa_hca_scaling.csv`、`analysis/csa_hca(csa128)_scaling.csv`|解释算子 CP 响应差异；[数据说明](../analysis/README.md)。|
|GQA/GDN 节点内长度扫描和固定 batch 扫描|`bench_data/bench_merged_cp{1,2,4,8}.csv`、`fixed_overhead.csv`、`bench_mfu_mem_log.csv`|F/B/F+B、吞吐、显存原始表；[来源和口径](../bench_data/README.md)。|
|二维响应使用的原始观测、fit 与误差|`bench_data/cp_response_observations.csv`；`experiments/sequence_dynamic_cp/results/historical_cp_response_20261012/`|99 条源观测与三个 profile（joint/single_b1/fixed_g8）。|
|跨节点 GQA CP8/16 配对|`experiments/sequence_dynamic_cp/results/cross_node_cp_evidence_20261012/`|逐 rank 原始值、配对响应、64/256Ki 锚点；供 CP16 响应政策使用。|
|H100 专属模块 CP 实测|`experiments/sequence_dynamic_cp/results/h100_module_cp_20261016_r1/`、`h100_module_cp_analysis_20261016_r1/`；`configs/cp_evidence_sources.json` 相对 sequence_dynamic_cp|原始逐 rank 记录、协议、时间与每卡效率复算；后续 H100 已测范围的依据，与 H800 分开保留。|
|H100 CP32 与方差复核|`experiments/sequence_dynamic_cp/results/h100_module_cp32_20261016_r1/`、`h100_module_cp32_gdn_recheck_20261016_r1/`、`h100_module_cp32_analysis_20261016_r1/`|两轮全部原始记录、3+7 合并规则、原 CP16 对照及清理证据；不可只保留新中位数或删除初轮慢步。|
|已有通信 overlap 证据|`experiments/sequence_dynamic_cp/results/nsys_gqa_cross_node_overlap_20261012/`|summary、layer_summary、unit_coverage；已删除大 profiling 数据库。|
|CP1 模块/stack 校准|`experiments/performance_model/results/cp1_20261009_p1_v2/`|measurements、inputs、coefficients、costs、residuals；`module_bench.py`、`fit_costs.py`、`cost_model.py` 仍在。|
|完整训练结果|`experiments/sequence_dynamic_cp/results/` 下保留的 N、joint、容量消融、显存修复及随机 batch 实验目录|输入、计划、协议、每步时间/显存、拒绝或失败状态；逐 rank 记录压缩为 `steps_by_rank.json.gz`，由 `results_io.py` 读取。|

[CALIBRATION.md](../experiments/sequence_dynamic_cp/results/CALIBRATION.md) 补充历史拓扑与测量口径。CSV 与 fit 都保留，不能把 `historical_cp_response.../summary.csv` 误认作原始扩展性数据。历史 GDN 为 32 heads，当前训练为 16 heads；旧响应是迁移先验；H100 已新增 256Ki 的 CP8/16 同硬件锚点，完整响应配置仍需跨长度与 packed 输入校准。

## 实现地图

以下模块均位于 `experiments/sequence_dynamic_cp/`，Megatron 路径另列。

|环节|入口|负责什么|
|---|---|---|
|输入|`joint_cp/inputs.py`、`two_dimensional_inputs.py`、`two_dimensional_packet.py`|显式长度 manifest、稳定 sample ID、确定性 token/label/mask、packet 供数。|
|计划表示|`joint_cp/ir.py`|Group/Layout/Microbatch/Plan、rank/sample 归属、布局及完整输入覆盖校验。|
|时间与显存模型|`joint_cp/model.py`、`response.py`、`analysis.py`|算子 CP 响应、拓扑搬运、组间最大耗时、跨 F/B 存活显存、整步成本与分项解释。|
|H100 响应配置|`build_h100_cp_profile.py`、`configs/joint_16gpu_h100.json`、`configs/joint_32gpu_h100.json`|从 H100 CP8/16/32 模块数据构建独立配置，分别建模 GQA/GDN；明确标记外推和未校准项，保留 H800。|
|独立二维调度器|`joint_cp/solver.py`；命令 `joint_cp_search.py`|从完整输入池选微批、packing、各层 CP、rank 组和 MLP token 放置；不依赖 N 的计划。exact 与 bounded 均保留。|
|大输入搜索和显存选择|`joint_cp_scale_search.py`、`joint_cp_scale_audit.py`、`joint_cp/memory_selection.py`|微批库/DP、CP1 短批重打包，保留完整执行计划。|
|固定/可变容量 N 对照|`native.py`、`native_n_generalization_theory.py`、`joint_advantage_search.py`、`random_batch_capacity_analysis.py`|调用 Megatron 原函数生成 N，逐输入与跨输入分布选择统一 K；有限候选比较和对称显存偏好。|
|计划回放与训练|`joint_cp/replay.py`、`adapter.py`、`transport.py`；`runtime.py`、`gpu_smoke.py`|冻结计划编译，层边界与 MLP 搬运，输入到优化器更新的完整训练。|
|实验准备与启动|`random_batch_capacity_gpu.py`、根目录 `scripts/run_manual_gdn_cp_case.sh`、`scripts/torchrun_launcher.py`|重建保留矩阵；显式指定单项启动；主机、Python、输出路径可配置。|

组内 S=ΣL、Q=ΣL²：GQA 计算项随 Q 和 S，GDN/MLP 随 S；各阶段取并行组最大时间，累加微批、路由、输出头和更新成本。显存模型保存跨层 F/B 激活，同时计入模型状态、workspace、搬运暂存及余量。`analysis.measure` 可对任意合法计划复算，`analysis.describe` 给出阶段归因。

保留的 H800 联合模型配置为 `configs/joint_16gpu_measured.json`；它没有被本次 H100 数据覆盖，不能当成已校准的 H100 配置。CP1 计算主体仍是解析 FLOP/吞吐系数；`response.py` 将历史 CP 响应转为有效残差，并用 CP8/16 配对处理跨机 GQA。已包含的算子通信不再重复计费，布局搬运单独计费。`performance_model` 的旧 CP1 stack 系数也保留，但不能误称为当前联合模型直接照搬的绝对 CP1 时间。

H100 分支采用 `T_o(S,Q,p)=A_o,p*W_o(S,Q)/W_o(256Ki,(256Ki)²)`，GQA 工作量同时含 S/Q、GDN 含 S。CP8 取两个节点中位数的中位数，GQA/GDN 为 463.346039/26.383841ms，CP16 为 305.552673/14.369376ms，新增 CP32 为 221.343719/10.643056ms。CP8→16 每卡效率保持率为 75.82%/91.81%，CP16→32 为 69.02%/67.50%；GDN 的前一段优势没有延续到后一段。CP1/2/4 显式按 CP8 每卡效率外推，attention 上限由配置中的实测 CP 域决定，最高 CP32；MLP/头/路由/更新/显存保留解析假设。完整公式与约束见 [JOINT_CP.md](../experiments/sequence_dynamic_cp/JOINT_CP.md)。调度目标计入全部分配卡，输出预测 GPU·ms 与每卡有效 TFLOP/s；不强制某类层选择特定 CP。

CP32 用户另行授权，窗口延长至 23:50。一次 GQA/GDN CP32 实验及一次仅 GDN 的方差复核共 5.231 GPU·小时，已于 23:45 清理完全部四节点。GDN CP32 采用两轮全部 3+7 次中位数，慢步保留；旧 CP16 不覆盖。证据为 `results/h100_module_cp32_20261016_r1/`、`h100_module_cp32_gdn_recheck_20261016_r1/`，主统计 `h100_module_cp32_analysis_20261016_r1/combined_analysis.json`，其中路径均相对 sequence_dynamic_cp。32 项测试及 CP32 真实 CLI/replay 检查通过；只验证了模块与 CPU 计划，未验证新的完整训练收益。

N 必须使用 Megatron `next_hdp_group_packing_aware`，beta=0、原 n²/CP 排序与 packing/tie-break，生成 K 与执行 K 一致。固定 N 只固定 K，仍按输入重新组批；可变 N 每个 batch 只选一个统一 K，候选可为任意合法对齐值，例如已有 K15.5Ki。连续 GDN 段及其 MLP 保持 contiguous，转换仅在 GQA/GDN 边界及模型 I/O。这个约束针对 baseline；二维保留 GQA MLP 重分布等全部自由度。

Megatron 的关键实现是 `megatron/core/mixed_context_parallel.py`、`context_parallel_layout/utils.py`、`datasets/data_schedule.py`、`packed_seq_params.py`、`transformer/attention.py`、`ssm/gated_delta_net/gdn.py`、`extensions/transformer_engine.py`、`transformer/transformer_block.py` 与 `transformer_config.py`。精确保留清单为 [MEGATRON_GIT_PATHS.txt](MEGATRON_GIT_PATHS.txt)。修改它之前读该仓库的 `AGENTS.md`。

## 新集群先做一个 CPU 接手检查

两个仓库放在同一父目录，或显式设置 `MEGATRON_ROOT`。用已有 Python 环境，从 prism-cp 根目录运行：

```bash
export PYTHON=/path/to/existing/environment/bin/python
export MEGATRON_ROOT=/shared/repos/Megatron-LM
"$PYTHON" experiments/sequence_dynamic_cp/handoff_check.py
```

该命令只读明确列出的数据和源码：核对 99 条拟合观测与原始 CSV，加载三个响应 profile，复算保留的 224Ki 计划及显存，完成一个 3:1 小规模独立求解与 replay 校验，并核对 5 份 N 原函数计划和 4 份 joint 布局。还检查交接文件是否列入 Git 路径清单；不启动 GPU，不遍历工作区，不生成快照或 hash。本轮 H100 接手已通过该 CPU 检查；首轮实际训练的依赖阻塞经用户安装 TileLang 后解除，第二轮完整训练合同已通过。

需要准备迁移实验时，按 [H100_MIGRATION.md](H100_MIGRATION.md) 的 `prepare --output ...` 命令建立新目录。H100 新输入使用 `joint_cp_search.py --manifest <inputs.json> --output <新目录> --world 16 --pattern 'GGG*GGG*' --hardware H100`；显式 `--model-profile <模型.json>` 是另一种互斥选择。结果包括评分/计划 `result.json`、`replay.json` 和 `executor.json`。它是 CPU 求解，结果带硬件配置与外推声明；不指定硬件仍为旧解析默认值，不能当成 H100 校准结果。真实 CLI 已通过小输入检查，证据在 `results/h100_model_validation_20261016_r1/cli_validation.json`。

## 接下来做什么

H800 已测 224Ki 正例仍成立：固定 N-K24 为 17.954s，可变 N-K14 为 16.089s，二维为 12.444s。但这份冻结计划没有迁移收益：H100 同输入 N-K14 为 6.519s、二维为 10.021s；不规则输入 N-K16 为 6.557s、二维为 9.612s；192Ki N-K24 为 4.815s、二维为 8.081s，较低显存的 N-K12 也只需 5.445s。全部比较在同一输入、同一主机对内进行，不能混用两种硬件的速度。

224Ki 同一 N-K14 计划经过 joint executor 为 6.601s，仅增加约 1.26% 时间；原二维计划的约 53.72% 时间增量不能主要归因于通用执行器固定开销。该控制单列，不作原生 N baseline。旧响应模型在 224Ki 和不规则输入选错；纯计算量在最终候选池中选对这两组，但在 192Ki 选择 N-K12，比已测最快 N-K24 慢约 13.09%。这支持优先修正模型迁移，不支持直接放弃响应模型或二维搜索空间。

后续模块实测已直接确认 GQA 的单组延迟方向反转：H800 256Ki 的 CP8/16 为 468.353/712.500ms，H100 为 458.957–467.735/305.553ms；H100 GDN chunkwise 为 25.944–26.824/14.369ms。但卡数翻倍后两者每卡效率均下降，均没有资源归一后的性能收益。历史 GDN 形状和实现不匹配，不能断言其跨硬件变化；全部软件/拓扑也未隔离为单一硬件因素。原始数据在 `results/h100_module_cp_20261016_r1/`，独立复算在 `results/h100_module_cp_analysis_20261016_r1/`，详见归档。

H100 新配置已完成 13 个冻结候选重评分和两个小输入的独立求解/replay。224Ki、192Ki 在模型可行候选中选中已测最快 N；不规则输入仍错误放行实际 guard 拒绝的 K24，实际有效的 K16/K20 排序也有误差。此次仅更新 CPU 模型与调度入口，未启动 GPU；不能将新计划预测称为性能收益。下一步补足跨长度、packed 输入和执行成本的独立校准并保留验证输入，再以同一候选集合、执行器与实际显存合同验证新计划。N 的容量候选需明确声明。随后才扩展实际在线 N/二维选择并计入决策/切换成本；当前在线选择未接入。

后续 CP32 数据已接入上述 32 卡配置，场景搜索在 `results/h100_joint_opportunity_20261016_r1/final_summary.json`。18 个 16 卡及 15 个 32 卡输入组合均保留正负结果；三个弱正例另经 32-token 步长 N 容量扫描。当前线索是 16 卡不规则中长输入，少数微批 GDN CP16 合并、GQA 两个 CP8 拆分，完整计划预测增益约 1.08% 或 0.63%，并降低模型峰值。32 卡未找到有意义且对照完整的优势；一个 +1.85% 候选有未评分的 N 容量，不能作正例。优先核对 packed 外推与完整显存误差，暂不把这些小幅预测直接作为 GPU 性能结论。

本轮证据入口为 `experiments/sequence_dynamic_cp/results/h100_capacity_analysis_20261016_r2/measured_comparison.json`（13 项主比较）与 `executor_control_comparison.json`（独立控制）；逐项运行计数及四节点退出证据分别为 `results/h100_capacity_20261016_r2_execution_summary.json`、`results/h100_capacity_20261016_r2_cleanup_verification.json`。H100 原始 JSONL 和日志全部保留，未自动加入迁移 Git 清单。原生 N 与二维生产执行源码均未修改；不要把首轮依赖失败或第二轮预算拒绝当成有效性能数据。
