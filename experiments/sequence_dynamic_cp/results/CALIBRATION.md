# 模型校准证据：H800 历史与 H100 新增

以下 H800 历史数据用于解释既有候选和结果，主快照为 `../configs/joint_16gpu_measured.json`，均保留原样。后续 H100 在已测范围使用本页 H100 实测段及 [硬件证据索引](../configs/cp_evidence_sources.json)，未测范围不静默借用 H800 值。数据到模型/调度器的完整关系见 [接手文档](../../../progress/CODEX_HANDOFF.md)。

- [bench_data](../../../bench_data/README.md)：节点内 GQA/GDN 的 CP1/2/4/8 原始长度扫描、固定 global batch=8 扫描及吞吐/显存表。`cp_response_observations.csv` 保留当前 fit 使用的 99 条去重 F+B 观测，与原始表逐项一致。
- [analysis](../../../analysis/README.md)：早期 GDN/CSA/HCA 扩展性两表，共 20 条记录；属于算子差异的历史证据，不作为当前 GQA 响应。
- `historical_cp_response_20261012/profiles/`：节点内 GQA/GDN 的三个拟合口径；同目录保留 fit_statistics、fit_validation CSV。`summary.csv` 是历史调度候选摘要，不是原始 CP 测量。`joint_cp/response.py` 直接读取 fit。
- `cross_node_cp_evidence_20261012/`：同系列 CP8/16 的原始逐 rank 测量、配对响应和输入锚点。GQA 64Ki 的 CP8/16 完整模块 F+B 为 39.752/184.915ms，256Ki 为 468.353/712.500ms；两点均是 CP8 更快。
- `nsys_gqa_cross_node_overlap_20261012/`：只保留已有 overlap 的 summary、layer_summary、unit_coverage 与核对摘要。大 sqlite、逐 kernel CSV、图片及源代码快照已移除。
- `../../performance_model/results/cp1_20261009_p1_v2/`：CP1 stack coefficients、测量表、拟合残差、costs 与输入。保留的 `module_bench.py`、`fit_costs.py`、`cost_model.py` 可用于后续经授权的模块测量与拟合。

旧 H800 网络锚点：节点内 8 卡 1GiB/rank AllReduce 为 11.479ms / 163.70GB/s busbw，跨机 16 卡为 74.189ms / 27.14GB/s；16-rank ring 的 64/128MiB 每轮约 18.0/36.0ms。AllReduce busbw 不能替代 ring 带宽。原网络 sweep 和诊断执行脚本已退休。

历史 GDN 响应来自 32 heads，当前模型为 16 heads；CP16 GDN、packed 响应、MLP/路由及 overlap 仍有未校准部分。H100 需独立记录硬件、拓扑与响应，不能把旧模型预估当成新硬件实测。

## H100 256Ki 模块锚点（2026-10-16）

`h100_module_cp_20261016_r1/` 保存一次 16 卡启动的 GQA/GDN chunkwise × CP8/16 四点，`h100_module_cp_analysis_20261016_r1/analysis.json` 与 `.csv` 可由同目录 `analyze.py` 复算。固定单序列、H4096、BF16，GQA Q32/KV8/head128/TE p2p/zigzag，GDN16×128/conv4/chunkwise/contiguous；实际模块 F+B，不含 MLP、优化器、参数梯度归约或跨布局搬运。

|模块|CP8 两个节点各自 F+B 中位数 ms|CP16 F+B 中位数 ms|仅延迟比 T8/T16|每卡 TFLOP/s／MFU 相对 CP8|
|---|---:|---:|---:|---:|
|GQA|458.957 / 467.735|305.553|1.502–1.531×|75.10%–76.54%|
|GDN chunkwise|25.944 / 26.824|14.369|1.806–1.867×|90.28%–93.34%|

每点 2 预热＋3 计量，逐次取组内 rank-max 后统计；CP8 两个独立组各处理一条相同序列，不合并副本吞吐。小输入输出/dX 的 CP8/16 一致性检查与所有计量 finite 检查通过。GDN 实际 chunkwise cp_context 已验证，具体 FLA 子 kernel dispatch 未直接记录。耗时 372.136s，约 1.654 GPU·小时，含编译；无重试，任务已退出。

同长度旧 H800 GQA 的 T16/T8 为 1.5213，本次为 0.6533–0.6658，说明单组延迟方向反转；不据此孤立归因于 GPU 型号。旧 GDN 形状和 CP 实现不匹配，不能计算其同配置跨硬件变化。本次只提供一个长度的两个 CP 点，尚未替换 `joint_16gpu_measured.json`，不能当成跨长度/packed 输入的完整 H100 校准。详细合同与限制见 [当日归档](../../../progress/ARCHIVE-20261016.md)。

## 按卡数归一的评价规则

用户指定以每卡有效 TFLOP/s 或 MFU 判断收益。固定有效工作量和同硬件时，CP8→16 的每卡效率比为 `T8/(2*T16)`，卡数翻倍而时间没有减半不计为性能收益。因此上述 GQA/GDN 的每卡效率分别下降 23.46%–24.90% / 6.66%–9.72%，两者均无资源归一后的收益。表中百分数是相对 CP8 的比值，不是绝对 MFU。具体派生值见 `h100_module_cp_analysis_20261016_r1/efficiency_comparison.csv`，其分析脚本已将延迟比与收益判定分开。

绝对 TFLOP/s／MFU 必须声明有效 FLOP 口径和对应单卡理论峰值；现有比例无需假设这两个值。完整计划需计入所有分配 GPU、空闲等待和有效输入工作；固定卡数及相同工作时，时间缩短才等价于每卡效率提升。

## H100 模型与调度接入

独立配置为 `../configs/joint_16gpu_h100.json`，构建入口 `../build_h100_cp_profile.py --output <新配置.json>`。原 H800 配置保持不变。H100 搜索显式传 `--hardware H100`；不带硬件参数的旧解析默认模型不能称为 H100 校准模型。

CP8 用两个节点各自中位数的中位数，GQA/GDN 为 463.346039/26.383841ms；CP16 为 305.552673/14.369376ms。按已有有效工作量公式缩放到 S=ΣL、Q=ΣL²，保留各算子独立 CP 响应，GDN CP8→16 每卡效率保持率 91.81% 高于 GQA 75.82%。这不是为调度结果拟合的系数，也不表明两模块 CP16 已提高每卡效率。

单长度不足以识别固定开销及各项系数。跨长度/packing 采用显式工作量外推，CP1/2/4 假设保持 CP8 每卡效率，attention CP>16 拒绝；MLP、头、路由、更新和显存是解析假设。原模块 CP 通信不重复叠加，完整计划按所有分配 GPU 的时间评价。公式见 [模型合同](../JOINT_CP.md)，独立 raw oracle、13 候选回溯、两个新计划及 CLI 检查见 `h100_model_validation_20261016_r1/`。新计划仅有 CPU 验证，实际收益和显存准入待验证。

## CP32 新增锚点与方差复核

后续新增 `../configs/joint_32gpu_h100.json`，最高 attention CP32；原 16 卡配置仍保留上述 CP16 上限。GQA32 三次中位数为 221.343719ms。GDN32 首轮三次波动较大，预声明新增七次后合并全部十次，中位数为 10.643056ms；不剔除慢步。复核 CP16 两组中位数 14.129312/14.235648ms 单列，原 CP16 锚点不覆盖。

主分析为 `h100_module_cp32_analysis_20261016_r1/combined_analysis.json`，原始数据在 `h100_module_cp32_20261016_r1/` 和 `h100_module_cp32_gdn_recheck_20261016_r1/`。相对旧 CP16 每卡效率保持率为 GQA 69.02%、GDN 67.50%；相对 CP8 为 52.33%/61.97%。两者均无资源归一收益，GDN 在 CP8→16 的明显优势未延续到下一段。GDN 初轮慢步、复核数据和两轮范围全部保留，范围不是置信区间。

构建新配置时传 `build_h100_cp_profile.py --cp32-analysis <combined_analysis.json> --output <新配置.json>`。`joint_cp_search.py --hardware H100 --world 32` 自动选择 CP32 配置，真实 CLI/replay 检查在分析目录 `independent_cli/`。33 个 CPU 场景组合的搜索及局限见 `h100_joint_opportunity_20261016_r1/final_summary.json` 与 `dense_native_summary.json`，不能将模型预测当成完整训练收益。
