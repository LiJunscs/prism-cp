# 联合 dynamic CP 实现与模型合同

新集群先读 [Codex 接手地图](../../progress/CODEX_HANDOFF.md)，其中列出 CP 原始数据、模型、调度器与训练执行的对应关系及 CPU 检查命令。

## 执行语义

当前模型 GDN:GQA=3:1，实验采用 8 层 `GGG*GGG*`。N 使用 Megatron 原始 `next_hdp_group_packing_aware`，beta=0；CPU 从该函数 AST 冻结计划，生成与执行使用同一个统一 K，执行采用 DUPC，不安装 joint adapter。固定 K 仍根据完整输入重新调度，可变 N 允许逐 global batch 选择任意合法对齐 K。

N 的连续 GDN 段连同 MLP 保持 contiguous，边界按需转为 GQA zigzag。二维独立选择微批、packing、分层 CP、rank/sample 归属及 token 放置。MLP 是 token 放置决策，包含其前向和反向搬运。相同布局跳过恒等路由；当前已测计划没有 GDN 段内往返，GQA MLP 可以另行重分布。

`joint_cp/adapter.py`、`transport.py` 复用 Megatron mixed GQA/GDN 内核及原有 residual/layer forward；输出头前恢复 ingress 归属，保持 labels/mask 对齐。合同为 dense MLP、TP=PP=1、dropout=0、无重计算，CP 组提前合法建立。

`runtime.py` 只保留原生 N 与 joint 冻结回放。完整更新包含输入、输出头与 loss、F/B、gradient finalize、norm/clip、非零 Adam、同步更新后的参数。首步 finalize 前仅释放一次空闲 allocator 缓存，两侧相同；不以空 optimizer 或跳过更新替代完整训练。

## 模型与求解

组内 S=ΣLᵢ、Q=ΣLᵢ²：

```text
T_GQA(p) = (aQ+bS)/p + R_GQA(p, input, placement, topology)
T_GDN(p) = cS/p + R_GDN(p, input, placement, topology)
T_plan = sum_microbatch sum_segment max_group(T_group) + T_routes + T_shared
```

R 包含对应算子的有效 CP 响应，已计入通信不重复叠加。MLP 按本地 token 工作量计费，路由计入 F/B。完整显存包络覆盖模型状态、跨 F/B 保存激活、workspace、重分布暂存及余量。局部单层峰值和总 tokens 都不能替代同时存活量。

`joint_cp/solver.py` 提供 exact 与 bounded；exact 仅在声明有限域内穷尽且未触发预算时有最优证书。`joint_cp_scale_audit.py` 的 DP 仅在给定微批库内最优。二维从完整输入池独立求解，不以 N 的 GQA 计划为约束；搜索失败不能伪装成原生 N。

`memory_selection.py` 对等长、平衡、rank 归属不变的 CP1 短批有限重打包，允许预测整步时间增加至多 0.1%，优先降低峰值；exact 保持原目标。568 条的低显存收益已 GPU 验证，1136 条修复仅有 CPU 推导。

H800 快照为 `configs/joint_16gpu_measured.json`，部分响应仍是迁移先验；原始数据与系数保持不变。其 `response.py` 分支读取冻结节点内 fit 与跨机 CP8/16 配对，不能根据最终调度收益反调系数。

H100 使用独立的 `configs/joint_16gpu_h100.json`（CP8/16）与 `joint_32gpu_h100.json`（另含实测 CP32），由 `build_h100_cp_profile.py` 从模块实测生成。令 L₀=262144，h 为 hidden、d 为 head_dim、D 为 GDN heads×head_dim；沿用现有有效 F+B FLOP 口径：

```text
W_GQA(S,Q) = 6[h(q_heads+2*kv_heads)d+h²]S + 6*q_heads*d*Q
W_GDN(S,Q) = 6[6hD+4*gdn_heads*gdn_dim²]S
T_o(S,Q,p) = A_o,p * W_o(S,Q)/W_o(L₀,L₀²)
C_o,p = W_o(L₀,L₀²)/(p*A_o,p)   # 有效工作量 / GPU·ms
T_o(S,Q,p) = W_o(S,Q)/(p*C_o,p)
```

|算子|A₈，ms|A₁₆，ms|A₃₂，ms|C₁₆/C₈|C₃₂/C₁₆|
|---|---:|---:|---:|---:|---:|
|GQA|463.346039|305.552673|221.343719|75.82098%|69.02221%|
|GDN chunkwise|26.383841|14.369376|10.643056|91.80580%|67.50588%|

CP8 系数取两个节点各自 rank-max 迭代中位数的中位数，CP16 取跨节点 rank-max 迭代中位数；两个节点的原始值仍单列保留。CP8 聚合是统一节点模型，不等于并行执行两组时实际测得的全 rank 最大值。GDN 的 CP8→16 效率保持率优于 GQA，但两者均低于 100%。单长度观测不能分别拟合线性、平方和固定开销，以上工作量缩放是显式假设。

CP32 GQA 取首轮 3 次计量中位数；GDN 首轮波动明显，经预声明 7 次复核后合并全部 10 次取中位数，不删慢步。CP16 旧锚点不变；复核的两个 CP16 组为 14.129312/14.235648ms，单列作敏感性依据。CP8→16 的 GDN 优势不能外推到 CP16→32：后一段两者效率均下降，按当前统计 GDN 略低。原始与合并统计见 `results/h100_module_cp32_analysis_20261016_r1/combined_analysis.json`。

单条 256Ki 以外的长度和 packing 标为 shape extrapolation；节点内 CP1/2/4 使用 `A_o,p=A_o,8*8/p` 的未验证假设，不借用 H800。attention 上限随所选配置为 CP16 或 CP32，CP>32 拒绝。MLP、输出头、布局搬运、更新与显存仍为未校准解析项。模块锚点已含其 CP 通信，不再叠加一次解析通信成本；布局间搬运独立计费。

调度仍联合选择组批、attention CP 与 MLP 放置，不按效率排序硬编码 GDN CP16/GQA CP8。每阶段取并行组最大时间，计入路由、共享训练项与完整显存；固定输入和分配 GPU 数 W 时最小化 `W*T_plan` 与最小化完整步时等价。输出 `resource_metrics` 包含所有分配卡的 GPU·ms、声明口径下预测的每卡 TFLOP/s，并将绝对 MFU 留空。跨卡数比较需比较 GPU·时间，不能只取较短步时。

CPU 搜索入口：`joint_cp_search.py --hardware H100 --manifest <inputs.json> --output <新目录> --world 16 --pattern 'GGG*GGG*'`；world=32 自动选择新的 CP32 配置。也可显式传 `--model-profile <配置.json>`，二者互斥。大输入入口 `joint_cp_scale_search.py --hardware H100` 需显式指定 `--scenarios 16gpu`、`32gpu_measured` 或 `32gpu_cp16_only`；最后一个为同一 CP32 实测模型下禁用 attention CP32 的控制，完整训练与 packed 并发仍未实测。结果携带锚点来源、外推覆盖统计和 `gpu_admitted=false`，CPU 可行性不能替代实际训练准入。

## 显存与评价

实际准入要求 `required <= min(0.9*total, total-8GiB)`，required 为 allocated 峰值加边界观测到的外部占用。reserved 辅助报告，不能与 allocated 相加。H800 的共同预算为 71.178833GiB；H100 按实际总显存应用同一规则。

可选显存交换偏好：共同固定参考下，吞吐增长 g=T_ref/T−1，显存增长 m=M/M_ref−1；m>0 时要求 g≥αm，当前 α=2。N/二维使用相同参考，物理预算始终独立。`memory_performance_policy.py` 保留此对称规则及时间/显存边界函数。

计时取每步全 rank 最大完整更新时间；显存取整个运行所有 rank 的最大 allocated。新最小实验每项 5 更新、2 预热＋3 计量。预算拒绝的预热步不能作速度结果。实际在线 N/二维选择、决策/切换成本未实现；目前支持冻结回放与离线选择。

最新结果与实验状态见 [归档](../../progress/ARCHIVE-20261016.md)，避免多处重复记录。GPU 暂停，不做全工作区扫描、hash、快照、环境盘点、GPU 预检或重复 profiling。
