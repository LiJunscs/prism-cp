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

H800 快照为 `configs/joint_16gpu_measured.json`，部分响应仍是迁移先验。`response.py` 读取冻结节点内 fit 与跨机 CP8/16 配对，不能根据最终调度收益反调系数。CP1、GDN head 数迁移、GDN CP16/packed 响应、overlap 和争用仍有误差；H100 应建立独立配置。

## 显存与评价

实际准入要求 `required <= min(0.9*total, total-8GiB)`，required 为 allocated 峰值加边界观测到的外部占用。reserved 辅助报告，不能与 allocated 相加。H800 的共同预算为 71.178833GiB；H100 按实际总显存应用同一规则。

可选显存交换偏好：共同固定参考下，吞吐增长 g=T_ref/T−1，显存增长 m=M/M_ref−1；m>0 时要求 g≥αm，当前 α=2。N/二维使用相同参考，物理预算始终独立。`memory_performance_policy.py` 保留此对称规则及时间/显存边界函数。

计时取每步全 rank 最大完整更新时间；显存取整个运行所有 rank 的最大 allocated。新最小实验每项 5 更新、2 预热＋3 计量。预算拒绝的预热步不能作速度结果。实际在线 N/二维选择、决策/切换成本未实现；目前支持冻结回放与离线选择。

最新结果与实验状态见 [归档](../../progress/ARCHIVE-20261016.md)，避免多处重复记录。GPU 暂停，不做全工作区扫描、hash、快照、环境盘点、GPU 预检或重复 profiling。
