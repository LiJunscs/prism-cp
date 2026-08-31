# 知识库主题：TMF CP Transition 理论模型与 Attention-only 1F1B 原型计划

> 日期：2026-08-29
> 更新：2026-08-31
> 状态：当前 benchmark correctness 已在冻结 BF16 范数门控内关闭；进入同版本性能桥接与 fixed-CP 1F1B reference 验证
> 最新路线：`TMF_PROFILING_DATA_AUDIT_AND_1F1B_ROADMAP_20260831.md`
> Agent 计划：`REMOTE_AGENT_TMF_1F1B_AND_MODEL_PLAN_20260831.md`

## 1. 为什么先做这两项

现有 `agent_run/20260822T100108Z_tmf_validation_profile/` 已包含 correctness、性能和 profiling 数据，但数据来自旧 commit/dirty patch。审计确认其 2,800 个逐迭代样本足以作为趋势先验和 regression band，不能直接作为当前实现或 1F1B 的最终性能证据。当前工作是：

1. 用闭式模型回答直接 CP transition 与分级 transition 的通信量、消息数和延迟权衡；
2. 建立最小 non-interleaved 1F1B 执行骨架，先只包含固定 CP 的 Attention stage；
3. 先做当前 commit 的小规模桥接复测和 GPipe-vs-1F1B 数值对照，再把 bundle boundary 插入 stage-local mesh；
4. 用逐 rank 事件图拟合性能模型，显式建模 cohort wait、PP P2P、transition、activation live-set 和 recompute。

## 2. CP transition 通信模型

> **2026-08-29 口径修正：** 后续论文比较必须使用完整布局路径
> `8→1→8→1→8→1` 与 `8→4→1→4→8`，不能只比较孤立的
> `8→1` 和 `8→4→1`。完整推导、逐边界表格、前向/反向总量及复现命令见
> `TMF_CP_TRANSITION_COMMUNICATION_ANALYSIS_20260829.md`。本节 2.1–2.2
> 仅保留为单次单调 transition 的局部引理，不再作为两种模型计划的最终对比。

设：

- `W`：GPU 数；
- `C_s -> C_d`：源/目标 CP，且二者嵌套；
- `r = max(C_s,C_d) / min(C_s,C_d)`；
- `U = S * B * H * bytes_per_element`：一份 unique global activation 的字节数；
- DP 与 CP 的乘积为 `W`，普通 redistribution 前后每 rank tensor 元素数均为 `U/W`。

### 2.1 普通 redistribution / bundle remesh

单次嵌套 transition 的聚合 off-rank payload：

```text
V_redistribute(r) = U * (1 - 1/r)
```

每 rank 的 off-rank peer message 数为 `r-1`，每条消息大小为 `U/(W*r)`；self-copy 不计网络流量。

若把总倍率 `R` 分解为 `r_1 * ... * r_k`，中间状态不复制，故：

```text
V_staged = U * sum_i(1 - 1/r_i)
V_direct = U * (1 - 1/R)
```

所以在同一带宽域中，普通 redistribution 的分级路径总字节数严格不小于直接路径。

示例：

```text
8 -> 1:       7/8 U = 0.875 U
8 -> 4 -> 1:  1/2 U + 3/4 U = 1.25 U
```

分级路径字节数高 `42.86%`，但每 rank off-rank message 数从 `7` 降为 `1+3=4`。因此小 tensor 或高启动延迟网络中，分级路径仍可能因较少 peer fan-out 获益；大 tensor、带宽主导时直接路径更优。

### 2.2 spatial repeat

CP 收缩时 physical replicas 增长。第 `i` 级之前已有复制倍率 `R_{i-1}`，该级网络 payload 为：

```text
V_repeat_i = U * R_{i-1} * (r_i - 1)
```

总量望远镜化简为：

```text
V_repeat_staged = U * (R - 1) = V_repeat_direct
```

因此：

```text
repeat 8 -> 1:       7 U
repeat 8 -> 4 -> 1:  1 U + 6 U = 7 U
```

分级与直接路径的理论总 payload 相同，但阶段数、消息粒度、临时 buffer、同步点与链路竞争不同。反向 mean-merge 具有相同的总量结论。

### 2.3 α–β 模型

基础串行 peer 模型：

```text
T = phases * barrier_or_launch_cost
  + messages * alpha
  + off_rank_bytes / effective_bandwidth
  + pack_unpack_cost
```

实际实现使用 batched P2P，`messages * alpha` 是上界近似；应增加并发修正系数和拓扑系数。单节点 NVSwitch 与多节点 IB/NVLink 混合域必须分别拟合，不能共用一组参数。

实现：`model_cp_transition.py` 可输出任意 CP path 的理论 payload、消息数和 α–β 估计。

## 3. Attention-only 1F1B v0

### 3.1 范围

- non-interleaved 1F1B；
- `world_size = pipeline_parallel_size * context_parallel_size`；
- 每个 PP stage 包含相同数量的 Attention 模块；
- 所有 stage 使用同一固定 CP，stage 间按相同 `cp_rank` 一一传递 activation/gradient；
- 不包含 GDN、异构 CP、bundle、repeat、optimizer、DP 或 recompute。

### 3.2 调度

stage `p` 的 warmup microbatches：

```text
warmup = min(P - p - 1, M)
```

之后进入 steady 1F1B：发送当前 forward 的同时接收最老 microbatch 的 output gradient，执行该 microbatch backward；中间 stage 再同时发送 input gradient 和接收下一个 forward activation。最后执行剩余 cooldown backward。

### 3.3 v0 验收

1. `M >= P` 时所有 rank 无死锁完成；
2. 每个 stage 恰好执行 `M` 次 forward 和 `M` 次 backward；
3. stage 间 tensor shape 始终为 `[S/CP, micro_batch, H]`；
4. loss、input gradient 和参数 gradient 均 finite；
5. 两次独立 launch 调度事件数量一致；
6. 输出 step latency、理论 bubble 比例和每 stage 事件序列；
7. v0 只证明 PP 骨架可运行，不证明 TMF correctness。

### 3.4 首轮命令

待可用 Megatron 环境中先跑 PP2×CP4：

```bash
torchrun --standalone --nproc_per_node=8 bench_1f1b_attention.py \
  --seq-length 4096 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --num-query-groups 8 \
  --attention-backend flash \
  --pipeline-parallel-size 2 \
  --context-parallel-size 4 \
  --micro-batches 8 \
  --micro-batch-size 1 \
  --layers-per-stage 2 \
  --trace-schedule
```

随后跑 PP4×CP2，以增加 warmup/cooldown 深度：

```bash
torchrun --standalone --nproc_per_node=8 bench_1f1b_attention.py \
  --seq-length 4096 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --num-query-groups 8 \
  --attention-backend flash \
  --pipeline-parallel-size 4 \
  --context-parallel-size 2 \
  --micro-batches 8 \
  --micro-batch-size 1 \
  --layers-per-stage 2 \
  --trace-schedule
```

正式 profiling 前增加 warmup/重复 step；v0 当前只执行一个 optimizer-step-shaped schedule，用于调度与通信闭环验证。

## 4. 后续插入 TMF 的接口

v1 在 Attention-only stage 内先加入显式 boundary hook，但仍保持固定 CP。v2 才允许某个 stage 内出现：

```text
Attention(high CP) -> TMF boundary -> low-CP block -> inverse boundary
```

需要额外维护：

- `(pipeline_microbatch_id, temporal_bundle_id, logical_sample_id)`；
- forward/backward 使用同一 layout plan；
- bundle rendezvous 不跨 optimizer step；
- PP activation queue 中保存 final CP 与 replication metadata；
- loss normalization 只由 logical samples 和 terminal replication 决定。

## 5. 工作顺序

```text
M0 通信闭式模型
 -> M1 模型脚本自检
 -> R0 当前版本 bundle/repeat 桥接复测
 -> P0 Attention-only GPipe 对照
 -> P1 fixed-CP 1F1B correctness
 -> P2 fixed-CP 逐 rank schedule/event profiling
 -> P3 stage-local identity boundary
 -> P4 bundle-aware 1F1B correctness
 -> P5 bundle-aware 1F1B profiling
 -> P6 事件性能模型拟合与 holdout
 -> P7 checkpoint/recompute 与 planner
```

不无差别重复历史矩阵，也不在不同 commit 之间混合绝对 latency。历史逐模块 profiling 只作 prior；当前模型的训练和验证数据按 `TMF_PROFILING_DATA_AUDIT_AND_1F1B_ROADMAP_20260831.md` 重新采集。
